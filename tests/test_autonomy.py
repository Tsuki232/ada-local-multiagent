"""Pruebas del agente autonomo real: saneado, guardas, curriculo, curacion y compuerta de entrenamiento."""

from __future__ import annotations

import json
import os
import shutil
import uuid

import pytest

from ada.autonomy import agent as agent_mod
from ada.autonomy import curator as curator_mod
from ada.autonomy import guard as guard_mod
from ada.autonomy.safe_expr import sanitize_expression
from ada.config import DATA_WORK_DIR
from ada.training import continuous_learner as cl


@pytest.fixture()
def sandbox(monkeypatch):
    """Redirige todo el estado autonomo a un subdirectorio temporal DENTRO de data/work (lo exige la barrera de rutas)."""
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    auto, train, mem = root / "autonomy", root / "training", root / "memory"
    for d in (auto, train, mem):
        d.mkdir(parents=True)
    for mod in (agent_mod, guard_mod):
        monkeypatch.setattr(mod, "AUTONOMY_DIR", auto)
    monkeypatch.setattr(agent_mod, "STATE_FILE", auto / "state.json")
    monkeypatch.setattr(agent_mod, "JOURNAL_FILE", auto / "journal.jsonl")
    monkeypatch.setattr(guard_mod, "STOP_FILE", auto / "STOP")
    monkeypatch.setattr(guard_mod, "PID_FILE", auto / "agent.pid")
    monkeypatch.setattr(guard_mod, "TRAINING_LOCK", auto / "training.lock")
    monkeypatch.setattr(cl, "MEMORY_DIR", mem)
    monkeypatch.setattr(cl, "TRAINING_DATA_DIR", train)
    monkeypatch.setattr(cl, "EPISODIC_MEMORY_FILE", mem / "episodic.jsonl")
    monkeypatch.setattr(cl, "CONTINUAL_DATASET_FILE", train / "dataset.jsonl")
    monkeypatch.setattr(curator_mod, "CURATED_TRAIN_FILE", train / "curated_train.jsonl")
    monkeypatch.setattr(curator_mod, "CURATED_EVAL_FILE", train / "curated_eval.jsonl")
    monkeypatch.setattr(curator_mod, "CURATION_REPORT_FILE", train / "curation_report.json")
    monkeypatch.setattr(curator_mod, "CONTINUAL_DATASET_FILE", train / "dataset.jsonl")
    yield root
    shutil.rmtree(root, ignore_errors=True)


class AlwaysAllow:
    def should_run(self):
        return True, "ok"

    def note_cycle(self):
        pass


# ----------------------------------------------------------------------------- saneado


@pytest.mark.parametrize(
    "expr",
    ["3*x**2 + 5*sin(x)", "x^3 - 2*x", "exp(x)*cos(x)", "sqrt(x + 1)/(x**2 + 1)", "log(x + 2)"],
)
def test_sanitize_accepts_normal_math(expr):
    assert sanitize_expression(expr, {"x"})


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('calc')",
        "x.__class__",
        "(lambda: 1)()",
        "open('data/raw/x')",
        "x**99",
        "x^13",
        "sin",  # funcion sin parentesis
        "y + x",  # variable no declarada
        "((x)",  # desbalanceado
        "x; import os",
        "a" * 200,
        "",
        None,
        "1e5*x",  # 'e' no es identificador permitido
        "(" * 9 + "x" + ")" * 9,
    ],
)
def test_sanitize_rejects_dangerous_or_malformed(expr):
    with pytest.raises(ValueError):
        sanitize_expression(expr, {"x"})


def test_validate_proposal_ranges():
    ok = agent_mod.validate_proposal("integral", 1, {"expression": "x**2", "lower": 0, "upper": 2})
    assert ok.params["upper"] == 2.0
    with pytest.raises(ValueError):
        agent_mod.validate_proposal("integral", 1, {"expression": "x**2", "lower": 2, "upper": 0})
    with pytest.raises(ValueError):
        agent_mod.validate_proposal("derivative", 1, {"expression": "x", "order": 2, "eval_point": 99})
    with pytest.raises(ValueError):
        agent_mod.validate_proposal("ode", 1, {"order": 1, "rhs": "dy + t", "y0": 0.1, "t_end": 2})  # dy no existe en orden 1


# ----------------------------------------------------------------------------- guardas


def test_pid_alive_does_not_kill_and_detects_dead():
    assert guard_mod.pid_alive(os.getpid()) is True
    assert guard_mod.pid_alive(4_000_000) is False
    assert guard_mod.pid_alive(os.getpid()) is True  # sigue viva tras la comprobacion


def test_guard_stop_file_and_budget(sandbox):
    g = guard_mod.AutonomyGuard(max_cycles_per_hour=2, check_bridge=False, gpu_busy_pct=101, max_temp_c=500)
    assert g.should_run() == (True, "ok")
    guard_mod.request_stop("test")
    allowed, why = g.should_run()
    assert allowed is False and "STOP" in why
    guard_mod.clear_stop()
    g.note_cycle()
    g.note_cycle()
    allowed, why = g.should_run()
    assert allowed is False and "presupuesto" in why


def test_guard_ignores_stale_training_lock_but_respects_live_one(sandbox):
    g = guard_mod.AutonomyGuard(check_bridge=False, gpu_busy_pct=101, max_temp_c=500)
    guard_mod.TRAINING_LOCK.write_text(json.dumps({"pid": 4_000_000}))  # proceso muerto
    assert g.should_run()[0] is True
    guard_mod.TRAINING_LOCK.write_text(json.dumps({"pid": os.getpid()}))  # proceso vivo
    allowed, why = g.should_run()
    assert allowed is False and "entrenamiento" in why


def test_single_instance_lock(sandbox):
    assert guard_mod.acquire_instance_lock() is True
    guard_mod.PID_FILE.write_text(str(os.getppid()))  # otro proceso vivo
    assert guard_mod.acquire_instance_lock() is False
    guard_mod.PID_FILE.write_text("4000000")  # lock huerfano
    assert guard_mod.acquire_instance_lock() is True
    guard_mod.release_instance_lock()


# ----------------------------------------------------------------------------- agente


def _fake_llm(payload):
    def llm(prompt, seed):
        return payload if isinstance(payload, str) else json.dumps(payload)

    return llm


def _verified(kind_result):
    return lambda kind, params: {"ok": True, "result": kind_result}


def test_agent_records_only_verified_samples(sandbox):
    calls = []

    def verifier(kind, params):
        calls.append((kind, params))
        return {"ok": True, "result": {"verified": False, "max_abs_discrepancy": 0.5, "relative_discrepancy": 0.5}}

    a = agent_mod.AutonomousAgent(
        llm=_fake_llm({"expression": "x**2", "order": 1, "eval_point": 1.0, "lower": 0, "upper": 1, "rhs": "-y", "y0": 1, "t_end": 2}),
        verifier=verifier,
        guard=AlwaysAllow(),
        record=True,
    )
    r = a.run_cycle()
    assert r["status"] == "failed_verification" and calls
    assert not cl.CONTINUAL_DATASET_FILE.exists(), "una muestra NO verificada jamas debe llegar al dataset"


def test_agent_rejects_malicious_proposal_without_calling_verifier(sandbox):
    calls = []
    a = agent_mod.AutonomousAgent(
        llm=_fake_llm({"expression": "__import__('os').system('calc')", "order": 1, "eval_point": 1, "lower": 0, "upper": 1}),
        verifier=lambda k, p: calls.append(1) or {"ok": False, "error": "no deberia llamarse"},
        guard=AlwaysAllow(),
        rng=__import__("random").Random(1),
        record=False,
    )
    for _ in range(6):  # cualquier tipo que se elija debe rechazarse
        r = a.run_cycle()
        assert r["status"] in {"rejected_proposal", "duplicate"}
    assert calls == [], "texto del LLM nunca debe alcanzar al verificador sin pasar el saneado"


def test_agent_llm_down_does_not_penalize_curriculum(sandbox):
    def boom(prompt, seed):
        raise ConnectionError("ollama caido")

    a = agent_mod.AutonomousAgent(llm=boom, verifier=_verified({}), guard=AlwaysAllow(), record=False)
    r = a.run_cycle()
    assert r["status"] == "error"
    assert all(v["proposed"] == 0 for v in a.state["curriculum"].values())


def test_curriculum_raises_and_lowers_level(sandbox):
    a = agent_mod.AutonomousAgent(llm=_fake_llm("{}"), verifier=_verified({}), guard=AlwaysAllow(), record=False)
    for _ in range(8):
        a._update_curriculum("derivative", True)
    assert a.state["curriculum"]["derivative"]["level"] == 2
    for _ in range(8):
        a._update_curriculum("derivative", False)
    assert a.state["curriculum"]["derivative"]["level"] == 1


def test_stop_file_prevents_cycle_and_is_journaled_once(sandbox):
    guard_mod.request_stop("test")
    g = guard_mod.AutonomyGuard(check_bridge=False)
    a = agent_mod.AutonomousAgent(llm=_fake_llm("{}"), verifier=_verified({}), guard=g, record=False)
    assert a.run_cycle()["status"] == "skipped"
    assert a.run_cycle()["status"] == "skipped"
    lines = agent_mod.JOURNAL_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "el mismo motivo de pausa no debe llenar el diario"


@pytest.mark.parametrize(
    "kind,params",
    [
        ("derivative", {"expression": "3*x**2 + 5*sin(x)", "variable": "x", "order": 1, "eval_point": 0.8}),
        ("integral", {"expression": "2*x**3 - 5*x**2 + 4*x - 1", "variable": "x", "lower": 0.0, "upper": 2.0}),
        ("ode", {"order": 2, "rhs": "-0.1*dy - 9*sin(y)", "y0": 0.4, "dy0": 0.0, "t_end": 3.0}),
    ],
)
def test_real_verification_end_to_end_and_sample_content(sandbox, kind, params):
    """Subproceso REAL + SymPy/SciPy reales: la muestra redactada contiene los numeros calculados."""
    out = agent_mod.default_verifier(kind, params)
    assert out["ok"], out
    res = out["result"]
    assert res["verified"] is True
    proposal = agent_mod.Proposal(kind=kind, level=1, params=params)
    q, a = agent_mod.compose_verified_sample(proposal, res)
    assert "[VERIFICADO:" in a and "$$" in a and q
    if kind in {"derivative", "ode"}:
        assert "\\begin{aligned}" in a
    if kind == "derivative":
        assert "cos" in a  # la derivada simbolica de 5 sin(x) aparece en la respuesta
    if kind == "integral":
        assert "\\frac{2}{3}" in a  # 2/3: resultado exacto calculado por SymPy


# ----------------------------------------------------------------------------- curador y gate


def _entry(human, gpt, **meta):
    return json.dumps({"conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}], "metadata": meta})


def test_curator_excludes_legacy_dedupes_and_reports_honestly(sandbox):
    rows = [
        _entry("[Auto-Reto Autónomo] Resolver EDO no lineal: y'' + 0.1y' + 9sin(y) = 0", "plantilla fija"),
        _entry("[Auto-Reto Autónomo] Resolver EDO no lineal: y'' + 0.2y' + 8sin(y) = 0", "plantilla fija"),
        _entry("Derivada de x^2", "2x", response_origin="verified_tool_template"),
        _entry("derivada de X^2", "2x duplicado", response_origin="verified_tool_template"),
        _entry("Hola ada", "Hola Diego"),
        "esto no es json",
    ]
    cl.CONTINUAL_DATASET_FILE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = curator_mod.curate_dataset(source=cl.CONTINUAL_DATASET_FILE)
    assert rep["total_curated"] == 2
    assert rep["real_interactions"] == 1 and rep["synthetic_verified"] == 1
    assert rep["counts"]["raw_legacy_template"] == 2
    assert rep["counts"]["duplicates"] == 1 and rep["counts"]["malformed"] == 1
    assert rep["ready_for_training"] is False
    assert any("legado" in w for w in rep["warnings"])


def test_curator_split_is_deterministic(sandbox):
    rows = [_entry(f"pregunta {i}", f"respuesta {i}", response_origin="verified_tool_template") for i in range(60)]
    cl.CONTINUAL_DATASET_FILE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    a = curator_mod.curate_dataset(source=cl.CONTINUAL_DATASET_FILE)
    first_eval = curator_mod.CURATED_EVAL_FILE.read_text(encoding="utf-8")
    b = curator_mod.curate_dataset(source=cl.CONTINUAL_DATASET_FILE)
    assert a["train"] == b["train"] and a["eval"] == b["eval"]
    assert curator_mod.CURATED_EVAL_FILE.read_text(encoding="utf-8") == first_eval
    assert a["train"] + a["eval"] == 60


def test_training_gate():
    from ada.training.qlora import decide_gate

    assert decide_gate(1.00, 0.90)["accepted"] is True
    assert decide_gate(1.00, 0.995)["accepted"] is False  # mejora insuficiente
    assert decide_gate(1.00, 1.20)["accepted"] is False  # empeora
    assert decide_gate(float("nan"), 0.5)["accepted"] is False
    assert decide_gate(0.0, 0.0)["accepted"] is False


def test_qlora_refuses_without_weights_and_never_downloads(sandbox, capsys):
    from ada.training import qlora

    path, msg = qlora.resolve_base_model("org-inexistente/modelo-que-no-existe")
    assert path is None and "no encontrados" in msg
