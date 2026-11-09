"""Pruebas de que Ada se hace notar (narracion, eventos, niveles), del arbitraje del puente y del autoentrenamiento."""

from __future__ import annotations

import json
import shutil
import uuid

import pytest

from ada.autonomy import agent as agent_mod
from ada.autonomy import guard as guard_mod
from ada.config import DATA_WORK_DIR
from ada.training import continuous_learner as cl


@pytest.fixture()
def sandbox(monkeypatch):
    """Todo el estado autonomo a un subdirectorio temporal DENTRO de data/work (lo exige la barrera de rutas)."""
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
    yield root
    shutil.rmtree(root, ignore_errors=True)


class AlwaysAllow:
    def should_run(self):
        return True, "ok"

    def note_cycle(self):
        pass


def _fake_llm(payload):
    return lambda prompt, seed: json.dumps(payload)


def _deriv_result():
    return {
        "verified": True,
        "verification_methods": ["sympy.diff", "mpmath.diff_50dps"],
        "expression_latex": "x^{2}",
        "latex_derivative": "2 x",
        "symbolic_value_at_point": 1.5,
        "numeric_value_at_point": 1.5,
        "relative_discrepancy": 1e-15,
    }


# ----------------------------------------------------------------------------- narracion, eventos y niveles


def test_agent_narrates_emits_events_and_reports_level_up(sandbox):
    events = []
    a = agent_mod.AutonomousAgent(
        llm=_fake_llm({"expression": "x**2", "order": 1, "eval_point": 0.75}),
        verifier=lambda kind, params: {"ok": True, "result": _deriv_result()},
        guard=AlwaysAllow(),
        record=False,
        narrate=True,
        narrator=lambda q, ans: "Comprobé que la derivada de x al cuadrado es 2x.",
        on_event=events.append,
    )
    a._choose_kind = lambda: "derivative"
    a.state["curriculum"]["derivative"]["window"] = [1] * 7  # el siguiente acierto completa 8 con 100 %
    r = a.run_cycle()
    assert r["status"] == "verified" and r["thought"].startswith("Comprobé")
    assert r["level_change"] == [1, 2], "al completar la ventana con acierto alto debe subir de nivel y decirlo"
    assert events and events[-1]["thought"] and events[-1]["level_change"] == [1, 2]
    journal = [json.loads(line) for line in agent_mod.JOURNAL_FILE.read_text(encoding="utf-8").splitlines()]
    assert journal[-1]["thought"] and journal[-1]["level_change"] == [1, 2]


def test_narrator_failure_does_not_break_the_cycle(sandbox):
    def boom(q, a):
        raise TimeoutError("ollama lento")

    a = agent_mod.AutonomousAgent(
        llm=_fake_llm({"expression": "x**3", "order": 1, "eval_point": 1.0}),
        verifier=lambda kind, params: {"ok": True, "result": _deriv_result()},
        guard=AlwaysAllow(),
        record=False,
        narrate=True,
        narrator=boom,
    )
    a._choose_kind = lambda: "derivative"
    r = a.run_cycle()
    assert r["status"] == "verified" and "thought" not in r and "TimeoutError" in r["thought_error"]


# ----------------------------------------------------------------------------- arbitraje del puente


def test_bridge_never_lets_the_agent_override_diegos_state():
    from ada.cognition import live_server as ls

    ls.set_live_thinking(False, source="user")
    ls.set_live_thinking(True, prompt="pregunta de Diego", source="user")
    ls.set_live_thinking(True, prompt="[autonomo]", source="autonomous")  # no debe pisar a Diego
    assert ls._LIVE_STATE["source"] == "user" and ls._LIVE_STATE["current_prompt"] == "pregunta de Diego"
    ls.set_live_thinking(False, source="autonomous")  # tampoco apagar lo de Diego
    assert ls._LIVE_STATE["is_thinking"] is True
    ls.set_live_thinking(False, source="user")
    ls.set_live_thinking(True, prompt="[autonomo]", source="autonomous")
    assert ls._LIVE_STATE["is_thinking"] and ls._LIVE_STATE["source"] == "autonomous"
    ls.set_live_thinking(False, source="autonomous")
    assert ls._LIVE_STATE["is_thinking"] is False


def test_guard_does_not_treat_its_own_announcement_as_diego_chatting(sandbox, monkeypatch):
    from ada.cognition import live_server as ls

    port = 8799
    assert ls.start_live_bridge(port=port)
    monkeypatch.setattr(guard_mod, "BRIDGE_STATUS_URL", f"http://127.0.0.1:{port}/api/status")
    g = guard_mod.AutonomyGuard(check_bridge=True, gpu_busy_pct=101, max_temp_c=500)
    try:
        ls.set_live_thinking(False, source="user")
        ls.set_live_thinking(True, prompt="[autonomo]", source="autonomous")
        assert g.should_run() == (True, "ok"), "el propio anuncio del agente no debe bloquearlo"
        ls.set_live_thinking(False, source="autonomous")
        ls.set_live_thinking(True, prompt="hola ada", source="user")
        allowed, why = g.should_run()
        assert allowed is False and "conversando" in why
    finally:
        ls.set_live_thinking(False, source="user")


# ----------------------------------------------------------------------------- autoentrenamiento


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.rc = None

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True
        self.rc = -15


READY = {"ready_for_training": True, "total_curated": 300, "real_interactions": 40, "thresholds": {"min_samples": 200, "min_real_interactions": 20}}
NOT_READY = {"ready_for_training": False, "total_curated": 30, "real_interactions": 11, "thresholds": {"min_samples": 200, "min_real_interactions": 20}}


def _trainer(monkeypatch, report, idle=4000.0, weights=True, now=1_000_000.0):
    from ada.autonomy import auto_trainer as at

    monkeypatch.setattr(at, "AUTONOMY_DIR", guard_mod.AUTONOMY_DIR)
    monkeypatch.setattr(at, "STOP_FILE", guard_mod.STOP_FILE)
    monkeypatch.setattr(at, "TRAIN_STATE_FILE", guard_mod.AUTONOMY_DIR / "training_state.json")
    monkeypatch.setattr(at, "ALLOW_DOWNLOAD_FILE", guard_mod.AUTONOMY_DIR / "ALLOW_DOWNLOAD")
    monkeypatch.setattr(at, "resolve_base_model", lambda b: (("/pesos", "local") if weights else (None, "faltan")))
    events, procs = [], []

    class G:
        def should_run(self):
            return True, "ok"

    t = at.AutoTrainer(
        log=events.append,
        idle_seconds=lambda: idle,
        guard=G(),
        spawn=lambda cmd: procs.append(_FakeProc()) or procs[-1],
        now=lambda: now,
        curate=lambda: report,
    )
    return at, t, events, procs


def test_autotrainer_waits_and_explains_when_data_is_not_enough(sandbox, monkeypatch):
    _, t, events, procs = _trainer(monkeypatch, NOT_READY)
    ev = t.tick()
    assert ev["status"] == "autotrain_wait" and "30/200" in ev["detail"] and not procs
    assert t.tick() is None, "el mismo motivo de espera no se repite en cada pasada"


def test_autotrainer_needs_one_authorization_to_download_weights(sandbox, monkeypatch):
    at, t, events, procs = _trainer(monkeypatch, READY, weights=False)
    action, why = t.decide()
    assert action == "wait" and "allow-download" in why
    at.allow_download()
    action, why = t.decide()
    assert action == "download" and "autorizado" in why


def test_autotrainer_trains_only_when_idle_and_respects_cooldown(sandbox, monkeypatch):
    _, t, _, _ = _trainer(monkeypatch, READY, idle=120.0)
    action, why = t.decide()
    assert action == "wait" and "sin usarme" in why

    at, t, events, procs = _trainer(monkeypatch, READY, idle=4000.0)
    ev = t.tick()
    assert ev["status"] == "autotrain_started" and len(procs) == 1
    st = json.loads(at.TRAIN_STATE_FILE.read_text(encoding="utf-8"))
    assert st["last_train_samples"] == 300

    _, t2, _, _ = _trainer(monkeypatch, READY, idle=4000.0, now=1_000_000.0 + 3600)
    action, why = t2.decide()
    assert action == "wait" and "entrené hace" in why
    _, t3, _, _ = _trainer(monkeypatch, READY, idle=4000.0, now=1_000_000.0 + 13 * 3600)
    action, why = t3.decide()
    assert action == "wait" and "muestras nuevas" in why, "sin datos nuevos no se reentrena aunque pase el cooldown"


def test_autotrainer_aborts_training_when_diego_comes_back(sandbox, monkeypatch):
    _, t, events, procs = _trainer(monkeypatch, READY, idle=4000.0)
    assert t.tick()["status"] == "autotrain_started"
    t.idle_seconds = lambda: 5.0  # Diego acaba de escribir
    ev = t.tick()
    assert ev["status"] == "autotrain_aborted" and procs[0].terminated


def test_autotrainer_reports_gate_rejection(sandbox, monkeypatch):
    _, t, events, procs = _trainer(monkeypatch, READY, idle=4000.0)
    t.tick()
    procs[0].rc = 4  # codigo del gate: adaptador descartado
    ev = t.tick()
    assert ev["status"] == "autotrain_finished" and "descartó" in ev["detail"]


def test_stop_file_blocks_autotraining(sandbox, monkeypatch):
    _, t, _, procs = _trainer(monkeypatch, READY)
    guard_mod.request_stop("test")
    action, why = t.decide()
    assert action == "wait" and "STOP" in why and not procs
