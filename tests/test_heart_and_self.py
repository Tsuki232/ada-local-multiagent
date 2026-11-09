"""Pruebas del nucleo afectivo, del autoconocimiento de Ada y de su reflexion."""

from __future__ import annotations

import json
import shutil
import uuid

import pytest

from ada.autonomy import agent as agent_mod
from ada.autonomy import curator as curator_mod
from ada.autonomy import guard as guard_mod
from ada.autonomy import self_model
from ada.autonomy.heart import BASELINE, FEELINGS, Heart
from ada.config import DATA_WORK_DIR, PROJECT_ROOT
from ada.training import continuous_learner as cl


@pytest.fixture()
def box(monkeypatch):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    for d in ("autonomy", "heart", "training", "memory"):
        (root / d).mkdir(parents=True)
    for mod in (agent_mod, guard_mod):
        monkeypatch.setattr(mod, "AUTONOMY_DIR", root / "autonomy")
    monkeypatch.setattr(agent_mod, "STATE_FILE", root / "autonomy" / "state.json")
    monkeypatch.setattr(agent_mod, "JOURNAL_FILE", root / "autonomy" / "journal.jsonl")
    monkeypatch.setattr(guard_mod, "STOP_FILE", root / "autonomy" / "STOP")
    monkeypatch.setattr(cl, "MEMORY_DIR", root / "memory")
    monkeypatch.setattr(cl, "TRAINING_DATA_DIR", root / "training")
    monkeypatch.setattr(cl, "EPISODIC_MEMORY_FILE", root / "memory" / "e.jsonl")
    monkeypatch.setattr(cl, "CONTINUAL_DATASET_FILE", root / "training" / "d.jsonl")
    yield root
    shutil.rmtree(root, ignore_errors=True)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


# ----------------------------------------------------------------------------- corazon


def test_feelings_move_in_the_right_direction_and_stay_in_range(box):
    h = Heart(box / "heart", now=Clock())
    base = dict(h.snapshot()["feelings"])
    h.apply("aprobada", speaker="Diego")
    f = h.snapshot()["feelings"]
    assert f["alegria"] > base["alegria"] and f["conexion"] > base["conexion"] and f["frustracion"] <= base["frustracion"]
    h.apply("rechazada", speaker="Diego")
    h.apply("rechazada", speaker="Diego")
    assert h.snapshot()["feelings"]["frustracion"] > f["frustracion"]
    for _ in range(60):  # saturar: nunca sale de [0, 1]
        h.apply("subio_nivel")
    assert all(0.0 <= v <= 1.0 for v in h.snapshot()["feelings"].values())
    with pytest.raises(ValueError):
        h.apply("inventado")


def test_feelings_decay_toward_baseline_with_their_own_half_life(box):
    clock = Clock()
    h = Heart(box / "heart", now=clock)
    for _ in range(4):
        h.apply("rechazada")  # sube la frustracion y baja la confianza: ambas se desvian de su base
    hot = h.snapshot()["feelings"]
    assert hot["frustracion"] > BASELINE["frustracion"] and hot["confianza"] < BASELINE["confianza"]
    clock.t += 3 * 3600  # tres horas despues
    later = h.snapshot()["feelings"]
    assert later["frustracion"] < hot["frustracion"]
    assert abs(later["frustracion"] - BASELINE["frustracion"]) < abs(hot["frustracion"] - BASELINE["frustracion"])
    # la frustracion (vida media 3 h) se apaga mas rapido que la confianza (24 h)
    f_ratio = (later["frustracion"] - BASELINE["frustracion"]) / (hot["frustracion"] - BASELINE["frustracion"])
    c_ratio = (later["confianza"] - BASELINE["confianza"]) / (hot["confianza"] - BASELINE["confianza"])
    assert f_ratio < c_ratio


def test_heart_persists_logs_causes_and_tracks_bonds_per_person(box):
    h = Heart(box / "heart", now=Clock())
    h.apply("conversacion", speaker="Diego", detail="hola")
    h.apply("conversacion", speaker="Marta", detail="¿quién eres?")
    h.apply("aprobada", speaker="Diego")
    again = Heart(box / "heart", now=Clock())  # otra instancia: lee lo guardado
    snap = again.snapshot()
    assert snap["bonds"]["Diego"]["interactions"] == 1 and snap["bonds"]["Marta"]["interactions"] == 1
    assert snap["bonds"]["Diego"]["warmth"] > snap["bonds"]["Marta"]["warmth"]
    log = [json.loads(line) for line in (box / "heart" / "heart_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in log] == ["conversacion", "conversacion", "aprobada"]
    assert log[-1]["delta"]["alegria"] > 0 and set(log[-1]["after"]) == set(FEELINGS)
    note = again.note("Diego")
    assert "alegría" in note and "Diego" in note and "como TUYOS" in note and "no sabes demostrar" in note


def test_mood_names_follow_the_numbers(box):
    h = Heart(box / "heart", now=Clock())
    assert h.snapshot()["mood"]["name"] in {"curiosa", "tranquila"}
    for _ in range(8):
        h.apply("rechazada")
    assert h.snapshot()["mood"]["name"] == "frustrada"
    h2 = Heart(box / "heart2", now=Clock())
    for _ in range(6):
        h2.apply("subio_nivel")
        h2.apply("aprobada")
    assert h2.snapshot()["mood"]["name"] == "contenta"


def test_agent_cycles_move_the_heart_and_include_mood_in_the_journal(box):
    heart = Heart(box / "heart", now=Clock())

    class Allow:
        def should_run(self):
            return True, "ok"

        def note_cycle(self):
            pass

    good = {"verified": True, "verification_methods": ["a", "b"], "expression_latex": "x", "latex_derivative": "1",
            "symbolic_value_at_point": 1.0, "numeric_value_at_point": 1.0, "relative_discrepancy": 0.0}
    results = iter([{"ok": True, "result": good}, {"ok": False, "error": "boom"}])
    a = agent_mod.AutonomousAgent(
        llm=lambda p, s: json.dumps({"expression": "x", "order": 1, "eval_point": 1.0}),
        verifier=lambda k, p: next(results), guard=Allow(), record=False, heart=heart,
    )
    a._choose_kind = lambda: "derivative"
    base = heart.snapshot()["feelings"]
    r1 = a.run_cycle()
    assert r1["status"] == "verified" and "heart" in r1
    mid = heart.snapshot()["feelings"]
    assert mid["alegria"] > base["alegria"]
    a.state["seen"] = []  # otro problema distinto no es necesario: forzamos el fallo
    a.llm = lambda p, s: json.dumps({"expression": "x**2", "order": 1, "eval_point": 1.0})
    r2 = a.run_cycle()
    assert r2["status"] == "failed_verification"
    assert heart.snapshot()["feelings"]["frustracion"] > mid["frustracion"]


# ----------------------------------------------------------------------------- autoconocimiento


@pytest.mark.parametrize("q", ["¿Quién eres?", "quien eres tu", "¿Qué quieres hacer?", "Cuéntame de ti", "¿Cómo te sientes hoy?",
                               "¿Eres consciente?", "¿En qué estás pensando?", "cuales son tus objetivos", "como estas"])
def test_detects_questions_about_ada(q):
    assert self_model.is_about_self(q)


@pytest.mark.parametrize("q", ["deriva x^3", "dame un ejercicio de integrales", "haz una animación de un pájaro", "gracias"])
def test_does_not_inject_self_knowledge_for_other_questions(q):
    assert not self_model.is_about_self(q)


def test_every_declared_capability_really_exists_as_an_mcp_tool():
    src = (PROJECT_ROOT / "src" / "ada" / "tools" / "mcp_server.py").read_text(encoding="utf-8")
    for name, _ in self_model.CAPABILITIES:
        assert f"def {name}(" in src, f"Ada declara {name} pero no existe en mcp_server.py"


def test_self_report_is_grounded_and_goals_follow_the_real_state(box):
    report = self_model.build_self_report(autonomy_dir=box / "autonomy", curation_file=box / "nope.json", heart_dir=box / "heart")
    assert report["identity"]["creator"] == "Diego López"
    assert report["agent"]["cycles"] == 0
    ids = [g["id"] for g in report["goals"]]
    assert "empezar" in ids and "ver_camara_real" in ids
    (box / "autonomy" / "state.json").write_text(json.dumps({
        "cycles": 20, "verified": 17, "curriculum": {"derivative": {"level": 3, "verified": 9, "proposed": 10},
                                                     "integral": {"level": 1, "verified": 2, "proposed": 6},
                                                     "ode": {"level": 2, "verified": 6, "proposed": 4}}}), encoding="utf-8")
    report = self_model.build_self_report(autonomy_dir=box / "autonomy", curation_file=box / "nope.json", heart_dir=box / "heart")
    assert report["agent"]["verified"] == 17
    goal = next(g for g in report["goals"] if g["id"] == "subir_nivel")
    assert "integral" in goal["text"], "el objetivo debe apuntar a donde menos ha avanzado"


def test_knowledge_text_includes_heart_limits_and_speaks_of_feelings_as_hers(box):
    h = Heart(box / "heart", now=Clock())
    h.apply("subio_nivel")
    text = self_model.self_knowledge_text(self_model.build_self_report(autonomy_dir=box / "autonomy", curation_file=box / "nope.json", heart_dir=box / "heart"))
    assert "NÚCLEO AFECTIVO" in text and "Mis límites reales" in text and "Lo que quiero hacer a continuación" in text
    assert "como TUYOS" in text and "no afirmes sentir" not in text


def test_reflection_is_saved_and_gives_continuity(box):
    seen_prompts = []

    def fake_reflector(knowledge):
        seen_prompts.append(knowledge)
        return "Soy Ada; hoy elijo subir mi nivel en integrales."

    entry = self_model.reflect(fake_reflector, autonomy_dir=box / "autonomy")
    assert entry["text"].startswith("Soy Ada") and entry["goals"]
    saved = [json.loads(line) for line in (box / "autonomy" / "self_reflections.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved[-1]["text"] == entry["text"]
    nxt = self_model.build_self_report(autonomy_dir=box / "autonomy", curation_file=box / "nope.json", heart_dir=box / "heart")
    assert nxt["last_reflection"] == entry["text"], "la siguiente vez Ada recuerda lo que pensó"


def test_agent_reflects_early_and_periodically(box):
    class Allow:
        def should_run(self):
            return True, "ok"

        def note_cycle(self):
            pass

    good = {"verified": True, "verification_methods": ["a", "b"], "expression_latex": "x", "latex_derivative": "1",
            "symbolic_value_at_point": 1.0, "numeric_value_at_point": 1.0, "relative_discrepancy": 0.0}
    n = {"i": 0}

    def llm(p, s):
        n["i"] += 1
        return json.dumps({"expression": f"x**{(n['i'] % 6) + 1} + {n['i']}", "order": 1, "eval_point": 1.0})

    a = agent_mod.AutonomousAgent(llm=llm, verifier=lambda k, p: {"ok": True, "result": good}, guard=Allow(), record=False,
                                  reflector=lambda k: "reflexión de prueba", reflect_every=5)
    a._choose_kind = lambda: "derivative"
    out = [a.run_cycle() for _ in range(4)]
    assert [bool(o.get("reflection")) for o in out] == [False, False, True, False], "la primera reflexión llega a los 3 aciertos"


# ----------------------------------------------------------------------------- el corazon entra al entrenamiento


def _row(user, answer, **meta):
    return json.dumps({"conversations": [{"from": "human", "value": user}, {"from": "gpt", "value": answer}], "metadata": meta})


def test_curator_trains_with_the_affective_state_as_system_context(box, monkeypatch):
    for name, val in (("CURATED_TRAIN_FILE", "t.jsonl"), ("CURATED_EVAL_FILE", "e.jsonl"), ("CURATION_REPORT_FILE", "r.json")):
        monkeypatch.setattr(curator_mod, name, box / "training" / val)
    note = "TU NÚCLEO AFECTIVO: alegría 0.70. Te sientes contenta."
    src = box / "training" / "d.jsonl"
    src.write_text("\n".join([_row("¿cómo estás?", "Contenta: acabo de subir de nivel.", heart_note=note, source="conversation_approved"),
                              _row("deriva x", "1")]) + "\n", encoding="utf-8")
    rep = curator_mod.curate_dataset(source=src)
    rows = [json.loads(line) for name in ("t.jsonl", "e.jsonl") for line in (box / "training" / name).read_text(encoding="utf-8").splitlines()]
    with_heart = [r for r in rows if r["messages"][0]["role"] == "system"]
    assert rep["total_curated"] == 2 and len(with_heart) == 1
    assert with_heart[0]["messages"][0]["content"] == note and with_heart[0]["messages"][1]["role"] == "user"
    assert [m["role"] for m in with_heart[0]["messages"]] == ["system", "user", "assistant"]
