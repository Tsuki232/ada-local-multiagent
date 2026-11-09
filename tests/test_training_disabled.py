"""El entrenamiento esta DESACTIVADO por decision de Diego (QLoRA 14B no cabe en 12 GB): ninguna via debe lanzarlo por accidente."""

from __future__ import annotations

import json
import shutil
import uuid

import pytest

from ada.autonomy import auto_trainer as at
from ada.autonomy import guard as guard_mod
from ada.autonomy import self_model
from ada.config import DATA_WORK_DIR, PROJECT_ROOT
from ada.training import qlora


@pytest.fixture()
def box(monkeypatch):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    (root / "autonomy").mkdir(parents=True)
    (root / "heart").mkdir()
    (root / "autonomy" / "TRAINING_DISABLED").write_text(json.dumps({"reason": "motivo de prueba"}), encoding="utf-8")
    monkeypatch.setattr(at, "AUTONOMY_DIR", root / "autonomy")
    monkeypatch.setattr(at, "STOP_FILE", root / "autonomy" / "STOP")
    monkeypatch.setattr(at, "TRAIN_STATE_FILE", root / "autonomy" / "training_state.json")
    monkeypatch.setattr(qlora, "AUTONOMY_DIR", root / "autonomy")
    yield root
    shutil.rmtree(root, ignore_errors=True)


READY = {"ready_for_training": True, "total_curated": 5000, "real_interactions": 500, "thresholds": {"min_samples": 200, "min_real_interactions": 20}}


def test_autotrainer_never_trains_while_disabled_even_with_perfect_conditions(box, monkeypatch):
    monkeypatch.setattr(at, "resolve_base_model", lambda b: ("/pesos", "local"))
    spawned = []

    class G:
        def should_run(self):
            return True, "ok"

    t = at.AutoTrainer(log=lambda e: None, idle_seconds=lambda: 99999.0, guard=G(), spawn=lambda cmd: spawned.append(cmd), now=lambda: 1e9, curate=lambda: READY)
    action, why = t.decide()
    assert action == "wait" and "DESACTIVADO" in why and "motivo de prueba" in why
    assert t.tick()["status"] == "autotrain_wait" and not spawned, "no debe lanzarse ningun proceso"
    assert at.training_disabled_reason() == "motivo de prueba"


def test_removing_the_switch_re_enables_the_trainer_decision(box, monkeypatch):
    (box / "autonomy" / "TRAINING_DISABLED").unlink()
    monkeypatch.setattr(at, "resolve_base_model", lambda b: ("/pesos", "local"))

    class G:
        def should_run(self):
            return True, "ok"

    t = at.AutoTrainer(log=lambda e: None, idle_seconds=lambda: 99999.0, guard=G(), spawn=lambda cmd: None, now=lambda: 1e9, curate=lambda: READY)
    assert t.decide()[0] == "train", "sin el interruptor, la logica de decision sigue intacta"


@pytest.mark.parametrize("flag", ["--execute", "--smoke"])
def test_manual_training_commands_refuse_while_disabled(box, flag, capsys):
    assert qlora.run([flag]) == 6
    assert "DESACTIVADO" in capsys.readouterr().out


def test_goals_and_self_description_match_the_decision(box):
    rep = self_model.build_self_report(autonomy_dir=box / "autonomy", curation_file=box / "no.json", heart_dir=box / "heart")
    ids = [g["id"] for g in rep["goals"]]
    assert "mejorar_sin_entrenar" in ids
    assert not ({"pesos_base", "primer_entrenamiento", "reunir_datos"} & set(ids)), "Ada no debe 'querer' descargar pesos ni entrenarse"
    assert rep["training"]["disabled"] is True
    text = self_model.self_knowledge_text(rep)
    assert "NO me entreno" in text and "dataset" not in text.split("Mi aprendizaje")[1].split("\n")[0]


def test_cli_and_daemon_do_not_enable_autotraining_by_default():
    cli = (PROJECT_ROOT / "scripts" / "ada_cli.py").read_text(encoding="utf-8")
    assert "auto_train=False" in cli and "auto_train=True" not in cli
    daemon = (PROJECT_ROOT / "scripts" / "ada_daemon.py").read_text(encoding="utf-8")
    assert "auto_train=args.auto_train" in daemon and "--auto-train" in daemon and "--no-train" not in daemon


def test_the_real_switch_exists_with_a_written_reason():
    flag = DATA_WORK_DIR / "autonomy" / "TRAINING_DISABLED"
    assert flag.is_file(), "falta data/work/autonomy/TRAINING_DISABLED"
    data = json.loads(flag.read_text(encoding="utf-8"))
    assert data["decided_by"] and "12 GB" in data["reason"]
    _ = guard_mod
