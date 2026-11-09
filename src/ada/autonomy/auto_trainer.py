"""Entrenador automatico: Ada decide SOLA cuando afinarse (QLoRA), sin que nadie se lo pida.

Entrena unicamente cuando TODO esto se cumple, y lo explica en voz alta cuando espera:
  * el curador dice que el dataset ya sirve (≥200 muestras, ≥20 reales) y hay ≥100 muestras nuevas desde la ultima vez;
  * pasaron ≥12 h desde el ultimo intento (el entrenamiento ocupa la GPU durante horas);
  * Diego lleva ≥30 min sin usar a Ada y la GPU esta libre y fria;
  * los pesos base estan en disco. Si faltan, descargarlos (≈30 GB, Qwen3-14B: el mismo modelo que sirve Ollama) exige UNA autorizacion tuya:
    `python scripts/ada_daemon.py --allow-download`. Sin ella Ada espera y lo dice.
Si Diego vuelve a usarla mientras entrena, el entrenamiento se ABORTA (la GPU es suya) y se reintenta mas tarde.
El resultado pasa por el gate de calidad de qlora.py: un adaptador que no mejora la perdida de evaluacion se descarta.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable

from ada.autonomy.curator import curate_dataset
from ada.autonomy.guard import AUTONOMY_DIR, STOP_FILE, AutonomyGuard
from ada.config import PROJECT_ROOT, ModelRoleConfig, assert_safe_write_path
from ada.training.qlora import resolve_base_model

TRAIN_STATE_FILE = AUTONOMY_DIR / "training_state.json"
ALLOW_DOWNLOAD_FILE = AUTONOMY_DIR / "ALLOW_DOWNLOAD"
TRAIN_LOG_FILE = AUTONOMY_DIR / "train.log"
BRIDGE_STATUS_URL = "http://127.0.0.1:8765/api/status"


def training_disabled_reason() -> str | None:
    """Si existe data/work/autonomy/TRAINING_DISABLED, el autoentrenamiento esta apagado y devuelve el motivo escrito alli."""
    f = AUTONOMY_DIR / "TRAINING_DISABLED"
    if not f.is_file():
        return None
    try:
        return str(json.loads(f.read_text(encoding="utf-8")).get("reason", "desactivado"))
    except (OSError, json.JSONDecodeError):
        return "desactivado"


def allow_download() -> None:
    assert_safe_write_path(ALLOW_DOWNLOAD_FILE)
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    ALLOW_DOWNLOAD_FILE.write_text(json.dumps({"authorized_at": time.time()}), encoding="utf-8")


def bridge_idle_seconds(now: float | None = None) -> float | None:
    """Segundos desde la ultima actividad REAL de Diego segun el puente; None si no hay puente (nadie esta usando el CLI)."""
    try:
        with urllib.request.urlopen(BRIDGE_STATUS_URL, timeout=0.8) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    now = now or time.time()
    last = max(float(d.get("last_user_activity_ts") or 0.0), float(d.get("bridge_started_at") or 0.0))
    if d.get("is_thinking") and d.get("source") != "autonomous":
        return 0.0
    return max(0.0, now - last)


class AutoTrainer:
    def __init__(
        self,
        *,
        log: Callable[[dict[str, Any]], None] | None = None,
        cooldown_h: float = 12.0,
        idle_min: float = 30.0,
        min_new_samples: int = 100,
        base_model: str | None = None,
        idle_seconds: Callable[[], float | None] = bridge_idle_seconds,
        guard: AutonomyGuard | None = None,
        spawn: Callable[[list[str]], subprocess.Popen] | None = None,
        now: Callable[[], float] = time.time,
        curate: Callable[[], dict[str, Any]] = curate_dataset,
    ) -> None:
        self.curate = curate
        self.log = log or (lambda e: None)
        self.cooldown_s = cooldown_h * 3600
        self.idle_s = idle_min * 60
        self.min_new_samples = min_new_samples
        self.base_model = base_model or ModelRoleConfig().orchestrator_hf
        self.idle_seconds = idle_seconds
        self.guard = guard or AutonomyGuard(check_bridge=False)
        self._spawn = spawn or self._default_spawn
        self.now = now
        self._proc: subprocess.Popen | None = None
        self._proc_kind = ""
        self._last_wait = ""

    # -- estado --------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if TRAIN_STATE_FILE.is_file():
            try:
                return json.loads(TRAIN_STATE_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return {"last_train_started": 0.0, "last_train_samples": 0, "history": []}

    def _save(self, st: dict[str, Any]) -> None:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
        assert_safe_write_path(TRAIN_STATE_FILE)
        st["history"] = st.get("history", [])[-20:]
        TRAIN_STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _default_spawn(cmd: list[str]) -> subprocess.Popen:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
        out = open(TRAIN_LOG_FILE, "a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), creationflags=flags,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    # -- decision pura (testeable) ----------------------------------------------
    def decide(self) -> tuple[str, str]:
        """('train' | 'download' | 'wait', motivo en español)."""
        if STOP_FILE.is_file():
            return "wait", "parada de emergencia activa (STOP)"
        reason = training_disabled_reason()
        if reason:
            return "wait", f"autoentrenamiento DESACTIVADO: {reason}"
        st = self._load()
        since = self.now() - float(st.get("last_train_started", 0.0))
        if since < self.cooldown_s:
            return "wait", f"entrené hace {since / 3600:.1f} h; espero {self.cooldown_s / 3600:.0f} h entre entrenamientos"
        report = self.curate()
        if not report["ready_for_training"]:
            t = report["thresholds"]
            return "wait", (f"aún no tengo datos suficientes para afinarme: {report['total_curated']}/{t['min_samples']} muestras, "
                            f"{report['real_interactions']}/{t['min_real_interactions']} conversaciones reales")
        new = report["total_curated"] - int(st.get("last_train_samples", 0))
        if new < self.min_new_samples:
            return "wait", f"solo {new} muestras nuevas desde el último entrenamiento (mínimo {self.min_new_samples})"
        path, _ = resolve_base_model(self.base_model)
        if path is None:
            if ALLOW_DOWNLOAD_FILE.is_file():
                return "download", f"tengo datos listos pero faltan los pesos base {self.base_model}: los descargo (autorizado)"
            return "wait", (f"tengo datos listos, pero necesito los pesos base {self.base_model} (≈30 GB). "
                            "Autoriza la descarga una vez con: python scripts/ada_daemon.py --allow-download")
        idle = self.idle_seconds()
        if idle is not None and idle < self.idle_s:
            return "wait", f"esperando a que lleves {self.idle_s / 60:.0f} min sin usarme (van {idle / 60:.0f})"
        ok, why = self.guard.should_run()
        if not ok:
            return "wait", why
        return "train", f"dataset listo ({report['total_curated']} muestras), GPU libre y llevas rato sin usarme: me afino"

    # -- accion -----------------------------------------------------------------
    def tick(self) -> dict[str, Any] | None:
        """Una pasada: vigila el proceso en curso o decide si lanzar otro. Devuelve el evento ocurrido, si hubo."""
        if self._proc is not None:
            rc = self._proc.poll()
            if rc is None:
                idle = self.idle_seconds()
                if self._proc_kind == "train" and idle is not None and idle < 60:
                    self._proc.terminate()
                    self._proc = None
                    ev = {"status": "autotrain_aborted", "detail": "Diego volvió a usarme: aborto el entrenamiento y lo reintento más tarde"}
                    self.log(ev)
                    return ev
                return None
            kind, self._proc = self._proc_kind, None
            ev = {"status": f"auto{kind}_finished", "detail": {0: "terminó bien (adaptador aceptado)" if kind == "train" else "descarga completa",
                                                                4: "terminó pero el gate descartó el adaptador: no mejoró la evaluación"}.get(rc, f"terminó con código {rc}; ver data/work/autonomy/train.log"),
                  "trained_ok": kind == "train" and rc == 0, "gate_rejected": kind == "train" and rc == 4}
            self.log(ev)
            return ev

        action, reason = self.decide()
        if action == "wait":
            if reason != self._last_wait:
                self._last_wait = reason
                ev = {"status": "autotrain_wait", "detail": reason}
                self.log(ev)
                return ev
            return None
        self._last_wait = ""
        st = self._load()
        if action == "download":
            self._proc = self._spawn([sys.executable, str(PROJECT_ROOT / "scripts" / "train_qlora.py"), "--download", "--base-model", self.base_model])
            self._proc_kind = "download"
        else:
            st["last_train_started"] = self.now()
            st["last_train_samples"] = self.curate()["total_curated"]
            st["history"].append({"started": st["last_train_started"], "samples": st["last_train_samples"]})
            self._save(st)
            self._proc = self._spawn([sys.executable, str(PROJECT_ROOT / "scripts" / "train_qlora.py"), "--execute", "--base-model", self.base_model])
            self._proc_kind = "train"
        ev = {"status": f"auto{self._proc_kind}_started", "detail": reason}
        self.log(ev)
        return ev
