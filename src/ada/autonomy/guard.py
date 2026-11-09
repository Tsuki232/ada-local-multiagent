"""Guardas de seguridad de la autonomia: interruptor de parada, presupuesto, cesion al usuario y bloqueo de instancia.

Principios: Diego siempre tiene prioridad sobre la GPU, existe un freno fisico (archivo STOP) que funciona
aunque el proceso este colgado, y el agente nunca escribe fuera de data/work (lo impone assert_safe_write_path).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

from ada.autonomy.telemetry import read_gpu_telemetry
from ada.config import DATA_WORK_DIR, assert_safe_write_path

AUTONOMY_DIR = DATA_WORK_DIR / "autonomy"
STOP_FILE = AUTONOMY_DIR / "STOP"
PID_FILE = AUTONOMY_DIR / "agent.pid"
TRAINING_LOCK = AUTONOMY_DIR / "training.lock"
BRIDGE_STATUS_URL = "http://127.0.0.1:8765/api/status"


def pid_alive(pid: int) -> bool:
    """Comprueba si un PID existe SIN enviarle senales (os.kill(pid, 0) mataria el proceso en Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _training_lock_active() -> bool:
    """El lock cuenta solo si el proceso que lo creo sigue vivo (un crash no debe bloquear al agente para siempre)."""
    if not TRAINING_LOCK.is_file():
        return False
    try:
        pid = int(json.loads(TRAINING_LOCK.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return pid_alive(pid)


def request_stop(reason: str = "manual") -> Path:
    """Crea el archivo STOP: el agente se detiene en su siguiente comprobacion."""
    assert_safe_write_path(STOP_FILE)
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(json.dumps({"reason": reason, "ts": time.time()}), encoding="utf-8")
    return STOP_FILE


def clear_stop() -> None:
    STOP_FILE.unlink(missing_ok=True)


def acquire_instance_lock() -> bool:
    """Garantiza una sola instancia del agente. Devuelve False si otra sigue viva."""
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    assert_safe_write_path(PID_FILE)
    if PID_FILE.is_file():
        try:
            other = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            other = 0
        if other and other != os.getpid() and pid_alive(other):
            return False
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_instance_lock() -> None:
    try:
        if PID_FILE.is_file() and PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError:
        pass


class AutonomyGuard:
    """Decide en cada ciclo si el agente PUEDE trabajar. Devuelve siempre (permitido, motivo)."""

    def __init__(
        self,
        max_cycles_per_hour: int = 40,
        gpu_busy_pct: float = 70.0,
        max_temp_c: float = 83.0,
        check_bridge: bool = True,
    ) -> None:
        self.max_cycles_per_hour = max_cycles_per_hour
        self.gpu_busy_pct = gpu_busy_pct
        self.max_temp_c = max_temp_c
        self.check_bridge = check_bridge
        self._history: deque[float] = deque()

    def note_cycle(self) -> None:
        self._history.append(time.time())

    def _cycles_last_hour(self) -> int:
        cutoff = time.time() - 3600
        while self._history and self._history[0] < cutoff:
            self._history.popleft()
        return len(self._history)

    def _user_is_chatting(self) -> bool:
        if not self.check_bridge:
            return False
        try:
            with urllib.request.urlopen(BRIDGE_STATUS_URL, timeout=0.6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # Un "pensando" de mas de 10 min es un estado atascado, no una conversacion real; y el que
            # anuncia el propio agente (source=autonomous) no cuenta como Diego conversando.
            return bool(data.get("is_thinking")) and data.get("source") != "autonomous" and float(data.get("elapsed_s", 0.0)) < 600
        except Exception:
            return False  # sin puente no hay nadie conversando por el CLI

    def should_run(self) -> tuple[bool, str]:
        if STOP_FILE.is_file():
            return False, "STOP: parada solicitada (borra data/work/autonomy/STOP para reanudar)"
        if _training_lock_active():
            return False, "entrenamiento QLoRA en curso: la GPU esta reservada"
        if self._cycles_last_hour() >= self.max_cycles_per_hour:
            return False, f"presupuesto agotado ({self.max_cycles_per_hour} ciclos/hora)"
        if self._user_is_chatting():
            return False, "Diego esta conversando con Ada: cedo la GPU"
        # Mediana de 3 muestras: en un escritorio compartido (navegador, juegos) un solo pico de 1 s no es carga sostenida.
        samples = []
        for i in range(3):
            if i:
                time.sleep(0.3)
            samples.append(read_gpu_telemetry(max_age_s=0.0))
        temps = [s["temperature_c"] for s in samples if s["temperature_c"] is not None]
        utils = sorted(s["gpu_util_pct"] for s in samples if s["gpu_util_pct"] is not None)
        if temps and max(temps) > self.max_temp_c:
            return False, f"GPU demasiado caliente ({max(temps):.0f} C)"
        if utils and utils[len(utils) // 2] > self.gpu_busy_pct:
            return False, f"GPU ocupada por otra carga ({utils[len(utils) // 2]:.0f}% mediana)"
        return True, "ok"
