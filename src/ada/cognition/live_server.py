"""Servidor HTTP ligero de puente de estado en tiempo real (IPC / Live Bridge) para ADA.

Permite que la interfaz web (ada_neural_brain_simulation.html) reaccione en vivo a 60 FPS
cuando Ada Q o Ada r1 están procesando tokens o razonando en la consola terminal de PowerShell.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ada.autonomy.guard import AUTONOMY_DIR
from ada.autonomy.telemetry import read_gpu_telemetry
from ada.cognition.autonomous_mind import get_autonomous_mind

logger = logging.getLogger("ada.cognition.live_server")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765

_STATE_LOCK = threading.Lock()
_LIVE_STATE: dict[str, Any] = {
    "is_thinking": False,
    "source": "user",
    "last_user_activity_ts": 0.0,
    "bridge_started_at": time.time(),
    "current_prompt": "",
    "model": "ada-q",
    "started_at": 0.0,
    "elapsed_s": 0.0,
    "tokens_per_sec": 0.0,  # se actualiza con el valor medido en cada respuesta
    "active_tokens": [],
    "total_grown_neurons": 133,
    "total_verified_cycles": 18,
    "latest_thought": "En reposo estático. Esperando consulta de Diego López o activación de auto-reflexión...",
    "gpu_util_pct": 1.0,
    "vram_used_mib": 11137.0,
    "vram_total_mib": 12227.0,
    "temperature_c": 47.0,
    "timestamp": time.time(),
}

_SERVER_INSTANCE: ThreadingHTTPServer | None = None
_SERVER_THREAD: threading.Thread | None = None


def _heart_snapshot() -> dict[str, Any] | None:
    """Sentimientos actuales de Ada (lee el nucleo afectivo persistido; no escribe nada)."""
    try:
        from ada.autonomy.heart import Heart

        return Heart().snapshot()
    except Exception:  # noqa: BLE001
        return None


def _vision_snapshot() -> dict[str, Any]:
    """Metricas de la sesion de vision en vivo de ESTE proceso (sin importar cv2 si nunca se uso)."""
    if "ada.vision.live_camera" not in sys.modules:
        return {"active": False}
    from ada.vision.live_camera import get_vision_session

    session = get_vision_session()
    return session.snapshot() if session else {"active": False}


def _agent_snapshot(journal_lines: int = 8) -> dict[str, Any] | None:
    """Estado persistido del agente autonomo y cola de su diario (lo escribe el daemon u otro proceso)."""
    state_file = AUTONOMY_DIR / "state.json"
    if not state_file.is_file():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state.pop("seen", None)
        journal: list[Any] = []
        jf = AUTONOMY_DIR / "journal.jsonl"
        if jf.is_file():
            with open(jf, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 16384))
                tail = fh.read().decode("utf-8", errors="ignore").splitlines()
            for line in tail[-journal_lines:]:
                try:
                    j = json.loads(line)
                    journal.append({k: j.get(k) for k in ("ts", "status", "kind", "level", "detail", "methods", "latency_s", "thought", "level_change", "question", "reflection")})
                except json.JSONDecodeError:
                    continue
        return {"state": state, "journal": journal}
    except (OSError, json.JSONDecodeError):
        return None


def get_live_state() -> dict[str, Any]:
    """Obtiene una copia consistente del estado en tiempo real, integrando la mente autonoma."""
    mind = get_autonomous_mind()
    telemetry = read_gpu_telemetry()
    with _STATE_LOCK:
        state = dict(_LIVE_STATE)
        if state["is_thinking"] and state["started_at"] > 0:
            state["elapsed_s"] = round(time.perf_counter() - state["started_at"], 2)

        # Telemetria fisica real (nvidia-smi); si no esta disponible se conserva el ultimo valor conocido.
        state["telemetry_source"] = telemetry["source"]
        for key in ("gpu_util_pct", "vram_used_mib", "vram_total_mib", "temperature_c"):
            if telemetry[key] is not None:
                state[key] = telemetry[key]

        state["agent"] = _agent_snapshot()
        state["vision"] = _vision_snapshot()
        state["heart"] = _heart_snapshot()
        state["total_grown_neurons"] = mind.total_grown_neurons
        state["total_verified_cycles"] = mind.total_verified_cycles
        latest = mind.get_latest_thoughts(limit=1)
        if latest and not state["is_thinking"]:
            state["latest_thought"] = latest[0]
        state["timestamp"] = time.time()
        return state


def set_live_thinking(
    is_thinking: bool,
    prompt: str = "",
    model: str = "ada-q",
    tokens: list[str] | None = None,
    last_response: str = "",
    tokens_per_sec: float | None = None,
    source: str = "user",
) -> None:
    """Actualiza de forma atomica el estado de inferencia de Ada. `tokens_per_sec` debe ser un valor medido.

    `source` es "user" (Diego conversa) o "autonomous" (el agente piensa solo). Lo autonomo NUNCA pisa lo de Diego:
    ni arranca mientras el usuario esta generando, ni apaga un estado que no es suyo.
    """
    with _STATE_LOCK:
        if source == "autonomous":
            if is_thinking and _LIVE_STATE["is_thinking"] and _LIVE_STATE.get("source") == "user":
                return
            if not is_thinking and _LIVE_STATE.get("source") != "autonomous":
                if tokens_per_sec is not None and not _LIVE_STATE["is_thinking"]:
                    _LIVE_STATE["tokens_per_sec"] = round(float(tokens_per_sec), 1)
                return
        _LIVE_STATE["source"] = source
        _LIVE_STATE["is_thinking"] = is_thinking
        _LIVE_STATE["model"] = model
        if source == "user":
            _LIVE_STATE["last_user_activity_ts"] = time.time()  # base para saber cuando Diego lleva rato sin usar a Ada
        if tokens_per_sec is not None:
            _LIVE_STATE["tokens_per_sec"] = round(float(tokens_per_sec), 1)
        if is_thinking:
            _LIVE_STATE["current_prompt"] = prompt
            _LIVE_STATE["started_at"] = time.perf_counter()
            _LIVE_STATE["elapsed_s"] = 0.0
            if tokens:
                _LIVE_STATE["active_tokens"] = tokens[:12]
            elif prompt:
                # Extraer hasta 10 tokens representativos del prompt para la matriz de atencion
                clean_words = [w for w in prompt.strip().split() if w]
                _LIVE_STATE["active_tokens"] = clean_words[:10]
        else:
            _LIVE_STATE["started_at"] = 0.0
            if last_response:
                _LIVE_STATE["latest_thought"] = (
                    f"Respuesta completada para: \"{_LIVE_STATE.get('current_prompt', '')}\". Red en reposo estático."
                )


class _LiveBridgeRequestHandler(BaseHTTPRequestHandler):
    """Manejador HTTP con soporte CORS completo para navegadores y file:///."""

    def log_message(self, format: str, *args: Any) -> None:
        # Silenciar logs ruidosos en la consola
        pass

    def _send_cors_headers(self, status_code: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status_code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control")
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._send_cors_headers(204, "text/plain")

    def do_GET(self) -> None:
        if self.path in {"/api/status", "/api/status/"}:
            data = get_live_state()
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send_cors_headers(200, "application/json; charset=utf-8")
            self.wfile.write(body)
        elif self.path.split("?")[0] == "/api/vision":
            body = json.dumps(_vision_snapshot(), ensure_ascii=False).encode("utf-8")
            self._send_cors_headers(200, "application/json; charset=utf-8")
            self.wfile.write(body)
        elif self.path.split("?")[0] == "/api/vision/frame.jpg":
            from ada.vision.live_camera import get_vision_session

            session = get_vision_session()
            jpeg = session.latest_jpeg() if session else None
            if jpeg is None:
                self.send_error(404, "Sin frame: la vision en vivo no esta activa")
            else:
                self._send_cors_headers(200, "image/jpeg")
                self.wfile.write(jpeg)
        elif self.path in {"/api/ping", "/api/ping/"}:
            body = b'{"status": "ok", "service": "ada_live_bridge"}'
            self._send_cors_headers(200, "application/json; charset=utf-8")
            self.wfile.write(body)
        else:
            self.send_error(404, "Endpoint no encontrado")

    def do_POST(self) -> None:
        if self.path in {"/api/status", "/api/status/"}:
            content_length = int(self.headers.get("Content-Length", 0))
            payload = self.rfile.read(content_length)
            try:
                data = json.loads(payload.decode("utf-8"))
                set_live_thinking(
                    is_thinking=bool(data.get("is_thinking", False)),
                    prompt=str(data.get("current_prompt", "")),
                    model=str(data.get("model", "ada-q")),
                    tokens=data.get("active_tokens"),
                    tokens_per_sec=data.get("tokens_per_sec"),
                    source="autonomous" if data.get("source") == "autonomous" else "user",
                )
                self._send_cors_headers(200, "application/json; charset=utf-8")
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_error(400, f"JSON invalido: {e}")
        else:
            self.send_error(404, "Endpoint no encontrado")


def is_port_in_use(port: int, host: str = _DEFAULT_HOST) -> bool:
    """Comprueba si el puerto local ya esta abierto."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_live_bridge(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> bool:
    """Inicia el servidor de puente de estado en un hilo secundario si no esta ya activo."""
    global _SERVER_INSTANCE, _SERVER_THREAD

    if _SERVER_INSTANCE is not None:
        return True

    if is_port_in_use(port, host):
        logger.info(f"El puerto {port} ya esta ocupado o activo.")
        return True

    try:
        server = ThreadingHTTPServer((host, port), _LiveBridgeRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="AdaLiveBridgeServer")
        thread.start()

        _SERVER_INSTANCE = server
        _SERVER_THREAD = thread
        logger.info(f"ADA Live Bridge iniciado exitosamente en http://{host}:{port}/api/status")
        return True
    except Exception as e:
        logger.warning(f"No se pudo iniciar ADA Live Bridge en {host}:{port}: {e}")
        return False


def stop_live_bridge() -> None:
    """Detiene el servidor de puente de estado."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    if _SERVER_INSTANCE is not None:
        try:
            _SERVER_INSTANCE.shutdown()
            _SERVER_INSTANCE.server_close()
        except Exception:
            pass
        _SERVER_INSTANCE = None
        _SERVER_THREAD = None
