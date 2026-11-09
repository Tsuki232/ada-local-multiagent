"""Vision en vivo de ADA: fuentes de frames del celular (o simulador), lazo de analisis y estado para el puente.

Cadena de fuentes (la primera que entregue un frame de verdad gana; se informa cual y por que):
  1. MJPEG por ADB      app de camara en el celular (p. ej. "IP Webcam") + `adb forward tcp:8080 tcp:8080`.
                        ~30 fps, es la camara real del celular, y OpenCV la lee de forma nativa.
  2. Pantalla por ADB   `adb exec-out screencap`: ~3-6 fps, sin instalar nada; ve la camara solo si la app de
                        camara esta abierta en la pantalla del celular.
  3. Simulador          laboratorio de pendulo sintetico (sin hardware).
Nunca se presenta una fuente como otra: `source_label` dice exactamente que se esta viendo.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any, Protocol

import cv2
import numpy as np

from ada.vision.adb_bridge import capture_adb_frame, forward_adb_port, list_adb_devices
from ada.vision.realtime_tracker import RealTimePhysicsTracker
from ada.vision.synthetic_stream import SyntheticPhysicsLaboratoryStream


class FrameSource(Protocol):
    label: str

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class SyntheticSource:
    label = "Simulador de laboratorio (sin celular)"

    def __init__(self, fps: float = 30.0) -> None:
        self._sim = SyntheticPhysicsLaboratoryStream(width=800, height=600)
        self._period = 1.0 / fps
        self._last = 0.0

    def read(self) -> np.ndarray | None:
        wait = self._period - (time.perf_counter() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.perf_counter()
        return self._sim.read_frame()

    def close(self) -> None:
        pass


class MJPEGSource:
    label = "Camara del celular (MJPEG por ADB)"

    def __init__(self, url: str) -> None:
        self.url = url
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray | None:
        if not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class ScreencapSource:
    label = "Pantalla del celular (screencap por ADB)"

    def __init__(self, serial: str | None) -> None:
        self.serial = serial

    def read(self) -> np.ndarray | None:
        return capture_adb_frame(self.serial)

    def close(self) -> None:
        pass


def _probe(source: FrameSource, tries: int = 4, timeout_s: float = 4.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    for _ in range(tries):
        if time.perf_counter() > deadline:
            break
        frame = source.read()
        if frame is not None and frame.size:
            return True
        time.sleep(0.15)
    return False


def open_best_source(serial: str | None = None, mjpeg_port: int = 8080) -> tuple[FrameSource, dict[str, Any]]:
    """Elige la mejor fuente disponible y devuelve (fuente, informe de lo intentado)."""
    attempts: list[dict[str, Any]] = []
    try:
        devices = list_adb_devices()
    except Exception as exc:  # adb ausente o roto: se informa y se sigue con el simulador
        devices = []
        attempts.append({"backend": "adb", "ok": False, "reason": f"adb no disponible: {exc}"})

    online = [d for d in devices if getattr(d, "state", "device") == "device"]
    if not online:
        attempts.append({"backend": "adb", "ok": False, "reason": "no hay celular Android conectado con depuracion USB autorizada"})
    else:
        dev = next((d for d in online if serial and d.serial == serial), online[0])
        if forward_adb_port(mjpeg_port, mjpeg_port, dev.serial):
            src: FrameSource = MJPEGSource(f"http://127.0.0.1:{mjpeg_port}/video")
            if _probe(src):
                attempts.append({"backend": "mjpeg", "ok": True})
                return src, {"chosen": src.label, "serial": dev.serial, "attempts": attempts}
            src.close()
            attempts.append({"backend": "mjpeg", "ok": False, "reason": f"nada responde en el puerto {mjpeg_port} del celular: abre una app de camara con servidor MJPEG (p. ej. IP Webcam) y pulsa 'Iniciar servidor'"})
        else:
            attempts.append({"backend": "mjpeg", "ok": False, "reason": "adb forward fallo"})
        scr: FrameSource = ScreencapSource(dev.serial)
        if _probe(scr):
            attempts.append({"backend": "screencap", "ok": True})
            return scr, {"chosen": scr.label, "serial": dev.serial, "attempts": attempts}
        attempts.append({"backend": "screencap", "ok": False, "reason": "adb no pudo capturar la pantalla"})

    sim = SyntheticSource()
    attempts.append({"backend": "synthetic", "ok": True})
    return sim, {"chosen": sim.label, "serial": None, "attempts": attempts}


class LiveVisionSession:
    """Lazo continuo: frame -> seguimiento y cinematica real -> metricas + ultimo frame anotado en JPEG."""

    def __init__(self, source: FrameSource, tracker: RealTimePhysicsTracker | None = None, jpeg_quality: int = 70) -> None:
        self.source = source
        self.tracker = tracker or RealTimePhysicsTracker()
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jpeg: bytes | None = None
        self._metrics: dict[str, Any] = {"active": False}
        self._frames = 0
        self._failures = 0
        self._fps = 0.0
        self.error: str | None = None

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AdaLiveVision")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.source.close()
        with self._lock:
            self._metrics = {**self._metrics, "active": False}

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        last_done = time.perf_counter()
        while not self._stop.is_set():
            t_read = time.perf_counter()
            frame = self.source.read()
            if frame is None:
                self._failures += 1
                if self._failures >= 30:
                    self.error = f"la fuente '{self.source.label}' dejo de entregar frames"
                    with self._lock:
                        self._metrics = {**self._metrics, "active": False, "error": self.error}
                    return
                time.sleep(0.1)
                continue
            self._failures = 0
            t_proc = time.perf_counter()
            annotated, kin = self.tracker.process_frame(frame)
            t_done = time.perf_counter()

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            inst = 1.0 / max(1e-6, t_done - last_done)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            last_done = t_done
            self._frames += 1

            metrics: dict[str, Any] = {
                "active": True,
                "source_label": self.source.label,
                "frames": self._frames,
                "fps": round(self._fps, 1),
                "read_ms": round((t_proc - t_read) * 1000, 1),
                "analysis_ms": round((t_done - t_proc) * 1000, 1),
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "target_found": kin is not None,
                "timestamp": time.time(),
            }
            if kin is not None:
                k = asdict(kin)
                metrics["kinematics"] = {key: (round(v, 4) if isinstance(v, float) else v) for key, v in k.items() if key != "timestamp"}
                metrics["verification_note"] = (
                    "El sello comprueba que dos integradores (DOP853 y Radau) coinciden entre si con los parametros estimados; "
                    "no valida la medicion de la camara."
                )
            with self._lock:
                if ok:
                    self._jpeg = buf.tobytes()
                self._metrics = metrics

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg


_SESSION: LiveVisionSession | None = None
_SESSION_REPORT: dict[str, Any] = {}


def start_vision(serial: str | None = None) -> dict[str, Any]:
    """Abre la mejor fuente y arranca el lazo. Si ya hay una sesion activa, la devuelve sin duplicar."""
    global _SESSION, _SESSION_REPORT
    if _SESSION and _SESSION.running:
        return {"already_running": True, **_SESSION_REPORT}
    source, report = open_best_source(serial)
    _SESSION = LiveVisionSession(source)
    _SESSION.start()
    _SESSION_REPORT = report
    return report


def stop_vision() -> None:
    global _SESSION
    if _SESSION:
        _SESSION.stop()
        _SESSION = None


def get_vision_session() -> LiveVisionSession | None:
    return _SESSION
