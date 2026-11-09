"""Puente ADB de alta velocidad para captura de video y camara de celular en tiempo real."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def find_adb_executable() -> str:
    """Localiza el binario de adb.exe en PATH o en las rutas estandar de Android Platform Tools."""
    on_path = shutil.which("adb")
    if on_path:
        return on_path

    known_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        / "Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe" / "platform-tools" / "adb.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path("C:/Program Files/Android/platform-tools/adb.exe"),
    ]
    for p in known_paths:
        if p.is_file():
            return str(p)
    return "adb"


def find_scrcpy_executable() -> str | None:
    """Localiza scrcpy.exe en PATH o en las rutas de WinGet."""
    on_path = shutil.which("scrcpy")
    if on_path:
        return on_path

    winget_pkg_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_pkg_dir.is_dir():
        for exe in winget_pkg_dir.glob("**/scrcpy.exe"):
            if exe.is_file():
                return str(exe)
    return None


@dataclass
class ADBDeviceInfo:
    serial: str
    state: str
    model: str = "Unknown"
    product: str = "Unknown"
    device: str = "Unknown"
    transport_id: str = ""
    battery_level: int | None = None
    screen_resolution: str = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "serial": self.serial,
            "state": self.state,
            "model": self.model,
            "product": self.product,
            "device": self.device,
            "transport_id": self.transport_id,
            "battery_level": self.battery_level,
            "screen_resolution": self.screen_resolution,
            "ready": self.state == "device",
        }


def list_adb_devices() -> list[ADBDeviceInfo]:
    """Lista todos los dispositivos Android conectados mediante ADB con detalles de estado."""
    adb_cmd = find_adb_executable()
    try:
        proc = subprocess.run(
            [adb_cmd, "devices", "-l"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc.returncode != 0:
            return []
    except Exception:
        return []

    lines = proc.stdout.strip().splitlines()
    devices: list[ADBDeviceInfo] = []

    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial = parts[0]
        state = parts[1]

        info = ADBDeviceInfo(serial=serial, state=state)
        for token in parts[2:]:
            if ":" in token:
                k, v = token.split(":", 1)
                if k == "model":
                    info.model = v
                elif k == "product":
                    info.product = v
                elif k == "device":
                    info.device = v
                elif k == "transport_id":
                    info.transport_id = v

        # Si el dispositivo esta autorizado y listo, consultamos bateria y resolucion rapida
        if state == "device":
            try:
                bat_proc = subprocess.run(
                    [adb_cmd, "-s", serial, "shell", "dumpsys", "battery"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                m = re.search(r"level:\s*(\d+)", bat_proc.stdout)
                if m:
                    info.battery_level = int(m.group(1))

                size_proc = subprocess.run(
                    [adb_cmd, "-s", serial, "shell", "wm", "size"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                m_size = re.search(r"Physical size:\s*([0-9x]+)", size_proc.stdout)
                if m_size:
                    info.screen_resolution = m_size.group(1)
            except Exception:
                pass

        devices.append(info)

    return devices


def capture_adb_frame(serial: str | None = None, max_dimension: int = 1280) -> np.ndarray | None:
    """Captura un frame crudo del celular conectado mediante `adb exec-out screencap -p`.
    
    Retorna una matriz numpy BGR lista para OpenCV o None si no hay dispositivo.
    """
    adb_cmd = find_adb_executable()
    cmd = [adb_cmd]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["exec-out", "screencap", "-p"])

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3.0, check=False)
        if proc.returncode != 0 or not proc.stdout:
            return None
        raw_bytes = proc.stdout
        # En Windows a veces screencap puede transformar CRLF, reemplazamos si es necesario
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            pass
        else:
            raw_bytes = raw_bytes.replace(b"\r\n", b"\n")

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame
    except Exception:
        return None


def forward_adb_port(local_port: int = 8080, remote_port: int = 8080, serial: str | None = None) -> bool:
    """Configura reenvio de puertos ADB (`adb forward tcp:X tcp:Y`) para streaming de camara ultra rapido."""
    adb_cmd = find_adb_executable()
    cmd = [adb_cmd]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["forward", f"tcp:{local_port}", f"tcp:{remote_port}"])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0, check=False)
        return proc.returncode == 0
    except Exception:
        return False
