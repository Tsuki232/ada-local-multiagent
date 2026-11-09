"""Telemetria real de la GPU via nvidia-smi (sin valores inventados)."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}

_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu"


def read_gpu_telemetry(max_age_s: float = 1.0) -> dict[str, Any]:
    """Lee uso de GPU, VRAM y temperatura reales. Si nvidia-smi falla devuelve source='unavailable' y None."""
    now = time.monotonic()
    with _CACHE_LOCK:
        if _CACHE["data"] is not None and now - _CACHE["ts"] < max_age_s:
            return dict(_CACHE["data"])

    data: dict[str, Any] = {
        "gpu_util_pct": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "vram_free_mib": None,
        "temperature_c": None,
        "source": "unavailable",
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip().splitlines()[0]
        util, used, total, temp = (float(p.strip()) for p in out.split(","))
        data.update(
            gpu_util_pct=util,
            vram_used_mib=used,
            vram_total_mib=total,
            vram_free_mib=round(total - used, 1),
            temperature_c=temp,
            source="nvidia-smi",
        )
    except Exception:
        pass

    with _CACHE_LOCK:
        _CACHE["ts"] = now
        _CACHE["data"] = dict(data)
    return data
