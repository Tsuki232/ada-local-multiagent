"""Gestor determinista de modelos locales en Ollama y control explicito de VRAM (12 GB)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from ada.config import ModelRoleConfig


@dataclass(frozen=True)
class LoadedModelStatus:
    """Estado de un modelo residente actualmente en VRAM segun /api/ps."""

    name: str
    size_bytes: int
    size_vram_bytes: int

    @property
    def size_vram_gib(self) -> float:
        return self.size_vram_bytes / (1024**3)


class OllamaVRAMManager:
    """Administra carga, descarga explicita (keep_alive=0) y conmutacion de modelos en Ollama."""

    def __init__(self, config: ModelRoleConfig | None = None, timeout_s: float = 120.0) -> None:
        self.config = config or ModelRoleConfig()
        self.host = self.config.ollama_api_host.rstrip("/")
        self.timeout_s = timeout_s

    def get_version(self) -> str:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{self.host}/api/version")
            resp.raise_for_status()
            return str(resp.json().get("version", "unknown"))

    def list_installed_models(self) -> list[str]:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [str(m["name"]) for m in models]

    def list_loaded_in_vram(self) -> list[LoadedModelStatus]:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.host}/api/ps")
            resp.raise_for_status()
            items = resp.json().get("models", [])
            return [
                LoadedModelStatus(
                    name=str(m.get("name", "")),
                    size_bytes=int(m.get("size", 0)),
                    size_vram_bytes=int(m.get("size_vram", 0)),
                )
                for m in items
            ]

    def unload_model(self, model_name: str) -> float:
        """Descarga inmediatamente un modelo de la VRAM enviando keep_alive=0. Devuelve segundos."""
        t0 = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.host}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
            )
            resp.raise_for_status()
        return time.perf_counter() - t0

    def unload_all(self) -> dict[str, float]:
        """Libera toda la VRAM ocupada actualmente por modelos en Ollama."""
        timings: dict[str, float] = {}
        for loaded in self.list_loaded_in_vram():
            timings[loaded.name] = self.unload_model(loaded.name)
        return timings

    def switch_to(self, target_model: str, keep_alive: str = "15m") -> dict[str, Any]:
        """Descarga cualquier otro modelo en VRAM y precalienta target_model midiendo tiempos reales."""
        unloaded: dict[str, float] = {}
        for loaded in self.list_loaded_in_vram():
            if loaded.name != target_model:
                unloaded[loaded.name] = self.unload_model(loaded.name)

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                f"{self.host}/api/generate",
                json={
                    "model": target_model,
                    "prompt": "",
                    "keep_alive": keep_alive,
                    "options": {"num_ctx": self.config.context_length},
                },
            )
            resp.raise_for_status()
        load_s = time.perf_counter() - t0
        active = self.list_loaded_in_vram()
        return {
            "target_model": target_model,
            "unloaded_previous_seconds": unloaded,
            "warmup_load_seconds": round(load_s, 3),
            "active_vram_models": [
                {"name": m.name, "vram_gib": round(m.size_vram_gib, 3)} for m in active
            ],
        }
