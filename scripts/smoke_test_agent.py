"""Prueba de humo Fase 1: Conmutacion VRAM en Ollama + 'Hola Mundo' con Tool Calling real en DeepSeek Harness."""

from __future__ import annotations

import json
import time
import uuid

from deepseek_harness import DeepSeekHarness

from ada.config import DSH_HOME_DIR, PROJECT_ROOT, ModelRoleConfig, resolve_dsh_launch_args
from ada.tools.ollama_manager import OllamaVRAMManager


def main() -> int:
    cfg = ModelRoleConfig()
    mgr = OllamaVRAMManager(cfg)
    ollama_ver = mgr.get_version()
    installed = mgr.list_installed_models()

    for required in (cfg.orchestrator_ollama, cfg.reasoner_ollama):
        if not any(m == required or m.startswith(f"{required}:") for m in installed):
            raise RuntimeError(f"Modelo requerido '{required}' no encontrado en Ollama. Instalados: {installed}")

    # 1. Medir conmutacion limpia de VRAM en 12 GB: cargar Razonador (deepseek-r1:14b) -> descargar -> cargar Orquestador (qwen3:14b)
    swap_to_reasoner = mgr.switch_to(cfg.reasoner_ollama)
    swap_to_orchestrator = mgr.switch_to(cfg.orchestrator_ollama)

    # 2. Ejecutar "Hola Mundo" agéntico real en DeepSeek Harness usando una herramienta de lectura sobre README.md
    DSH_HOME_DIR.mkdir(parents=True, exist_ok=True)
    patch_path = str((PROJECT_ROOT / "config" / "harness" / "cordis.patch.yml").resolve())
    session_id = f"smoke-phase1-{uuid.uuid4().hex[:8]}"

    prompt = (
        "Hola, eres ADA. Usa tu herramienta de lectura de archivos para leer la primera linea "
        "de README.md en este workspace y responde en espanol en maximo 2 oraciones incluyendo "
        "esa primera linea textual."
    )

    t0 = time.perf_counter()
    tool_events: list[str] = []

    def _on_notif(notif) -> None:
        method = getattr(notif, "method", "")
        if "tool" in method.lower():
            tool_events.append(method)

    launch_args, base_env = resolve_dsh_launch_args(profile="sdk", patches=(patch_path,))
    run_env = {
        **base_env,
        "OLLAMA_API_KEY": "ollama-local",
        "DSH_PERMISSION_MODE": "read-only",
    }

    with DeepSeekHarness(
        provider="ollama-local",
        model=cfg.orchestrator_ollama,
        reasoning_effort="off",
        cwd=str(PROJECT_ROOT),
        dsh_home=str(DSH_HOME_DIR),
        patches=(patch_path,),
        env=run_env,
        initialize_timeout_seconds=60.0,
        _launch_args=launch_args,
    ) as harness:
        result = harness.run(prompt, session_id=session_id, on_notification=_on_notif)
    elapsed_s = time.perf_counter() - t0

    if result.finish_reason != "completed" or not result.final_response.strip():
        raise RuntimeError(
            f"El turno en DeepSeek Harness fallo: finish_reason={result.finish_reason}, "
            f"final_response={result.final_response!r}"
        )

    summary = {
        "ollama_version": ollama_ver,
        "installed_models": installed,
        "vram_swap_benchmark": {
            "to_reasoner_deepseek_r1_14b": swap_to_reasoner,
            "to_orchestrator_qwen3_14b": swap_to_orchestrator,
        },
        "deepseek_harness_hello_world": {
            "session_id": result.session_id,
            "provider": "ollama-local",
            "model": cfg.orchestrator_ollama,
            "sandbox_mode": "read-only",
            "finish_reason": result.finish_reason,
            "elapsed_seconds": round(elapsed_s, 2),
            "events_count": len(result.events),
            "notifications_count": len(result.notifications),
            "tool_notification_methods": sorted(set(tool_events)),
            "final_response": result.final_response.strip(),
        },
        "status": "OK",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
