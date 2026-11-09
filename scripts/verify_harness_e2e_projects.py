"""Evaluacion sin entrenamiento (Zero-Shot Out-of-the-Box) de qwen3:14b + MCP ADA en DeepSeek Harness."""

from __future__ import annotations

import json
import sys
import time
import uuid

from deepseek_harness import DeepSeekHarness

from ada.config import DSH_HOME_DIR, OUTPUTS_DIR, PROJECT_ROOT, ModelRoleConfig, assert_safe_write_path, resolve_dsh_launch_args
from ada.ingest.demo_assets import create_graphical_ode_image, create_sample_sqlite_database
from ada.tools.ollama_manager import OllamaVRAMManager


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = ModelRoleConfig()
    mgr = OllamaVRAMManager(cfg)
    mgr.switch_to(cfg.orchestrator_ollama)

    db_path = create_sample_sqlite_database()
    img_path = create_graphical_ode_image()

    DSH_HOME_DIR.mkdir(parents=True, exist_ok=True)
    patch_path = str((PROJECT_ROOT / "config" / "harness" / "cordis.patch.yml").resolve())
    launch_args, base_env = resolve_dsh_launch_args(profile="sdk", patches=(patch_path,))
    # Sin OLLAMA_API_KEY en el entorno: verifica tambien autenticacion keyless via headers!
    run_env = {
        **base_env,
        "DSH_PERMISSION_MODE": "read-only",
    }

    tool_calls_seen: list[str] = []

    def _on_notif(notif) -> None:
        raw = str(notif)
        if "mcp__ada__" in raw:
            tool_calls_seen.append(raw[:160])

    with DeepSeekHarness(
        provider="ollama-local",
        model=cfg.orchestrator_ollama,
        reasoning_effort="off",
        cwd=str(PROJECT_ROOT),
        dsh_home=str(DSH_HOME_DIR),
        patches=(patch_path,),
        env=run_env,
        initialize_timeout_seconds=90.0,
        _launch_args=launch_args,
    ) as harness:
        # --- PRUEBA E2E PROYECTO 1: Limpieza + Prediccion sobre Base SQLite Real ---
        t1 = time.perf_counter()
        prompt_p1 = (
            f"Tenemos la base SQLite real en '{db_path.as_posix()}' (tabla 'turbine_telemetry'). "
            "1) Usa `mcp__ada__ada_clean_filter_adapt_data` con table_or_sheet='turbine_telemetry', "
            "where_sql='vibration_mms < 12.0', missing_strategy='keep' y output_filename='e2e_turbine_clean.parquet'. "
            "2) Luego usa `mcp__ada__ada_predict_from_database` sobre el archivo limpio resultante para predecir "
            "target_column='power_output_kw'. Resume brevemente en espanol el mejor modelo, su R^2, las alertas del Critico "
            "y la ruta del dashboard HTML generado."
        )
        res_p1 = harness.run(prompt_p1, session_id=f"e2e-proj1-{uuid.uuid4().hex[:6]}", on_notification=_on_notif)
        dt1 = round(time.perf_counter() - t1, 2)

        # --- PRUEBA E2E PROYECTO 2: EDO Grafica + Pendulo '¿Cuanto vale la aprox lineal?' + Control Electronico + Animacion ---
        t2 = time.perf_counter()
        prompt_p2 = (
            f"1) Lee la ecuacion diferencial escrita en la imagen '{img_path.as_posix()}' usando `mcp__ada__ada_read_graphical_equation`. "
            "2) Evalua el proyecto del pendulo '¿Cuanto vale la aproximacion lineal?' para theta0_deg=30.0, tolerance_pct=1.0 y n_cycles=10 "
            "usando `mcp__ada__ada_pendulum_approximation_lab` para generar la animacion HTML5 interactiva. "
            "3) Toma control del dispositivo electronico virtual (`device_port='virtual://pendulum-mcu'`) usando "
            "`mcp__ada__ada_electronic_devices_control` con action='run', allow_actuation=true, control_mode='nonlinear_computed_torque', "
            "target_angle_deg=15.0, initial_angle_deg=-20.0, duration_s=4.0. "
            "Responde en espanol reportando los valores verificados y las rutas de las animaciones HTML generadas."
        )
        res_p2 = harness.run(prompt_p2, session_id=f"e2e-proj2-{uuid.uuid4().hex[:6]}", on_notification=_on_notif)
        dt2 = round(time.perf_counter() - t2, 2)

    report = {
        "model_evaluated_without_training": cfg.orchestrator_ollama,
        "project_1_e2e": {
            "finish_reason": res_p1.finish_reason,
            "elapsed_seconds": dt1,
            "events_count": len(res_p1.events),
            "final_response": res_p1.final_response.strip(),
        },
        "project_2_e2e": {
            "finish_reason": res_p2.finish_reason,
            "elapsed_seconds": dt2,
            "events_count": len(res_p2.events),
            "final_response": res_p2.final_response.strip(),
        },
        "mcp_notifications_detected": len(tool_calls_seen),
    }
    out_report = assert_safe_write_path(OUTPUTS_DIR / "e2e_zero_shot_evaluation.json")
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
