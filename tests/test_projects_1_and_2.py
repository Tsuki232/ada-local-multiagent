"""Tests reproducibles extremo a extremo para los Proyectos 1 y 2 de ADA."""

from __future__ import annotations

from pathlib import Path

import pytest

from ada.calculus.verified_math import (
    extract_equation_from_image,
    solve_and_verify_ode,
    verify_definite_integral,
    verify_derivative,
)
from ada.clean.pipeline import clean_and_adapt_dataset
from ada.config import DATA_RAW_DIR, SecurityViolationError, assert_safe_write_path
from ada.ingest.loader import profile_dataset
from ada.pendulum.hardware_bridge import (
    VIRTUAL_DEVICE_URI,
    list_electronic_devices,
    run_pendulum_hardware_control,
)
from ada.pendulum.simulator import (
    analyze_pendulum_linear_approximation,
    build_pendulum_interactive_html,
)
from ada.stats.predictor import evaluate_and_predict_database
from ada.ingest.demo_assets import (
    create_graphical_ode_image,
    create_sample_sqlite_database,
)


def test_immutability_guard_blocks_data_raw() -> None:
    with pytest.raises(SecurityViolationError):
        assert_safe_write_path(DATA_RAW_DIR / "forbidden.csv")


def test_project1_database_eval_clean_and_predict() -> None:
    db_path = create_sample_sqlite_database()

    prof = profile_dataset(db_path, table_or_sheet="turbine_telemetry")
    assert prof["verified"] is True
    assert prof["n_rows"] == 245
    assert prof["duplicate_rows"] == 5

    cleaned = clean_and_adapt_dataset(
        db_path,
        table_or_sheet="turbine_telemetry",
        where_sql="vibration_mms < 12.0",
        order_by_sql="motor_rpm ASC",
        drop_duplicates=True,
        missing_strategy="keep",
        output_filename="turbine_telemetry_clean.parquet",
    )
    assert cleaned["verified"] is True
    assert cleaned["source_immutable_verified"] is True
    assert cleaned["duplicates_removed"] == 5
    assert Path(cleaned["output_path"]).is_file()

    pred = evaluate_and_predict_database(
        cleaned["output_path"],
        target_column="power_output_kw",
        task_type="regression",
        seed=42,
    )
    assert pred["verified"] is True
    assert pred["best_model"] in {"RidgeLinear", "HistGradientBoosting"}
    best_r2 = pred["models"][pred["best_model"]]["r2"]
    dummy_r2 = pred["models"]["DummyBaseline"]["r2"]
    assert best_r2 > 0.90
    assert best_r2 > dummy_r2 + 0.50
    assert pred["predictions_scope"] == "unlabeled_rows_in_source"
    assert len(pred["sample_predictions"]) >= 5
    assert Path(pred["predictions_csv_path"]).is_file()
    assert Path(pred["interactive_dashboard_html"]).is_file()


def test_project2_graphical_ode_pendulum_and_hardware_control() -> None:
    # 1. Calculo simbolico vs numerico verificado
    d_res = verify_derivative("sin(theta) - theta", variable="theta", order=3, eval_point=0.4)
    assert d_res["verified"] is True
    i_res = verify_definite_integral("sin(x)**2", variable="x", lower_limit=0.0, upper_limit=3.141592653589793)
    assert i_res["verified"] is True

    # 2. Lectura OCR de EDO escrita graficamente + integracion verificada DOP853 vs Radau
    img_path = create_graphical_ode_image()
    ocr_out = extract_equation_from_image(img_path)
    cand_rhs = ocr_out["parsed_ode_hint"]["candidate_ode_rhs"]
    assert "sin" in cand_rhs.lower()

    ode_res = solve_and_verify_ode(
        cand_rhs,
        order=2,
        t0=0.0,
        t_end=4.0,
        y0=0.5,
        dy0=0.0,
    )
    assert ode_res["verified"] is True
    assert ode_res["max_abs_discrepancy"] < 1e-7

    # 3. Proyecto del Pendulo: '¿Cuanto vale la aproximacion lineal?' + Animacion HTML5 60 FPS
    pend = analyze_pendulum_linear_approximation(
        length_m=1.0,
        gravity_ms2=9.80665,
        theta0_deg=30.0,
        tolerance_pct=1.0,
        n_cycles=10,
    )
    assert pend["verified"] is True
    assert pend["dual_period_relative_discrepancy"] < 1e-10
    assert 13.8 < pend["critical_angles_for_tolerance"]["max_angle_for_force_error_deg"] < 14.2
    assert 22.8 < pend["critical_angles_for_tolerance"]["max_angle_for_period_error_deg"] < 23.3

    anim = build_pendulum_interactive_html(theta0_deg=35.0, tolerance_pct=1.0)
    assert Path(anim["interactive_html_path"]).is_file()

    # 4. Control seguro de dispositivos electronicos (enclavamiento por defecto + lazo cerrado PD/NoLineal)
    devs = list_electronic_devices()
    assert devs["virtual_device_available"]["port"] == VIRTUAL_DEVICE_URI

    locked = run_pendulum_hardware_control(
        device_port=VIRTUAL_DEVICE_URI,
        allow_actuation=False,
        initial_angle_deg=20.0,
        duration_s=1.0,
    )
    assert locked["actuation_enabled"] is False

    controlled = run_pendulum_hardware_control(
        device_port=VIRTUAL_DEVICE_URI,
        control_mode="nonlinear_computed_torque",
        allow_actuation=True,
        target_angle_deg=15.0,
        initial_angle_deg=-20.0,
        duration_s=5.0,
    )
    assert controlled["actuation_enabled"] is True
    assert controlled["final_tracking_error_deg"] < 1.5
    assert Path(controlled["telemetry_csv_path"]).is_file()
    assert Path(controlled["animated_replay_html"]).is_file()


@pytest.mark.asyncio
async def test_mcp_server_registers_all_9_tools() -> None:
    from ada.tools.mcp_server import mcp
    from ada.viz.neural_canvas import build_neural_network_live_canvas

    canvas_res = build_neural_network_live_canvas(open_in_browser=False)
    assert canvas_res["verified"] is True
    assert canvas_res["max_attention_discrepancy"] < 1e-9
    assert Path(canvas_res["interactive_html_path"]).is_file()

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "ada_profile_database",
        "ada_clean_filter_adapt_data",
        "ada_predict_from_database",
        "ada_verify_calculus",
        "ada_solve_verified_ode",
        "ada_read_graphical_equation",
        "ada_pendulum_approximation_lab",
        "ada_electronic_devices_control",
        "ada_consult_ada_r1",
        "ada_neural_network_live_canvas",
        "ada_build_math_textbook",
    }
    assert expected.issubset(names)
