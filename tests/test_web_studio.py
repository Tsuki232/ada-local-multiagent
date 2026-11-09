"""Pruebas verificadas del motor ADA Web Design Studio v2.0 y Auditor DOM/CSS/Sandbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from ada.viz.web_studio import (
    audit_html_quality,
    build_canvas_animation_studio,
    build_interactive_webapp,
)
from ada.viz.brain_simulator import build_neural_brain_simulation


def test_auditor_catches_notebook_sheet_and_orphan_classes(tmp_path: Path) -> None:
    broken_html = tmp_path / "broken_notebook.html"
    broken_html.write_text(
        """<!DOCTYPE html>
<html>
<head><style>
  body { background-color: #f0f4f8; color: #222; }
  .card { background: #ffffff; padding: 20px; }
</style></head>
<body>
  <div class="grid">
    <div class="card">
      <div class="bar-row"><div class="bar-fill" style="width:50%"></div></div>
      <canvas id="c" width="400" height="200"></canvas>
    </div>
  </div>
  <script>
    const t = localStorage.getItem("theme");
  </script>
</body></html>""",
        encoding="utf-8",
    )
    report = audit_html_quality(broken_html)
    assert report["passed"] is False
    assert report["grade"] == "FAIL"
    assert "grid" in report["orphan_classes"]
    assert "bar-fill" in report["orphan_classes"]
    assert any("localStorage" in err for err in report["critical_errors"])


def test_build_interactive_webapp_passes_a_plus_audit() -> None:
    res = build_interactive_webapp(
        title="ADA — Estudio de Telemetría y Optimización",
        subtitle="Panel interactivo Bento Glass verificado a 60 FPS",
        output_filename="test_ada_studio_verification.html",
        kpis=[
            {"label": "Precisión Global", "value": "98.42%", "delta": "+4.1% vs Base", "meta": "IC 95% [97.8, 99.0]"},
            {"label": "Latencia RTX 5070", "value": "5.21 s", "delta": "1999.6 tok/s", "meta": "100% GDDR7"},
        ],
        charts=[
            {
                "tab_label": "Curva Temporal",
                "title": "Convergencia Real vs Modelo",
                "type": "line",
                "x": [0, 1, 2, 3, 4, 5],
                "x_label": "Época",
                "y_label": "Rendimiento",
                "series": [
                    {"name": "Observado", "data": [72.0, 84.5, 91.2, 95.4, 97.6, 98.4]},
                    {"name": "Teórico", "data": [70.0, 83.0, 90.5, 95.0, 97.4, 98.5]},
                ],
            },
            {
                "tab_label": "Comparativa",
                "title": "Arena de Arquitecturas",
                "type": "arena",
                "y_label": "Score",
                "items": [
                    {"label": "Baseline", "value": 0.742, "isWinner": False},
                    {"label": "Ada Q Studio", "value": 0.984, "isWinner": True, "secondary": "Óptimo verificado"},
                ],
            },
        ],
        callouts=[
            {"title": "Doble Verificación", "text": "Convergencia validada sin fuga de datos.", "severity": "ok"}
        ],
        table_rows=[
            {"Modulo": "ADACharts HiDPI", "FPS": 60, "Score": 100},
            {"Modulo": "Bento Glass CSS", "FPS": 60, "Score": 100},
        ],
    )
    assert res["verified"] is True
    audit = res["quality_audit"]
    assert audit["passed"] is True
    assert audit["grade"] == "A+"
    assert audit["quality_score"] >= 95
    assert audit["orphan_classes"] == []


def test_webapp_guardrail_redirects_animations_and_canvas_studio_passes_a_plus() -> None:
    with pytest.raises(ValueError, match="ada_build_canvas_animation"):
        build_interactive_webapp(
            title="Animacion",
            charts=[{"type": "animation", "x": [], "series": []}],
        )

    anim = build_canvas_animation_studio(
        title="ADA — Escena Animada 60 FPS",
        output_filename="test_canvas_scene_60fps.html",
        controls=[{"key": "speed", "label": "Velocidad", "min": 0.5, "max": 2.5, "step": 0.1, "value": 1.0}],
        draw_frame_js="""
        const g = ctx.createLinearGradient(0, 0, 0, h);
        g.addColorStop(0, '#0f172a');
        g.addColorStop(1, '#1e293b');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc((t * 80 * (params.speed || 1)) % w, h * 0.4 + Math.sin(t * 4) * 20, 14, 0, Math.PI * 2);
        ctx.fill();
        """,
    )
    assert anim["verified"] is True
    assert anim["quality_audit"]["grade"] == "A+"
    assert anim["quality_audit"]["orphan_classes"] == []

    # El crítico visual debe rechazar dibujos estáticos que no usen t/dt en el cuerpo
    with pytest.raises(ValueError, match="CRÍTICO VISUAL"):
        build_canvas_animation_studio(
            title="Estatico",
            output_filename="test_static_reject.html",
            draw_frame_js="""
            function drawFrame(ctx, w, h, t, dt) {
                ctx.fillStyle = 'skyblue';
                ctx.fillRect(0, 0, w, h);
                ctx.fillStyle = 'orange';
                ctx.beginPath(); ctx.arc(100, 100, 20, 0, Math.PI * 2); ctx.fill();
            }
            """,
        )


def test_neural_brain_connectome_simulation_passes_a_plus() -> None:
    res = build_neural_brain_simulation(
        output_filename="test_brain_connectome_60fps.html",
        open_in_browser=False,
    )
    assert res["status"] == "OK"
    audit = audit_html_quality(res["interactive_html_path"])
    assert audit["passed"] is True
    assert audit["grade"] == "A+"
    assert audit["quality_score"] == 100
    assert audit["orphan_classes"] == []
