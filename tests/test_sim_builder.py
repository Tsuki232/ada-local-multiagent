"""Pruebas del constructor de simulaciones con autorreparacion (validacion real en Node)."""

from __future__ import annotations

import json
import shutil

import pytest

from ada.config import PROJECT_ROOT
from ada.viz import sim_builder
from ada.viz.sim_builder import build_simulation, check_in_node

pytestmark = pytest.mark.skipif(sim_builder._node_exe() is None, reason="Node.js no disponible")

CTRL = [{"key": "g", "label": "Gravedad", "min": 0, "max": 20, "step": 0.1, "value": 9.8, "unit": "m/s²"}]
INIT = "state.y = 50; state.vy = 0;"
GOOD = (
    "ctx.clearRect(0,0,w,h);\n"
    "const bg = ctx.createLinearGradient(0,0,0,h); bg.addColorStop(0,'#123'); bg.addColorStop(1,'#456'); ctx.fillStyle = bg; ctx.fillRect(0,0,w,h);\n"
    "state.vy += params.g * 30 * dt; state.y += state.vy * dt; if (state.y > h-20) { state.y = h-20; state.vy *= -0.8; }\n"
    "const rg = ctx.createRadialGradient(w/2, state.y, 2, w/2, state.y, 20); rg.addColorStop(0,'#fff'); rg.addColorStop(1,'#fa0');\n"
    "ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(w/2, state.y, 20, 0, Math.PI*2); ctx.fill();"
)


def test_validator_accepts_a_real_working_simulation():
    assert check_in_node(GOOD, INIT, CTRL) is None


@pytest.mark.parametrize(
    "draw,expected",
    [
        ("ctx.fillRect(10,10,50,50); ctx.fillRect(20,20,5,5); ctx.beginPath(); ctx.arc(5,5,3,0,6); ctx.fill();" * 3, "ERROR_STATIC"),
        ("ctx.fillRect(0,0,1,1); algoQueNoExiste.mover();", "ERROR_FRAME"),
        ("ADAScene.funcionInventada(ctx, 1, 2);", "ERROR_FRAME"),
        ("state.z = t;", "ERROR_EMPTY"),  # no dibuja nada
        ("state.x = (state.x || 1) * 1e200; ctx.fillRect(state.x%50,0,5,5); ctx.arc(t*50,5,5,0,6); ctx.fill();".replace("1e200", "1e100") * 1, "ERROR_NAN"),
    ],
)
def test_validator_catches_real_failures_with_exact_reason(draw, expected):
    err = check_in_node(draw, "", [])
    assert err is not None and err.startswith(expected), err


def test_validator_catches_infinite_loops_with_a_timeout():
    err = check_in_node("while(true){}", "", [], timeout_s=3)
    assert err is not None and err.startswith("ERROR_TIMEOUT")


@pytest.fixture()
def cleanup_outputs():
    yield
    for p in (PROJECT_ROOT / "outputs").glob("sim_test_*.html"):
        p.unlink(missing_ok=True)
    shutil.rmtree(PROJECT_ROOT / "data" / "work" / "sim_check", ignore_errors=True)


def _chat(responses):
    it = iter(responses)
    seen = []

    def chat(messages):
        seen.append(messages)
        return next(it)

    chat.seen = seen
    return chat


def _spec(draw, title="sim_test_pelota"):
    return json.dumps({"title": title, "subtitle": "prueba", "init_state_js": INIT, "draw_frame_js": draw, "controls_json": json.dumps(CTRL)})


def test_repair_loop_feeds_the_exact_error_back_and_succeeds(cleanup_outputs):
    static = "const bg = ctx.createLinearGradient(0,0,0,h); ctx.fillStyle = bg; ctx.fillRect(0,0,w,h); ctx.beginPath(); ctx.arc(100,100,20,0,6); ctx.fill(); ctx.arc(1,1,1,0,6); ctx.fill(); dt; t;"
    chat = _chat([_spec(static), _spec(GOOD)])
    res = build_simulation("pelota rebotando", chat=chat)
    assert res["ok"] and res["attempts"] == 2 and res["audit"]["passed"]
    assert res["history"][0]["error"].startswith("ERROR_STATIC") and res["history"][1]["error"] is None
    second_call_user_msg = chat.seen[1][-1]["content"]
    assert "ERROR_STATIC" in second_call_user_msg, "el reintento debe recibir el error EXACTO"
    assert (PROJECT_ROOT / "outputs" / "sim_sim_test_pelota.html").is_file() or (PROJECT_ROOT / res["interactive_html_path"]).is_file()
    (PROJECT_ROOT / res["interactive_html_path"]).unlink(missing_ok=True)


def test_repair_loop_reports_failure_honestly_when_attempts_run_out(cleanup_outputs):
    bad = _spec("ctx.fillRect(0,0,1,1); algoQueNoExiste();")
    res = build_simulation("algo imposible", chat=_chat([bad, bad, bad]), max_attempts=3)
    assert res["ok"] is False and res["attempts"] == 3 and "ERROR_FRAME" in res["last_error"]
    assert "No logré" in res["message"], "debe decir que falló, no fingir exito"


def test_garbage_model_output_is_a_retryable_error_not_a_crash(cleanup_outputs):
    res = build_simulation("x", chat=_chat(["esto no es json", "[1,2,3]", _spec(GOOD, "sim_test_ok")]), max_attempts=3)
    assert res["ok"] and res["attempts"] == 3
    (PROJECT_ROOT / res["interactive_html_path"]).unlink(missing_ok=True)


def test_validator_internal_failure_aborts_without_blaming_the_model(monkeypatch, cleanup_outputs):
    monkeypatch.setattr(sim_builder, "check_in_node", lambda *a, **k: f"{sim_builder.VALIDATOR_FAILURE}: node roto")
    chat = _chat([_spec(GOOD), _spec(GOOD), _spec(GOOD)])
    res = build_simulation("x", chat=chat, max_attempts=3)
    assert res["ok"] is False and res["attempts"] == 1 and "validador" in res["message"]
