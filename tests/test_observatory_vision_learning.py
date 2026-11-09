"""Pruebas: formulas de libro, memoria aprobada, mapa real del modelo, Observatorio y vision en vivo."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from ada.calculus.math_markdown import normalize_math_markdown
from ada.config import DATA_WORK_DIR, PROJECT_ROOT
from ada.training import continuous_learner as cl

BAD = r"""1 Aplicar la regla de derivación término a término:
    • $ \frac{d}{dx}[3x^4] = 12x^3 $
    • $ \frac{d}{dx}[-2x^3] = -6x^2 $
    • $ \frac{d}{dx}[5x] = 5 $
    • $ \frac{d}{dx}[-7] = 0 $
 2 Combinar los resultados: $$ f'(x) = 12x^3 - 6x^2 + 5 $$"""


# ----------------------------------------------------------------------------- formulas de libro


def test_bullets_with_inline_math_become_one_aligned_block():
    out = normalize_math_markdown(BAD)
    assert "• $" not in out
    assert out.count("\\begin{aligned}") == 1, "las cuatro viñetas deben fusionarse en UN bloque"
    assert "\\frac{d}{dx}[3x^4] &= 12x^3 \\\\" in out
    assert "\\frac{d}{dx}[-7] &= 0\n\\end{aligned}" in out
    assert "f'(x) = 12x^3 - 6x^2 + 5" in out and "Combinar los resultados:" in out


def test_normalizer_is_idempotent_and_leaves_other_content_alone():
    once = normalize_math_markdown(BAD)
    assert normalize_math_markdown(once) == once
    plain = "Texto normal con $x$ en línea y sin viñetas.\n```python\nprint('$ a = b $')\n```"
    assert normalize_math_markdown(plain) == plain, "ni el texto con math en línea ni los bloques de código se tocan"
    already = "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$"
    assert normalize_math_markdown(already) == already


def test_equals_inside_braces_is_not_the_alignment_point():
    out = normalize_math_markdown("- $ \\int_{x=0}^{1} x\\,dx = \\tfrac{1}{2} $\n- $ y = 2 $")
    assert "\\int_{x=0}^{1} x\\,dx &= \\tfrac{1}{2}" in out


# ----------------------------------------------------------------------------- memoria aprobada


@pytest.fixture()
def memory_sandbox(monkeypatch):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    mem, train = root / "memory", root / "training"
    mem.mkdir(parents=True)
    train.mkdir(parents=True)
    monkeypatch.setattr(cl, "MEMORY_DIR", mem)
    monkeypatch.setattr(cl, "TRAINING_DATA_DIR", train)
    monkeypatch.setattr(cl, "EPISODIC_MEMORY_FILE", mem / "episodic.jsonl")
    monkeypatch.setattr(cl, "CONTINUAL_DATASET_FILE", train / "dataset.jsonl")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_memory_retrieval_ignores_stopwords_and_unapproved(memory_sandbox):
    cl.record_interaction("derivada de x al cubo", "3x^2", user_feedback="positive", tags=["feedback_diego"])
    cl.record_interaction("derivada de x al cubo (mala)", "4x", user_feedback="negative", quality_score=0.0)
    cl.record_interaction("derivada de seno sin evidencia", "cos", quality_score=0.5)
    cl.record_interaction("[Auto-Reto] derivada de x al cubo", "plantilla", verified_methods=["sympy"], tags=["auto_juego"])
    got = cl.retrieve_relevant_memories("cual es la derivada de x al cubo", min_overlap=2, only_approved=True)
    assert [m["assistant_response"] for m in got] == ["3x^2"]
    # solo palabras vacias en comun: no debe recuperar nada
    assert cl.retrieve_relevant_memories("que de la el", min_overlap=1) == []


def _load_cli():
    spec = importlib.util.spec_from_file_location("ada_cli_under_test", PROJECT_ROOT / "scripts" / "ada_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conversation_turn_enters_training_only_with_evidence_and_gets_normalized(memory_sandbox):
    cli = _load_cli()
    from ada.autonomy.heart import Heart

    cli.HEART = Heart(directory=memory_sandbox / "heart")  # el corazon REAL de Ada no debe recibir conversaciones de test
    clean = cli._record_turn("deriva el polinomio", BAD, used_tool=False)
    assert "\\begin{aligned}" in clean and "• $" not in clean
    assert not cl.CONTINUAL_DATASET_FILE.exists(), "una respuesta sin evidencia NO debe entrar al dataset de entrenamiento"
    assert cl.EPISODIC_MEMORY_FILE.exists(), "pero sí a la memoria episódica"

    cli._record_turn("integra 2x", "Resultado $$ x^2 + C $$ [VERIFICADO: sympy + scipy]", used_tool=False)
    rows = [json.loads(line) for line in cl.CONTINUAL_DATASET_FILE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and "VERIFICADO" in rows[0]["conversations"][1]["value"]
    assert rows[0]["metadata"]["source"] == "conversation"


# ----------------------------------------------------------------------------- mapa real del modelo y Observatorio


def _model_map():
    from ada.modelmap.extract import load_model_map

    loaded = load_model_map()
    if loaded is None:
        pytest.skip("mapa del modelo no extraido (python -m ada.modelmap.extract)")
    return loaded


def test_real_model_map_is_complete_and_matches_the_gguf():
    meta, arr = _model_map()
    assert meta["d_ff"] == 17408, "el FFN real de Qwen3-14B es 17408, no 13824"
    assert meta["parameters_match_reported"] is True
    assert meta["parameter_count_from_tensors"] == meta["parameter_count_reported"]
    assert arr["ffn_influence"].shape == (meta["layers"], meta["d_ff"])
    assert np.isfinite(arr["ffn_influence"]).all() and (arr["ffn_influence"] > 0).all()
    assert arr["head_q_norm"].shape == (meta["layers"], meta["heads_q"])
    assert arr["head_k_norm"].shape == (meta["layers"], meta["heads_kv"])


def test_observatory_builds_light_page_with_all_neurons_and_passes_audit():
    from ada.viz.observatory import build_observatory
    from ada.viz.web_studio import audit_html_quality

    meta, _ = _model_map()
    rep = build_observatory(output_filename="test_observatory.html")
    assert rep["ffn_neurons_rendered"] == meta["layers"] * meta["d_ff"] == 696320
    assert rep["bytes"] < 100_000, "la página debe ser ligera; los datos van en un archivo aparte"
    audit = audit_html_quality(rep["absolute_path"])
    assert audit["passed"] and audit["grade"] in {"A", "A+"} and audit["orphan_classes"] == []

    out_dir = PROJECT_ROOT / "outputs"
    try:
        data_js = (out_dir / "test_observatory_data.js").read_text(encoding="utf-8")
        payload = json.loads(data_js.removeprefix("window.ADA_OBS = ").rstrip().rstrip(";"))
        assert len(base64.b64decode(payload["ffn"])) == 696320, "cada neurona real = un byte = un píxel"
        assert payload["meta"]["params_match"] is True
    finally:  # el test no debe dejar archivos en outputs/
        (out_dir / "test_observatory.html").unlink(missing_ok=True)
        (out_dir / "test_observatory_data.js").unlink(missing_ok=True)


# ----------------------------------------------------------------------------- vision en vivo


class _FakeMJPEG(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silencio
        pass

    def do_GET(self):
        from ada.vision.synthetic_stream import SyntheticPhysicsLaboratoryStream

        sim = SyntheticPhysicsLaboratoryStream(width=320, height=240)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for _ in range(200):
                ok, buf = cv2.imencode(".jpg", sim.read_frame())
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf) + buf.tobytes() + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def test_mjpeg_source_reads_real_frames_over_http():
    from ada.vision.live_camera import MJPEGSource

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMJPEG)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    src = MJPEGSource(f"http://127.0.0.1:{srv.server_address[1]}/video")
    try:
        frames = [f for f in (src.read() for _ in range(8)) if f is not None]
        assert len(frames) >= 5
        assert frames[0].shape == (240, 320, 3)
    finally:
        src.close()
        srv.shutdown()


def test_no_phone_falls_back_to_simulator_and_says_why(monkeypatch):
    from ada.vision import live_camera

    monkeypatch.setattr(live_camera, "list_adb_devices", lambda: [])
    src, report = live_camera.open_best_source()
    assert "Simulador" in report["chosen"] and "Simulador" in src.label
    assert any(a["backend"] == "adb" and not a["ok"] and "celular" in a["reason"] for a in report["attempts"])


def test_live_session_measures_kinematics_and_bridge_serves_it():
    from ada.cognition.live_server import start_live_bridge
    from ada.vision import live_camera

    port = 8799
    assert start_live_bridge(port=port)
    live_camera.stop_vision()
    live_camera._SESSION = live_camera.LiveVisionSession(live_camera.SyntheticSource(fps=30))
    session = live_camera._SESSION
    session.start()
    try:
        deadline = time.time() + 12
        while time.time() < deadline and session.snapshot().get("frames", 0) < 40:
            time.sleep(0.2)
        snap = session.snapshot()
        assert snap["active"] and snap["frames"] >= 40 and snap["fps"] > 3
        assert snap["target_found"] and "theta_deg" in snap["kinematics"]
        assert "no valida la medicion" in snap["verification_note"]

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/vision", timeout=5) as r:
            assert json.loads(r.read())["active"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/vision/frame.jpg", timeout=5) as r:
            body = r.read()
            assert r.headers["Content-Type"] == "image/jpeg"
            img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            assert img is not None and img.shape[:2] == (600, 800)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
            st = json.loads(r.read())
            assert "vision" in st and "agent" in st
    finally:
        live_camera.stop_vision()
