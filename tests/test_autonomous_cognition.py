"""Pruebas unitarias del motor de cognicion autonoma continua y plasticidad de ADA."""

import pytest
from ada.cognition.autonomous_mind import AutonomousMindEngine, get_autonomous_mind


def test_autonomous_mind_singleton_and_status():
    mind = get_autonomous_mind()
    status = mind.get_status()
    assert isinstance(status, dict)
    assert "total_grown_neurons" in status
    assert "total_verified_cycles" in status
    assert status["total_grown_neurons"] >= 128


def test_single_cognitive_cycle_grows_neurons_and_verifies():
    mind = AutonomousMindEngine(cycle_interval_sec=1.0)
    prev_neurons = mind.total_grown_neurons
    prev_cycles = mind.total_verified_cycles

    thought = mind._perform_single_cognitive_cycle()
    assert thought is not None
    assert thought.verified is True
    assert mind.total_grown_neurons >= prev_neurons
    assert mind.total_verified_cycles == prev_cycles + 1

    mind._record_thought(thought)
    latest = mind.get_latest_thoughts(limit=2)
    assert len(latest) >= 1
    assert latest[-1]["thought_type"] in {"ode_self_play", "math_discovery", "neuron_expansion"}


def test_terminal_math_formatting():
    from ada.calculus.terminal_math import format_latex_to_terminal, format_text_for_terminal_cli

    raw_latex = r"\frac{d}{dt} \left( \int_{a(t)}^{b(t)} f(x,t) \, dx \right) = f(b(t),t) \cdot \frac{db}{dt} - f(a(t),t) \cdot \frac{da}{dt} + \int_{a(t)}^{b(t)} \frac{\partial f}{\partial t} \, dx"
    formatted = format_latex_to_terminal(raw_latex)
    assert r"\frac" not in formatted
    assert r"\left" not in formatted
    assert r"\cdot" not in formatted
    assert "d/dt" in formatted
    assert "∫" in formatted
    assert "∂f/∂t" in formatted

    multiline = r"""
Deducción:
$$
\begin{aligned}
\frac{d}{dx}\left[3x^4 - 2x^3 + 5x - 7\right] &= \frac{d}{dx}[3x^4] - \frac{d}{dx}[2x^3] + \frac{d}{dx}[5x] - \frac{d}{dx}[7] \\
&= 12x^3 - 6x^2 + 5
\end{aligned}
$$
"""
    clean_text = format_text_for_terminal_cli(multiline)
    assert "$$" not in clean_text
    assert r"\frac" not in clean_text
    assert "12x³ - 6x² + 5" in clean_text

    # Test exacto del ejercicio de sustitucion de Diego:
    diego_sample = r"\frac{1}{2} \cdot \frac{u^{3/2}}{3/2} + C = \frac{1}{3} u^{3/2} + C"
    parsed_diego = format_latex_to_terminal(diego_sample)
    assert "frac{" not in parsed_diego
    assert "(1) / (3)" not in parsed_diego
    assert "⅓" in parsed_diego or "1/3" in parsed_diego
    assert "½" in parsed_diego or "1/2" in parsed_diego


def test_math_textbook_studio_passes_audit_a_plus():
    from ada.viz.math_textbook import build_math_textbook_html
    from ada.viz.web_studio import audit_html_quality

    rep = build_math_textbook_html(open_in_browser=False)
    audit = audit_html_quality(rep["interactive_html_path"])
    assert audit["passed"] is True
    assert audit["grade"] in {"A+", "A"}
    assert len(audit["orphan_classes"]) == 0


def test_live_server_bridge_endpoints():
    import urllib.request
    import json
    from ada.cognition.live_server import start_live_bridge, set_live_thinking, get_live_state

    # Puerto dedicado al test: el 8765 puede estar ocupado por un puente real de otra sesion de ADA,
    # y entonces el endpoint HTTP respondería con el estado de OTRO proceso.
    test_port = 8799
    started = start_live_bridge(port=test_port)
    assert started is True

    set_live_thinking(True, prompt="y que me cuentas?", model="ada-q", tokens_per_sec=41.7)
    state = get_live_state()
    assert state["is_thinking"] is True
    assert state["current_prompt"] == "y que me cuentas?"
    assert state["tokens_per_sec"] == 41.7

    # Probar endpoint HTTP
    with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/api/status", timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert data["is_thinking"] is True
        assert data["current_prompt"] == "y que me cuentas?"
        assert "tokens_per_sec" in data

    set_live_thinking(False, prompt="y que me cuentas?", model="ada-q", last_response="Listo.")
    state_after = get_live_state()
    assert state_after["is_thinking"] is False
    # Telemetria real: un porcentaje valido medido por nvidia-smi (o el ultimo conocido si no hay GPU NVIDIA).
    assert 0.0 <= state_after["gpu_util_pct"] <= 100.0
    assert state_after["telemetry_source"] in {"nvidia-smi", "unavailable"}
