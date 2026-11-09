"""Visualizadores web interactivos HTML5 Canvas a 60 FPS y ADA Design Studio v2.0."""

from ada.viz.math_textbook import build_math_textbook_html
from ada.viz.neural_canvas import build_neural_network_live_canvas
from ada.viz.web_studio import (
    audit_html_quality,
    build_canvas_animation_studio,
    build_interactive_webapp,
    render_prediction_studio_html,
)

__all__ = [
    "audit_html_quality",
    "build_canvas_animation_studio",
    "build_interactive_webapp",
    "build_math_textbook_html",
    "build_neural_network_live_canvas",
    "render_prediction_studio_html",
]
