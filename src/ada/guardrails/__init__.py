"""Guardarrailes de ADA en CODIGO (no en el prompt): corrigen y vigilan sin gastar contexto.

- tool_args: normaliza argumentos de herramientas que el modelo escribe con sinonimos y comprueba que los DATOS enviados sean los de la pregunta.
- grounding: avisa de cifras de la respuesta final que no salen de ninguna herramienta ni de la pregunta, y de un SENTIDO (subio/redujo)
  que contradice las medias antes/despues.
"""

from ada.guardrails.grounding import direction_check, direction_hint, numeric_grounding
from ada.guardrails.tool_args import check_data_fidelity, normalize_inference_args

__all__ = ["check_data_fidelity", "direction_check", "direction_hint", "normalize_inference_args", "numeric_grounding"]
