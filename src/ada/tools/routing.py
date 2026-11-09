"""Seleccion de herramientas POR PREGUNTA: Ada solo recibe los esquemas que puede necesitar, no todos.

Por que: cada esquema de herramienta se re-envia en CADA peticion y consume contexto util para pensar y responder. Con ~27 herramientas en el
harness son ~7 000 tokens fijos; en el CLI el costo crece con cada herramienta nueva. Aqui se agrupan en "paquetes" y se elige por intencion.

Regla de seguridad: ante la duda NUNCA se deja a Ada sin la herramienta de inferencia, porque sin ella calcularia de memoria.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Paquetes de herramientas por tema. El orden dentro de cada paquete es la prioridad si hay que recortar.
PACKS: dict[str, list[str]] = {
    "stats": ["ada_inferential_stats", "ada_probability_bayes", "ada_predict_from_database"],
    "viz": ["ada_build_canvas_animation", "ada_math_textbook_studio", "ada_neural_brain_simulation", "ada_neural_network_live_canvas"],
}

# Herramienta minima que se envia cuando ninguna intencion encaja (nunca se deja a Ada sin poder verificar estadistica).
FALLBACK = ["ada_inferential_stats"]
MAX_TOOLS = 4

_KEYWORDS: dict[str, re.Pattern[str]] = {
    "stats": re.compile(
        r"p[- ]?valor|prueba de|hipotesis|significativ|\bt de\b|student|welch|anova|chi|correlaci|regresi|\bmedia\b|medias|mediana|desviaci|varianza|"
        r"intervalo|confianza|\bmuestras?\b|grupos?\b|proporci|potencia|tamano de|probabilidad|bayes|evento|condicional|independien|distribuci|binomial|poisson|"
        r"normal\b|\bdatos\b|tabla|alfa|\balpha\b|compar|difer|predic|clasific|dataset|\bcsv\b|\bvs\b|versus|efecto|riesgo|prevalencia|sensibilidad|especificidad|"
        r"falso positivo|aleatori|esperanza|hipergeom|teorema|\bcuantos? sujetos|comparaciones multiples|holm|bonferroni|"
        r"\bp\(|\bunion\b|interseccion|complemento|\bprior\b|previa|posterior|verosimilitud|falsos? negativos?|\bevidencias?\b"
    ),
    "viz": re.compile(
        r"animaci|simula|escena|paisaje|dibuj|canvas|cerebro|red neuronal|transformer|cuaderno|libro de texto|\bformulas?\b|visualiza|grafic[ao]|juego|"
        r"pajarito|sistema solar|particulas|fuegos artificiales|ciudad|conectoma|neuronas"
    ),
}
_NUMBERS = re.compile(r"-?\d+(?:[.,]\d+)?")


def _norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def intents(user_text: str) -> list[str]:
    """Paquetes que encajan con la pregunta. Una lista de >= 6 numeros cuenta como estadistica aunque no haya palabras clave."""
    t = _norm(user_text)
    found = [name for name, rx in _KEYWORDS.items() if rx.search(t)]
    if "stats" not in found and len(_NUMBERS.findall(t)) >= 6:
        found.append("stats")
    return found


def select_tools(user_text: str, tools: list[dict[str, Any]], max_tools: int = MAX_TOOLS) -> list[dict[str, Any]]:
    """Subconjunto de `tools` (esquemas OpenAI) relevante para la pregunta, con tope `max_tools`."""
    by_name = {t["function"]["name"]: t for t in tools}
    wanted: list[str] = []
    for pack in intents(user_text):
        wanted += [n for n in PACKS[pack] if n not in wanted]
    if not wanted:
        wanted = list(FALLBACK)
    # Con estadistica presente la inferencia nunca se recorta; el resto se corta por prioridad.
    if "ada_inferential_stats" in wanted:
        wanted.remove("ada_inferential_stats")
        wanted.insert(0, "ada_inferential_stats")
    return [by_name[n] for n in wanted[:max_tools] if n in by_name]
