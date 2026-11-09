"""Verificador de fundamentacion numerica: cifras de la respuesta final que no salen de ninguna herramienta ni de la pregunta.

Es un AVISO (no bloquea): Ada puede redondear o derivar legitimamente (cuadrados, porcentajes, dobles). Se vigilan solo las cifras con >= 3 cifras
significativas, que son las que delatan un p-valor o un estadistico inventado; los numeros chicos y redondos (gl, n, 0.05) no se marcan.
"""

from __future__ import annotations

import json
import re

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")


def _prep(text: str) -> str:
    return re.sub(r"(\d),(\d)", r"\1.\2", text)  # "1,5" sin espacio es decimal


def _floats(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(_prep(text)):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _sig_digits(token: str) -> int:
    mant = token.lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0")
    return len(mant.rstrip("0")) or 1


def _decimals(token: str) -> int:
    return len(token.lower().split("e")[0].split(".")[1]) if "." in token else 0


def _grounded(token: str, pool: set[float]) -> bool:
    """El numero esta respaldado si, redondeando algun valor del pool (o uno derivado: cuadrado, doble, x100, /100, 1-v) como lo redondeo el
    texto (mismos decimales o mismas cifras significativas), se obtiene exactamente ese numero."""
    tok = float(token)
    sig, dec = _sig_digits(token), _decimals(token)
    for v in pool:
        for cand in (v, v * v, 2 * v, 100 * v, v / 100, 1 - v, abs(v)):
            if abs(round(cand, dec) - tok) < 1e-12 or abs(float(f"{cand:.{sig}g}") - tok) < 1e-12 * max(1.0, abs(tok)):
                return True
    return False


_BEFORE = re.compile(r"\bantes\b|\bpre[- ]?(?:test|intervenci)|\bbasal\b", re.I)
_AFTER = re.compile(r"\bdespu[eé]s\b|\btras\b|\bpost[- ]?(?:test|intervenci)", re.I)
_DOWN = re.compile(r"\b(?:redujo|redujeron|reducci[oó]n|reducir|disminu\w+|descen\w+|baj[oó]|bajaron|cay[oó]|decreci\w+|menor(?:es)? (?:despu[eé]s|tras))", re.I)
_UP = re.compile(r"\b(?:aument\w+|increment\w+|subi[oó]|subieron|crec\w+|elev\w+|mayor(?:es)? (?:despu[eé]s|tras))", re.I)
_HEDGED = re.compile(r"\bno hay (?:evidencia|indicio)|\bsin evidencia|\bno se (?:observ|encontr|detect)|\bno\b(?:\W+\w+){0,5}?\W+(?:estad[ií]sticamente\W+)?significativ|\bno mostr\w+|\bsi (?:hubiera|hubiese)", re.I)
_NEGATED = re.compile(r"\bno\s+(?:\w+\s+){0,2}$", re.I)


def _first_json(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def direction_hint(tool_outputs: list[str]) -> str | None:
    """La frase `direction` del ultimo resultado de dos grupos (para guiar una reescritura), o None."""
    for out in reversed(tool_outputs):
        res = _first_json(out)
        if res and res.get("analysis") == "two_groups" and isinstance(res.get("direction"), str):
            return res["direction"]
    return None


def direction_check(answer: str, tool_outputs: list[str], question: str = "") -> list[str]:
    """Contradicciones entre el SENTIDO que afirma la respuesta (subio/redujo) y las medias de la herramienta.

    Solo aplica a comparaciones de dos grupos de tipo ANTES/DESPUES (x = el primero que menciona la pregunta, que `check_data_fidelity` ya
    exige en ese orden); sin esa estructura no se sabe que grupo es "antes" y no se afirma nada. Es un AVISO como `numeric_grounding`.
    """
    if not (_BEFORE.search(question) and _AFTER.search(question)):
        return []
    truth = None
    for out in tool_outputs:
        res = _first_json(out)
        if not res or res.get("analysis") != "two_groups":
            continue
        desc = res.get("descriptives") or {}
        try:
            mx, my = float(desc["x"]["mean"]), float(desc["y"]["mean"])
        except (KeyError, TypeError, ValueError):
            continue
        if mx != my:
            truth = "up" if my > mx else "down"
    if truth is None:
        return []
    wrong, right = [], 0
    body = re.sub(r"\[VERIFICADO:.*?\]", "", answer, flags=re.S)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        if _HEDGED.search(sentence):   # "no hay evidencia de que aumentara": hipotetiza, no afirma el sentido
            continue
        for kind, rx in (("down", _DOWN), ("up", _UP)):
            for m in rx.finditer(sentence):
                if _NEGATED.search(sentence[: m.start()]):
                    continue
                if kind == truth:
                    right += 1
                else:
                    wrong.append(sentence.strip()[:120])
    if wrong and len(wrong) > right:
        word = "SUBIERON" if truth == "up" else "BAJARON"
        return [f"la respuesta afirma lo contrario de los datos: las medias {word} de 'antes' a 'después' ({wrong[0]!r})"]
    return []


def numeric_grounding(answer: str, tool_outputs: list[str], question: str = "") -> list[str]:
    """Cifras de `answer` (>= 3 cifras significativas) sin respaldo en `tool_outputs` ni en `question`. Lista vacia = todo respaldado."""
    if not tool_outputs:
        return []
    pool: set[float] = set()
    for text in [*tool_outputs, question]:
        pool.update(_floats(text))
    body = re.sub(r"[\^_]\{?\d+\}?", "", re.sub(r"\[VERIFICADO:.*?\]", "", answer, flags=re.S))  # sello y exponentes/indices de LaTeX fuera
    out: list[str] = []
    for m in _NUM.finditer(_prep(body)):
        tok = m.group(0)
        if _sig_digits(tok) < 3 or tok in out:
            continue
        if not _grounded(tok, pool):
            out.append(tok)
    return out
