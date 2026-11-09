"""Sanitizacion estricta de expresiones matematicas escritas por el LLM antes de llegar a SymPy.

`sympy.sympify` usa `eval` internamente, asi que un texto generado por un modelo jamas debe pasar sin filtrar.
Solo se aceptan numeros, operadores, parentesis, un conjunto cerrado de funciones y variables declaradas.
"""

from __future__ import annotations

import re

ALLOWED_FUNCTIONS = frozenset({"sin", "cos", "tan", "exp", "log", "sqrt"})
_CHARSET = re.compile(r"^[0-9A-Za-z_+\-*/^().\s]+$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPONENT = re.compile(r"(?:\*\*|\^)\s*\(?\s*(\d+)")
MAX_EXPR_LEN = 120
MAX_PAREN_DEPTH = 8
MAX_EXPONENT = 12


def sanitize_expression(expr: object, allowed_vars: set[str] | frozenset[str]) -> str:
    """Devuelve la expresion normalizada o lanza ValueError con el motivo del rechazo."""
    if not isinstance(expr, str):
        raise ValueError("la expresion debe ser texto")
    text = expr.strip().replace("^", "**")
    if not text:
        raise ValueError("expresion vacia")
    if len(text) > MAX_EXPR_LEN:
        raise ValueError(f"expresion demasiado larga (>{MAX_EXPR_LEN})")
    if not _CHARSET.match(text.replace("**", "^")):
        raise ValueError("caracteres no permitidos")
    if "__" in text:
        raise ValueError("doble guion bajo no permitido")

    for name in _IDENT.findall(text):
        if name not in ALLOWED_FUNCTIONS and name not in allowed_vars:
            raise ValueError(f"identificador no permitido: {name!r}")

    # Una funcion solo es valida si va seguida de '(' (evita usar 'sin' como variable suelta).
    for fn in ALLOWED_FUNCTIONS:
        for m in re.finditer(rf"\b{fn}\b", text):
            if not text[m.end():].lstrip().startswith("("):
                raise ValueError(f"la funcion {fn} requiere parentesis")

    depth = max_depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("parentesis desbalanceados")
    if depth != 0:
        raise ValueError("parentesis desbalanceados")
    if max_depth > MAX_PAREN_DEPTH:
        raise ValueError("anidamiento excesivo")

    for m in _EXPONENT.finditer(text):
        if int(m.group(1)) > MAX_EXPONENT:
            raise ValueError(f"exponente mayor que {MAX_EXPONENT}")

    return text
