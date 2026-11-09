"""Normaliza Markdown matematico "roto" a deducciones continuas de libro de texto.

Convierte listas de vinetas/numeradas donde cada linea es solo una formula en linea (`• $ ... $`) en UN bloque
`$$ \\begin{aligned} ... \\end{aligned} $$` alineado en el signo '=' (estilo Stewart/Spivak), y eleva las
formulas sueltas a bloques de pantalla. Es idempotente: aplicarlo dos veces no cambia el resultado.
"""

from __future__ import annotations

import re

# Una linea que es SOLO una formula en linea, con prefijo opcional de vineta o numeracion.
_PURE_MATH_LINE = re.compile(r"^\s*(?:[•·\-*+]|\d+[.)])?\s*(?<!\$)\$(?!\$)\s*(?P<m>[^$]+?)\s*(?<!\$)\$(?!\$)\s*[.;,]?\s*$")
_INLINE_DISPLAY_LINE = re.compile(r"^(?P<pre>.*?)\$\$\s*(?P<m>[^$]+?)\s*\$\$(?P<post>.*)$")


def _first_top_level_equals(expr: str) -> int:
    depth = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = expr[i - 1] if i else ""
            nxt = expr[i + 1] if i + 1 < len(expr) else ""
            if prev not in "<>!=\\:" and nxt != "=":
                return i
        i += 1
    return -1


def _align(expr: str) -> str:
    expr = expr.strip()
    if "&" in expr or "\\begin" in expr:
        return expr
    k = _first_top_level_equals(expr)
    return expr if k < 0 else f"{expr[:k].rstrip()} &= {expr[k + 1:].lstrip()}"


def _block(formulas: list[str]) -> list[str]:
    if len(formulas) == 1:
        return ["$$", formulas[0].strip(), "$$"]
    body = " \\\\\n".join(_align(f) for f in formulas)
    return ["$$", "\\begin{aligned}", body, "\\end{aligned}", "$$"]


def normalize_math_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    in_display = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        if stripped == "$$":
            in_display = not in_display
        if in_fence or in_display or stripped == "$$":
            out.append(line)
            i += 1
            continue

        m = _PURE_MATH_LINE.match(line)
        if m and "\\begin" not in m.group("m"):
            run = [m.group("m")]
            j = i + 1
            while j < len(lines):
                mj = _PURE_MATH_LINE.match(lines[j])
                if not mj or "\\begin" in mj.group("m"):
                    break
                run.append(mj.group("m"))
                j += 1
            out.extend(_block(run))
            i = j
            continue

        d = _INLINE_DISPLAY_LINE.match(line)
        if d and "\\begin" not in d.group("m"):
            pre, post = d.group("pre").rstrip(), d.group("post").strip()
            if pre:
                out.append(pre)
            out.extend(_block([d.group("m")]))
            if post:
                out.append(post)
            i += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out)
