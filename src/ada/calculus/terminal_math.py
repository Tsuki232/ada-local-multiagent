"""Traductor y formateador de formulas matematicas para consola terminal (LaTeX a Unicode limpio de libro de texto).

Elimina codigos LaTeX crudos (frac, delimitadores, barras invertidas) y genera notacion
matematica elegante de universidad tanto para bloques display como para formulas en linea.
"""

from __future__ import annotations

import re


SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "x": "ˣ", "y": "ʸ", "t": "ᵗ", "i": "ⁱ", "k": "ᵏ", "m": "ᵐ",
}

SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ", "t": "ₜ", "k": "ₖ", "i": "ᵢ", "n": "ₙ",
}

COMMON_VULGAR_FRACTIONS = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
    ("1", "5"): "⅕",
    ("2", "5"): "⅖",
    ("3", "5"): "⅗",
    ("4", "5"): "⅘",
    ("1", "6"): "⅙",
    ("5", "6"): "⅚",
    ("1", "8"): "⅛",
    ("3", "8"): "⅜",
    ("5", "8"): "⅝",
    ("7", "8"): "⅞",
}


def _extract_braced(text: str, start_idx: int) -> tuple[str, int]:
    """Extrae el contenido de un bloque delimitado por llaves balanceadas { ... } comenzando en start_idx."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return "", start_idx
    depth = 0
    content_chars = []
    i = start_idx
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            if depth > 1:
                content_chars.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars), i + 1
            else:
                content_chars.append(c)
        else:
            content_chars.append(c)
        i += 1
    return "".join(content_chars), i


def _parse_fractions_recursive(s: str) -> str:
    """Parsea de forma recursiva todas las fracciones \\frac{A}{B} o frac{A}{B} con soporte de llaves anidadas."""
    pattern = r"\\?frac\s*\{"
    while True:
        m = re.search(pattern, s)
        if not m:
            break
        start_num = m.end() - 1
        num, after_num = _extract_braced(s, start_num)
        while after_num < len(s) and s[after_num].isspace():
            after_num += 1
        if after_num < len(s) and s[after_num] == "{":
            den, after_den = _extract_braced(s, after_num)
        else:
            den = ""
            after_den = after_num

        # Procesar recursivamente fracciones internas
        num_clean = _parse_fractions_recursive(num).strip()
        den_clean = _parse_fractions_recursive(den).strip()

        # Simplificar formatos comunes de libro de texto
        if (num_clean, den_clean) in COMMON_VULGAR_FRACTIONS:
            formatted_frac = COMMON_VULGAR_FRACTIONS[(num_clean, den_clean)]
        elif den_clean == "3/2":
            formatted_frac = f"⅔ {num_clean}" if num_clean else "⅔"
        elif den_clean == "1/2":
            formatted_frac = f"2 · {num_clean}" if num_clean else "2"
        elif den_clean in {"2", "3", "4", "5", "6", "8"} and num_clean.isdigit():
            formatted_frac = f"{num_clean}/{den_clean}"
        elif num_clean in {"du", "df", "dy", "dx", "da", "db"} and den_clean in {"dx", "dt", "du", "2"}:
            formatted_frac = f"{num_clean}/{den_clean}"
        elif re.match(r"^[a-zA-Z0-9]+$", num_clean) and re.match(r"^[a-zA-Z0-9]+$", den_clean):
            formatted_frac = f"{num_clean}/{den_clean}"
        else:
            num_part = f"({num_clean})" if any(op in num_clean for op in "+-") else num_clean
            den_part = f"({den_clean})" if any(op in den_clean for op in "+-/*") else den_clean
            formatted_frac = f"{num_part}/{den_part}"

        s = s[:m.start()] + formatted_frac + s[after_den:]
    return s


def _convert_exponents(s: str) -> str:
    """Convierte exponentes a superindices reales o notacion ^(p/q) limpia para exponentes fraccionarios."""
    def repl(m: re.Match) -> str:
        content = m.group(1) or m.group(2)
        if "/" in content:
            return f"^({content})"
        return "".join(SUPERSCRIPTS.get(c, c) for c in content)
    return re.sub(r"\^(?:\{([^}]+)\}|([0-9a-zA-Z+-]))", repl, s)


def _convert_subscripts(s: str) -> str:
    """Convierte subindices a caracteres subindice Unicode."""
    def repl(m: re.Match) -> str:
        content = m.group(1) or m.group(2)
        return "".join(SUBSCRIPTS.get(c, c) for c in content)
    return re.sub(r"_(?:\{([^}]+)\}|([0-9a-zA-Z+-]))", repl, s)


def format_latex_to_terminal(latex_str: str) -> str:
    """Convierte codigo LaTeX de formulas en representacion matematica Unicode limpia y legible para CLI."""
    s = latex_str.strip()

    # Quitar delimitadores de ecuacion si los tiene
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Quitar entornos begin{aligned} y end{aligned}
    s = re.sub(r"\\begin\{(?:aligned|align\*?|equation\*?)\}", "", s)
    s = re.sub(r"\\end\{(?:aligned|align\*?|equation\*?)\}", "", s)
    s = re.sub(r"\\\\", "\n    ", s)
    s = s.replace("&=", "=").replace("&", "")

    # Delimitadores nulos o barras de evaluacion
    s = re.sub(r"\\left\.", "", s)
    s = re.sub(r"\\right\.", "", s)
    s = re.sub(r"\\right\|", "|", s)

    # Derivadas comunes
    s = re.sub(r"\\frac\{d\}\{dt\}", "d/dt ", s)
    s = re.sub(r"\\frac\{d\}\{dx\}", "d/dx ", s)
    s = re.sub(r"\\frac\{d\^2\s*([a-zA-Z]+)\}\{dt\^2\}", r"d²\1/dt²", s)
    s = re.sub(r"\\frac\{d\^2\s*([a-zA-Z]+)\}\{dx\^2\}", r"d²\1/dx²", s)
    s = re.sub(r"\\frac\{db\}\{dt\}", "b'(t)", s)
    s = re.sub(r"\\frac\{da\}\{dt\}", "a'(t)", s)
    s = re.sub(r"\\frac\{\\partial\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z]+)\}", r"∂\1/∂\2", s)
    s = re.sub(r"\\frac\{dy\}\{dt\}", "dy/dt", s)
    s = re.sub(r"\\frac\{dy\}\{dx\}", "dy/dx", s)

    # Integrales con limites y sin limites
    s = re.sub(r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r"∫[\1 → \2]", s)
    s = re.sub(r"\\int", "∫", s)

    # Parsea todas las fracciones (incluyendo anidadas) con el nuevo motor
    s = _parse_fractions_recursive(s)

    # Parentesis y corchetes escalados
    s = re.sub(r"\\left\(", "(", s)
    s = re.sub(r"\\right\)", ")", s)
    s = re.sub(r"\\left\[", "[", s)
    s = re.sub(r"\\right\]", "]", s)
    s = re.sub(r"\\left\\{", "{", s)
    s = re.sub(r"\\right\\}", "}", s)

    # Simbolos, flechas y operadores matematicos
    symbols_map = [
        (r"\\implies", " ⟹ "),
        (r"\\iff", " ⟺ "),
        (r"\\cdot", " · "),
        (r"\\times", " × "),
        (r"\\pm", "±"),
        (r"\\mp", "∓"),
        (r"\\partial", "∂"),
        (r"\\nabla", "∇"),
        (r"\\infty", "∞"),
        (r"\\approx", "≈"),
        (r"\\neq", "≠"),
        (r"\\leq", "≤"),
        (r"\\geq", "≥"),
        (r"\\in", "∈"),
        (r"\\to", " → "),
        (r"\\rightarrow", " → "),
        (r"\\sum", "∑"),
        (r"\\prod", "∏"),
        # Letras griegas
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\gamma", "γ"),
        (r"\\delta", "δ"),
        (r"\\epsilon", "ε"),
        (r"\\zeta", "ζ"),
        (r"\\eta", "η"),
        (r"\\theta", "θ"),
        (r"\\lambda", "λ"),
        (r"\\mu", "μ"),
        (r"\\pi", "π"),
        (r"\\sigma", "σ"),
        (r"\\tau", "τ"),
        (r"\\phi", "φ"),
        (r"\\omega", "ω"),
        (r"\\Delta", "Δ"),
        (r"\\Omega", "Ω"),
        # Funciones y raices
        (r"\\sin", "sin"),
        (r"\\cos", "cos"),
        (r"\\tan", "tan"),
        (r"\\exp", "exp"),
        (r"\\ln", "ln"),
        (r"\\log", "log"),
        (r"\\sqrt\{([^}]+)\}", r"√(\1)"),
        (r"\\sqrt", "√"),
        # Espaciado LaTeX
        (r"\\,", " "),
        (r"\\;", " "),
        (r"\\quad", "   "),
        (r"\\qquad", "     "),
        (r"\\!", ""),
    ]
    for pat, rep in symbols_map:
        s = re.sub(pat, rep, s)

    # Exponentes y subindices
    s = _convert_exponents(s)
    s = _convert_subscripts(s)

    # Limpiar cualquier residuo de llaves de texto LaTeX y cajas boxed
    s = re.sub(r"\\?boxed\{([^}]+)\}", r"[ \1 ]", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\math[a-zA-Z]+\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)

    # Eliminar dobles espacios y limpiar
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


def format_text_for_terminal_cli(raw_text: str) -> str:
    """Procesa una respuesta completa de Ada detectando formulas LaTeX $$...$$ y transformandolas en bloques Unicode limpios."""
    # 1. Bloques de ecuacion completa $$ ... $$
    def repl_block(m: re.Match) -> str:
        formula = format_latex_to_terminal(m.group(1))
        lines = [f"    {line.strip()}" for line in formula.splitlines() if line.strip()]
        return "\n\n" + "\n".join(lines) + "\n\n"

    res = re.sub(r"\$\$(.+?)\$\$", repl_block, raw_text, flags=re.DOTALL)

    # 2. Bloques de ecuacion \\[ ... \\]
    def repl_bracket(m: re.Match) -> str:
        formula = format_latex_to_terminal(m.group(1))
        lines = [f"    {line.strip()}" for line in formula.splitlines() if line.strip()]
        return "\n\n" + "\n".join(lines) + "\n\n"

    res = re.sub(r"\\\[(.+?)\\\]", repl_bracket, res, flags=re.DOTALL)

    # 3. Formulas en linea $ ... $ si contienen barras invertidas, exponentes o variables
    def repl_inline(m: re.Match) -> str:
        inner = m.group(1)
        return format_latex_to_terminal(inner)

    res = re.sub(r"\$([^$\n]+?)\$", repl_inline, res)

    return res
