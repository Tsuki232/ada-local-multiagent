"""Normalizacion de argumentos y comprobacion de fidelidad de datos para las herramientas estadisticas.

El modelo de 14B a veces escribe `alternative: "bilateral"` o transcribe mal una lista larga. Eso se corrige o se detecta AQUI,
en codigo y sin ocupar contexto: solo cuando hay algo que decir se anade una linea al resultado de la herramienta.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_ANALYSIS_ALIASES = {
    "t_test": "two_groups", "ttest": "two_groups", "t-test": "two_groups", "t test": "two_groups", "welch": "two_groups", "mann_whitney": "two_groups",
    "anova": "k_groups", "kruskal": "k_groups", "chi2": "chi_square", "chi-square": "chi_square", "chisquare": "chi_square", "chi cuadrado": "chi_square", "fisher": "chi_square",
    "correlacion": "correlation", "pearson": "correlation", "spearman": "correlation", "regresion": "regression", "linear_regression": "regression",
    "power": "power_t", "potencia": "power_t", "holm": "adjust_pvalues", "bonferroni": "adjust_pvalues", "z": "z_test", "t_one_sample": "one_sample",
    "sample_size_mean": "sample_size", "tamano_muestra": "sample_size",
}
_ALTERNATIVE = {
    "two-sided": {"two-sided", "two_sided", "two sided", "bilateral", "dos colas", "dos_colas", "two-tailed", "two tailed", "!=", "distinto", "diferente", "diferencia"},
    "greater": {"greater", "mayor", "mayor que", "derecha", "right", "right-tailed", "upper", ">", "larger", "unilateral derecha"},
    "less": {"less", "menor", "menor que", "izquierda", "left", "left-tailed", "lower", "<", "smaller", "unilateral izquierda"},
}
_METHOD = {
    "auto": {"auto", "automatico", "automatica", "default"}, "welch": {"welch", "t de welch", "t-welch", "welch t"},
    "student": {"student", "t de student", "pooled", "agrupada", "t-student"}, "mann_whitney": {"mann_whitney", "mann-whitney", "mannwhitney", "u", "u de mann-whitney", "wilcoxon rank-sum"},
}
_TRUE, _FALSE = {"true", "si", "verdadero", "yes", "1", "pareado", "pareados", "paired"}, {"false", "no", "falso", "0", "independiente", "independientes"}


def _plain(x: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(x).strip().lower()) if unicodedata.category(c) != "Mn")


def _lookup(value: Any, table: dict[str, set[str]]) -> str | None:
    v = _plain(value)
    for canon, names in table.items():
        if v == canon or v in names:
            return canon
    return None


def normalize_inference_args(args: dict[str, Any], context: str = "") -> tuple[dict[str, Any], list[str]]:
    """Devuelve (args corregidos, notas). Solo toca sinonimos inequivocos; lo que no reconoce lo deja tal cual para que la herramienta dé su error claro."""
    out = dict(args)
    notes: list[str] = []
    an = out.get("analysis")
    if isinstance(an, str) and an not in {"two_groups", "k_groups", "chi_square", "two_proportions", "one_proportion", "correlation", "regression", "power_t", "adjust_pvalues",
                                          "one_sample", "ci_mean", "ci_proportion", "variance_ratio", "z_test", "sample_size"}:
        fixed = _ANALYSIS_ALIASES.get(_plain(an))
        if fixed:
            notes.append(f"analysis '{an}' -> '{fixed}'")
            out["analysis"] = fixed
    a = out.get("alpha")
    if isinstance(a, str):
        try:
            v = float(a.replace("%", "").replace(",", ".").strip())
            out["alpha"] = v / 100 if "%" in a or v > 1 else v
            notes.append(f"alpha '{a}' -> {out['alpha']}")
        except ValueError:
            pass
    data = out.get("data")
    if isinstance(data, dict):
        data = dict(data)
        if "alternative" in data and data["alternative"] not in {"two-sided", "greater", "less"}:
            fixed = _lookup(data["alternative"], _ALTERNATIVE)
            if fixed:
                notes.append(f"alternative '{data['alternative']}' -> '{fixed}'")
                data["alternative"] = fixed
        if "method" in data and data["method"] not in {"auto", "welch", "student", "mann_whitney"}:
            fixed = _lookup(data["method"], _METHOD)
            if fixed:
                notes.append(f"method '{data['method']}' -> '{fixed}'")
                data["method"] = fixed
        if "paired" in data and not isinstance(data["paired"], bool):
            v = _plain(data["paired"])
            if v in _TRUE or v in _FALSE:
                data["paired"] = v in _TRUE
                notes.append(f"paired '{args['data']['paired']}' -> {data['paired']}")
        if out.get("analysis") == "k_groups" and context and isinstance(data.get("groups"), dict):
            restored = _restore_group_names(data["groups"], context)
            if restored:
                notes.append("nombres de grupo " + ", ".join(f"'{a}' -> '{b}'" for a, b in zip(data["groups"], restored)))
                data["groups"] = restored
        out["data"] = data
    return out, notes


_LABELLED = re.compile(r"^\s*(?P<label>[^:\n]{1,40}?)\s*(?:\(\s*n\s*=\s*\d+\s*\))?\s*:\s*(?P<nums>-?\d+(?:\.\d+)?(?:\s*[;,]\s*-?\d+(?:\.\d+)?)+)\s*$", re.M)


def _restore_group_names(groups: dict, context: str) -> dict | None:
    """Si el modelo envio los valores de cada grupo EXACTOS y en orden pero con otros nombres (A, B, C en vez de 'Marca 1', 'Marca 2'...),
    devuelve los grupos con los nombres de la pregunta. Si los valores no coinciden uno a uno no toca nada. Los nombres salen en el post-hoc."""
    text = re.sub(r"(\d),(\d)", r"\1.\2", context)
    labelled = [(m.group("label").strip(), [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group("nums"))]) for m in _LABELLED.finditer(text)]
    sent = list(groups.items())
    if len(labelled) < len(sent) or len(sent) < 2:
        return None
    for start in range(len(labelled) - len(sent) + 1):
        window = labelled[start:start + len(sent)]
        if all(isinstance(v, (list, tuple)) and len(v) == len(w[1]) and all(abs(float(a) - b) < 1e-9 for a, b in zip(v, w[1])) for (_, v), w in zip(sent, window)):
            names = [w[0] for w in window]
            if len(set(names)) == len(names) and names != [k for k, _ in sent]:
                return {nm: v for nm, (_, v) in zip(names, sent)}
            return None
    return None


# ----------------------------------------------------------------------------- fidelidad de datos


def _question_numbers(text: str) -> list[float]:
    t = re.sub(r"(\d),(\d)", r"\1.\2", text)          # "1,5" (sin espacio) es decimal; "1, 5" es separador de lista
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]


def _arrays(o: Any) -> list[list[float]]:
    """Todas las listas numericas del dato enviado (grupos, columnas, filas de tabla...)."""
    out: list[list[float]] = []
    if isinstance(o, dict):
        for v in o.values():
            out += _arrays(v)
    elif isinstance(o, (list, tuple)):
        if o and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in o):
            out.append([float(v) for v in o])
        else:
            for v in o:
                out += _arrays(v)
    return out


def _contains(arr: list[float], sub: list[float]) -> bool:
    n = len(sub)
    return any(all(abs(arr[i + j] - sub[j]) < 1e-9 for j in range(n)) for i in range(len(arr) - n + 1))


def check_data_fidelity(question: str, args: dict[str, Any], min_list: int = 5) -> list[str]:
    """Compara los DATOS enviados a la herramienta con las listas de la pregunta. Devuelve problemas (vacio = fiel).

    Detecta: valores alterados (que no aparecen en la pregunta) y listas largas de la pregunta que no llegaron completas y en orden.
    """
    data = args.get("data")
    if not isinstance(data, dict):
        return []
    sent = _arrays({k: v for k, v in data.items() if k not in {"alpha", "p0", "mu0", "sigma", "margin", "effect_size", "power", "n_per_group", "p"}})
    if not sent:
        return []
    qnums = _question_numbers(question)
    qset = set(qnums) | {round(v / 100, 9) for v in qnums} | {round(v * 100, 9) for v in qnums}
    issues: list[str] = []
    altered = sorted({v for arr in sent for v in arr if not any(abs(v - q) < 1e-9 for q in qset)})
    if altered:
        issues.append(f"valores enviados que NO aparecen en la pregunta: {altered[:5]}")
    for line in re.split(r"\n+", question):
        for nums in _number_runs(line, min_list):
            if not any(_contains(arr, nums) for arr in sent):
                first = ", ".join(f"{v:g}" for v in nums[:3])
                issues.append(f"la lista de {len(nums)} valores de la pregunta que empieza por [{first}, ...] no llegó completa y en el mismo orden")
    return issues


_RUN = re.compile(r"-?\d+(?:\.\d+)?(?:(?:\s*[,;]\s*|\s+)-?\d+(?:\.\d+)?)*")


def _number_runs(line: str, min_list: int) -> list[list[float]]:
    """Listas de datos de una linea: >= min_list numeros CONSECUTIVOS separados solo por comas, punto y coma o espacios.

    Una frase como "35 de 100 respondieron... 46 de 100 ... α = 0.01" tiene 5 numeros pero NO es una lista de datos (hay palabras entre ellos);
    tratarla como lista obligaba a repetir una llamada correcta (falso positivo medido en el eval de 96 preguntas).
    """
    t = re.sub(r"(\d),(\d)", r"\1.\2", line)
    runs = []
    for m in _RUN.finditer(t):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(0))]
        if len(nums) >= min_list:
            runs.append(nums)
    return runs
