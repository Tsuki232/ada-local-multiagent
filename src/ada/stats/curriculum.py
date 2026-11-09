"""Curriculo de inferencia estadistica para ENTRENAR a Ada: escenarios con respuesta correcta por construccion.

Cada ejemplo ensena la conducta completa: leer el problema, LLAMAR a `ada_inferential_stats` con los datos tal como estan en la pregunta,
y redactar la respuesta final con TODOS los numeros tomados del resultado verificado (nada se inventa ni se calcula de memoria).
Los datos del enunciado son los mismos redondeados que se envian a la herramienta, asi lo que Ada ve coincide con lo que se calculo.

Formato: mensajes de texto uniformes (system/user/assistant/tool). La llamada a la herramienta va renderizada como Qwen la espera:
<tool_call>{"name": ..., "arguments": {...}}</tool_call>, de modo que el esquema es consistente y coincide con la inferencia.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable

import numpy as np

from ada.stats.inference import STATS_SYSTEM_RULE, compact_result, run_inference

SYSTEM = "Te llamas Ada Q. Tu creador y arquitecto es Diego López. Responde siempre en español, con rigor técnico directo.\n" + STATS_SYSTEM_RULE

DOMAINS = [
    {"x": "el fertilizante A", "y": "el fertilizante B", "var": "rendimiento (t/ha)", "mean": 5.0, "sd": 0.8, "dec": 2, "unit": "t/ha"},
    {"x": "el método tradicional", "y": "el método nuevo", "var": "tiempo de reacción (ms)", "mean": 320.0, "sd": 40.0, "dec": 0, "unit": "ms"},
    {"x": "el grupo control", "y": "el grupo con tratamiento", "var": "presión sistólica (mmHg)", "mean": 128.0, "sd": 12.0, "dec": 0, "unit": "mmHg"},
    {"x": "el turno de mañana", "y": "el turno de tarde", "var": "piezas defectuosas por lote", "mean": 8.0, "sd": 2.5, "dec": 0, "unit": "piezas"},
    {"x": "la sede norte", "y": "la sede sur", "var": "satisfacción del cliente (0-100)", "mean": 72.0, "sd": 9.0, "dec": 1, "unit": "puntos"},
    {"x": "la dieta estándar", "y": "la dieta experimental", "var": "pérdida de peso (kg)", "mean": 3.5, "sd": 1.6, "dec": 1, "unit": "kg"},
    {"x": "el lote 1", "y": "el lote 2", "var": "resistencia a la tracción (MPa)", "mean": 410.0, "sd": 25.0, "dec": 0, "unit": "MPa"},
    {"x": "la plataforma A", "y": "la plataforma B", "var": "latencia de respuesta (s)", "mean": 1.8, "sd": 0.4, "dec": 2, "unit": "s"},
    {"x": "el programa de estudio 1", "y": "el programa de estudio 2", "var": "calificación del examen", "mean": 6.8, "sd": 1.2, "dec": 1, "unit": "puntos"},
    {"x": "el sensor viejo", "y": "el sensor nuevo", "var": "error de medición (mm)", "mean": 0.9, "sd": 0.3, "dec": 2, "unit": "mm"},
]
GROUP_NAMES = [["Dieta A", "Dieta B", "Dieta C", "Dieta D"], ["Marca 1", "Marca 2", "Marca 3", "Marca 4"], ["Dosis baja", "Dosis media", "Dosis alta", "Placebo"], ["Turno 1", "Turno 2", "Turno 3", "Turno 4"]]


# ----------------------------------------------------------------------------- generacion de datos


def _draw(rng: np.random.Generator, kind: str, n: int, mean: float, sd: float, dec: int) -> list[float]:
    if kind == "skew":
        s2 = math.log(1 + (sd / mean) ** 2) if mean > 0 else 0.25
        a = rng.lognormal(math.log(max(mean, 1e-6)) - s2 / 2, math.sqrt(s2) * 1.5, n)
    elif kind == "outlier":
        a = rng.normal(mean, sd, n)
        a[rng.integers(0, n)] = mean + sd * rng.choice([-1, 1]) * rng.uniform(4, 6)
    else:
        a = rng.normal(mean, sd, n)
    return [round(float(v), dec) for v in a]


def _fmt_list(vals: list[float], dec: int) -> str:
    return ", ".join(f"{v:.{dec}f}" for v in vals)


def _num(v: float, sig: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    return s.replace("e-0", "e-").replace("e+0", "e+") if "e" in s else s


def _p(p: float) -> str:
    """p-valor con 3 cifras significativas (APA: p < 0.001 para valores minimos). Nunca redondea a '1': eso falsearia el valor."""
    if p < 0.001:
        return "p < 0.001"
    if p > 0.999:
        return "p > 0.999"
    return f"p = {p:.3g}"


# ----------------------------------------------------------------------------- escenarios


class Scenario:
    def __init__(self, analysis: str, data: dict[str, Any], alpha: float, question: str, ctx: dict[str, Any]) -> None:
        self.analysis, self.data, self.alpha, self.question, self.ctx = analysis, data, alpha, question, ctx


def _alpha(rng: random.Random) -> float:
    return rng.choices([0.05, 0.01, 0.10], weights=[85, 10, 5])[0]


def sc_two_groups(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    d = rng.choice(DOMAINS)
    kind = rng.choices(["normal", "skew", "outlier"], weights=[60, 28, 12])[0]
    n1, n2 = rng.choice([(8, 8), (12, 12), (15, 15), (20, 20), (10, 18), (18, 10), (22, 12), (6, 9)])
    effect = rng.choices([0, 0.3, 0.6, 1.0, 1.5], weights=[34, 14, 18, 20, 14])[0]
    sd2 = d["sd"] * rng.choices([1, 2.5, 4], weights=[70, 20, 10])[0]
    x = _draw(nprng, kind, n1, d["mean"], d["sd"], d["dec"])
    y = _draw(nprng, kind, n2, d["mean"] + effect * d["sd"] * rng.choice([-1, 1]), sd2, d["dec"])
    alt = rng.choices(["two-sided", "greater", "less"], weights=[82, 9, 9])[0]
    alpha = _alpha(rng)
    ask = {"two-sided": f"¿Hay una diferencia significativa en {d['var']} entre {d['x']} y {d['y']}?",
           "greater": f"¿Es significativamente mayor {d['var']} en {d['x']} que en {d['y']}?",
           "less": f"¿Es significativamente menor {d['var']} en {d['x']} que en {d['y']}?"}[alt]
    q = (f"Medí {d['var']} en {d['x']} y en {d['y']}.\n{d['x'].capitalize()} (n={n1}): {_fmt_list(x, d['dec'])}\n{d['y'].capitalize()} (n={n2}): {_fmt_list(y, d['dec'])}\n"
         f"{ask} Usa α = {alpha}.")
    return Scenario("two_groups", {"x": x, "y": y, "paired": False, "alternative": alt}, alpha, q, d)


def sc_paired(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    d = rng.choice(DOMAINS)
    n = rng.choice([8, 10, 12, 15, 20])
    before = _draw(nprng, "normal", n, d["mean"], d["sd"], d["dec"])
    eff = rng.choices([0, 0.3, 0.7, 1.2], weights=[30, 20, 25, 25])[0] * d["sd"] * 0.5
    noise_kind = rng.choices(["normal", "skew"], weights=[75, 25])[0]
    noise = nprng.normal(0, d["sd"] * 0.4, n) if noise_kind == "normal" else (nprng.exponential(d["sd"] * 0.4, n) - d["sd"] * 0.4)
    after = [round(float(b + eff + e), d["dec"]) for b, e in zip(before, noise)]
    alpha = _alpha(rng)
    q = (f"Medí {d['var']} en {n} sujetos antes y después de una intervención (mismos sujetos, mismo orden).\nAntes: {_fmt_list(before, d['dec'])}\n"
         f"Después: {_fmt_list(after, d['dec'])}\n¿Cambió significativamente {d['var']} tras la intervención? Usa α = {alpha}.")
    return Scenario("two_groups", {"x": before, "y": after, "paired": True, "alternative": "two-sided"}, alpha, q, {**d, "paired": True})


def sc_k_groups(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    d = rng.choice(DOMAINS)
    names = rng.choice(GROUP_NAMES)[: rng.choice([3, 3, 4])]
    n = rng.choice([6, 8, 10])
    kind = rng.choices(["normal", "skew"], weights=[70, 30])[0]
    spread = rng.choices([1, 3], weights=[75, 25])[0]
    shift = rng.choices([0, 0.4, 0.9], weights=[35, 30, 35])[0]
    groups = {nm: _draw(nprng, kind, n, d["mean"] + (i == 1) * shift * d["sd"] * 1.5, d["sd"] * (spread if i == 2 else 1), d["dec"]) for i, nm in enumerate(names)}
    alpha = _alpha(rng)
    body = "\n".join(f"{nm} (n={n}): {_fmt_list(v, d['dec'])}" for nm, v in groups.items())
    q = f"Comparé {d['var']} entre {len(names)} grupos.\n{body}\n¿Hay diferencias significativas entre los grupos? Si las hay, ¿cuáles? Usa α = {alpha}."
    return Scenario("k_groups", {"groups": groups}, alpha, q, d)


def sc_chi(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    shape = rng.choice([(2, 2), (2, 2), (2, 3), (3, 3)])
    base = nprng.integers(8, 40, size=shape).astype(float)
    if rng.random() < 0.5:
        base[0, 0] += rng.randint(10, 29)
    if rng.random() < 0.3:
        base = nprng.integers(1, 9, size=shape).astype(float)
    table = [[int(v) for v in row] for row in base]
    rows, cols = rng.choice([(["Fumadores", "No fumadores"], ["Con síntoma", "Sin síntoma"]), (["Método A", "Método B"], ["Aprobó", "Reprobó"]), (["Turno 1", "Turno 2", "Turno 3"], ["Defectuoso", "Correcto"]),
                             (["Región norte", "Región sur", "Región centro"], ["Compra", "No compra", "Indeciso"])])
    rows, cols = rows[: shape[0]] + [f"Fila {i + 1}" for i in range(len(rows), shape[0])], cols[: shape[1]] + [f"Col {j + 1}" for j in range(len(cols), shape[1])]
    alpha = _alpha(rng)
    body = "\n".join(f"{r}: " + ", ".join(f"{c} = {v}" for c, v in zip(cols, row)) for r, row in zip(rows, table))
    q = f"Tengo esta tabla de conteos:\n{body}\n¿Existe asociación entre las filas y las columnas? Usa α = {alpha}."
    return Scenario("chi_square", {"table": table}, alpha, q, {"rows": rows, "cols": cols})


def sc_two_prop(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    n1, n2 = rng.choice([(50, 50), (100, 100), (200, 180), (30, 40), (400, 400)])
    p1 = rng.choice([0.1, 0.25, 0.4, 0.5]); p2 = min(0.95, p1 + rng.choice([0, 0.04, 0.1, 0.2]))
    a, b = int(nprng.binomial(n1, p1)), int(nprng.binomial(n2, p2))
    alpha = _alpha(rng)
    q = f"En el grupo A, {a} de {n1} respondieron al tratamiento; en el grupo B, {b} de {n2}. ¿Difieren significativamente las proporciones de respuesta? Usa α = {alpha}."
    return Scenario("two_proportions", {"successes": [a, b], "n": [n1, n2]}, alpha, q, {})


def sc_one_prop(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    n = rng.choice([15, 20, 30, 50, 100]); p0 = rng.choice([0.5, 0.3, 0.7, 0.25])
    k = int(nprng.binomial(n, min(0.95, p0 + rng.choice([0, 0.1, 0.25]))))
    alpha = _alpha(rng)
    q = f"De {n} pacientes, {k} mejoraron. ¿La tasa de mejoría es distinta de {p0:.0%}? Usa α = {alpha}."
    return Scenario("one_proportion", {"successes": k, "n": n, "p0": p0}, alpha, q, {})


def sc_corr(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    n = rng.choice([10, 12, 15, 20, 25]); rho = rng.choice([0, 0.3, 0.6, -0.5, 0.85])
    z = nprng.normal(0, 1, n); w = rho * z + math.sqrt(1 - rho ** 2) * nprng.normal(0, 1, n)
    skew = rng.random() < 0.3
    xs = [round(float(math.exp(v / 1.2) * 10 if skew else 50 + 10 * v), 2) for v in z]
    ys = [round(float(math.exp(v / 1.2) * 5 if skew else 20 + 4 * v), 2) for v in w]
    alpha = _alpha(rng)
    q = f"Medí dos variables en {n} unidades.\nX: {_fmt_list(xs, 2)}\nY: {_fmt_list(ys, 2)}\n¿Hay una correlación significativa entre X e Y? Usa α = {alpha}."
    return Scenario("correlation", {"x": xs, "y": ys}, alpha, q, {})


def sc_reg(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    n = rng.choice([14, 18, 22]); k = rng.choice([1, 2]); X = {f"x{i + 1}": [round(float(v), 2) for v in nprng.normal(10, 3, n)] for i in range(k)}
    coefs = [rng.choice([0, 0.8, -1.5, 2.0]) for _ in range(k)]
    y = [round(float(5 + sum(c * X[f"x{i + 1}"][j] for i, c in enumerate(coefs)) + nprng.normal(0, 4)), 2) for j in range(n)]
    alpha = _alpha(rng)
    cols = "\n".join(f"{nm}: {_fmt_list(v, 2)}" for nm, v in X.items())
    q = f"Quiero explicar Y con {', '.join(X)}.\nY: {_fmt_list(y, 2)}\n{cols}\nAjusta una regresión lineal e indica qué predictores son significativos. Usa α = {alpha}."
    return Scenario("regression", {"y": y, "X": X}, alpha, q, {})


def sc_power(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    mode = rng.choice(["n", "power", "effect"]); alpha = rng.choice([0.05, 0.05, 0.01])
    if mode == "n":
        d_ = rng.choice([0.2, 0.3, 0.5, 0.8]); pw = rng.choice([0.8, 0.9])
        data = {"effect_size": d_, "n_per_group": None, "power": pw}
        q = f"Voy a comparar dos grupos con una prueba t. Quiero detectar un efecto d de Cohen = {d_} con potencia {pw} y α = {alpha}. ¿Cuántos sujetos necesito por grupo?"
    elif mode == "power":
        d_ = rng.choice([0.3, 0.5, 0.8]); n_ = rng.choice([15, 25, 40, 64, 100])
        data = {"effect_size": d_, "n_per_group": n_, "power": None}
        q = f"Tengo {n_} sujetos por grupo y quiero comparar dos medias con una prueba t bilateral (α = {alpha}). ¿Qué potencia tengo para detectar d = {d_}?"
    else:
        n_ = rng.choice([20, 30, 50, 80]); pw = rng.choice([0.8, 0.9])
        data = {"effect_size": None, "n_per_group": n_, "power": pw}
        q = f"Solo puedo reclutar {n_} sujetos por grupo. Con α = {alpha} y potencia {pw}, ¿cuál es el efecto mínimo detectable (d de Cohen) en una prueba t de dos muestras?"
    return Scenario("power_t", data, alpha, q, {"mode": mode})


def sc_adjust(rng: random.Random, nprng: np.random.Generator) -> Scenario:
    m = rng.choice([4, 5, 6, 8]); ps = sorted(round(float(v), 4) for v in np.concatenate([nprng.uniform(0.0005, 0.04, max(1, m // 3)), nprng.uniform(0.02, 0.9, m - max(1, m // 3))]))
    method = rng.choice(["holm", "bonferroni", "fdr_bh"]); alpha = rng.choice([0.05, 0.05, 0.10])
    nm = {"holm": "Holm", "bonferroni": "Bonferroni", "fdr_bh": "Benjamini-Hochberg (FDR)"}[method]
    q = f"Hice {m} pruebas de hipótesis y obtuve estos p-valores: {', '.join(str(p) for p in ps)}. Corrígelos por comparaciones múltiples con {nm} y dime cuáles siguen siendo significativos con α = {alpha}."
    return Scenario("adjust_pvalues", {"pvalues": ps, "method": method}, alpha, q, {"method_name": nm})


SCENARIOS: list[tuple[str, Callable[..., Scenario], int]] = [
    ("two_groups", sc_two_groups, 34), ("paired", sc_paired, 10), ("k_groups", sc_k_groups, 15), ("chi_square", sc_chi, 10), ("two_proportions", sc_two_prop, 5),
    ("one_proportion", sc_one_prop, 3), ("correlation", sc_corr, 7), ("regression", sc_reg, 5), ("power_t", sc_power, 6), ("adjust_pvalues", sc_adjust, 5),
]


# ----------------------------------------------------------------------------- redaccion de la respuesta (todo sale del resultado)


def _stat_block(sc: Scenario, r: dict[str, Any]) -> str:
    t, st, df = r["test"], r.get("statistic"), r.get("df")
    if sc.analysis == "two_groups" and "Welch" in t:
        ds = r["descriptives"]
        n1, n2 = r["n"]
        return (f"$$\nt = \\frac{{\\bar x - \\bar y}}{{\\sqrt{{s_x^2/n_x + s_y^2/n_y}}}} = \\frac{{{_num(ds['x']['mean'])} - {_num(ds['y']['mean'])}}}"
                f"{{\\sqrt{{{_num(ds['x']['sd'] ** 2)}/{n1} + {_num(ds['y']['sd'] ** 2)}/{n2}}}}} = {_num(st)},\\qquad gl = {_num(df)}\n$$")
    if sc.analysis == "two_groups" and "pareada" in t:
        return f"$$\nt = \\frac{{\\bar d}}{{s_d/\\sqrt{{n}}}} = {_num(st)},\\qquad gl = {_num(df)}\n$$"
    if sc.analysis == "chi_square" and "Chi" in t:
        return f"$$\n\\chi^2 = \\sum \\frac{{(O - E)^2}}{{E}} = {_num(st)},\\qquad gl = {df}\n$$"
    if sc.analysis == "k_groups" and "ANOVA" in t:
        return f"$$\nF = \\frac{{CM_{{entre}}}}{{CM_{{dentro}}}} = {_num(st)},\\qquad gl = ({_num(df[0])},\\ {_num(df[1])})\n$$"
    if st is None:
        return ""
    gl = f",\\qquad gl = {_num(df)}" if df is not None and not isinstance(df, list) else ""
    return f"$$\n\\text{{estadístico}} = {_num(st)}{gl}\n$$"


def _conclusion(sc: Scenario, r: dict[str, Any]) -> str:
    sig = r["p_value"] < sc.alpha
    d, a = sc.ctx, sc.alpha
    if sc.analysis == "two_groups":
        if d.get("paired"):
            base = f"Con α = {a}, {'hay evidencia suficiente de que ' + d['var'] + ' cambió tras la intervención' if sig else 'no hay evidencia suficiente de un cambio en ' + d['var']}."
        else:
            alt = r["alternative"]
            rel = {"two-sided": "difiere entre", "greater": "es mayor en " + d["x"] + " que en", "less": "es menor en " + d["x"] + " que en"}[alt]
            tail = f" {d['x']} y {d['y']}" if alt == "two-sided" else f" {d['y']}"
            base = f"Con α = {a}, {'hay evidencia suficiente de que ' + d['var'] + ' ' + rel + tail if sig else 'no hay evidencia suficiente de que ' + d['var'] + ' ' + rel + tail}."
    elif sc.analysis == "k_groups":
        base = f"Con α = {a}, {'hay evidencia de que al menos un grupo difiere' if sig else 'no hay evidencia de diferencias entre los grupos'}."
    elif sc.analysis == "chi_square":
        base = f"Con α = {a}, {'hay evidencia de asociación entre las filas y las columnas' if sig else 'no hay evidencia de asociación entre las filas y las columnas'}."
    elif sc.analysis in {"two_proportions", "one_proportion"}:
        base = f"Con α = {a}, {'las proporciones difieren significativamente' if sig else 'no hay evidencia suficiente de que las proporciones difieran'}."
    elif sc.analysis == "correlation":
        base = f"Con α = {a}, {'la correlación es significativamente distinta de cero' if sig else 'no hay evidencia de correlación'}."
    elif sc.analysis == "regression":
        base = f"Con α = {a}, el modelo {'es' if sig else 'no es'} globalmente significativo."
    else:
        base = r.get("decision", "")
    extra = ""
    if not sig and sc.analysis not in {"power_t", "adjust_pvalues"}:
        extra = " No rechazar H0 no demuestra que no haya efecto: puede faltar potencia."
    elif sig and sc.analysis not in {"power_t", "adjust_pvalues", "regression"}:
        extra = " Significativo no significa relevante: valora el tamaño del efecto."
    return base + extra


def compose_answer(sc: Scenario, r: dict[str, Any]) -> str:
    L = [f"**Prueba elegida.** {r['test']}. " + " ".join(r.get("rationale", []))]
    if sc.analysis == "power_t":
        L = [f"**Cálculo.** {r['test']}. " + " ".join(r.get("rationale", [])), f"**Resultado.** {r['decision']}."]
        if r["solved_for"] == "n_per_group":
            L.append(f"Se necesitan **{r['result']} sujetos por grupo** ({2 * r['result']} en total).")
        L.append(f"Comprobación independiente (t no central): potencia = {_num(r['power'])} con n = {r['n_per_group']} por grupo y d = {_num(r['effect_size'])}. [VERIFICADO: {' + '.join(r['verification_methods'])}]")
        return "\n\n".join(L)
    if sc.analysis == "adjust_pvalues":
        rows = "; ".join(f"{_num(p)} → {_num(q)}" for p, q in zip(r["p_raw"], r["p_adjusted"]))
        L = [f"**Corrección.** {r['test']}. " + " ".join(r.get("rationale", [])), f"**p-valores ajustados** (crudo → ajustado): {rows}.",
             f"**Conclusión.** {r['decision']} Significativos: {[i + 1 for i, v in enumerate(r['reject']) if v] or 'ninguno'} (posición en tu lista). [VERIFICADO: {' + '.join(r['verification_methods'])}]"]
        return "\n\n".join(L)
    if sc.analysis == "regression":
        m = r["model"]
        rows = "; ".join(f"{c['term']}: β = {_num(c['coef'])} (EE {_num(c['se'])}, t = {_num(c['t'])}, {_p(c['p'])}, IC [{_num(c['ci_low'])}, {_num(c['ci_high'])}])" for c in r["coefficients"])
        sigs = [c["term"] for c in r["coefficients"][1:] if c["p"] < sc.alpha]
        L.append(f"**Coeficientes.** {rows}.")
        L.append(f"$$\nF({m['df'][0]},\\ {m['df'][1]}) = {_num(m['F'])},\\quad {_p(m['F_p'])},\\qquad R^2 = {_num(m['r2'])},\\qquad R^2_{{aj}} = {_num(m['r2_adj'])}\n$$")
        if any("HC3" in w for w in r.get("warnings", [])):
            L.append("**Con errores robustos (HC3)**, que son los que debes usar por la heterocedasticidad: " + "; ".join(f"{c['term']}: {_p(c['p_robust_HC3'])}" for c in r["coefficients"]) + ".")
            sigs = [c["term"] for c in r["coefficients"][1:] if c["p_robust_HC3"] < sc.alpha]
        L.append(f"**Conclusión.** Predictores significativos al {sc.alpha:.0%}: {', '.join(sigs) if sigs else 'ninguno'}. {_conclusion(sc, r)}")
    else:
        L.append("**Resultado.**\n\n" + _stat_block(sc, r) + f"\n\n{_p(r['p_value'])}.")
        if r.get("effect_size"):
            e = r["effect_size"]
            L.append(f"**Tamaño del efecto.** {e['name']} = {_num(e['value'])}" + (f" ({e['interpretation']})" if e.get("interpretation") else "") + ".")
        if r.get("ci"):
            c = r["ci"]
            L.append(f"**Intervalo de confianza al {c['level']:.0%}** para {c['of']}: [{_num(c['low'])}, {_num(c['high'])}].")
        if r.get("post_hoc"):
            ph = r["post_hoc"]
            sig = [f"{p['a']} vs {p['b']} (p ajustado = {_num(p['p_adj'])})" for p in ph["pairs"] if p["reject"]]
            L.append(f"**Comparaciones post hoc** ({ph['method']}): " + ("; ".join(sig) if sig else "ninguna diferencia por pares es significativa") + ".")
        L.append(f"**Conclusión.** {_conclusion(sc, r)}")
    if r.get("warnings"):
        L.append("**Advertencias.** " + " ".join(r["warnings"]))
    L.append(f"[VERIFICADO: {' + '.join(r['verification_methods'])}]")
    return "\n\n".join(L)


# ----------------------------------------------------------------------------- ejemplos completos


def build_example(sc: Scenario) -> dict[str, Any] | None:
    res = run_inference(sc.analysis, sc.data, alpha=sc.alpha)
    if not res.get("verified"):
        return None  # solo se ensena con resultados verificados por doble via
    call = {"name": "ada_inferential_stats", "arguments": {"analysis": sc.analysis, "data": sc.data, "alpha": sc.alpha}}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": sc.question},
            {"role": "assistant", "content": "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"},
            {"role": "tool", "content": json.dumps(compact_result(res), ensure_ascii=False)},
            {"role": "assistant", "content": compose_answer(sc, res)},
        ],
        "origin": "verified_curriculum",
        "meta": {"analysis": sc.analysis, "test": res["test"], "p_value": res.get("p_value"), "alpha": sc.alpha},
    }


def generate_scenarios(n: int, seed: int) -> list[Scenario]:
    rng, nprng = random.Random(seed), np.random.default_rng(seed)
    names, fns, weights = zip(*SCENARIOS)
    out = []
    while len(out) < n:
        out.append(rng.choices(fns, weights=weights)[0](rng, nprng))
    return out


def training_questions(path: Any) -> set[str]:
    """Preguntas del archivo REAL de entrenamiento: la evaluacion debe excluirlas para ser inedita."""
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["messages"][1]["content"])
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return out


def generate_curriculum(n: int = 900, seed: int = 20260104) -> list[dict[str, Any]]:
    """n ejemplos completos y verificados, sin duplicados."""
    examples, seen = [], set()
    attempts = 0
    while len(examples) < n and attempts < n * 5:
        batch = generate_scenarios(max(50, n - len(examples)), seed + attempts)
        attempts += len(batch)
        for sc in batch:
            if sc.question in seen:
                continue
            try:
                ex = build_example(sc)
            except ValueError:
                continue
            if ex is not None:
                seen.add(sc.question)
                examples.append(ex)
                if len(examples) >= n:
                    break
    return examples
