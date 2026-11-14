"""Inferencia estadistica VERIFICADA de ADA: cada resultado se calcula por dos vias independientes.

Para cada analisis hay una via principal (SciPy/statsmodels) y una via independiente (formula manual con NumPy/SciPy.stats
distribuciones, o una prueba de permutacion). `verified` es True solo si ambas coinciden dentro de la tolerancia. Antes de elegir
la prueba se comprueban los supuestos (normalidad, varianzas, tamanos esperados) y se explica la eleccion.

Esto es lo que Ada debe LLAMAR en vez de calcular de memoria: un modelo de lenguaje no hace aritmetica fiable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint, proportions_ztest
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW, ttest_ind

STATS_SYSTEM_RULE = (
    "REGLA DE INFERENCIA ESTADÍSTICA: nunca calcules de memoria estadísticos, p-valores, intervalos, tamaños del efecto ni tamaños de muestra. "
    "Para cualquier prueba de hipótesis, comparación de grupos, tabla de contingencia, proporciones, correlación, regresión con inferencia, potencia o corrección "
    "por comparaciones múltiples, llama SIEMPRE a `ada_inferential_stats` con los datos y presenta: la prueba elegida y por qué (supuestos), el estadístico con sus "
    "grados de libertad, el p-valor, el tamaño del efecto con su interpretación, el intervalo de confianza, la conclusión y las advertencias que devuelva la herramienta. "
    "Un resultado con verified=false no se presenta como válido."
)

SEED = 42
N_PERM = 20000
N_BOOT = 4000


# ----------------------------------------------------------------------------- utilidades


def _arr(x: Any, name: str = "datos", min_n: int = 2) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < min_n:
        raise ValueError(f"{name}: se necesitan al menos {min_n} valores numericos validos (hay {a.size}).")
    return a


def _agree(a: float, b: float, tol: float = 1e-8) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b)))


def _verdict(p: float, alpha: float) -> str:
    return f"Se rechaza H0 (p = {p:.4g} < α = {alpha})." if p < alpha else f"No se rechaza H0 (p = {p:.4g} ≥ α = {alpha})."


def _interp_d(d: float) -> str:
    a = abs(d)
    return "insignificante" if a < 0.2 else "pequeño" if a < 0.5 else "mediano" if a < 0.8 else "grande"


def _interp_r(r: float) -> str:
    a = abs(r)
    return "insignificante" if a < 0.1 else "pequeño" if a < 0.3 else "mediano" if a < 0.5 else "grande"


def _normal_enough(a: np.ndarray, alpha_assump: float = 0.05) -> tuple[bool, dict[str, Any]]:
    """Normalidad: Shapiro-Wilk; con n >= 50 se tolera por TLC si la asimetria es moderada."""
    n = a.size
    if n < 3:
        return False, {"n": int(n), "shapiro_p": None, "skew": None, "ok": False, "note": "n < 3: no evaluable"}
    p = float(stats.shapiro(a[:5000]).pvalue)
    sk = float(stats.skew(a)) if n > 2 else 0.0
    ok = p >= alpha_assump or (n >= 50 and abs(sk) < 1.0)
    return ok, {"n": int(n), "shapiro_p": round(p, 6), "skew": round(sk, 4), "ok": bool(ok)}


def _boot_ci(f, rng: np.random.Generator, *arrs: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    vals = []
    for _ in range(N_BOOT):
        vals.append(f(*[a[rng.integers(0, a.size, a.size)] for a in arrs]))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def _perm_p_two_sample(x: np.ndarray, y: np.ndarray, stat, alternative: str, rng: np.random.Generator) -> float:
    pooled = np.concatenate([x, y])
    obs = stat(x, y)
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        s = stat(pooled[: x.size], pooled[x.size:])
        cnt += (abs(s) >= abs(obs) - 1e-12) if alternative == "two-sided" else (s >= obs - 1e-12 if alternative == "greater" else s <= obs + 1e-12)
    return (cnt + 1) / (N_PERM + 1)


def _perm_p_paired(d: np.ndarray, alternative: str, rng: np.random.Generator) -> float:
    obs = np.sum(np.sign(d) * stats.rankdata(np.abs(d)))
    cnt = 0
    for _ in range(N_PERM):
        sgn = rng.choice([-1.0, 1.0], size=d.size)
        s = np.sum(sgn * stats.rankdata(np.abs(d)))
        cnt += (abs(s) >= abs(obs) - 1e-9) if alternative == "two-sided" else (s >= obs - 1e-9 if alternative == "greater" else s <= obs + 1e-9)
    return (cnt + 1) / (N_PERM + 1)


def _brunner_munzel_manual(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Estadistico W, grados de libertad y p̂ = P(X>Y)+½P(X=Y) segun Brunner y Munzel (2000), a partir de rangos."""
    n1, n2 = x.size, y.size
    R = stats.rankdata(np.concatenate([x, y]))
    R1, R2 = R[:n1], R[n1:]
    r1, r2 = stats.rankdata(x), stats.rankdata(y)
    m1, m2 = R1.mean(), R2.mean()
    s1 = ((R1 - r1 - m1 + (n1 + 1) / 2) ** 2).sum() / (n1 - 1)
    s2 = ((R2 - r2 - m2 + (n2 + 1) / 2) ** 2).sum() / (n2 - 1)
    den = n1 * s1 + n2 * s2
    if den <= 0:
        return float("nan"), float("nan"), 0.5
    w = n1 * n2 * (m2 - m1) / ((n1 + n2) * math.sqrt(den))
    df = den ** 2 / ((n1 * s1) ** 2 / (n1 - 1) + (n2 * s2) ** 2 / (n2 - 1))
    ph = float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
    return float(w), float(df), ph


def _mc_tol(p: float) -> float:
    return max(0.02, 4.5 * math.sqrt(max(p * (1 - p), 1e-4) / N_PERM))


# ----------------------------------------------------------------------------- 1. dos grupos


def compare_two_groups(x: Any, y: Any, *, paired: bool = False, alternative: str = "two-sided", alpha: float = 0.05, method: str = "auto", sensitivity: bool = True) -> dict[str, Any]:
    """Compara dos grupos eligiendo la prueba segun los supuestos. Devuelve estadistico, p, tamano del efecto, IC y verificacion dual.

    `sensitivity=False` (solo con method='welch'|'student') omite el analisis de sensibilidad automatico: su prueba robusta usa bootstrap/permutaciones
    de coste n1*n2 y no escala a decenas de miles de filas. Se AVISA en `warnings`; quien lo pide debe aportar su propia comprobacion robusta."""
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")
    if method not in {"auto", "welch", "student", "mann_whitney"}:
        raise ValueError("method debe ser 'auto', 'welch', 'student' o 'mann_whitney'.")
    if method != "auto":
        if paired:
            raise ValueError("`method` solo aplica a grupos independientes (en datos pareados la prueba se elige por los supuestos).")
        return _two_groups_forced(_arr(x, "grupo x"), _arr(y, "grupo y"), method, alternative, alpha, sensitivity=sensitivity)
    x, y = _arr(x, "grupo x"), _arr(y, "grupo y")
    rng = np.random.default_rng(SEED)
    why: list[str] = []
    warnings: list[str] = []
    out: dict[str, Any] = {"analysis": "two_groups", "paired": paired, "alternative": alternative, "alpha": alpha}

    if paired:
        if x.size != y.size:
            raise ValueError("En datos pareados x e y deben tener el mismo tamaño.")
        d = x - y
        ok, info = _normal_enough(d)
        out["assumptions"] = {"normality_of_differences": info}
        n = d.size
        if ok:
            res = stats.ttest_rel(x, y, alternative=alternative)
            t, p, df = float(res.statistic), float(res.pvalue), n - 1
            manual_t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
            manual_p = {"two-sided": 2 * stats.t.sf(abs(manual_t), df), "greater": stats.t.sf(manual_t, df), "less": stats.t.cdf(manual_t, df)}[alternative]
            verified = _agree(t, manual_t) and _agree(p, float(manual_p))
            lo, hi = DescrStatsW(d).tconfint_mean(alpha=alpha) if alternative == "two-sided" else DescrStatsW(d).tconfint_mean(alpha=2 * alpha)
            dz = float(d.mean() / d.std(ddof=1))
            why.append(f"Diferencias aproximadamente normales (Shapiro p={info['shapiro_p']}) -> t de Student pareada.")
            out.update(test="t de Student pareada", statistic=t, df=df, p_value=p, effect_size={"name": "d de Cohen (dz)", "value": dz, "interpretation": _interp_d(dz)},
                       ci={"level": 1 - alpha, "of": "diferencia media", "low": float(lo), "high": float(hi)}, estimate=float(d.mean()),
                       verification_methods=["scipy.stats.ttest_rel", "formula manual t = d̄/(s_d/√n)"], discrepancy=abs(p - float(manual_p)))
        else:
            if np.all(d == 0):
                raise ValueError("Todas las diferencias son 0: no hay información para Wilcoxon.")
            res = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
            p = float(res.pvalue)
            pperm = _perm_p_paired(d[d != 0], alternative, rng)
            verified = abs(p - pperm) <= _mc_tol(p)
            ranks = stats.rankdata(np.abs(d[d != 0]))
            wp, wm = ranks[d[d != 0] > 0].sum(), ranks[d[d != 0] < 0].sum()
            rb = float((wp - wm) / (wp + wm))
            lo, hi = _boot_ci(lambda a: float(np.median(a)), rng, d, level=1 - alpha)
            why.append(f"Diferencias NO normales (Shapiro p={info['shapiro_p']}) -> Wilcoxon de rangos con signo.")
            out.update(test="Wilcoxon de rangos con signo", statistic=float(res.statistic), df=None, p_value=p,
                       effect_size={"name": "correlación biserial por rangos", "value": rb, "interpretation": _interp_r(rb)},
                       ci={"level": 1 - alpha, "of": "mediana de las diferencias (bootstrap percentil)", "low": lo, "high": hi}, estimate=float(np.median(d)),
                       verification_methods=["scipy.stats.wilcoxon", f"prueba de permutacion de signos ({N_PERM} permutaciones)"], discrepancy=abs(p - pperm))
    else:
        okx, ix = _normal_enough(x)
        oky, iy = _normal_enough(y)
        lev = stats.levene(x, y, center="median")
        equal_var = bool(lev.pvalue >= 0.05)
        out["assumptions"] = {"normality_x": ix, "normality_y": iy, "levene_p": round(float(lev.pvalue), 6), "equal_variances": equal_var}
        n1, n2 = x.size, y.size
        if okx and oky:
            # Welch SIEMPRE (Delacre et al., 2017): elegir t agrupada solo cuando Levene no rechaza infla los falsos positivos
            # (el contraste previo falla a veces) y Welch pierde practicamente nada si las varianzas son iguales.
            equal_var = False
            usevar = "unequal"
            res = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
            t, p = float(res.statistic), float(res.pvalue)
            sm_t, sm_p, sm_df = ttest_ind(x, y, alternative={"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alternative], usevar=usevar)
            verified = _agree(t, float(sm_t)) and _agree(p, float(sm_p))
            cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
            lo, hi = cm.tconfint_diff(alpha=alpha if alternative == "two-sided" else 2 * alpha, usevar=usevar)
            sp = math.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)) if equal_var else math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            d = float((x.mean() - y.mean()) / sp)
            g = d * (1 - 3 / (4 * (n1 + n2) - 9))
            name = "t de Welch"
            why.append(f"Ambos grupos aproximadamente normales (Shapiro p={ix['shapiro_p']} y {iy['shapiro_p']}) -> t de Welch, que no exige varianzas iguales "
                       f"(Levene p={lev.pvalue:.4g}, informativo; Welch es la opción recomendada aunque sean iguales).")
            out.update(test=name, statistic=t, df=float(sm_df), p_value=p, effect_size={"name": "g de Hedges", "value": float(g), "interpretation": _interp_d(g)},
                       ci={"level": 1 - alpha, "of": "diferencia de medias (x - y)", "low": float(lo), "high": float(hi)}, estimate=float(x.mean() - y.mean()),
                       verification_methods=["scipy.stats.ttest_ind", f"statsmodels.stats.weightstats.ttest_ind(usevar='{usevar}')"], discrepancy=abs(p - float(sm_p)))
        elif lev.pvalue < 0.05:
            # No normales Y dispersion distinta: Mann-Whitney NO es valido (infla los falsos positivos, ~15-20 % medido en simulacion).
            res = stats.brunnermunzel(x, y, alternative=alternative)
            p, stat = float(res.pvalue), float(res.statistic)
            mw, mdf, ph = _brunner_munzel_manual(x, y)
            mp = {"two-sided": 2 * stats.t.sf(abs(mw), mdf), "greater": stats.t.cdf(mw, mdf), "less": stats.t.sf(mw, mdf)}[alternative]
            # signo: scipy y la formula manual usan orientacion opuesta entre si; se contrasta la estadistica en valor absoluto y el p-valor
            verified = _agree(abs(stat), abs(mw), 1e-7) and _agree(p, float(mp), 1e-6)
            lo, hi = _boot_ci(lambda a, b: float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()), rng, x, y, level=1 - alpha)
            bad = [n for n, ok in (("x", okx), ("y", oky)) if not ok]
            why.append(f"Normalidad no sostenida en {', '.join(bad)} y varianzas distintas (Levene p={lev.pvalue:.4g}) -> Mann-Whitney no es válido aquí; "
                       "se usa Brunner-Munzel (robusto a dispersión y tamaños desiguales).")
            if min(n1, n2) < 20:
                warnings.append(f"Brunner-Munzel es algo liberal con muestras menores de 20 (n1={n1}, n2={n2}): en simulación puede rechazar H0 más de lo nominal; interpreta con cautela.")
            out.update(test="Brunner-Munzel", statistic=mw, df=float(mdf), p_value=p,
                       effect_size={"name": "probabilidad de superioridad P(X>Y)+½P(X=Y)", "value": float(ph), "interpretation": "0.5 = sin efecto; "
                                    + ("pequeño" if abs(ph - 0.5) < 0.14 else "mediano" if abs(ph - 0.5) < 0.21 else "grande")},
                       ci={"level": 1 - alpha, "of": "probabilidad de superioridad (bootstrap percentil)", "low": lo, "high": hi}, estimate=float(ph),
                       verification_methods=["scipy.stats.brunnermunzel", "formula manual de Brunner-Munzel (rangos globales e internos)"], discrepancy=abs(p - float(mp)))
        else:
            res = stats.mannwhitneyu(x, y, alternative=alternative, method="auto")
            p, u1 = float(res.pvalue), float(res.statistic)
            pperm = _perm_p_two_sample(x, y, lambda a, b: float(stats.rankdata(np.concatenate([a, b]))[: a.size].sum()) - a.size * (a.size + b.size + 1) / 2, alternative, rng)
            verified = abs(p - pperm) <= _mc_tol(p)
            rb = float(2 * u1 / (n1 * n2) - 1)
            lo, hi = _boot_ci(lambda a, b: float(np.median(a) - np.median(b)), rng, x, y, level=1 - alpha)
            bad = [n for n, ok in (("x", okx), ("y", oky)) if not ok]
            why.append(f"Normalidad no sostenida en {', '.join(bad)} (Shapiro p={ix['shapiro_p']} y {iy['shapiro_p']}) -> Mann-Whitney U (no paramétrica).")
            out.update(test="U de Mann-Whitney", statistic=u1, df=None, p_value=p, effect_size={"name": "correlación biserial por rangos", "value": rb, "interpretation": _interp_r(rb)},
                       ci={"level": 1 - alpha, "of": "diferencia de medianas (bootstrap percentil)", "low": lo, "high": hi}, estimate=float(np.median(x) - np.median(y)),
                       verification_methods=["scipy.stats.mannwhitneyu", f"prueba de permutacion de rangos ({N_PERM} permutaciones)"], discrepancy=abs(p - pperm))
        if min(n1, n2) < 15:
            warnings.append(f"Muestras pequeñas (n1={n1}, n2={n2}): la potencia es baja y los supuestos son difíciles de comprobar.")

    out.update(rationale=why, warnings=warnings, verified=bool(verified), decision=_verdict(out["p_value"], alpha), n=[int(x.size), int(y.size)],
               descriptives={"x": {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(np.median(x))},
                             "y": {"mean": float(y.mean()), "sd": float(y.std(ddof=1)), "median": float(np.median(y))}})
    return out


# ----------------------------------------------------------------------------- 2. k grupos


def compare_k_groups(groups: dict[str, Any] | list[Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """ANOVA / Welch / Kruskal-Wallis segun supuestos, con tamano del efecto y comparaciones post hoc corregidas."""
    items = list(groups.items()) if isinstance(groups, dict) else [(f"G{i + 1}", g) for i, g in enumerate(groups)]
    if len(items) < 3:
        raise ValueError("Para k grupos se necesitan al menos 3 (con 2 usa compare_two_groups).")
    names = [n for n, _ in items]
    data = [_arr(g, f"grupo {n}", min_n=2) for n, g in items]
    allv = np.concatenate(data)
    N, k = allv.size, len(data)
    norm = [_normal_enough(g)[0] for g in data]
    lev = stats.levene(*data, center="median")
    equal_var = bool(lev.pvalue >= 0.05)
    out: dict[str, Any] = {"analysis": "k_groups", "groups": names, "n": [int(g.size) for g in data], "alpha": alpha,
                           "assumptions": {"normality": dict(zip(names, norm)), "levene_p": round(float(lev.pvalue), 6), "equal_variances": equal_var}}
    why, warnings = [], []
    grand = allv.mean()
    ss_b = float(sum(g.size * (g.mean() - grand) ** 2 for g in data))
    ss_w = float(sum(((g - g.mean()) ** 2).sum() for g in data))

    if all(norm) and equal_var:
        res = stats.f_oneway(*data)
        F, p = float(res.statistic), float(res.pvalue)
        sm_res = anova_oneway(data, use_var="equal")
        manual_F = (ss_b / (k - 1)) / (ss_w / (N - k))
        manual_p = float(stats.f.sf(manual_F, k - 1, N - k))
        verified = _agree(F, manual_F) and _agree(p, manual_p) and _agree(p, float(sm_res.pvalue), 1e-7)
        eta2 = ss_b / (ss_b + ss_w)
        why.append(f"Normalidad en todos los grupos y varianzas homogéneas (Levene p={lev.pvalue:.4g}) -> ANOVA de un factor.")
        out.update(test="ANOVA de un factor", statistic=F, df=[k - 1, N - k], p_value=p, effect_size={"name": "η²", "value": eta2, "interpretation": "pequeño" if eta2 < 0.06 else "mediano" if eta2 < 0.14 else "grande"},
                   verification_methods=["scipy.stats.f_oneway", "statsmodels anova_oneway(equal) + formula manual F = CMentre/CMdentro"], discrepancy=abs(p - manual_p))
        if p < alpha:
            tk = pairwise_tukeyhsd(allv, np.repeat(names, [g.size for g in data]), alpha=alpha)
            rows = [{"a": str(r[0]), "b": str(r[1]), "diff": float(r[2]), "p_adj": float(r[3]), "low": float(r[4]), "high": float(r[5]), "reject": bool(r[6])} for r in tk._results_table.data[1:]]
            out["post_hoc"] = {"method": "Tukey HSD", "pairs": rows}
    elif all(norm):
        sm_res = anova_oneway(data, use_var="unequal")
        ns = np.array([g.size for g in data], float)
        vs = np.array([g.var(ddof=1) for g in data])
        ms = np.array([g.mean() for g in data])
        w = ns / vs
        mw = (w * ms).sum() / w.sum()
        tmp = ((1 - w / w.sum()) ** 2 / (ns - 1)).sum()
        F = ((w * (ms - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
        df2 = (k ** 2 - 1) / (3 * tmp)
        p = float(stats.f.sf(F, k - 1, df2))
        verified = _agree(float(sm_res.statistic), float(F), 1e-7) and _agree(float(sm_res.pvalue), p, 1e-7)
        omega2 = max(0.0, (ss_b - (k - 1) * ss_w / (N - k)) / (ss_b + ss_w + ss_w / (N - k)))
        why.append(f"Normalidad pero varianzas distintas (Levene p={lev.pvalue:.4g}) -> ANOVA de Welch.")
        out.update(test="ANOVA de Welch", statistic=float(F), df=[k - 1, float(df2)], p_value=p, effect_size={"name": "ω²", "value": float(omega2), "interpretation": "pequeño" if omega2 < 0.06 else "mediano" if omega2 < 0.14 else "grande"},
                   verification_methods=["statsmodels anova_oneway(use_var='unequal')", "formula manual de Welch (1951)"], discrepancy=abs(p - float(sm_res.pvalue)))
        if p < alpha:
            pairs = _pairwise_welch_holm(names, data, alpha)
            out["post_hoc"] = {"method": "t de Welch por pares con corrección de Holm", "pairs": pairs}
    else:
        res = stats.kruskal(*data)
        H, p = float(res.statistic), float(res.pvalue)
        ranks = stats.rankdata(allv)
        pos, mH = 0, 0.0
        for g in data:
            r = ranks[pos: pos + g.size]
            mH += r.sum() ** 2 / g.size
            pos += g.size
        _, counts = np.unique(allv, return_counts=True)
        tie = 1 - (counts ** 3 - counts).sum() / (N ** 3 - N)
        manual_H = (12 / (N * (N + 1)) * mH - 3 * (N + 1)) / tie
        manual_p = float(stats.chi2.sf(manual_H, k - 1))
        verified = _agree(H, manual_H, 1e-8) and _agree(p, manual_p, 1e-8)
        eps2 = H / ((N ** 2 - 1) / (N + 1))
        bad = [n for n, ok in zip(names, norm) if not ok]
        why.append(f"Normalidad no sostenida en {', '.join(bad)} -> Kruskal-Wallis (no paramétrica).")
        out.update(test="Kruskal-Wallis", statistic=H, df=k - 1, p_value=p, effect_size={"name": "ε²", "value": float(eps2), "interpretation": "pequeño" if eps2 < 0.06 else "mediano" if eps2 < 0.14 else "grande"},
                   verification_methods=["scipy.stats.kruskal", "formula manual H con correccion por empates"], discrepancy=abs(p - manual_p))
        if p < alpha:
            pairs = []
            for i in range(k):
                for j in range(i + 1, k):
                    pairs.append((names[i], names[j], float(stats.mannwhitneyu(data[i], data[j]).pvalue)))
            adj = multipletests([q[2] for q in pairs], alpha=alpha, method="holm")[1]
            out["post_hoc"] = {"method": "Mann-Whitney por pares con corrección de Holm",
                               "pairs": [{"a": a, "b": b, "p_raw": pr, "p_adj": float(pa), "reject": bool(pa < alpha)} for (a, b, pr), pa in zip(pairs, adj)]}
    if min(g.size for g in data) < 8:
        warnings.append("Algún grupo tiene menos de 8 observaciones: la comprobación de supuestos es poco fiable.")
    out.update(rationale=why, warnings=warnings, verified=bool(verified), decision=_verdict(out["p_value"], alpha),
               descriptives={n: {"mean": float(g.mean()), "sd": float(g.std(ddof=1)), "median": float(np.median(g))} for n, g in zip(names, data)})
    return out


def _pairwise_welch_holm(names: list[str], data: list[np.ndarray], alpha: float) -> list[dict[str, Any]]:
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            r = stats.ttest_ind(data[i], data[j], equal_var=False)
            pairs.append((names[i], names[j], float(data[i].mean() - data[j].mean()), float(r.pvalue)))
    adj = multipletests([q[3] for q in pairs], alpha=alpha, method="holm")[1]
    return [{"a": a, "b": b, "diff": d, "p_raw": pr, "p_adj": float(pa), "reject": bool(pa < alpha)} for (a, b, d, pr), pa in zip(pairs, adj)]


# ----------------------------------------------------------------------------- 3. tablas y proporciones


def chi_square_independence(table: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Independencia en tabla de contingencia: chi-cuadrado (o Fisher exacta en 2x2 con esperados < 5) + V de Cramer."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or min(t.shape) < 2 or (t < 0).any():
        raise ValueError("La tabla debe ser una matriz r x c (r,c >= 2) de conteos no negativos.")
    n = t.sum()
    exp = np.outer(t.sum(1), t.sum(0)) / n
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    manual = float(((t - exp) ** 2 / exp).sum())
    manual_p = float(stats.chi2.sf(manual, dof))
    verified = _agree(float(chi2), manual) and _agree(float(p), manual_p)
    v = math.sqrt(manual / (n * (min(t.shape) - 1)))
    low_exp = int((exp < 5).sum())
    out: dict[str, Any] = {"analysis": "chi_square", "alpha": alpha, "n": int(n), "expected_min": float(exp.min()), "cells_expected_lt_5": low_exp, "warnings": [], "rationale": []}
    if t.shape == (2, 2) and (exp < 5).any():
        orr, pf = stats.fisher_exact(t)
        out.update(test="Prueba exacta de Fisher", statistic=float(orr), statistic_name="razón de momios", df=None, p_value=float(pf),
                   effect_size={"name": "V de Cramér (phi)", "value": v, "interpretation": _interp_r(v)}, verified=bool(verified),
                   verification_methods=["scipy.stats.fisher_exact", "chi-cuadrado manual (solo contraste de consistencia)"], discrepancy=abs(float(p) - manual_p))
        out["rationale"].append(f"Tabla 2x2 con frecuencias esperadas < 5 (mínimo {exp.min():.2f}) -> Fisher exacta en lugar de chi-cuadrado.")
        p_used = float(pf)
    else:
        out.update(test="Chi-cuadrado de independencia", statistic=float(chi2), df=int(dof), p_value=float(p),
                   effect_size={"name": "V de Cramér", "value": v, "interpretation": "pequeño" if v < 0.3 else "mediano" if v < 0.5 else "grande"}, verified=bool(verified),
                   verification_methods=["scipy.stats.chi2_contingency", "formula manual Σ(O-E)²/E"], discrepancy=abs(float(p) - manual_p))
        out["rationale"].append("Frecuencias esperadas suficientes -> chi-cuadrado de Pearson.")
        if low_exp:
            out["warnings"].append(f"{low_exp} celdas con frecuencia esperada < 5: la aproximación chi-cuadrado puede no ser fiable.")
        p_used = float(p)
    out["expected"] = exp.round(3).tolist()
    out["decision"] = _verdict(p_used, alpha)
    return out


def two_proportions(successes: list[int], nobs: list[int], *, alpha: float = 0.05) -> dict[str, Any]:
    """Compara dos proporciones: z de dos proporciones, IC de Newcombe para la diferencia y razon de momios."""
    (a, b), (n1, n2) = successes, nobs
    if not (0 <= a <= n1 and 0 <= b <= n2 and n1 > 0 and n2 > 0):
        raise ValueError("Exitos fuera de rango (0 <= exitos <= n).")
    p1, p2 = a / n1, b / n2
    z, p = proportions_ztest([a, b], [n1, n2])
    pp = (a + b) / (n1 + n2)
    mz = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else float("nan")
    mp = 2 * stats.norm.sf(abs(mz))
    lo, hi = confint_proportions_2indep(a, n1, b, n2, method="newcomb", alpha=alpha)
    odds = (a * (n2 - b)) / max(b * (n1 - a), 1e-12)
    small = min(n1 * pp, n1 * (1 - pp), n2 * pp, n2 * (1 - pp)) < 5
    return {"analysis": "two_proportions", "test": "z de dos proporciones", "statistic": float(z), "df": None, "p_value": float(p), "alpha": alpha,
            "estimate": p1 - p2, "proportions": [p1, p2], "ci": {"level": 1 - alpha, "of": "diferencia de proporciones (Newcombe)", "low": float(lo), "high": float(hi)},
            "effect_size": {"name": "razón de momios", "value": float(odds), "interpretation": f"h de Cohen = {abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))):.3f}"},
            "verified": _agree(float(z), float(mz)) and _agree(float(p), float(mp)),
            "verification_methods": ["statsmodels.proportions_ztest", "z manual con proporcion conjunta"], "discrepancy": abs(float(p) - float(mp)),
            "rationale": ["Dos proporciones independientes -> z con proporción conjunta bajo H0."],
            "warnings": (["Frecuencias esperadas < 5: usa la prueba exacta de Fisher (chi_square con tabla 2x2)."] if small else []), "decision": _verdict(float(p), alpha)}


def one_proportion(successes: int, n: int, p0: float = 0.5, *, alpha: float = 0.05) -> dict[str, Any]:
    """Prueba binomial exacta para una proporcion, con IC de Wilson."""
    if not (0 <= successes <= n and n > 0 and 0 < p0 < 1):
        raise ValueError("Parametros fuera de rango.")
    res = stats.binomtest(successes, n, p0, alternative="two-sided")
    pm = float(sum(stats.binom.pmf(k, n, p0) for k in range(n + 1) if stats.binom.pmf(k, n, p0) <= stats.binom.pmf(successes, n, p0) * (1 + 1e-7)))
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return {"analysis": "one_proportion", "test": "Binomial exacta", "statistic": successes, "df": None, "p_value": float(res.pvalue), "alpha": alpha, "estimate": successes / n,
            "p0": p0, "ci": {"level": 1 - alpha, "of": "proporción (Wilson)", "low": float(lo), "high": float(hi)},
            "effect_size": {"name": "diferencia con p0", "value": successes / n - p0, "interpretation": ""}, "verified": _agree(float(res.pvalue), pm, 1e-7),
            "verification_methods": ["scipy.stats.binomtest", "suma manual de probabilidades binomiales <= observada"], "discrepancy": abs(float(res.pvalue) - pm),
            "rationale": ["Una proporción con n finito -> prueba binomial exacta (sin aproximación normal)."], "warnings": [], "decision": _verdict(float(res.pvalue), alpha)}


# ----------------------------------------------------------------------------- 4. correlacion


def correlation_test(x: Any, y: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Pearson si ambas variables son aprox. normales; si no, Spearman. IC por transformacion z de Fisher."""
    x, y = _arr(x, "x", 4), _arr(y, "y", 4)
    if x.size != y.size:
        raise ValueError("x e y deben tener el mismo tamaño.")
    n = x.size
    okx, ix = _normal_enough(x)
    oky, iy = _normal_enough(y)
    why = []
    if okx and oky:
        r, p = stats.pearsonr(x, y)
        mr = float(np.corrcoef(x, y)[0, 1])
        mt = mr * math.sqrt((n - 2) / max(1 - mr ** 2, 1e-15))
        mp = 2 * stats.t.sf(abs(mt), n - 2)
        name, methods = "r de Pearson", ["scipy.stats.pearsonr", "formula manual t = r√((n-2)/(1-r²))"]
        why.append(f"Ambas variables aproximadamente normales (Shapiro p={ix['shapiro_p']} y {iy['shapiro_p']}) -> Pearson.")
    else:
        r, p = stats.spearmanr(x, y)
        mr = float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])
        mt = mr * math.sqrt((n - 2) / max(1 - mr ** 2, 1e-15))
        mp = 2 * stats.t.sf(abs(mt), n - 2)
        name, methods = "ρ de Spearman", ["scipy.stats.spearmanr", "Pearson sobre rangos + t manual"]
        why.append("Normalidad no sostenida en alguna variable -> Spearman (monótona, por rangos).")
    z = np.arctanh(float(r))
    se = 1 / math.sqrt(max(n - 3, 1))
    crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))
    return {"analysis": "correlation", "test": name, "statistic": float(r), "df": n - 2, "p_value": float(p), "alpha": alpha, "n": int(n),
            "effect_size": {"name": name, "value": float(r), "interpretation": _interp_r(float(r))}, "ci": {"level": 1 - alpha, "of": "correlación (z de Fisher)", "low": lo, "high": hi},
            "verified": _agree(float(r), mr, 1e-7) and _agree(float(p), float(mp), 1e-6), "verification_methods": methods, "discrepancy": abs(float(p) - float(mp)),
            "rationale": why, "warnings": ["Correlación no implica causalidad."] + (["n < 10: IC muy ancho."] if n < 10 else []), "decision": _verdict(float(p), alpha)}


# ----------------------------------------------------------------------------- 5. regresion


def regression_inference(y: Any, X: dict[str, Any], *, alpha: float = 0.05) -> dict[str, Any]:
    """OLS con tabla de coeficientes (EE, t, p, IC), prueba F, errores robustos HC3 y diagnosticos (BP, DW, VIF, Shapiro, Cook)."""
    yv = _arr(y, "y", 5)
    cols = {k: _arr(v, k, 2) for k, v in X.items()}
    if any(c.size != yv.size for c in cols.values()):
        raise ValueError("Todas las variables deben tener la misma longitud que y.")
    n, k = yv.size, len(cols)
    if n <= k + 1:
        raise ValueError(f"n={n} es insuficiente para {k} predictores (+ intercepto).")
    Xd = pd.DataFrame(cols)
    Xc = sm.add_constant(Xd, has_constant="add")
    fit = sm.OLS(yv, Xc).fit()
    robust = sm.OLS(yv, Xc).fit(cov_type="HC3")
    A = Xc.to_numpy()
    beta = np.linalg.solve(A.T @ A, A.T @ yv)
    resid = yv - A @ beta
    df_res = n - A.shape[1]
    s2 = float(resid @ resid / df_res)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    tstat = beta / se
    pv = 2 * stats.t.sf(np.abs(tstat), df_res)
    verified = bool(np.allclose(fit.params, beta, rtol=1e-8, atol=1e-10) and np.allclose(fit.bse, se, rtol=1e-8) and np.allclose(fit.pvalues, pv, rtol=1e-6, atol=1e-12))
    ci = fit.conf_int(alpha=alpha)
    names = ["intercepto"] + list(cols)
    table = [{"term": nm, "coef": float(fit.params.iloc[i] if hasattr(fit.params, "iloc") else fit.params[i]), "se": float(fit.bse.iloc[i] if hasattr(fit.bse, "iloc") else fit.bse[i]),
              "t": float(fit.tvalues.iloc[i] if hasattr(fit.tvalues, "iloc") else fit.tvalues[i]), "p": float(fit.pvalues.iloc[i] if hasattr(fit.pvalues, "iloc") else fit.pvalues[i]),
              "ci_low": float(np.asarray(ci)[i, 0]), "ci_high": float(np.asarray(ci)[i, 1]),
              "p_robust_HC3": float(robust.pvalues.iloc[i] if hasattr(robust.pvalues, "iloc") else robust.pvalues[i])} for i, nm in enumerate(names)]
    warnings = []
    bp = het_breuschpagan(fit.resid, Xc)
    dw = float(durbin_watson(fit.resid))
    sw = float(stats.shapiro(fit.resid[:5000]).pvalue)
    vifs = {nm: float(variance_inflation_factor(A, i + 1)) for i, nm in enumerate(cols)} if k >= 2 else {}
    cooks = fit.get_influence().cooks_distance[0]
    n_infl = int((cooks > 4 / n).sum())
    if bp[1] < 0.05:
        warnings.append(f"Heterocedasticidad (Breusch-Pagan p={bp[1]:.3g}): usa los p con errores robustos HC3.")
    if sw < 0.05:
        warnings.append(f"Residuos no normales (Shapiro p={sw:.3g}): los p/IC exactos pueden ser poco fiables con n pequeño.")
    if vifs and max(vifs.values()) > 10:
        warnings.append(f"Multicolinealidad (VIF máx = {max(vifs.values()):.1f} > 10).")
    if not (1.5 <= dw <= 2.5):
        warnings.append(f"Posible autocorrelación de residuos (Durbin-Watson = {dw:.2f}).")
    if n < 10 * k:
        warnings.append(f"n={n} para {k} predictores (< 10 por predictor): riesgo de sobreajuste.")
    return {"analysis": "regression", "test": "Regresión lineal por MCO", "n": int(n), "alpha": alpha, "coefficients": table,
            "model": {"r2": float(fit.rsquared), "r2_adj": float(fit.rsquared_adj), "F": float(fit.fvalue), "F_p": float(fit.f_pvalue), "df": [int(fit.df_model), int(fit.df_resid)], "sigma": math.sqrt(s2)},
            "diagnostics": {"breusch_pagan_p": float(bp[1]), "durbin_watson": dw, "shapiro_resid_p": sw, "vif": vifs, "cook_n_over_4_over_n": n_infl},
            "statistic": float(fit.fvalue), "p_value": float(fit.f_pvalue), "verified": verified,
            "verification_methods": ["statsmodels.OLS", "ecuaciones normales con NumPy: β=(X'X)⁻¹X'y, EE=√(s²·diag((X'X)⁻¹))"],
            "discrepancy": float(np.max(np.abs(np.asarray(fit.bse) - se))), "rationale": ["Regresión lineal múltiple por mínimos cuadrados ordinarios con inferencia t por coeficiente y F global."],
            "warnings": warnings, "decision": _verdict(float(fit.f_pvalue), alpha).replace("H0", "H0: todos los β = 0")}


# ----------------------------------------------------------------------------- 6. potencia y multiples comparaciones


def _power_manual(d: float, n_per_group: float, alpha: float) -> float:
    df = 2 * n_per_group - 2
    nc = d * math.sqrt(n_per_group / 2)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def power_two_sample_t(*, effect_size: float | None = None, n_per_group: float | None = None, power: float | None = None, alpha: float = 0.05) -> dict[str, Any]:
    """Potencia / tamano de muestra / efecto minimo detectable para t de dos muestras (bilateral). Deja exactamente UNO como incognita."""
    unknown = [nm for nm, v in (("effect_size", effect_size), ("n_per_group", n_per_group), ("power", power)) if v is None]
    if len(unknown) != 1:
        raise ValueError("Deja exactamente uno de effect_size, n_per_group o power como incognita (None).")
    pw = TTestIndPower()
    solved = unknown[0]
    val = float(pw.solve_power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"))
    d = val if solved == "effect_size" else effect_size
    n = math.ceil(val) if solved == "n_per_group" else (n_per_group if solved != "n_per_group" else val)
    check = _power_manual(d, n, alpha)
    target = power if solved != "power" else val
    verified = _agree(check, val, 1e-6) if solved == "power" else (check >= target - 5e-3 if solved == "n_per_group" else _agree(check, target, 1e-4))
    return {"analysis": "power_t", "test": "Potencia t de dos muestras (bilateral)", "solved_for": solved, "alpha": alpha,
            "effect_size": d, "n_per_group": n, "power": check if solved != "power" else val, "result": val if solved != "n_per_group" else int(math.ceil(val)),
            "verified": bool(verified), "verification_methods": ["statsmodels.TTestIndPower.solve_power", "t no central: P(|T|>t_crit) con scipy.stats.nct"],
            "discrepancy": abs(check - (val if solved == "power" else target)), "rationale": ["Se resuelve la incógnita con la distribución t no central y se comprueba reinsertando el resultado."],
            "warnings": [], "decision": f"Resultado: {solved} = {val:.4f}" + (f" -> {int(math.ceil(val))} por grupo" if solved == "n_per_group" else "")}


def adjust_pvalues(pvalues: list[float], *, method: str = "holm", alpha: float = 0.05) -> dict[str, Any]:
    """Correccion por comparaciones multiples: holm | bonferroni | fdr_bh."""
    p = np.asarray(pvalues, dtype=float)
    if p.size < 2 or ((p < 0) | (p > 1)).any():
        raise ValueError("Se necesitan >= 2 p-valores en [0, 1].")
    if method not in {"holm", "bonferroni", "fdr_bh"}:
        raise ValueError("method debe ser holm, bonferroni o fdr_bh.")
    rej, padj, _, _ = multipletests(p, alpha=alpha, method=method)
    m = p.size
    order = np.argsort(p)
    manual = np.empty(m)
    if method == "bonferroni":
        manual = np.minimum(p * m, 1.0)
    elif method == "holm":
        run = 0.0
        for rank, idx in enumerate(order):
            run = max(run, (m - rank) * p[idx])
            manual[idx] = min(run, 1.0)
    else:
        run = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            run = min(run, p[idx] * m / (rank + 1))
            manual[idx] = run
    return {"analysis": "adjust_pvalues", "test": f"Corrección {method}", "alpha": alpha, "p_raw": p.tolist(), "p_adjusted": [float(v) for v in padj], "reject": [bool(v) for v in rej],
            "verified": bool(np.allclose(padj, manual, rtol=1e-9, atol=1e-12)), "verification_methods": ["statsmodels.multipletests", f"implementación manual de {method}"],
            "discrepancy": float(np.max(np.abs(padj - manual))), "rationale": [f"{m} contrastes simultáneos: se controla {'la tasa de error por familia (FWER)' if method != 'fdr_bh' else 'la tasa de falsos descubrimientos (FDR)'}."],
            "warnings": [], "decision": f"{int(rej.sum())} de {m} contrastes siguen siendo significativos tras la corrección."}


ANALYSES = {
    "two_groups": lambda d, a: compare_two_groups(d["x"], d["y"], paired=bool(d.get("paired", False)), alternative=d.get("alternative", "two-sided"), alpha=a, method=d.get("method", "auto")),
    "one_sample": lambda d, a: one_sample_t(d["x"], float(d.get("mu0", 0.0)), alternative=d.get("alternative", "two-sided"), alpha=a),
    "ci_mean": lambda d, a: ci_mean(d["x"], level=1 - a),
    "ci_proportion": lambda d, a: ci_proportion(int(d["successes"]), int(d["n"]), level=1 - a),
    "variance_ratio": lambda d, a: variance_ratio_F(d["x"], d["y"], alpha=a),
    "z_test": lambda d, a: z_test_known_sigma(float(d["mean"]), int(d["n"]), float(d["mu0"]), float(d["sigma"]), alternative=d.get("alternative", "two-sided"), alpha=a),
    "sample_size": lambda d, a: sample_size(d["kind"], margin=float(d["margin"]), level=1 - a, sigma=d.get("sigma"), p=float(d.get("p", 0.5))),
    "k_groups": lambda d, a: compare_k_groups(d["groups"], alpha=a),
    "chi_square": lambda d, a: chi_square_independence(d["table"], alpha=a),
    "two_proportions": lambda d, a: two_proportions(d["successes"], d["n"], alpha=a),
    "one_proportion": lambda d, a: one_proportion(int(d["successes"]), int(d["n"]), float(d.get("p0", 0.5)), alpha=a),
    "correlation": lambda d, a: correlation_test(d["x"], d["y"], alpha=a),
    "regression": lambda d, a: regression_inference(d["y"], d["X"], alpha=a),
    "power_t": lambda d, a: power_two_sample_t(effect_size=d.get("effect_size"), n_per_group=d.get("n_per_group"), power=d.get("power"), alpha=a),
    "adjust_pvalues": lambda d, a: adjust_pvalues(d["pvalues"], method=d.get("method", "holm"), alpha=a),
}



# ----------------------------------------------------------------------------- 1b. prueba FORZADA (method=) y analisis de sensibilidad


def _two_groups_forced(x: np.ndarray, y: np.ndarray, method: str, alternative: str, alpha: float, sensitivity: bool = True) -> dict[str, Any]:
    """El usuario fija la prueba (p. ej. la guia exige Welch). Nunca se sustituye en silencio: si los supuestos no se sostienen se AVISA y,
    para Welch/Student, se devuelve ademas el resultado robusto como `sensitivity` (analisis de sensibilidad)."""
    rng = np.random.default_rng(SEED)
    okx, ix = _normal_enough(x)
    oky, iy = _normal_enough(y)
    lev = stats.levene(x, y, center="median")
    n1, n2 = x.size, y.size
    out: dict[str, Any] = {"analysis": "two_groups", "paired": False, "alternative": alternative, "alpha": alpha, "method_requested": method,
                           "assumptions": {"normality_x": ix, "normality_y": iy, "levene_p": round(float(lev.pvalue), 6), "equal_variances": bool(lev.pvalue >= 0.05)}}
    warnings: list[str] = []
    if method in {"welch", "student"}:
        usevar = "unequal" if method == "welch" else "pooled"
        res = stats.ttest_ind(x, y, equal_var=(method == "student"), alternative=alternative)
        sm_t, sm_p, sm_df = ttest_ind(x, y, alternative={"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alternative], usevar=usevar)
        t, p = float(res.statistic), float(res.pvalue)
        cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
        lo, hi = cm.tconfint_diff(alpha=alpha if alternative == "two-sided" else 2 * alpha, usevar=usevar)
        v1, v2 = x.var(ddof=1), y.var(ddof=1)
        if method == "welch":
            se2 = v1 / n1 + v2 / n2
            man_df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))   # Welch-Satterthwaite a mano
            man_t = (x.mean() - y.mean()) / math.sqrt(se2)
            sp = math.sqrt((v1 + v2) / 2)
        else:
            man_df = n1 + n2 - 2
            sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / man_df)
            man_t = (x.mean() - y.mean()) / (sp * math.sqrt(1 / n1 + 1 / n2))
        man_p = {"two-sided": 2 * stats.t.sf(abs(man_t), man_df), "greater": stats.t.sf(man_t, man_df), "less": stats.t.cdf(man_t, man_df)}[alternative]
        verified = _agree(t, float(sm_t)) and _agree(p, float(sm_p)) and _agree(float(sm_df), float(man_df), 1e-8) and _agree(p, float(man_p), 1e-8)
        d = float((x.mean() - y.mean()) / sp)
        g = d * (1 - 3 / (4 * (n1 + n2) - 9))
        name = "t de Welch" if method == "welch" else "t de Student (varianzas agrupadas)"
        out.update(test=name, statistic=t, df=float(man_df), p_value=p, effect_size={"name": "g de Hedges", "value": float(g), "interpretation": _interp_d(g)},
                   ci={"level": 1 - alpha, "of": "diferencia de medias (x - y)", "low": float(lo), "high": float(hi)}, estimate=float(x.mean() - y.mean()),
                   verification_methods=["scipy.stats.ttest_ind", f"statsmodels ttest_ind(usevar='{usevar}')", "formula manual" + (" (Welch-Satterthwaite)" if method == "welch" else " (varianza agrupada)")],
                   discrepancy=abs(p - float(man_p)), rationale=[f"Prueba FORZADA por el usuario: {name}."])
        if method == "student" and lev.pvalue < 0.05:
            warnings.append(f"Las varianzas difieren (Levene p={lev.pvalue:.4g}): la t agrupada infla los falsos positivos; Welch es la opción segura.")
        if not (okx and oky):
            bad = [n for n, ok in (("x", okx), ("y", oky)) if not ok]
            if sensitivity:
                warnings.append(f"Normalidad dudosa en {', '.join(bad)} (Shapiro p={ix['shapiro_p']} y {iy['shapiro_p']}): se calculó la prueba pedida y se incluye `sensitivity` con la alternativa robusta.")
                sens = compare_two_groups(x, y, alternative=alternative, alpha=alpha, method="mann_whitney" if lev.pvalue >= 0.05 else "auto")
                out["sensitivity"] = {k: sens[k] for k in ("test", "statistic", "p_value", "effect_size", "decision", "verified") if k in sens}
                out["sensitivity"]["agrees_with_requested"] = bool((sens["p_value"] < alpha) == (p < alpha))
            else:
                warnings.append(f"Normalidad dudosa en {', '.join(bad)} (Shapiro p={ix['shapiro_p']} y {iy['shapiro_p']}): se calculó la prueba pedida; el análisis de sensibilidad automático se omitió a petición (sensitivity=False) y debe aportarse aparte.")
    else:  # mann_whitney forzado
        res = stats.mannwhitneyu(x, y, alternative=alternative, method="auto")
        p, u1 = float(res.pvalue), float(res.statistic)
        pperm = _perm_p_two_sample(x, y, lambda a, b: float(stats.rankdata(np.concatenate([a, b]))[: a.size].sum()) - a.size * (a.size + b.size + 1) / 2, alternative, rng)
        verified = abs(p - pperm) <= _mc_tol(p)
        rb = float(2 * u1 / (n1 * n2) - 1)
        lo, hi = _boot_ci(lambda a, b: float(np.median(a) - np.median(b)), rng, x, y, level=1 - alpha)
        out.update(test="U de Mann-Whitney", statistic=u1, df=None, p_value=p, effect_size={"name": "correlación biserial por rangos", "value": rb, "interpretation": _interp_r(rb)},
                   ci={"level": 1 - alpha, "of": "diferencia de medianas (bootstrap percentil)", "low": lo, "high": hi}, estimate=float(np.median(x) - np.median(y)),
                   verification_methods=["scipy.stats.mannwhitneyu", f"prueba de permutacion de rangos ({N_PERM} permutaciones)"], discrepancy=abs(p - pperm),
                   rationale=["Prueba FORZADA por el usuario: Mann-Whitney."])
        if lev.pvalue < 0.05:
            warnings.append(f"Las dispersiones difieren (Levene p={lev.pvalue:.4g}): Mann-Whitney no contrasta medianas aquí y puede inflar los falsos positivos; considera Brunner-Munzel (method='auto').")
    if min(n1, n2) < 15:
        warnings.append(f"Muestras pequeñas (n1={n1}, n2={n2}): la potencia es baja y los supuestos son difíciles de comprobar.")
    out.update(warnings=warnings, verified=bool(verified), decision=_verdict(out["p_value"], alpha), n=[int(n1), int(n2)],
               descriptives={"x": {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(np.median(x))}, "y": {"mean": float(y.mean()), "sd": float(y.std(ddof=1)), "median": float(np.median(y))}})
    return out


# ----------------------------------------------------------------------------- 4b. t de una muestra, IC, razon de varianzas, z, tamanos de muestra


def one_sample_t(x: Any, mu0: float = 0.0, *, alternative: str = "two-sided", alpha: float = 0.05) -> dict[str, Any]:
    """t de una muestra contra mu0 con IC de la media y comprobacion de normalidad; con no normalidad y n pequeno agrega Wilcoxon como sensibilidad."""
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'.")
    x = _arr(x, "muestra", 3)
    n = x.size
    res = stats.ttest_1samp(x, mu0, alternative=alternative)
    t, p = float(res.statistic), float(res.pvalue)
    s = x.std(ddof=1)
    mt = (x.mean() - mu0) / (s / math.sqrt(n))
    mp = {"two-sided": 2 * stats.t.sf(abs(mt), n - 1), "greater": stats.t.sf(mt, n - 1), "less": stats.t.cdf(mt, n - 1)}[alternative]
    lo, hi = DescrStatsW(x).tconfint_mean(alpha=alpha if alternative == "two-sided" else 2 * alpha)
    ok, info = _normal_enough(x)
    d = float((x.mean() - mu0) / s)
    warnings = []
    out: dict[str, Any] = {"analysis": "one_sample", "test": "t de una muestra", "statistic": t, "df": n - 1, "p_value": p, "alpha": alpha, "mu0": mu0, "alternative": alternative,
                           "estimate": float(x.mean()), "ci": {"level": 1 - alpha, "of": "media", "low": float(lo), "high": float(hi)},
                           "effect_size": {"name": "d de Cohen", "value": d, "interpretation": _interp_d(d)}, "n": int(n), "assumptions": {"normality": info},
                           "verified": _agree(t, float(mt)) and _agree(p, float(mp), 1e-8), "verification_methods": ["scipy.stats.ttest_1samp", "formula manual t = (x̄-μ0)/(s/√n)"],
                           "discrepancy": abs(p - float(mp)), "rationale": [f"Una muestra contra μ0 = {mu0}: t de Student con n-1 = {n - 1} gl."], "decision": _verdict(p, alpha)}
    if not ok and n < 30:
        warnings.append(f"Normalidad dudosa con n={n} (Shapiro p={info['shapiro_p']}): se incluye Wilcoxon de una muestra como sensibilidad.")
        diff = x - mu0
        if np.any(diff != 0):
            w = stats.wilcoxon(diff, alternative=alternative)
            out["sensitivity"] = {"test": "Wilcoxon de una muestra", "p_value": float(w.pvalue), "agrees_with_requested": bool((w.pvalue < alpha) == (p < alpha))}
    if n < 10:
        warnings.append("n < 10: el IC es muy ancho y los supuestos son difíciles de comprobar.")
    out["warnings"] = warnings
    return out


def ci_mean(x: Any, *, level: float = 0.95) -> dict[str, Any]:
    """IC t para una media: x̄ ± t(1-α/2; n-1)·s/√n, contrastado con statsmodels."""
    if not 0 < level < 1:
        raise ValueError("level debe estar en (0, 1).")
    x = _arr(x, "muestra", 3)
    n, a = x.size, 1 - level
    m, s = float(x.mean()), float(x.std(ddof=1))
    half = float(stats.t.ppf(1 - a / 2, n - 1) * s / math.sqrt(n))
    lo, hi = DescrStatsW(x).tconfint_mean(alpha=a)
    ok, info = _normal_enough(x)
    return {"analysis": "ci_mean", "test": "Intervalo de confianza para la media (t)", "estimate": m, "ci": {"level": level, "of": "media", "low": m - half, "high": m + half}, "df": n - 1, "n": int(n),
            "margin_of_error": half, "verified": _agree(m - half, float(lo), 1e-9) and _agree(m + half, float(hi), 1e-9), "verification_methods": ["formula manual x̄ ± t·s/√n", "statsmodels DescrStatsW.tconfint_mean"],
            "discrepancy": abs(m - half - float(lo)), "rationale": [f"Normalidad {'aceptable' if ok else 'dudosa'} (Shapiro p={info['shapiro_p']})."],
            "warnings": ([] if ok or n >= 30 else [f"Normalidad dudosa con n={n}: el IC t puede no tener la cobertura nominal."]),
            "decision": f"Con {level:.0%} de confianza, la media está entre {m - half:.4g} y {m + half:.4g}. Eso no dice que el {level:.0%} de los datos caiga ahí."}


def ci_proportion(successes: int, n: int, *, level: float = 0.95) -> dict[str, Any]:
    """IC de una proporcion por Wilson y por Clopper-Pearson (exacto), cada uno verificado contra una formula independiente."""
    if not (n > 0 and 0 <= successes <= n and 0 < level < 1):
        raise ValueError("Parametros fuera de rango (0 <= exitos <= n, n > 0, 0 < level < 1).")
    a, ph, z = 1 - level, successes / n, float(stats.norm.ppf(1 - (1 - level) / 2))
    wl, wh = proportion_confint(successes, n, alpha=a, method="wilson")
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    cl, ch = proportion_confint(successes, n, alpha=a, method="beta")
    bl = 0.0 if successes == 0 else float(stats.beta.ppf(a / 2, successes, n - successes + 1))
    bh = 1.0 if successes == n else float(stats.beta.ppf(1 - a / 2, successes + 1, n - successes))
    verified = _agree(float(wl), centre - half, 1e-9) and _agree(float(wh), centre + half, 1e-9) and _agree(float(cl), bl, 1e-8) and _agree(float(ch), bh, 1e-8)
    return {"analysis": "ci_proportion", "test": "IC de una proporción (Wilson y Clopper-Pearson)", "estimate": ph, "n": int(n),
            "ci": {"level": level, "of": "proporción (Wilson)", "low": float(wl), "high": float(wh)},
            "intervals": {"wilson": [float(wl), float(wh)], "clopper_pearson": [float(cl), float(ch)]}, "verified": bool(verified),
            "verification_methods": ["statsmodels proportion_confint (wilson, beta)", "formula manual de Wilson y cuantiles de la beta (scipy)"], "discrepancy": abs(float(wl) - (centre - half)),
            "rationale": ["Wilson es la recomendación general; Clopper-Pearson es exacto pero conservador (más ancho)."],
            "warnings": (["Muestra pequeña o proporción extrema: prefiere Wilson o Clopper-Pearson; evita el intervalo de Wald."] if min(successes, n - successes) < 5 else []),
            "decision": f"Con {level:.0%} de confianza, la proporción está entre {float(wl):.4g} y {float(wh):.4g} (Wilson)."}


def variance_ratio_F(x: Any, y: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Razon de varianzas F = s1²/s2² (bilateral) con IC. MUY sensible a la no normalidad: advierte y compara con Levene."""
    x, y = _arr(x, "grupo x", 3), _arr(y, "grupo y", 3)
    d1, d2 = x.size - 1, y.size - 1
    v1, v2 = float(x.var(ddof=1)), float(y.var(ddof=1))
    F = v1 / v2
    p = float(min(1.0, 2 * min(stats.f.cdf(F, d1, d2), stats.f.sf(F, d1, d2))))
    mp = float(min(1.0, 2 * min(stats.f(d1, d2).cdf(F), stats.f(d1, d2).sf(F))))
    lo, hi = F / float(stats.f.ppf(1 - alpha / 2, d1, d2)), F / float(stats.f.ppf(alpha / 2, d1, d2))
    lev = stats.levene(x, y, center="median")
    okx, _ = _normal_enough(x)
    oky, _ = _normal_enough(y)
    warnings = [] if (okx and oky) else ["La prueba F de varianzas es MUY sensible a la no normalidad: si los datos no son normales, usa Levene (se informa abajo)."]
    return {"analysis": "variance_ratio", "test": "F de razón de varianzas", "statistic": F, "df": [d1, d2], "p_value": p, "alpha": alpha, "estimate": F,
            "ci": {"level": 1 - alpha, "of": "razón de varianzas σ1²/σ2²", "low": lo, "high": hi}, "levene": {"p_value": float(lev.pvalue), "statistic": float(lev.statistic)},
            "verified": _agree(p, mp, 1e-10), "verification_methods": ["scipy.stats.f (cdf/sf)", "scipy.stats.f(d1, d2) (objeto de distribución congelado)"], "discrepancy": abs(p - mp),
            "rationale": ["Se compara F = s1²/s2² con la distribución F(n1-1, n2-1); el p-valor bilateral duplica la cola menor."], "warnings": warnings, "decision": _verdict(p, alpha)}


def z_test_known_sigma(mean: float, n: int, mu0: float, sigma: float, *, alternative: str = "two-sided", alpha: float = 0.05) -> dict[str, Any]:
    """z para una media con sigma POBLACIONAL conocida (contraste con el caso t, donde sigma se estima)."""
    if n < 1 or sigma <= 0 or alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("Parametros invalidos: n >= 1, sigma > 0 y alternative valida.")
    se = sigma / math.sqrt(n)
    z = (mean - mu0) / se
    p = {"two-sided": 2 * stats.norm.sf(abs(z)), "greater": stats.norm.sf(z), "less": stats.norm.cdf(z)}[alternative]
    dist = stats.norm(loc=mu0, scale=se)
    mp = {"two-sided": 2 * min(dist.cdf(mean), dist.sf(mean)), "greater": dist.sf(mean), "less": dist.cdf(mean)}[alternative]
    zc = float(stats.norm.ppf(1 - alpha / 2))
    return {"analysis": "z_test", "test": "z de una media (σ conocida)", "statistic": float(z), "df": None, "p_value": float(p), "alpha": alpha, "n": int(n), "mu0": mu0, "sigma": sigma,
            "estimate": mean, "ci": {"level": 1 - alpha, "of": "media (σ conocida)", "low": mean - zc * se, "high": mean + zc * se},
            "verified": _agree(float(p), float(mp), 1e-10), "verification_methods": ["z manual con scipy.stats.norm.sf/cdf", "scipy.stats.norm(loc=μ0, scale=σ/√n) (distribución de x̄ bajo H0)"],
            "discrepancy": abs(float(p) - float(mp)), "rationale": ["σ es CONOCIDA (dato del enunciado, no estimada): el estadístico es z, no t."],
            "warnings": ["Si σ no es realmente conocida, usa la t de una muestra."], "decision": _verdict(float(p), alpha)}


def sample_size(kind: str, *, margin: float, level: float = 0.95, sigma: float | None = None, p: float = 0.5) -> dict[str, Any]:
    """Tamano de muestra para estimar con margen de error E: kind = mean (n=(zσ/E)²) | two_means (por grupo, 2(zσ/E)²) | proportion (n=z²p(1-p)/E²)."""
    if kind not in {"mean", "two_means", "proportion"} or margin <= 0 or not 0 < level < 1:
        raise ValueError("kind debe ser mean|two_means|proportion, margin > 0 y 0 < level < 1.")
    if kind != "proportion" and (sigma is None or sigma <= 0):
        raise ValueError("Con mean/two_means se necesita sigma > 0 (estimala de un piloto o de la literatura).")
    if kind == "proportion" and not 0 < p < 1:
        raise ValueError("p debe estar en (0, 1) (usa 0.5 si no sabes: es el caso más conservador).")
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    var = {"mean": sigma ** 2 if sigma else 0, "two_means": 2 * sigma ** 2 if sigma else 0, "proportion": p * (1 - p)}[kind]
    exact = z * z * var / (margin * margin)
    n = math.ceil(exact)

    def half_width(m: int) -> float:  # metodo B: semiancho del intervalo con n observaciones, por la distribucion normal directamente
        lo, hi = stats.norm.interval(level, loc=0, scale=math.sqrt(var / m))
        return float((hi - lo) / 2)

    verified = half_width(n) <= margin + 1e-12 and (n == 1 or half_width(n - 1) > margin)
    return {"analysis": "sample_size", "test": f"Tamaño de muestra ({kind})", "n_required": int(n), "n_exact": exact, "margin": margin, "level": level, "sigma": sigma, "p": p if kind == "proportion" else None,
            "unit": "por grupo" if kind == "two_means" else "en total", "verified": bool(verified), "verification_methods": ["fórmula cerrada n = z²·var/E²", "semiancho del IC con scipy.stats.norm.interval: n cumple y n-1 no"],
            "discrepancy": 0.0 if verified else 1.0, "rationale": ["Se redondea SIEMPRE hacia arriba: quedarse corto rompe el margen pedido."],
            "warnings": ["Es el tamaño para ESTIMAR con ese margen, no para detectar un efecto (para eso usa power_t)." ] + (["σ es un supuesto: si es mayor, necesitarás más muestra."] if kind != "proportion" else []),
            "decision": f"Se necesitan n = {n} ({'por grupo' if kind == 'two_means' else 'en total'}) para un margen de {margin} al {level:.0%}."}


_COMPACT_KEYS = ["analysis", "test", "statistic", "df", "p_value", "alpha", "effect_size", "ci", "estimate", "rationale", "warnings", "decision", "verified",
                 "verification_methods", "n", "post_hoc", "descriptives", "coefficients", "model", "diagnostics", "p_raw", "p_adjusted", "reject",
                 "solved_for", "result", "n_per_group", "power", "proportions", "p0", "expected_min", "cells_expected_lt_5", "statistic_name",
                 "sensitivity", "intervals", "margin_of_error", "n_required", "levene", "method_requested"]


def compact_result(res: dict[str, Any]) -> dict[str, Any]:
    """Formato UNICO del resultado que ve Ada (CLI, MCP y entrenamiento): lo necesario para redactar la respuesta, con cifras a 6 cifras significativas.

    Incluye `verification_methods` para que el sello [VERIFICADO: ...] de la respuesta salga del resultado y no se invente.
    """
    def rnd(o: Any) -> Any:
        if isinstance(o, float):
            return float(f"{o:.6g}") if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rnd(v) for v in o]
        return o

    out = rnd({k: res[k] for k in _COMPACT_KEYS if k in res})
    direction = _direction_note(res)
    if direction:
        out["direction"] = direction
    return out


def _direction_note(res: dict[str, Any]) -> str | None:
    """Frase explicita de QUE GRUPO ES MAYOR. El signo de `estimate` (x - y) se leyo al reves en una respuesta real
    ("redujo" cuando las medias SUBIERON); con la frase en el resultado no hay que interpretar signos."""
    if res.get("analysis") != "two_groups":
        return None
    desc = res.get("descriptives") or {}
    for stat, label in (("mean", "media"), ("median", "mediana")):
        try:
            a, b = float(desc["x"][stat]), float(desc["y"][stat])
        except (KeyError, TypeError, ValueError):
            continue
        if a == b:
            return f"La {label} de x y la de y son iguales ({a:.4g})."
        hi, lo = ("y", "x") if b > a else ("x", "y")
        return f"La {label} de {hi} ({max(a, b):.4g}) es MAYOR que la de {lo} ({min(a, b):.4g}); x - y = {a - b:.4g}."
    return None


def run_inference(analysis: str, data: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Punto de entrada unico (lo usan el CLI, el MCP y el generador de curriculo)."""
    if analysis not in ANALYSES:
        raise ValueError(f"analysis desconocido: {analysis!r}. Disponibles: {', '.join(ANALYSES)}")
    if not (0 < alpha < 1):
        raise ValueError("alpha debe estar en (0, 1).")
    res = ANALYSES[analysis](data, alpha)
    res.setdefault("alpha", alpha)
    return res
