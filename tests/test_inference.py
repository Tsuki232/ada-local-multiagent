"""Pruebas de la inferencia estadistica verificada: valores de libro, calibracion por simulacion y entradas invalidas.

Los valores esperados se calcularon A MANO o son clasicos (Cohen, Holm, Benjamini-Hochberg): no salen del propio motor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from ada.stats import inference as inf
from ada.stats.inference import run_inference


@pytest.fixture()
def fast(monkeypatch):
    """Menos permutaciones/bootstrap para las simulaciones (la verificacion cruzada sigue siendo valida, solo mas ruidosa)."""
    monkeypatch.setattr(inf, "N_PERM", 200)
    monkeypatch.setattr(inf, "N_BOOT", 60)


# ----------------------------------------------------------------------------- valores calculados a mano


def test_pooled_t_matches_hand_calculation():
    # x=[1..5], y=[3..7]: varianzas iguales (2.5). t = (3-5)/sqrt(2.5*(2/5)) = -2, df = 8, p = 2*P(T8 > 2)
    # Con n iguales y varianzas iguales, Welch coincide exactamente con la t agrupada (t=-2, gl=8).
    r = run_inference("two_groups", {"x": [1, 2, 3, 4, 5], "y": [3, 4, 5, 6, 7]})
    assert r["test"] == "t de Welch" and r["verified"]
    assert r["statistic"] == pytest.approx(-2.0, abs=1e-12) and r["df"] == pytest.approx(8.0)
    assert r["p_value"] == pytest.approx(2 * stats.t.sf(2.0, 8), rel=1e-12) and r["p_value"] == pytest.approx(0.0805, abs=1e-4)
    # d = (3-5)/sqrt(2.5) = -1.2649 ; g = d * (1 - 3/(4*10-9))
    assert r["effect_size"]["value"] == pytest.approx(-2 / math.sqrt(2.5) * (1 - 3 / 31), abs=1e-9)
    assert r["ci"]["low"] < -2 < r["ci"]["high"] or r["ci"]["low"] < r["estimate"] < r["ci"]["high"]
    assert r["estimate"] == pytest.approx(-2.0)


def test_chi_square_matches_hand_calculation():
    # [[10,20],[20,10]]: n=60, esperadas 15 -> chi2 = 4*(25/15) = 6.6667, gl=1, V = sqrt(6.6667/60) = 1/3
    r = run_inference("chi_square", {"table": [[10, 20], [20, 10]]})
    assert r["statistic"] == pytest.approx(20 / 3, rel=1e-12) and r["df"] == 1 and r["verified"]
    assert r["p_value"] == pytest.approx(0.00982, abs=2e-5)
    assert r["effect_size"]["value"] == pytest.approx(1 / 3, rel=1e-9)


def test_fisher_is_chosen_for_small_expected_counts_and_matches_the_classic_p():
    r = run_inference("chi_square", {"table": [[8, 2], [1, 5]]})
    assert r["test"] == "Prueba exacta de Fisher" and r["p_value"] == pytest.approx(0.03497, abs=1e-4)
    assert r["statistic"] == pytest.approx(20.0) and r["cells_expected_lt_5"] > 0


def test_two_proportions_and_exact_binomial_known_values():
    r = run_inference("two_proportions", {"successes": [45, 30], "n": [100, 100]})
    assert r["statistic"] == pytest.approx(0.15 / math.sqrt(0.375 * 0.625 * 0.02), rel=1e-12) and r["verified"]
    assert r["p_value"] == pytest.approx(0.02846, abs=1e-4)
    b = run_inference("one_proportion", {"successes": 14, "n": 20, "p0": 0.5})
    assert b["p_value"] == pytest.approx(0.1153, abs=2e-4) and b["verified"]  # 2*P(X>=14 | n=20, .5)


def test_power_matches_cohens_classic_64_per_group():
    r = run_inference("power_t", {"effect_size": 0.5, "power": 0.8})
    assert r["result"] == 64 and r["verified"], "d=0.5, potencia 0.80, alfa 0.05 bilateral -> 64 por grupo (Cohen)"
    p = run_inference("power_t", {"effect_size": 0.5, "n_per_group": 64})
    assert p["power"] == pytest.approx(0.8015, abs=1e-3) and p["verified"]
    d = run_inference("power_t", {"n_per_group": 50, "power": 0.8})
    assert d["result"] == pytest.approx(0.5659, abs=1e-3)


def test_multiple_comparison_corrections_match_textbook():
    holm = run_inference("adjust_pvalues", {"pvalues": [0.001, 0.02, 0.04, 0.3], "method": "holm"})
    assert holm["p_adjusted"] == pytest.approx([0.004, 0.06, 0.08, 0.3], abs=1e-12) and holm["verified"]
    bh = run_inference("adjust_pvalues", {"pvalues": [0.001, 0.02, 0.04, 0.3], "method": "fdr_bh"})
    assert bh["p_adjusted"] == pytest.approx([0.004, 0.04, 0.04 * 4 / 3, 0.3], abs=1e-12) and bh["verified"]
    bonf = run_inference("adjust_pvalues", {"pvalues": [0.01, 0.4], "method": "bonferroni"})
    assert bonf["p_adjusted"] == pytest.approx([0.02, 0.8])


def test_simple_regression_matches_the_classic_worked_example():
    # (1,2),(2,4),(3,5),(4,4),(5,5): b1=0.6, b0=2.2, SSE=2.4, s2=0.8, Sxx=10 -> se=sqrt(0.08), t=2.1213, R2=0.6
    r = run_inference("regression", {"y": [2, 4, 5, 4, 5], "X": {"x": [1, 2, 3, 4, 5]}})
    coef = {c["term"]: c for c in r["coefficients"]}
    assert coef["x"]["coef"] == pytest.approx(0.6, abs=1e-12) and coef["intercepto"]["coef"] == pytest.approx(2.2, abs=1e-12)
    assert coef["x"]["se"] == pytest.approx(math.sqrt(0.08), rel=1e-10) and coef["x"]["t"] == pytest.approx(0.6 / math.sqrt(0.08), rel=1e-10)
    assert coef["x"]["p"] == pytest.approx(2 * stats.t.sf(2.12132, 3), rel=1e-4) and r["model"]["r2"] == pytest.approx(0.6, abs=1e-12)
    assert r["verified"] and r["model"]["F"] == pytest.approx(0.6 / math.sqrt(0.08) * 0.6 / math.sqrt(0.08), rel=1e-9)


def test_anova_is_chosen_for_clean_data_and_welch_for_unequal_variances():
    # Cuantiles exactos de la normal: forma normal garantizada, asi la eleccion de prueba no depende del azar de una semilla.
    z = stats.norm.ppf((np.arange(30) + 0.5) / 30)
    rng = np.random.default_rng(3)
    a = run_inference("k_groups", {"groups": {"A": 5 + z, "B": 6.5 + z + 0.0 * rng.normal(), "C": 5 + z[::-1] * 1.0}})
    assert a["test"] == "ANOVA de un factor" and a["verified"] and a["p_value"] < 0.001 and a["post_hoc"]["method"] == "Tukey HSD"
    sig = {(p["a"], p["b"]) for p in a["post_hoc"]["pairs"] if p["reject"]}
    assert ("A", "B") in sig and ("B", "C") in sig and ("A", "C") not in sig, "Tukey debe aislar a B como el distinto"
    w = run_inference("k_groups", {"groups": {"A": 5 + 0.3 * z, "B": 8 + 6 * z, "C": 5.2 + 2.5 * z}})
    assert w["test"] == "ANOVA de Welch" and w["verified"]


# ----------------------------------------------------------------------------- calibracion por simulacion (la prueba de fondo)


def _rate(fn, n=300, seed=11):
    rng = np.random.default_rng(seed)
    return np.mean([fn(rng) for _ in range(n)])


@pytest.mark.parametrize("scenario", ["normal_iguales", "normal_varianzas_distintas", "asimetrica_lognormal"])
def test_type_I_error_is_controlled_under_the_null(fast, scenario):
    """Con H0 verdadera, la tasa de rechazo al 5 % debe estar cerca de 5 % (el motor elige la prueba segun supuestos)."""
    def draw(rng):
        if scenario == "normal_iguales":
            return rng.normal(0, 1, 25), rng.normal(0, 1, 25)
        if scenario == "normal_varianzas_distintas":
            return rng.normal(0, 1, 40), rng.normal(0, 4, 15)
        return rng.lognormal(0, 1, 30), rng.lognormal(0, 1, 30)

    rate = _rate(lambda rng: inf.compare_two_groups(*draw(rng))["p_value"] < 0.05)
    assert 0.02 <= rate <= 0.09, f"tasa de falsos positivos {rate:.3f} fuera de lo razonable en {scenario}"


def test_power_matches_the_power_calculator(fast):
    """Con d=0.5 y n=64 por grupo, la tasa REAL de rechazo debe acercarse a la potencia teorica (~0.80)."""
    rate = _rate(lambda rng: inf.compare_two_groups(rng.normal(0.5, 1, 64), rng.normal(0, 1, 64))["p_value"] < 0.05, n=400, seed=5)
    assert 0.72 <= rate <= 0.88, f"potencia empirica {rate:.3f} vs 0.80 teorica"


def test_confidence_interval_coverage_is_close_to_95_percent(fast):
    true_diff = 1.0

    def covers(rng):
        r = inf.compare_two_groups(rng.normal(true_diff, 2, 30), rng.normal(0, 2, 30))
        return r["ci"]["low"] <= true_diff <= r["ci"]["high"]

    assert 0.91 <= _rate(covers, n=300, seed=21) <= 0.99


def test_every_result_in_simulation_passes_dual_verification(fast):
    rng = np.random.default_rng(99)
    ok = [inf.compare_two_groups(rng.lognormal(0, 1, 20), rng.lognormal(0.3, 1, 20))["verified"] for _ in range(40)]
    ok += [inf.compare_two_groups(rng.normal(0, 1, 20), rng.normal(0.4, 1, 20))["verified"] for _ in range(40)]
    assert all(ok), "la via principal y la independiente deben coincidir siempre"


def test_deterministic_for_the_same_input():
    d = {"x": [3, 5, 2, 8, 9, 4, 6], "y": [7, 9, 12, 11, 8, 15, 10]}
    a, b = run_inference("two_groups", d), run_inference("two_groups", d)
    assert a["p_value"] == b["p_value"] and a["ci"] == b["ci"]


# ----------------------------------------------------------------------------- entradas invalidas

@pytest.mark.parametrize(
    "analysis,data",
    [
        ("two_groups", {"x": [1], "y": [1, 2, 3]}),
        ("two_groups", {"x": [1, 2, 3], "y": [1, 2], "paired": True}),
        ("two_groups", {"x": [1, 2, 3], "y": [4, 5, 6], "alternative": "raro"}),
        ("two_groups", {"x": [1, 2, 3], "y": [1, 2, 3], "paired": True}),
        ("k_groups", {"groups": {"A": [1, 2, 3], "B": [2, 3, 4]}}),
        ("chi_square", {"table": [[1, -2], [3, 4]]}),
        ("chi_square", {"table": [[1, 2, 3]]}),
        ("two_proportions", {"successes": [11, 3], "n": [10, 10]}),
        ("power_t", {"effect_size": 0.5}),
        ("power_t", {}),
        ("adjust_pvalues", {"pvalues": [0.5]}),
        ("adjust_pvalues", {"pvalues": [0.1, 1.5]}),
        ("regression", {"y": [1, 2, 3], "X": {"a": [1, 2, 3]}}),
        ("inventado", {}),
    ],
)
def test_invalid_inputs_raise_clear_errors_instead_of_returning_garbage(analysis, data):
    with pytest.raises(ValueError):
        run_inference(analysis, data)


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        run_inference("two_groups", {"x": [1, 2, 3], "y": [2, 3, 4]}, alpha=1.5)


def test_nan_values_are_dropped_not_propagated():
    r = run_inference("two_groups", {"x": [1, 2, 3, 4, 5, float("nan")], "y": [3, 4, 5, 6, 7]})
    assert r["n"] == [5, 5] and np.isfinite(r["p_value"])


# ----------------------------------------------------------------------------- Brunner-Munzel en lugar de Mann-Whitney con dispersion distinta


@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_brunner_munzel_replaces_mann_whitney_when_spreads_differ_and_verifies_in_all_directions(alt):
    x = np.random.default_rng(5).lognormal(0, 0.4, 25)
    y = np.random.default_rng(6).lognormal(0.5, 1.6, 12)
    r = inf.compare_two_groups(x, y, alternative=alt)
    assert r["test"] == "Brunner-Munzel" and r["verified"]
    assert "Mann-Whitney no es válido" in r["rationale"][0]
    assert any("liberal" in w for w in r["warnings"]), "con n<20 debe advertir que Brunner-Munzel es algo liberal"


def test_mann_whitney_is_still_used_when_spreads_are_similar():
    rng = np.random.default_rng(2)
    r = inf.compare_two_groups(rng.lognormal(0, 1, 40), rng.lognormal(0.3, 1, 40))
    assert r["test"] == "U de Mann-Whitney" and r["verified"]


def test_mann_whitney_inflation_is_actually_reduced_vs_always_using_it(fast):
    """Escenario donde MW infla (N(0,1) n=40 vs N(0,4) n=15): el motor debe quedar claramente por debajo de usar MW siempre."""
    rng = np.random.default_rng(11)
    engine = mwu = 0
    n = 400
    for _ in range(n):
        x, y = rng.normal(0, 1, 40), rng.normal(0, 4, 15)
        engine += inf.compare_two_groups(x, y)["p_value"] < 0.05
        mwu += stats.mannwhitneyu(x, y).pvalue < 0.05
    assert engine / n < mwu / n - 0.04 and engine / n < 0.085, f"motor {engine / n:.3f} vs Mann-Whitney siempre {mwu / n:.3f}"
