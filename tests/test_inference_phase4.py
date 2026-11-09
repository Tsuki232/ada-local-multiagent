"""Fase 4: prueba forzable (method=), t de una muestra, IC, razon de varianzas, z con sigma conocida y tamanos de muestra. Valores a mano."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from ada.stats.inference import compact_result, run_inference


# ----------------------------------------------------------------------------- method= forzable


def test_welch_can_be_forced_and_gives_sensitivity_analysis_when_data_are_skewed():
    skew_x = [1, 2, 2, 3, 3, 4, 4, 5, 60, 90]
    y = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    r = run_inference("two_groups", {"x": skew_x, "y": y, "method": "welch"})
    assert r["test"] == "t de Welch" and r["method_requested"] == "welch" and r["verified"], "la guia exige Welch: se calcula Welch aunque 'auto' elegiria otra"
    assert any("sensitivity" in w for w in r["warnings"]) and "sensitivity" in r
    assert r["sensitivity"]["test"] and isinstance(r["sensitivity"]["agrees_with_requested"], bool), "la alternativa robusta se agrega, NO sustituye en silencio"
    auto = run_inference("two_groups", {"x": skew_x, "y": y})
    assert auto["test"] != "t de Welch", "sin forzar, los datos asimetricos van a una prueba robusta"


def test_forced_welch_matches_hand_calculation_with_fractional_df():
    x, y = [4, 6, 5, 7, 8], [10, 14, 9, 12, 15, 11, 13]  # n1=5, n2=7
    r = run_inference("two_groups", {"x": x, "y": y, "method": "welch"})
    x, y = np.array(x, float), np.array(y, float)
    v1, v2, n1, n2 = x.var(ddof=1), y.var(ddof=1), 5, 7
    se2 = v1 / n1 + v2 / n2
    t = (x.mean() - y.mean()) / math.sqrt(se2)
    nu = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))   # Welch-Satterthwaite
    assert r["statistic"] == pytest.approx(t, rel=1e-10) and r["df"] == pytest.approx(nu, rel=1e-10)
    assert nu != int(nu), "con varianzas distintas los gl son fraccionarios (no n1+n2-2)"
    assert r["ci"]["low"] < x.mean() - y.mean() < r["ci"]["high"]
    assert r["p_value"] == pytest.approx(2 * stats.t.sf(abs(t), nu), rel=1e-9)


def test_forced_student_uses_pooled_variance_and_warns_when_variances_differ():
    r = run_inference("two_groups", {"x": [1, 2, 3, 4, 5], "y": [3, 4, 5, 6, 7], "method": "student"})
    assert r["df"] == 8 and r["statistic"] == pytest.approx(-2.0) and r["verified"]
    z = stats.norm.ppf((np.arange(25) + 0.5) / 25)
    bad = run_inference("two_groups", {"x": (0.2 * z).tolist(), "y": (5 * z + 1).tolist(), "method": "student"})
    assert any("Welch es la opción segura" in w for w in bad["warnings"])


def test_forced_mann_whitney_and_invalid_methods():
    r = run_inference("two_groups", {"x": [1, 2, 3, 4, 5, 6], "y": [4, 5, 6, 7, 8, 9], "method": "mann_whitney"})
    assert r["test"] == "U de Mann-Whitney" and r["verified"]
    for bad in ({"x": [1, 2, 3], "y": [2, 3, 4], "method": "tukey"}, {"x": [1, 2, 3], "y": [2, 3, 5], "method": "welch", "paired": True}):
        with pytest.raises(ValueError):
            run_inference("two_groups", bad)


def test_auto_is_unchanged_by_the_new_parameter():
    d = {"x": [1, 2, 3, 4, 5], "y": [3, 4, 5, 6, 7]}
    assert run_inference("two_groups", d)["test"] == "t de Welch" and "method_requested" not in run_inference("two_groups", d)


# ----------------------------------------------------------------------------- t de una muestra e IC


def test_one_sample_t_matches_hand_calculation():
    # x = 2,4,4,4,5,5,7,9: media 5, SC = 32, s = sqrt(32/7) = 2.138090, se = 0.755929 ; mu0 = 4 -> t = 1.322876, gl = 7
    r = run_inference("one_sample", {"x": [2, 4, 4, 4, 5, 5, 7, 9], "mu0": 4})
    assert r["statistic"] == pytest.approx(1 / (math.sqrt(32 / 7) / math.sqrt(8)), rel=1e-12) and r["statistic"] == pytest.approx(1.322876, abs=1e-6)
    assert r["df"] == 7 and r["verified"] and r["p_value"] == pytest.approx(2 * stats.t.sf(1.322876, 7), rel=1e-6)
    half = stats.t.ppf(0.975, 7) * math.sqrt(32 / 7) / math.sqrt(8)
    assert r["ci"]["low"] == pytest.approx(5 - half, rel=1e-9) and r["ci"]["high"] == pytest.approx(5 + half, rel=1e-9)
    assert r["effect_size"]["value"] == pytest.approx(1 / math.sqrt(32 / 7))


def test_one_sample_adds_wilcoxon_sensitivity_for_skewed_small_samples():
    r = run_inference("one_sample", {"x": [1, 1, 2, 2, 3, 3, 4, 40], "mu0": 2})
    assert "sensitivity" in r and r["sensitivity"]["test"] == "Wilcoxon de una muestra" and any("Normalidad dudosa" in w for w in r["warnings"])


def test_ci_mean_matches_textbook():
    # 10,12,14,16,18: media 14, s = sqrt(10), n=5 -> se = sqrt(10)/sqrt(5) = sqrt(2) ; t(0.975; 4 gl) = 2.776445 (tablas) -> margen = 2.776445*1.414214 = 3.926486
    r = run_inference("ci_mean", {"x": [10, 12, 14, 16, 18]}, alpha=0.05)
    assert r["ci"]["low"] == pytest.approx(14 - 2.776445105 * math.sqrt(2), abs=1e-6) and r["ci"]["high"] == pytest.approx(14 + 2.776445105 * math.sqrt(2), abs=1e-6) and r["verified"]
    assert r["ci"]["low"] == pytest.approx(10.073514, abs=1e-5) and r["margin_of_error"] == pytest.approx(3.926486, abs=1e-5) and "no dice que el 95" in r["decision"], "advierte el error de interpretacion clasico"
    wide = run_inference("ci_mean", {"x": [10, 12, 14, 16, 18]}, alpha=0.01)
    assert wide["margin_of_error"] > r["margin_of_error"], "mas confianza -> intervalo mas ancho"


def test_ci_proportion_wilson_and_clopper_pearson():
    r = run_inference("ci_proportion", {"successes": 8, "n": 20})
    z, ph, n = 1.959964, 0.4, 20
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert r["intervals"]["wilson"] == pytest.approx([centre - half, centre + half], abs=1e-6) and r["verified"]
    assert r["intervals"]["clopper_pearson"] == pytest.approx([0.1912, 0.6395], abs=1e-3), "valor de tablas para 8/20"
    assert r["intervals"]["clopper_pearson"][0] < r["intervals"]["wilson"][0] and r["intervals"]["clopper_pearson"][1] > r["intervals"]["wilson"][1], "CP es mas ancho (conservador)"
    edge = run_inference("ci_proportion", {"successes": 0, "n": 15})
    assert edge["verified"] and edge["intervals"]["clopper_pearson"][0] == 0.0 and edge["warnings"]


def test_variance_ratio_F_by_hand():
    x = [8, 12] * 5   # media 10, SC = 40, var = 40/9
    y = [9, 11] * 5   # media 10, SC = 10, var = 10/9   -> F = 4 exacto
    r = run_inference("variance_ratio", {"x": x, "y": y})
    assert r["statistic"] == pytest.approx(4.0) and r["df"] == [9, 9] and r["verified"]
    assert r["p_value"] == pytest.approx(2 * stats.f.sf(4.0, 9, 9), rel=1e-10)
    assert r["ci"]["low"] < 4.0 < r["ci"]["high"]


def test_z_test_with_known_sigma_classic_example():
    # media muestral 52, n=36, mu0=50, sigma=6 -> z = 2/(6/6) = 2.0 ; p bilateral = 0.0455003
    r = run_inference("z_test", {"mean": 52, "n": 36, "mu0": 50, "sigma": 6})
    assert r["statistic"] == pytest.approx(2.0) and r["p_value"] == pytest.approx(0.0455003, abs=1e-6) and r["verified"]
    assert r["ci"]["low"] == pytest.approx(52 - 1.959964) and r["ci"]["high"] == pytest.approx(52 + 1.959964)
    one = run_inference("z_test", {"mean": 52, "n": 36, "mu0": 50, "sigma": 6, "alternative": "greater"})
    assert one["p_value"] == pytest.approx(0.0227501, abs=1e-6)


def test_sample_sizes_match_the_classic_textbook_values():
    assert run_inference("sample_size", {"kind": "mean", "margin": 3, "sigma": 15})["n_required"] == 97            # (1.96*15/3)^2 = 96.04
    two = run_inference("sample_size", {"kind": "two_means", "margin": 2, "sigma": 10})
    assert two["n_required"] == 193 and two["unit"] == "por grupo"                                              # 2*(1.96*10/2)^2 = 192.08
    prop = run_inference("sample_size", {"kind": "proportion", "margin": 0.05})
    assert prop["n_required"] == 385 and prop["verified"] and prop["n_exact"] == pytest.approx(384.1459, abs=1e-3)  # el clasico 385
    assert run_inference("sample_size", {"kind": "mean", "margin": 3, "sigma": 15}, alpha=0.01)["n_required"] > 97, "mas confianza -> mas muestra"
    assert all(run_inference("sample_size", {"kind": k, "margin": 0.1 if k == "proportion" else 1, "sigma": 5})["verified"] for k in ("mean", "two_means", "proportion"))


@pytest.mark.parametrize(
    "analysis,data",
    [
        ("one_sample", {"x": [1, 2]}),
        ("one_sample", {"x": [1, 2, 3], "alternative": "raro"}),
        ("ci_mean", {"x": [1, 2]}),
        ("ci_proportion", {"successes": 5, "n": 3}),
        ("ci_proportion", {"successes": -1, "n": 10}),
        ("variance_ratio", {"x": [1, 2], "y": [1, 2, 3]}),
        ("z_test", {"mean": 1, "n": 10, "mu0": 0, "sigma": 0}),
        ("z_test", {"mean": 1, "n": 0, "mu0": 0, "sigma": 1}),
        ("sample_size", {"kind": "mean", "margin": 1}),
        ("sample_size", {"kind": "inventado", "margin": 1, "sigma": 1}),
        ("sample_size", {"kind": "mean", "margin": -1, "sigma": 1}),
        ("sample_size", {"kind": "proportion", "margin": 0.05, "p": 1.0}),
    ],
)
def test_new_operations_reject_invalid_input_with_clear_errors(analysis, data):
    with pytest.raises(ValueError):
        run_inference(analysis, data)


def test_new_results_survive_compaction_without_losing_what_the_answer_needs():
    c = compact_result(run_inference("ci_proportion", {"successes": 8, "n": 20}))
    assert "intervals" in c and "verification_methods" in c and "decision" in c
    s = compact_result(run_inference("sample_size", {"kind": "mean", "margin": 3, "sigma": 15}))
    assert s["n_required"] == 97 and "verification_methods" in s
    f = compact_result(run_inference("two_groups", {"x": [1, 2, 3, 4, 50], "y": [3, 4, 5, 6, 7], "method": "welch"}))
    assert "sensitivity" in f and f["method_requested"] == "welch"
