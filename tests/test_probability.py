"""Pruebas del motor de probabilidad y Bayes. Valores esperados calculados A MANO (tablas con conteos exactos), no con el propio motor."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ada.stats.probability import run_probability

A = {"column": "a", "op": "==", "value": 1}
B = {"column": "b", "op": "==", "value": 1}


def frame_from_counts(n11: int, n10: int, n01: int, n00: int) -> pd.DataFrame:
    """Dataset con exactamente n11 filas (a=1,b=1), n10 (a=1,b=0), n01 (a=0,b=1) y n00 (a=0,b=0)."""
    rows = [(1, 1)] * n11 + [(1, 0)] * n10 + [(0, 1)] * n01 + [(0, 0)] * n00
    return pd.DataFrame(rows, columns=["a", "b"])


# n=100: P(A)=40/100, P(B)=30/100, P(A∩B)=10/100  -> P(A∪B)=60/100 (a mano: 40+30-10)
DF = frame_from_counts(10, 30, 20, 40)


def test_event_probability_by_hand():
    r = run_probability("event", df=DF, A=A)
    assert r["probability"] == 0.4 and r["counts"]["A"] == 40 and r["verified"] and r["discrepancy"] == 0
    assert r["ci"]["low"] < 0.4 < r["ci"]["high"]
    assert "pandas" in r["verification_methods"][0] and "DuckDB" in r["verification_methods"][1]


def test_complement_by_hand():
    r = run_probability("complement", df=DF, A=A)
    assert r["probability"] == pytest.approx(0.6) and r["counts"]["notA"] == 60 and r["verified"]


def test_union_uses_inclusion_exclusion_and_shows_why_adding_is_wrong():
    r = run_probability("union", df=DF, A=A, B=B)
    assert r["probability"] == pytest.approx(0.60)            # 0.40 + 0.30 - 0.10
    assert r["naive_sum"] == pytest.approx(0.70) and r["overlap"] == pytest.approx(0.10)
    assert r["verified"] and "dos veces" in r["decision"]


def test_intersection_and_conditional_by_hand():
    assert run_probability("intersection", df=DF, A=A, B=B)["probability"] == pytest.approx(0.10)
    c = run_probability("conditional", df=DF, A=A, B=B)        # P(A|B) = 10/30
    assert c["probability"] == pytest.approx(1 / 3) and c["n_given"] == 30 and c["verified"]
    assert c["p_A"] == pytest.approx(0.4), "P(A|B) = 0.333 difiere de P(A) = 0.4"


def test_independent_events_are_not_rejected_but_warns_it_is_not_proof():
    ind = frame_from_counts(12, 28, 18, 42)  # n=100, P(A)=.4, P(B)=.3, P(A∩B)=.12 = .4*.3 EXACTO
    r = run_probability("independence", df=ind, A=A, B=B)
    assert r["p_AB"] == pytest.approx(r["p_A_times_p_B"]) and r["p_value"] > 0.99
    assert "No se rechaza la independencia" in r["decision"] and "NO demuestra" in r["decision"]
    assert r["verified"]


def test_dependent_events_are_detected():
    dep = frame_from_counts(40, 10, 10, 40)
    r = run_probability("independence", df=dep, A=A, B=B)
    assert r["p_value"] < 0.001 and "Se rechaza la independencia" in r["decision"] and r["odds_ratio"] == pytest.approx(16.0)


def test_small_expected_counts_switch_to_fisher():
    r = run_probability("independence", df=frame_from_counts(8, 2, 1, 5), A=A, B=B)
    assert "Fisher" in r["test_used"]


# ----------------------------------------------------------------------------- Bayes


def test_bayes_from_priors_classic_case_is_exactly_one_sixth():
    # prevalencia 1 %, sensibilidad 99 %, falsos positivos 5 %  ->  P(D|+) = 0.0099 / (0.0099 + 0.0495) = 1/6
    r = run_probability("bayes_priors", params={"prior": "0.01", "sensitivity": "0.99", "false_positive_rate": "0.05"})
    assert r["posterior_fraction"] == "1/6" and r["posterior"] == pytest.approx(1 / 6) and r["verified"] and r["discrepancy"] == 0
    nf = r["natural_frequencies"]
    assert nf["enfermos_positivos"] == pytest.approx(99) and nf["sanos_positivos"] == pytest.approx(495)  # por cada 10 000
    assert r["warnings"], "con prevalencia baja debe advertir que los falsos positivos dominan"


def test_bayes_from_counts_equals_direct_counting():
    # n=1000: 10 con D (9 con C), 990 sin D (99 con C) -> P(D|C) = 9/108 = 1/12 EXACTO
    rows = [(1, 1)] * 9 + [(1, 0)] * 1 + [(0, 1)] * 99 + [(0, 0)] * 891
    df = pd.DataFrame(rows, columns=["d", "c"])
    r = run_probability("bayes", df=df, A={"column": "d", "op": "==", "value": 1}, B={"column": "c", "op": "==", "value": 1})
    assert r["posterior"] == pytest.approx(1 / 12) and r["direct_conditional"] == pytest.approx(1 / 12)
    assert r["prior"] == pytest.approx(0.01) and r["p_C_given_D"] == pytest.approx(0.9) and r["p_C_given_notD"] == pytest.approx(0.1)
    assert r["verified"] and r["discrepancy"] == 0
    assert any("no es una probabilidad poblacional" in w for w in r["warnings"])


def test_bayes_sequential_matches_hand_calculation_and_flags_independence_assumption():
    # dos pruebas positivas independientes (sens .99, fpr .05), prior .01: odds = (1/99)*(99/5)^2 = 99/25 -> P = 99/124
    ev = [{"sensitivity": "0.99", "false_positive_rate": "0.05"}] * 2
    r = run_probability("bayes_sequential", params={"prior": "0.01", "evidences": ev})
    assert r["posterior_fraction"] == "99/124" and r["verified"]
    assert [round(s["despues"], 4) for s in r["steps"]] == [round(1 / 6, 4), round(99 / 124, 4)]
    assert any("independientes" in w for w in r["warnings"])
    neg = run_probability("bayes_sequential", params={"prior": "0.5", "evidences": [{"sensitivity": "0.9", "false_positive_rate": "0.1", "positive": False}]})
    assert neg["posterior"] == pytest.approx(0.1) and neg["verified"]  # P(D|-) = .5*.1/(.5*.1+.5*.9)


@pytest.mark.parametrize("bad", [{"prior": "1.5", "sensitivity": "0.9", "false_positive_rate": "0.1"}, {"prior": "x", "sensitivity": "0.9", "false_positive_rate": "0.1"},
                                 {"prior": "0.1", "sensitivity": "-0.2", "false_positive_rate": "0.1"}, {"prior": None, "sensitivity": "0.9", "false_positive_rate": "0.1"}])
def test_bayes_priors_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        run_probability("bayes_priors", params=bad)


# ----------------------------------------------------------------------------- eventos: semantica, cuantiles y seguridad


def test_missing_values_are_handled_identically_by_both_methods():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0], "s": ["a", "b", None, "a", "c", None, "b", "a"]})
    cases = [
        {"column": "x", "op": ">", "value": 3}, {"column": "x", "op": "!=", "value": 2}, {"column": "x", "op": "==", "value": 4},
        {"column": "x", "op": "isnull"}, {"column": "x", "op": "notnull"}, {"column": "x", "op": "between", "value": [2, 7]},
        {"column": "s", "op": "in", "value": ["a", "b"]}, {"column": "s", "op": "!=", "value": "a"},
        {"not": {"column": "x", "op": ">", "value": 3}}, {"and": [{"column": "x", "op": ">", "value": 1}, {"column": "s", "op": "==", "value": "a"}]},
        {"or": [{"column": "x", "op": "isnull"}, {"column": "s", "op": "isnull"}]},
    ]
    for ev in cases:
        r = run_probability("event", df=df, A=ev)
        assert r["verified"] and r["discrepancy"] == 0, f"pandas y DuckDB discrepan con {ev}: {r['counts']}"
    # semantica de dos valores: un dato faltante NO cumple '!=' ni '>' y 'not' es el complemento exacto
    assert run_probability("event", df=df, A={"column": "x", "op": "!=", "value": 2})["counts"]["A"] == 5
    # x > 3 se cumple en 4 filas (4, 6, 7, 8); su complemento exacto son las otras 4: 1, 2 y los DOS faltantes (8 - 4)
    assert run_probability("event", df=df, A={"not": {"column": "x", "op": ">", "value": 3}})["counts"]["A"] == 4


def test_quantile_threshold_is_resolved_recorded_and_reproducible():
    df = pd.DataFrame({"cc": np.arange(1, 101)})
    ev = {"column": "cc", "op": ">", "quantile": 0.75}
    r1, r2 = run_probability("event", df=df, A=ev), run_probability("event", df=df, A=ev)
    assert r1["resolved"]["A"]["value"] == pytest.approx(75.25) and r1["counts"]["A"] == 25   # 76..100
    assert r1["resolved"] == r2["resolved"] and r1["verified"]


@pytest.mark.parametrize(
    "bad",
    [
        {"column": "no_existe", "op": ">", "value": 1},
        {"column": "a", "op": "LIKE", "value": "x"},
        {"column": "a", "op": "between", "value": [1]},
        {"column": "a", "op": "in", "value": []},
        {"column": "a", "op": ">", "value": [1, 2]},
        {"column": "a", "op": ">", "quantile": 1.5},
        {"column": "a", "op": "==", "quantile": 0.5},
        {"and": [A]},
        "a > 1",
        {"column": 'a"; DROP TABLE t; --', "op": "==", "value": 1},
        {"column": "a", "op": "==", "value": None},
    ],
)
def test_malformed_or_malicious_events_raise_clear_errors(bad):
    with pytest.raises(ValueError):
        run_probability("event", df=DF, A=bad)


def test_deeply_nested_events_are_rejected():
    ev = A
    for _ in range(8):
        ev = {"not": ev}
    with pytest.raises(ValueError, match="anidado"):
        run_probability("event", df=DF, A=ev)


def test_sql_injection_through_values_is_harmless_because_values_are_bound():
    df = pd.DataFrame({"s": ["x", "y", "x'; DROP TABLE t; --"]})
    r = run_probability("event", df=df, A={"column": "s", "op": "==", "value": "x'; DROP TABLE t; --"})
    assert r["counts"]["A"] == 1 and r["verified"]


def test_conditional_on_impossible_event_is_an_error():
    with pytest.raises(ValueError, match="indefinida"):
        run_probability("conditional", df=DF, A=A, B={"column": "a", "op": "==", "value": 99})


def test_requires_both_events_and_a_dataset_and_safe_path():
    with pytest.raises(ValueError):
        run_probability("union", df=DF, A=A)
    with pytest.raises(ValueError):
        run_probability("event", A=A)
    with pytest.raises(ValueError, match="dentro de data"):
        run_probability("event", path=r"C:\Windows\win.ini", A=A)
    with pytest.raises(ValueError):
        run_probability("inventada", df=DF, A=A)


def test_bayes_sequential_with_fractions_is_exact_not_float():
    r = run_probability("bayes_priors", params={"prior": "1/3", "sensitivity": "3/4", "false_positive_rate": "1/10"})
    assert Fraction(r["posterior_fraction"]) == Fraction(1, 3) * Fraction(3, 4) / (Fraction(1, 3) * Fraction(3, 4) + Fraction(2, 3) * Fraction(1, 10))
