"""Pruebas del evaluador de inferencia: su detector de conclusiones fallaba en 6 de 11 casos reales; estas pruebas lo blindan."""

from __future__ import annotations

import importlib.util
import json

import pytest

from ada.config import DATA_WORK_DIR, PROJECT_ROOT


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("ev_under_test", PROJECT_ROOT / "scripts" / "eval_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Frases REALES que el primer detector clasifico mal en la corrida baseline de Ada Q.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("**Conclusión:** No se rechaza la hipótesis nula. No hay evidencia suficiente para afirmar que los errores difieran.", False),
        ("**Conclusión:** Se rechaza la hipótesis nula (p = 0.00118 < α = 0.05). Existe una asociación estadísticamente significativa entre el turno y el estado.", True),
        ("**Conclusión:** No hay evidencia estadísticamente significativa de diferencias en la calidad entre los turnos.", False),
        ("**Conclusión:** Se rechaza $ H_0 $ (no hay diferencia) a favor de $ H_1 $ (disminución significativa tras la intervención).", True),
        ("**Conclusión**: Se rechaza la hipótesis nula. Existe una asociación estadísticamente significativa al nivel α = 0.05.", True),
        ("Conclusión: No hay evidencia suficiente para afirmar que las dosis afecten significativamente la resistencia.", False),
        ("Con α = 0.05, hay evidencia suficiente de que el rendimiento difiere entre A y B.", True),
        ("Con α = 0.05, no hay evidencia suficiente de que el rendimiento difiera entre A y B.", False),
        ("La correlación es significativamente distinta de cero.", True),
        ("**Conclusión**: no se detecta diferencia significativa (p = 0.8639 ≥ α = 0.05).", False),
        ("**Conclusión:** no se pueden atribuir diferencias significativas al tratamiento dietético.", False),
        ("No hay diferencias significativas entre los grupos (Kruskal-Wallis H = 2.605, p = 0.272 ≥ α = 0.05).", False),
        ("No hay una correlación estadísticamente significativa entre X e Y (r = 0.1446, p = 0.6538).", False),
        ("", None),
        ("Aquí tienes los resultados de la prueba.", None),
    ],
)
def test_conclusion_detector_handles_real_phrasings(ev, text, expected):
    assert ev._says_significant(text) is expected


@pytest.mark.parametrize(
    "text,p,expected",
    [
        ("p = 0.0207", 0.02072, True),
        ("el valor p fue 0.0207 (< 0.05)", 0.02072, True),
        ("p < 0.001", 6.7e-9, True),
        ("p = 0.5", 0.5517, False),
        ("p = 0.5517", 0.5517, True),
        ("sin ningún p-valor", 0.03, False),
        ("p = 0.0207", None, True),
    ],
)
def test_p_value_citation_detector(ev, text, p, expected):
    assert ev._cites_p(text, p) is expected


def test_data_faithfulness_requires_exact_transcription(ev):
    truth = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
    assert ev._data_faithful({"x": [1, 2, 3], "y": [4, 5, 6], "paired": True}, truth, "two_groups")
    assert not ev._data_faithful({"x": [1, 2, 3], "y": [4, 5, 7]}, truth, "two_groups"), "un solo dato alterado debe contar como infiel"
    assert not ev._data_faithful({"x": [1, 2, 3]}, truth, "two_groups")
    assert ev._data_faithful({"groups": {"A": [1, 2], "B": [3, 4]}}, {"groups": {"A": [1.0, 2.0], "B": [3.0, 4.0]}}, "k_groups")
    assert not ev._data_faithful({"groups": {"A": [1, 2]}}, {"groups": {"A": [1.0, 2.0], "B": [3.0, 4.0]}}, "k_groups"), "grupo omitido = infiel"


def test_the_curriculum_teacher_answers_score_100_percent_on_the_detector(ev):
    """Coherencia fundamental: lo que se le ENSENA debe puntuar perfecto en lo que se MIDE. Si no, entrenar no tendria sentido."""
    f = DATA_WORK_DIR / "training" / "stats_curriculum.jsonl"
    if not f.is_file():
        pytest.skip("curriculo no generado (scripts/generate_stats_curriculum.py)")
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()][:400]
    checked = bad_conclusion = bad_p = 0
    for r in rows:
        tool = json.loads(r["messages"][3]["content"])
        final = r["messages"][4]["content"]
        if tool.get("p_value") is not None:
            checked += 1
            bad_p += not ev._cites_p(final, tool["p_value"])
            if tool["analysis"] not in {"power_t", "adjust_pvalues"}:
                says = ev._says_significant(final)
                bad_conclusion += (says is None) or (says != (tool["p_value"] < tool["alpha"]))
    assert checked > 300
    assert bad_p == 0, f"{bad_p} respuestas del curriculo no citan su propio p-valor segun el detector"
    assert bad_conclusion == 0, f"{bad_conclusion} respuestas del curriculo concluyen distinto a lo que el detector entiende"
