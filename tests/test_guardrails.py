"""Guardarrailes en codigo: normalizacion de argumentos, fidelidad de datos, cifras sin respaldo y el lazo de herramientas del CLI."""

from __future__ import annotations

import importlib.util
import json
import shutil
import uuid

import pytest

from ada.autonomy.heart import Heart
from ada.config import DATA_WORK_DIR, PROJECT_ROOT
from ada.guardrails import check_data_fidelity, normalize_inference_args, numeric_grounding
from ada.training import continuous_learner as cl

# ----------------------------------------------------------------------------- normalizacion


@pytest.mark.parametrize("raw,expected", [("bilateral", "two-sided"), ("dos colas", "two-sided"), ("two_sided", "two-sided"), ("mayor", "greater"), (">", "greater"),
                                          ("izquierda", "less"), ("Menor que", "less"), ("greater", "greater")])
def test_alternative_synonyms_are_normalized(raw, expected):
    out, notes = normalize_inference_args({"analysis": "two_groups", "data": {"x": [1], "y": [2], "alternative": raw}})
    assert out["data"]["alternative"] == expected
    assert bool(notes) == (raw != expected), "solo se anota una nota si realmente se corrigio algo"


def test_method_paired_alpha_and_analysis_aliases():
    out, notes = normalize_inference_args({"analysis": "t_test", "alpha": "5%", "data": {"x": [1], "y": [2], "method": "t de Welch", "paired": "sí"}})
    assert out["analysis"] == "two_groups" and out["alpha"] == 0.05 and out["data"]["method"] == "welch" and out["data"]["paired"] is True
    assert len(notes) == 4
    assert normalize_inference_args({"analysis": "anova", "data": {}})[0]["analysis"] == "k_groups"
    assert normalize_inference_args({"alpha": "0,01", "data": {}})[0]["alpha"] == 0.01


def test_unrecognized_values_are_left_for_the_tool_to_reject_and_input_is_not_mutated():
    args = {"analysis": "inventado", "data": {"alternative": "quiza"}}
    out, notes = normalize_inference_args(args)
    assert out["analysis"] == "inventado" and out["data"]["alternative"] == "quiza" and notes == []
    assert args == {"analysis": "inventado", "data": {"alternative": "quiza"}}


# ----------------------------------------------------------------------------- fidelidad de datos

Q = "Medí A y B.\nGrupo A (n=5): 12, 14, 15, 11, 13\nGrupo B (n=5): 18, 20, 19, 22, 21\n¿Difieren? Usa α = 0.05."


def test_faithful_data_has_no_issues_and_ignores_options_like_alpha():
    assert check_data_fidelity(Q, {"data": {"x": [12, 14, 15, 11, 13], "y": [18, 20, 19, 22, 21], "alpha": 0.05, "p0": 0.5}}) == []


def test_omitted_value_is_detected():
    issues = check_data_fidelity(Q, {"data": {"x": [12, 14, 15, 11], "y": [18, 20, 19, 22, 21]}})
    assert any("no llegó completa" in i and "12" in i for i in issues)


def test_altered_value_and_wrong_order_are_detected():
    assert any("NO aparecen" in i and "31" in i for i in check_data_fidelity(Q, {"data": {"x": [12, 14, 15, 11, 13], "y": [18, 20, 19, 22, 31]}}))
    assert check_data_fidelity(Q, {"data": {"x": [11, 12, 13, 14, 15], "y": [18, 20, 19, 22, 21]}}), "reordenar los datos tambien es infiel (apareamiento)"


def test_groups_tables_and_decimal_comma():
    q = "Grupo A: 1,5 2,5 3,5 4,5 5,5\nGrupo B: 2 3 4 5 6\nGrupo C: 7 8 9 10 11"
    ok = {"data": {"groups": {"A": [1.5, 2.5, 3.5, 4.5, 5.5], "B": [2, 3, 4, 5, 6], "C": [7, 8, 9, 10, 11]}}}
    assert check_data_fidelity(q, ok) == [], "'1,5' sin espacio es decimal, no dos numeros"
    bad = {"data": {"groups": {"A": [1.5, 2.5, 3.5, 4.5, 5.5], "B": [2, 3, 4, 5, 6]}}}
    assert any("7" in i for i in check_data_fidelity(q, bad)), "un grupo omitido debe detectarse"


Q_GROUPS = ("Comparé piezas defectuosas por lote entre 3 grupos.\nMarca 1 (n=4): 10, 4, 5, 8\nMarca 2 (n=4): 17, 14, 10, 7\nMarca 3 (n=4): 6, 13, 5, 22\n"
            "¿Hay diferencias significativas entre los grupos? Usa α = 0.05.")


def test_group_names_are_restored_when_values_match_exactly():
    """Medido en el eval de 96: los valores llegaban intactos pero con nombres A/B/C, y el post-hoc perdia 'Marca 1/2/3'."""
    sent = {"analysis": "k_groups", "data": {"groups": {"A": [10, 4, 5, 8], "B": [17, 14, 10, 7], "C": [6, 13, 5, 22]}}}
    fixed, notes = normalize_inference_args(sent, Q_GROUPS)
    assert list(fixed["data"]["groups"]) == ["Marca 1", "Marca 2", "Marca 3"] and fixed["data"]["groups"]["Marca 2"] == [17, 14, 10, 7]
    assert notes and "'A' -> 'Marca 1'" in notes[0]
    assert list(sent["data"]["groups"]) == ["A", "B", "C"], "no muta la entrada"
    wrong = {"analysis": "k_groups", "data": {"groups": {"A": [10, 4, 5, 8], "B": [17, 14, 10, 8], "C": [6, 13, 5, 22]}}}   # un valor alterado: no se renombra
    assert list(normalize_inference_args(wrong, Q_GROUPS)[0]["data"]["groups"]) == ["A", "B", "C"]
    right = {"analysis": "k_groups", "data": {"groups": {"Marca 1": [10, 4, 5, 8], "Marca 2": [17, 14, 10, 7], "Marca 3": [6, 13, 5, 22]}}}
    assert normalize_inference_args(right, Q_GROUPS)[1] == []
    assert normalize_inference_args(sent)[1] == [], "sin contexto no hay nada que restaurar"
    q_comma = "Turno 1 (n=3): 6,3; 7,0; 7,7\nTurno 2 (n=3): 4,4; 7,3; 6,2\n¿Difieren?"   # coma decimal
    dec = {"analysis": "k_groups", "data": {"groups": {"A": [6.3, 7.0, 7.7], "B": [4.4, 7.3, 6.2]}}}
    assert list(normalize_inference_args(dec, q_comma)[0]["data"]["groups"]) == ["Turno 1", "Turno 2"]


def test_prose_with_five_numbers_is_not_treated_as_a_data_list():
    """Falso positivo medido en el eval de 96: '35 de 100 ... 46 de 100 ... α = 0.01' tiene 5 numeros pero no es una lista de datos."""
    q = "En el grupo A, 35 de 100 respondieron al tratamiento; en el grupo B, 46 de 100. ¿Difieren las proporciones? Usa α = 0.01."
    assert check_data_fidelity(q, {"analysis": "two_proportions", "data": {"successes": [35, 46], "n": [100, 100]}}) == []
    assert check_data_fidelity(q, {"analysis": "two_proportions", "data": {"successes": [35, 47], "n": [100, 100]}}), "un valor alterado se sigue detectando"
    lst = "Datos: 4, 8, 15, 16, 23, 42"
    assert check_data_fidelity(lst, {"analysis": "ci_mean", "data": {"x": [4, 8, 15, 16, 23]}}), "una lista de datos recortada se sigue detectando"
    assert check_data_fidelity(lst, {"analysis": "ci_mean", "data": {"x": [4, 8, 15, 16, 23, 42]}}) == []


def test_data_pasted_in_an_earlier_turn_counts_as_context():
    ctx = "Mis datos:\nA: 3 5 7 9 11\nB: 4 6 8 10 12"
    follow_up = "Ahora hazlo con t de Student."
    assert check_data_fidelity(ctx + "\n" + follow_up, {"data": {"x": [3, 5, 7, 9, 11], "y": [4, 6, 8, 10, 12]}}) == []
    assert check_data_fidelity(follow_up, {"data": {"x": [3, 5, 7, 9, 11], "y": [4, 6, 8, 10, 12]}}), "sin el contexto previo, los datos aparecerian como inventados"


def test_short_questions_without_long_lists_are_not_flagged_for_omissions():
    assert check_data_fidelity("Tengo 3 y 4 exitos de 10 y 12.", {"data": {"successes": [3, 4], "n": [10, 12]}}) == []


# ----------------------------------------------------------------------------- cifras sin respaldo

TOOL = json.dumps({"p_value": 0.020718, "statistic": -2.3456, "df": 8.0, "effect_size": {"value": -1.4837}, "ci": {"low": -5.1234, "high": -0.4321}})


def test_numbers_that_come_from_the_tool_are_grounded_even_when_rounded():
    ans = "t = -2.346 con 8 gl, p = 0.0207; g de Hedges = -1.484; IC [-5.123, -0.432]."
    assert numeric_grounding(ans, [TOOL], Q) == []


def test_invented_p_value_and_effect_are_flagged():
    assert numeric_grounding("t = -2.346, p = 0.0431 y d = 0.987.", [TOOL], Q) == ["0.0431", "0.987"]


def test_derived_numbers_are_accepted_but_unrelated_ones_are_not():
    assert numeric_grounding("El 95 % de confianza y p = 0.0207 (x100: 2.07 %).", [json.dumps({"p_value": 0.020718})], "") == []
    assert numeric_grounding("La media fue 14.7203.", [json.dumps({"p_value": 0.02})], "") == ["14.7203"]


def test_small_round_numbers_and_the_seal_are_not_flagged_and_no_tool_means_no_claim():
    assert numeric_grounding("Con α = 0.05 y n = 12, gl = 8 [VERIFICADO: scipy 1.18.1 + statsmodels 0.15.0]", [TOOL], "") == []
    assert numeric_grounding("p = 0.9999", [], "") == [], "sin herramientas no hay nada contra lo que verificar"


def test_numbers_from_the_question_are_grounded():
    assert numeric_grounding("Con tus 22 observaciones y media 18.350 ...", [TOOL], "media 18.350 de 22 observaciones") == []


# ----------------------------------------------------------------------------- lazo de herramientas del CLI (modelo simulado)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("ada_cli_guardrails", PROJECT_ROOT / "scripts" / "ada_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(args, id_="c1"):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": id_, "type": "function", "function": {"name": "ada_inferential_stats", "arguments": json.dumps(args)}}]}


def _text(t):
    return {"role": "assistant", "content": t}


GOOD = {"analysis": "two_groups", "data": {"x": [12, 14, 15, 11, 13], "y": [18, 20, 19, 22, 21], "alternative": "two-sided"}}


class ScriptedModel:
    """Sustituye al modelo: devuelve respuestas guardadas y recuerda con que `tools` se le pregunto."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, model=None, tools=None):
        self.calls.append({"tools": tools, "n_messages": len(messages)})
        return {"choices": [{"message": self.replies.pop(0)}], "usage": {"completion_tokens": 10}}


def _hist():
    return [{"role": "system", "content": "sistema"}, {"role": "user", "content": Q}]


def test_chain_of_tool_rounds_ends_with_text_instead_of_silence(cli, monkeypatch):
    """El fallo medido: tras corregir un argumento Ada volvia a llamar a la herramienta y el CLI no imprimia NADA."""
    bad = {**GOOD, "data": {**GOOD["data"], "alternative": "bilateral"}}
    model = ScriptedModel([_call(GOOD, "c2"), _text("Concluyo con los datos correctos: p = 0.0007.")])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _hist()
    history.append(_call(bad))
    _, final, _, outputs = cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q)
    assert final["content"].startswith("Concluyo") and len(outputs) == 2
    assert outputs[0].startswith("[argumentos corregidos automáticamente"), "'bilateral' se normalizo y la herramienta SI se ejecuto"
    assert "alternative 'bilateral' -> 'two-sided'" in outputs[0]
    assert [m["role"] for m in history].count("tool") == 2


def test_empty_final_answer_triggers_a_text_only_retry(cli, monkeypatch):
    model = ScriptedModel([_text(""), _text("Respuesta redactada tras el recordatorio.")])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _hist()
    history.append(_call(GOOD))
    _, final, _, _ = cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q)
    assert final["content"].startswith("Respuesta redactada")
    assert model.calls[-1]["tools"] == [], "el reintento va SIN herramientas para forzar texto"


def test_endless_tool_calling_is_capped_and_then_forced_to_write(cli, monkeypatch):
    model = ScriptedModel([_call(GOOD, f"c{i}") for i in range(2, 5)] + [_text("Por fin redacto.")])  # tras 3 rondas el modelo insiste; el recordatorio sin herramientas lo obliga a escribir
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _hist()
    history.append(_call(GOOD, "c1"))
    _, final, _, outputs = cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q)
    assert final["content"] == "Por fin redacto." and len(outputs) == cli.MAX_TOOL_ROUNDS
    assert model.calls[-1]["tools"] == []


def test_altered_data_produces_a_fidelity_warning_in_the_tool_message(cli, monkeypatch):
    altered = {"analysis": "two_groups", "data": {"x": [12, 14, 15, 11, 13], "y": [18, 20, 19, 22, 31]}}
    model = ScriptedModel([_text("ok")])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _hist()
    history.append(_call(altered))
    _, _, _, outputs = cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q)
    assert outputs[0].startswith("ADVERTENCIA DE FIDELIDAD") and "NO presentes estos resultados" in outputs[0]
    clean = cli._run_tool_checked("ada_inferential_stats", GOOD, Q)[0]
    assert not clean.startswith("ADVERTENCIA") and not clean.startswith("["), "con datos fieles no se anade ningun ruido (cero tokens extra)"


def test_empty_first_reply_is_retried_instead_of_printing_nothing(cli, monkeypatch):
    """Medido con Ada Q real: ~1 de cada 8 veces Qwen3 agota su razonamiento oculto y devuelve ni texto ni herramienta."""
    model = ScriptedModel([_text(""), _call(GOOD)])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    res = cli._ask_nonempty(_hist(), "ada-q", "")
    assert res["choices"][0]["message"].get("tool_calls") and len(model.calls) == 2


def test_empty_retries_are_capped_and_a_normal_first_reply_costs_one_call(cli, monkeypatch):
    model = ScriptedModel([_text("")] * (cli.MAX_EMPTY_RETRIES + 1))
    monkeypatch.setattr(cli, "send_chat_completion", model)
    cli._ask_nonempty(_hist(), "ada-q", "")
    assert len(model.calls) == cli.MAX_EMPTY_RETRIES + 1
    ok = ScriptedModel([_text("Hola.")])
    monkeypatch.setattr(cli, "send_chat_completion", ok)
    cli._ask_nonempty(_hist(), "ada-q", "")
    assert len(ok.calls) == 1


PRE_POST_ARGS = {"analysis": "two_groups", "data": {"x": [11, 7, 8, 8, 7, 6, 6, 8, 7, 8, 11, 12], "y": [11, 9, 12, 9, 9, 6, 7, 7, 7, 11, 11, 13], "paired": True}}


def _pre_post_history():
    return [{"role": "system", "content": "sistema"}, {"role": "user", "content": Q_PRE_POST}]


def test_wrong_direction_triggers_one_guided_rewrite_and_drops_the_wrong_answer(cli, monkeypatch):
    """El caso medido (Wilcoxon, #66): las medias SUBIERON y Ada escribio 'reduccion'. Ahora se le pide UNA reescritura con la frase de direccion."""
    wrong = _text("Hubo una reducción significativa de las piezas defectuosas tras la intervención (p = 0.0247).")
    right = _text("Las piezas defectuosas aumentaron de 8.25 a 9.33 tras la intervención (p = 0.0247).")
    model = ScriptedModel([wrong, right])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _pre_post_history()
    history.append(_call(PRE_POST_ARGS))
    _, final, _, _ = cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q_PRE_POST)
    assert final["content"] == right["content"] and history[-1]["content"] == right["content"]
    assert not any(m.get("content") == wrong["content"] for m in history), "la respuesta equivocada no se queda en el historial"
    assert model.calls[-1]["tools"] == [] and len(model.calls) == 2


def test_correct_direction_costs_no_extra_call_and_a_second_wrong_answer_is_not_looped(cli, monkeypatch):
    ok = ScriptedModel([_text("Las piezas defectuosas aumentaron de 8.25 a 9.33 tras la intervención.")])
    monkeypatch.setattr(cli, "send_chat_completion", ok)
    history = _pre_post_history()
    history.append(_call(PRE_POST_ARGS))
    cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q_PRE_POST)
    assert len(ok.calls) == 1
    bad = ScriptedModel([_text("Hubo una reducción significativa."), _text("Sigue habiendo una reducción significativa.")])
    monkeypatch.setattr(cli, "send_chat_completion", bad)
    history = _pre_post_history()
    history.append(_call(PRE_POST_ARGS))
    cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q_PRE_POST)
    assert len(bad.calls) == 2, "como máximo UNA reescritura: sin bucles"


def test_clean_flow_costs_exactly_one_followup_call(cli, monkeypatch):
    model = ScriptedModel([_text("Listo.")])
    monkeypatch.setattr(cli, "send_chat_completion", model)
    history = _hist()
    history.append(_call(GOOD))
    cli._resolve_tool_rounds(history, "ada-q", "", history[-1], Q)
    assert len(model.calls) == 1, "sin problemas no hay llamadas extra al modelo"


# ----------------------------------------------------------------------------- sentido del cambio (subio / redujo)

from ada.guardrails import direction_check  # noqa: E402
from ada.stats.inference import compact_result, run_inference  # noqa: E402

Q_PRE_POST = ("Medí piezas defectuosas por lote en 12 sujetos antes y después de una intervención (mismos sujetos, mismo orden).\n"
              "Antes: 11, 7, 8, 8, 7, 6, 6, 8, 7, 8, 11, 12\nDespués: 11, 9, 12, 9, 9, 6, 7, 7, 7, 11, 11, 13\n¿Cambió significativamente tras la intervención?")


def _pre_post_output():
    x = [11, 7, 8, 8, 7, 6, 6, 8, 7, 8, 11, 12]   # media 99/12 = 8.25
    y = [11, 9, 12, 9, 9, 6, 7, 7, 7, 11, 11, 13]  # media 112/12 = 9.3333
    return json.dumps(compact_result(run_inference("two_groups", {"x": x, "y": y, "paired": True}, alpha=0.05)), ensure_ascii=False)


def test_tool_result_states_which_group_is_larger_in_words():
    res = compact_result(run_inference("two_groups", {"x": [1, 2, 3], "y": [4, 5, 6]}, alpha=0.05))   # medias 2 y 5, x - y = -3
    assert res["direction"] == "La media de y (5) es MAYOR que la de x (2); x - y = -3."
    same = compact_result(run_inference("two_groups", {"x": [1, 2, 3], "y": [3, 2, 1]}, alpha=0.05))
    assert same["direction"].startswith("La media de x y la de y son iguales")


def test_direction_contradicting_the_means_is_flagged():
    """El fallo medido: las medias SUBIERON (8.25 -> 9.33) y Ada escribio 'La intervención redujo...'."""
    out = _pre_post_output()
    assert '"direction"' in out and "MAYOR" in out
    issues = direction_check("La intervención redujo significativamente el número de piezas defectuosas. Conclusión: fue efectiva para disminuir los defectos.", [out], Q_PRE_POST)
    assert issues and "SUBIERON" in issues[0]


def test_correct_negated_and_hedged_direction_is_not_flagged():
    out = _pre_post_output()
    assert direction_check("Las piezas defectuosas aumentaron de 8.25 a 9.33 tras la intervención.", [out], Q_PRE_POST) == []
    assert direction_check("La intervención no redujo los defectos; aumentaron.", [out], Q_PRE_POST) == []
    assert direction_check("No hay evidencia de que los defectos disminuyeran.", [out], Q_PRE_POST) == []
    assert direction_check("La intervención no mostró impacto estadísticamente significativo en la reducción de piezas defectuosas.", [out], Q_PRE_POST) == [], "falso positivo medido en el eval de 96"
    assert direction_check("Hubo una reducción significativa en el número de piezas defectuosas tras la intervención.", [out], Q_PRE_POST), "una afirmación directa sigue marcándose"
    assert direction_check("El tratamiento redujo los defectos.", [out], "Compara el grupo A y el grupo B con estos datos.") == [], "sin estructura antes/después no se afirma nada"
    assert direction_check("redujo", [], Q_PRE_POST) == [], "sin resultado de herramienta no hay nada que contrastar"


# ----------------------------------------------------------------------------- lo que tiene cifras sin respaldo no entra al entrenamiento


@pytest.fixture()
def sandbox(monkeypatch, cli):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    (root / "memory").mkdir(parents=True)
    (root / "training").mkdir()
    monkeypatch.setattr(cl, "MEMORY_DIR", root / "memory")
    monkeypatch.setattr(cl, "TRAINING_DATA_DIR", root / "training")
    monkeypatch.setattr(cl, "EPISODIC_MEMORY_FILE", root / "memory" / "e.jsonl")
    monkeypatch.setattr(cl, "CONTINUAL_DATASET_FILE", root / "training" / "d.jsonl")
    monkeypatch.setattr(cli, "HEART", Heart(directory=root / "heart"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_answers_with_unbacked_numbers_are_not_saved_as_verified_training_data(cli, sandbox):
    cli._record_turn("pregunta", "p = 0.0431", used_tool=True, ungrounded=True)
    assert not cl.CONTINUAL_DATASET_FILE.exists(), "cifras sin respaldo -> solo memoria episodica, jamas dataset de entrenamiento"
    cli._record_turn("otra", "p = 0.0207", used_tool=True, ungrounded=False)
    assert cl.CONTINUAL_DATASET_FILE.exists()
