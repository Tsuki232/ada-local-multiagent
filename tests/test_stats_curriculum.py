"""Pruebas del curriculo de inferencia: que cada ejemplo sea correcto por construccion y no ensene nada inventado."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import Counter

import pytest

from ada.autonomy import curator as curator_mod
from ada.config import DATA_WORK_DIR
from ada.stats import curriculum as cur
from ada.stats import inference as inf
from ada.stats.inference import ANALYSES, run_inference


@pytest.fixture(scope="module")
def examples():
    old = (inf.N_PERM, inf.N_BOOT)
    inf.N_PERM, inf.N_BOOT = 300, 80  # verificacion cruzada mas ruidosa pero suficiente para probar la estructura
    try:
        yield cur.generate_curriculum(120, seed=12345)
    finally:
        inf.N_PERM, inf.N_BOOT = old


def _call(ex):
    body = ex["messages"][2]["content"]
    m = re.fullmatch(r"<tool_call>\n(.*)\n</tool_call>", body, re.S)
    assert m, "la llamada debe tener el formato exacto de Qwen"
    return json.loads(m.group(1))


def _numbers(o):
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return []
    if isinstance(o, (int, float)):
        return [float(o)]
    if isinstance(o, dict):
        return [n for v in o.values() for n in _numbers(v)]
    return [n for v in o for n in _numbers(v)]


def test_every_example_has_the_uniform_conversation_schema(examples):
    assert len(examples) == 120
    for ex in examples:
        assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
        assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) and m["content"] for m in ex["messages"]), "esquema uniforme: solo role y content (texto)"
        assert "ada_inferential_stats" in ex["messages"][0]["content"], "el system debe contener la regla de usar la herramienta"


def test_tool_call_is_valid_and_data_is_exactly_what_the_user_sees(examples):
    for ex in examples:
        call = _call(ex)
        assert call["name"] == "ada_inferential_stats" and call["arguments"]["analysis"] in ANALYSES
        question_numbers = {float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ex["messages"][1]["content"])}
        question_numbers |= {round(n / 100, 6) for n in question_numbers}  # "50 %" en la pregunta -> p0 = 0.5 en la herramienta (conversion legitima)
        args = call["arguments"]["data"]
        missing = [n for n in _numbers(args) if n not in question_numbers]
        assert not missing, f"datos enviados a la herramienta que NO aparecen en la pregunta: {missing[:5]}"


def test_tool_message_matches_a_fresh_recalculation_and_final_answer_cites_it(examples):
    inf_n = (inf.N_PERM, inf.N_BOOT)
    inf.N_PERM, inf.N_BOOT = 300, 80
    try:
        for ex in examples[:40]:
            call = _call(ex)["arguments"]
            tool = json.loads(ex["messages"][3]["content"])
            fresh = run_inference(call["analysis"], call["data"], alpha=call["alpha"])
            assert tool["test"] == fresh["test"] and tool["verified"] is True
            if "p_value" in fresh:
                assert tool["p_value"] == pytest.approx(fresh["p_value"], rel=1e-5), "el resultado guardado debe coincidir con recalcular"
            final = ex["messages"][4]["content"]
            assert fresh["test"] in final and "[VERIFICADO:" in final
    finally:
        inf.N_PERM, inf.N_BOOT = inf_n


def test_final_answers_are_clean_spanish_with_textbook_math_and_correct_conclusion_direction(examples):
    for ex in examples:
        final, tool = ex["messages"][4]["content"], json.loads(ex["messages"][3]["content"])
        for ascii_artifact in ("pequenas", "parametrica", "dificiles", "$ ", "• $"):
            assert ascii_artifact not in final.replace("$$", ""), f"artefacto {ascii_artifact!r} en la respuesta"
        if tool["analysis"] in {"two_groups", "k_groups", "chi_square", "two_proportions", "one_proportion", "correlation"}:
            sig = tool["p_value"] < tool["alpha"]
            concl = final.split("**Conclusión.**")[1].split("\n\n")[0]
            if sig:
                assert "No rechazar H0" not in concl and "Significativo no significa relevante" in concl, f"significativo mal concluido: {concl}"
            else:
                assert "No rechazar H0 no demuestra" in concl and "Significativo no significa" not in concl, f"no significativo mal concluido: {concl}"
        if tool["analysis"] == "two_groups" and tool["test"] == "t de Welch":
            assert "\\frac" in final, "la t de Welch se presenta con su fórmula de libro"


def test_curriculum_is_diverse_in_analyses_and_tests(examples):
    analyses = Counter(e["meta"]["analysis"] for e in examples)
    assert len(analyses) >= 8 and analyses["two_groups"] >= 25
    tests = Counter(e["meta"]["test"] for e in examples)
    assert len(tests) >= 8, f"poca variedad de pruebas: {dict(tests)}"
    assert any("Mann-Whitney" in t or "Brunner" in t or "Wilcoxon" in t or "Kruskal" in t for t in tests), "debe incluir casos no parametricos"


def test_questions_are_unique_and_generation_is_deterministic():
    old = (inf.N_PERM, inf.N_BOOT)
    inf.N_PERM, inf.N_BOOT = 200, 60
    try:
        a, b = cur.generate_curriculum(25, seed=777), cur.generate_curriculum(25, seed=777)
        assert [e["messages"][1]["content"] for e in a] == [e["messages"][1]["content"] for e in b]
        assert len({e["messages"][1]["content"] for e in a}) == 25
        assert a[3]["messages"][3]["content"] == b[3]["messages"][3]["content"]
    finally:
        inf.N_PERM, inf.N_BOOT = old


def test_curator_merges_the_curriculum_dedupes_and_warns_when_it_dominates(examples):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        cfile = root / "curriculum.jsonl"
        cfile.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples[:60]) + json.dumps(examples[0]) + "\n", encoding="utf-8")
        src = root / "dataset.jsonl"
        src.write_text(json.dumps({"conversations": [{"from": "human", "value": "hola"}, {"from": "gpt", "value": "buenas"}], "metadata": {}}) + "\n", encoding="utf-8")
        for name, val in (("CURATED_TRAIN_FILE", "t.jsonl"), ("CURATED_EVAL_FILE", "e.jsonl"), ("CURATION_REPORT_FILE", "r.json")):
            setattr(curator_mod, name, root / val)
        rep = curator_mod.curate_dataset(source=src, curriculum_files=[cfile])
        assert rep["curriculum_inference"] == 60 and rep["real_interactions"] == 1 and rep["counts"]["duplicates"] == 1
        assert any("olvidar cómo conversar" in w for w in rep["warnings"]), "debe avisar si el curriculo domina el dataset"
        assert rep["ready_for_training"] is False, "con solo 1 conversacion real NO debe declararse listo"
        rows = [json.loads(line) for n in ("t.jsonl", "e.jsonl") for line in (root / n).read_text(encoding="utf-8").splitlines()]
        assert any(r["messages"][3]["role"] == "tool" for r in rows if len(r["messages"]) == 5)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        import importlib
        importlib.reload(curator_mod)


# ----------------------------------------------------------------------------- fundamento: ningun numero ni sello sin respaldo


def _all_numbers(o):
    if isinstance(o, bool) or o is None:
        return []
    if isinstance(o, (int, float)):
        return [float(o)]
    if isinstance(o, str):
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", o)]
    if isinstance(o, dict):
        return [n for v in o.values() for n in _all_numbers(v)]
    return [n for v in o for n in _all_numbers(v)]


def _grounded(token: str, pool: set[float]) -> bool:
    t = float(token)
    dec = len(token.split(".")[1]) if "." in token else 0
    sig = len(token.replace("-", "").replace(".", "").lstrip("0")) or 1
    for v in pool:
        for cand in (v, v * v, 2 * v, 100 * v, 1 - v, abs(v), v / 100):
            if abs(cand - t) <= 0.5 * 10 ** (-dec) + 1e-12 or float(f"{cand:.{sig}g}") == t:
                return True
    return False


def test_verified_seal_in_the_answer_is_exactly_the_one_in_the_tool_result(examples):
    for ex in examples:
        tool = json.loads(ex["messages"][3]["content"])
        assert "verification_methods" in tool, "la herramienta debe devolver los metodos: sin ellos Ada tendria que inventar el sello"
        final = ex["messages"][4]["content"]
        seal = re.findall(r"\[VERIFICADO: (.*?)\]$", final.strip(), re.S)
        assert seal and seal[0] == " + ".join(tool["verification_methods"]), "el sello debe copiar los metodos del resultado"


def test_every_number_in_the_final_answer_is_traceable_to_the_tool_result_or_the_question(examples):
    """Si algun numero no se puede rastrear, el curriculo estaria ensenando a Ada a escribir cifras sin respaldo."""
    ungrounded = []
    for ex in examples:
        tool, question = json.loads(ex["messages"][3]["content"]), ex["messages"][1]["content"]
        pool = set(_all_numbers(tool)) | set(_all_numbers(question)) | set(_all_numbers(_call(ex)["arguments"]))
        final = re.sub(r"\[VERIFICADO: .*?\]\s*$", "", ex["messages"][4]["content"].strip(), flags=re.S)
        final = re.sub(r"[\^_]\{?\d+\}?", "", final)  # exponentes e indices de LaTeX (R^2, s_x^2, \chi^2)
        # Cotas APA ("p < 0.001", "p > 0.999"): no son cifras inventadas, pero deben ser CIERTAS para algun p del resultado.
        p_values = [tool["p_value"]] if tool.get("p_value") is not None else []
        p_values += [c["p"] for c in tool.get("coefficients", [])] + [c["p_robust_HC3"] for c in tool.get("coefficients", [])]
        p_values += [q["p_adj"] for q in (tool.get("post_hoc") or {}).get("pairs", [])] + list(tool.get("p_adjusted", []))
        for bm in re.finditer(r"p\s*([<>])\s*(0\.001|0\.999)", final):
            ok = any((pv < 0.001) if bm.group(1) == "<" and bm.group(2) == "0.001" else (pv > 0.999) if bm.group(1) == ">" and bm.group(2) == "0.999" else False for pv in p_values)
            assert ok, f"cota falsa en la respuesta: {bm.group(0)} con p={p_values}"
        final = re.sub(r"p\s*[<>]\s*0\.(?:001|999)", "", final)
        for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])", final):
            tok = m.group(0)
            if "e" in tok:
                continue
            if float(tok) in (0.0, 1.0, 2.0) or not _grounded(tok, pool):
                if float(tok) in (0.0, 1.0, 2.0):
                    continue
                ungrounded.append((ex["meta"]["analysis"], tok, final[max(0, m.start() - 40): m.end() + 20].replace("\n", " ")))
    assert not ungrounded, f"{len(ungrounded)} numeros sin respaldo, p. ej.: {ungrounded[:4]}"
