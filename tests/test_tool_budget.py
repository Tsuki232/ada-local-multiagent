"""Presupuesto de contexto: Ada debe conservar espacio para pensar y responder completo. Estos tests FALLAN si el costo fijo crece de mas.

Principios: (1) ninguna herramienta NUEVA pasa de ~320 tokens; (2) el arranque fijo del CLI queda bajo ~10 % de la ventana;
(3) el harness (MCP) no puede crecer por encima de lo medido; (4) cada pregunta solo recibe las herramientas que puede necesitar;
(5) la nota del corazon no se manda en cada turno.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import uuid

import pytest

from ada.autonomy.heart import Heart
from ada.config import DATA_WORK_DIR, PROJECT_ROOT, ModelRoleConfig
from ada.tools import budget
from ada.tools.routing import FALLBACK, MAX_TOOLS, PACKS, intents, select_tools

pytestmark = pytest.mark.skipif(not budget.available(), reason="tokenizador de Qwen3-14B no disponible (data/work/base_models)")

CTX = ModelRoleConfig().context_length
NEW_TOOL_MAX = 320            # tokens por herramienta nueva
CLI_FULL_MAX = 2300           # sistema + TODAS las herramientas del CLI
CLI_ROUTED_MAX = 1650         # sistema + herramientas ELEGIDAS para una pregunta tipica
HARNESS_TOTAL_MAX = 8400      # esquemas de TODAS las herramientas MCP + instrucciones (medido: 8 351 = 8 136 + 215 de ada_scrape_tables, aprobada por Diego)
MINE = {"ada_scrape_tables", "ada_inferential_stats", "ada_probability_bayes", "ada_simulate", "ada_observatory", "ada_self_report", "ada_live_vision_session", "ada_autonomous_agent"}


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("ada_cli_budget", PROJECT_ROOT / "scripts" / "ada_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- CLI


def test_cli_fixed_overhead_with_every_tool_stays_under_budget(cli):
    full = budget.rendered_overhead(cli.SYSTEM_PROMPT, cli.TOOLS_SCHEMA)
    assert full <= CLI_FULL_MAX, f"el arranque fijo del CLI ya es {full} tokens (tope {CLI_FULL_MAX}, ventana {CTX})"
    assert full <= 0.10 * CTX, f"el arranque fijo usa {full / CTX:.1%} de la ventana (limite 10 %)"


def test_every_cli_tool_schema_is_small(cli):
    heavy = {t["function"]["name"]: budget.schema_tokens(t) for t in cli.TOOLS_SCHEMA}
    over = {k: v for k, v in heavy.items() if v > NEW_TOOL_MAX}
    assert not over, f"esquemas de herramientas demasiado grandes: {over}"


@pytest.mark.parametrize(
    "prompt,must,must_not",
    [
        ("Compara estos dos grupos: A: 12, 14, 15, 11 y B: 18, 20, 19, 22. ¿Hay diferencia significativa? α = 0.05", {"ada_inferential_stats"}, {"ada_build_canvas_animation"}),
        ("¿Cuál es la probabilidad condicional de defecto dado complejidad alta? Usa Bayes.", {"ada_inferential_stats", "ada_probability_bayes"}, {"ada_build_canvas_animation"}),
        ("Haz una animación de un pajarito volando sobre un edificio", {"ada_build_canvas_animation"}, {"ada_inferential_stats"}),
        ("muéstrame el cerebro de la red neuronal", {"ada_neural_brain_simulation"}, {"ada_inferential_stats"}),
    ],
)
def test_routing_sends_only_relevant_tools(cli, prompt, must, must_not):
    names = {t["function"]["name"] for t in select_tools(prompt, cli.TOOLS_SCHEMA)}
    assert must <= names and not (must_not & names), names
    assert len(names) <= MAX_TOOLS


def test_greeting_gets_only_the_minimum_and_never_zero_stats_safety(cli):
    names = [t["function"]["name"] for t in select_tools("Hola Ada, ¿cómo va todo?", cli.TOOLS_SCHEMA)]
    assert names == FALLBACK, "sin intencion clara se envia solo la herramienta de inferencia (Ada nunca debe quedar sin poder verificar estadistica)"


def test_a_pasted_list_of_numbers_counts_as_statistics_even_without_keywords(cli):
    assert "stats" in intents("12.5 13.1 11.8 14.2 12.9 13.7 15.1 ¿y esto?")


def test_routed_prompts_stay_light_in_tokens(cli):
    for prompt in ("Hola Ada", "¿hay diferencia significativa entre A: 1,2,3,4,5 y B: 6,7,8,9,10?", "simula un sistema solar con animación"):
        tools = select_tools(prompt, cli.TOOLS_SCHEMA)
        used = budget.rendered_overhead(cli.SYSTEM_PROMPT, tools)
        assert used <= CLI_ROUTED_MAX, f"{prompt!r} -> {used} tokens fijos con {[t['function']['name'] for t in tools]}"


def test_every_statistics_question_in_training_and_evaluation_routes_to_the_inference_tool(cli):
    """Si el enrutador fallara aqui, Ada se quedaria sin la herramienta justo cuando debe verificar: 100 % obligatorio."""
    from ada.stats.curriculum import generate_scenarios

    f = DATA_WORK_DIR / "training" / "stats_curriculum.jsonl"
    questions = [json.loads(line)["messages"][1]["content"] for line in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []
    questions += [s.question for s in generate_scenarios(150, 777_001)]
    assert len(questions) >= 150
    missing = [q[:90] for q in questions if "ada_inferential_stats" not in {t["function"]["name"] for t in select_tools(q, cli.TOOLS_SCHEMA)}]
    assert not missing, f"{len(missing)} preguntas sin herramienta de inferencia, p. ej. {missing[:2]}"


def test_probability_questions_route_to_the_probability_tool(cli):
    qs = ["P(A) si A es complejidad > 10", "Usa el teorema de Bayes con prevalencia 1 %, sensibilidad 99 %", "¿Son independientes estos dos eventos?",
          "probabilidad de la unión de A y B", "calcula la probabilidad condicional"]
    for q in qs:
        assert "ada_probability_bayes" in {t["function"]["name"] for t in select_tools(q, cli.TOOLS_SCHEMA)}, q


def test_every_routed_tool_name_really_exists_in_the_cli(cli):
    names = {t["function"]["name"] for t in cli.TOOLS_SCHEMA}
    for pack, tools in PACKS.items():
        assert set(tools) <= names, f"el paquete {pack!r} nombra herramientas que el CLI no tiene: {set(tools) - names}"


# ----------------------------------------------------------------------------- extras por turno


def test_heart_note_is_not_sent_on_every_turn_but_returns_when_it_matters(cli, monkeypatch):
    root = DATA_WORK_DIR / "_pytest_tmp" / uuid.uuid4().hex
    try:
        monkeypatch.setattr(cli, "HEART", Heart(directory=root / "heart"))
        monkeypatch.setattr(cli, "_LAST_HEART_NOTE", {"note": "", "mood": None, "since": 99})
        first = cli._memory_note("deriva x al cubo")
        assert "NÚCLEO AFECTIVO" in first, "la primera vez se informa su estado"
        notes = [cli._memory_note("deriva x al cubo") for _ in range(8)]
        assert all("NÚCLEO AFECTIVO" not in n for n in notes), "con el mismo estado de animo no se repite en cada turno"
        assert cli._LAST_HEART_NOTE["note"] == "", "y el dataset tampoco la guarda en un turno donde no se inyecto"
        assert "NÚCLEO AFECTIVO" in cli._memory_note("deriva x al cubo"), "se refresca tras 8 turnos sin ella"
        for _ in range(6):
            cli.HEART.apply("rechazada")
        assert "NÚCLEO AFECTIVO" in cli._memory_note("deriva x al cubo"), "si cambia el animo, se informa"
        assert "NÚCLEO AFECTIVO" in cli._memory_note("¿cómo te sientes?"), "si preguntan por ella, siempre"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_per_turn_extras_are_bounded():
    from ada.autonomy.self_model import self_knowledge_text

    assert budget.count(Heart(directory=DATA_WORK_DIR / "_pytest_tmp" / "no_existe").note("Diego")) <= 200
    assert budget.count(self_knowledge_text()) <= 950, "el autoconocimiento no debe crecer: solo se inyecta si preguntan por Ada"


def test_worst_case_turn_leaves_the_context_for_a_full_answer(cli):
    """Peor caso razonable: arranque con las 4 herramientas mas pesadas elegidas + autoconocimiento + 3 recuerdos de 250 tokens."""
    from ada.autonomy.self_model import self_knowledge_text

    heavy4 = sorted(cli.TOOLS_SCHEMA, key=budget.schema_tokens, reverse=True)[:MAX_TOOLS]
    worst = budget.rendered_overhead(cli.SYSTEM_PROMPT, heavy4) + budget.count(self_knowledge_text()) + 3 * 250
    assert worst <= 0.18 * CTX, f"peor caso {worst} tokens = {worst / CTX:.1%} de la ventana; deja {CTX - worst} para historial y respuesta"


# ----------------------------------------------------------------------------- harness (MCP)


def test_harness_total_does_not_grow_beyond_the_measured_budget():
    from ada.tools import mcp_server

    tools = asyncio.run(mcp_server.mcp.list_tools())
    sizes = {t.name: budget.count(json.dumps({"name": t.name, "description": t.description, "inputSchema": t.input_schema}, ensure_ascii=False)) for t in tools}
    instr = getattr(mcp_server.mcp, "instructions", None) or getattr(mcp_server, "ADA_SYSTEM_INSTRUCTIONS", "") or ""
    total = sum(sizes.values()) + budget.count(instr)
    assert total <= HARNESS_TOTAL_MAX, f"el harness carga {total} tokens fijos (tope {HARNESS_TOTAL_MAX}); mas pesadas: {sorted(sizes.items(), key=lambda kv: -kv[1])[:4]}"
    over = {k: v for k, v in sizes.items() if k in MINE and v > NEW_TOOL_MAX}
    assert not over, f"herramientas nuevas con descripcion demasiado larga: {over}"
