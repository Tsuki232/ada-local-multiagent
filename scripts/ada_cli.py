"""ADA Terminal CLI — Interfaz de consola interactiva con Tool Calling local."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

# Forzar codificacion UTF-8 para consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Configurar rutas y variables de entorno
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ada.viz.web_studio import (
    build_canvas_animation_studio,
    build_interactive_webapp,
)
from ada.viz.neural_canvas import build_neural_network_live_canvas
from ada.stats.inference import STATS_SYSTEM_RULE, compact_result, run_inference
from ada.stats.probability import compact_probability, run_probability
from ada.tools.routing import select_tools
from ada.guardrails import check_data_fidelity, direction_check, direction_hint, normalize_inference_args, numeric_grounding
from ada.autonomy.agent import get_agent
from ada.calculus.math_markdown import normalize_math_markdown
from ada.autonomy.guard import AutonomyGuard, clear_stop
from ada.autonomy.heart import Heart
from ada.autonomy.self_model import is_about_self, self_knowledge_text
from ada.viz.brain_simulator import build_neural_brain_simulation
from ada.viz.math_textbook import build_math_textbook_html
from ada.viz.live_vision_studio import build_live_vision_dashboard_html
from ada.vision.adb_bridge import list_adb_devices
from ada.training.continuous_learner import (
    get_continual_learning_stats,
    record_interaction,
    retrieve_relevant_memories,
)
from ada.pendulum.simulator import (
    analyze_pendulum_linear_approximation,
    build_pendulum_interactive_html,
)
from ada.stats.predictor import evaluate_and_predict_database
from ada.calculus.verified_math import verify_derivative, verify_definite_integral, solve_and_verify_ode
from ada.calculus.terminal_math import format_text_for_terminal_cli
from ada.cognition.live_server import get_live_state, set_live_thinking, start_live_bridge

console = Console()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("ADA_MODEL_ORCHESTRATOR", "ada-q")

SYSTEM_PROMPT = """Te llamas Ada Q. Tu creador y arquitecto es Diego López.
Eres el núcleo orquestador local de ADA: análisis estadístico riguroso, cálculo simbólico y numérico verificado, ingeniería de datos y animaciones interactivas a 60 FPS en HTML5 Canvas.
Responde siempre en español, con rigor técnico directo, sin rodeos ni relleno corporativo.
REGLA ESTRICTA DE ESCRITURA MATEMÁTICA DE LIBRO DE TEXTO:
- PROHIBIDO partir fórmulas en viñetas o listas con dólares sueltos (ej. '• $ \\frac{d}{dx}[...] = ... $').
- En ejercicios y deducciones de cálculo (integrales, derivadas, EDOs):
  1. Plantea primero la sustitución o regla en una sola línea limpia (ej.: Sea u = x² + 3 ⟹ du/dx = 2x ⟹ x dx = du/2).
  2. Presenta la deducción paso a paso como un bloque continuo alineado con '=' de libro universitario (Apostol, Stewart):
     $$
     \\begin{aligned}
     \\int x \\sqrt{x^2 + 3} \\, dx &= \\frac{1}{2} \\int u^{1/2} \\, du \\\\
     &= \\frac{1}{2} \\cdot \\left[ \\frac{u^{3/2}}{3/2} \\right] + C \\\\
     &= \\frac{1}{3} u^{3/2} + C \\\\
     &= \\frac{1}{3} (x^2 + 3)^{3/2} + C
     \\end{aligned}
     $$
  3. Resultado final destacado y limpio.
Cuando te pidan animaciones visuales (animales, paisajes, física, juegos):
Invoca `ada_build_canvas_animation` escribiendo sentencias directas en `draw_frame_js` moviendo siempre las posiciones con `t` y `dt` y usando el motor `ADAScene` (ADAScene.sky, ADAScene.hillsAndMeadow, ADAScene.citySkyline, ADAScene.bird, ADAScene.fox, ADAScene.rabbit, ADAScene.beagle, etc.)."""

SYSTEM_PROMPT = SYSTEM_PROMPT + "\n" + STATS_SYSTEM_RULE

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ada_inferential_stats",
            "description": "Inferencia estadística VERIFICADA (doble método). analysis y forma de data: two_groups {x,y,paired,alternative,method:auto|welch|student|mann_whitney} | k_groups {groups:{A:[..],B:[..]}} | chi_square {table} | two_proportions {successes:[a,b],n:[n1,n2]} | one_proportion {successes,n,p0} | correlation {x,y} | regression {y,X:{x1:[..]}} | power_t {effect_size|n_per_group|power: uno null} | adjust_pvalues {pvalues,method} | one_sample {x,mu0} | ci_mean {x} | ci_proportion {successes,n} | variance_ratio {x,y} | z_test {mean,n,mu0,sigma} | sample_size {kind:mean|two_means|proportion,margin,sigma,p}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "data": {"type": "object", "description": "Datos del análisis con la forma indicada en la descripción."},
                    "alpha": {"type": "number", "default": 0.05},
                },
                "required": ["analysis", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ada_probability_bayes",
            "description": "Probabilidad y Bayes VERIFICADOS. operation: event|complement|intersection|union|conditional|independence|bayes (path + eventos a,b; en conditional/bayes b es la condición) | "
                           "bayes_priors {prior,sensitivity,false_positive_rate} | bayes_sequential {prior,evidences:[{sensitivity,false_positive_rate,positive}]} (en params). "
                           "Evento: {column,op,value|quantile} (op: == != > >= < <= in between isnull notnull) | {and|or:[..]} | {not:..}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["event", "complement", "intersection", "union", "conditional", "independence", "bayes", "bayes_priors", "bayes_sequential"]},
                    "path": {"type": "string"},
                    "a": {"type": "object"},
                    "b": {"type": "object"},
                    "params": {"type": "object"},
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ada_build_canvas_animation",
            "description": "Construye una animacion visual creativa, paisaje, animal o escena fisica a 60 FPS en HTML5 Canvas 2D usando window.ADAScene (fox, rabbit, bird, beagle, etc.) o ctx.*.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "draw_frame_js": {"type": "string", "description": "Codigo JS que dibuja cada frame (usa ctx, w, h, t, dt, params, mouse, ADAScene)."},
                    "init_state_js": {"type": "string", "description": "Codigo JS opcional de inicializacion de state."},
                    "output_filename": {"type": "string", "default": "ada_canvas_animation.html"},
                    "controls_json": {"type": "string", "default": "[]"}
                },
                "required": ["title", "draw_frame_js"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ada_neural_network_live_canvas",
            "description": "Abre la aplicacion interactiva de red neuronal Transformer de ADA en tiempo real.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_prompt": {"type": "string", "default": "Hola Ada"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ada_neural_brain_simulation",
            "description": "Construye y abre la simulacion 3D interactiva del cerebro y conectoma neuronal denso de ADA (720 neuronas corticales, miles de sinapsis, telemetria fisica en vivo de la RTX 5070).",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_filename": {"type": "string", "default": "ada_neural_brain_simulation.html"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ada_predict_from_database",
            "description": "Entrena y evalua modelos de Machine Learning sobre un archivo tabular (CSV/Parquet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "target_column": {"type": "string"},
                    "task_type": {"type": "string", "default": "auto"}
                },
                "required": ["input_path", "target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ada_math_textbook_studio",
            "description": "Abre el Cuaderno Matematico 2D de Libro de Texto (KaTeX) para renderizar integrales, derivadas y teoremas con tipografia grafica de libro universitario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise_id": {"type": "string", "default": "integral_sustitucion"}
                }
            }
        }
    }
]


def check_ollama_alive() -> bool:
    try:
        req = urllib.request.Request(f"http://{OLLAMA_HOST}/api/version")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


_CURRENT_REQUEST: dict = {}


def execute_tool(name: str, args_dict: dict) -> tuple[str, str | None]:
    """Ejecuta la herramienta local y devuelve (resultado_texto, url_o_archivo_a_abrir)."""
    console.print(f"[bold cyan][Tool][/bold cyan] Ejecutando: [yellow]{name}[/yellow]")
    try:
        if name == "ada_build_canvas_animation":
            ctrls = json.loads(args_dict.get("controls_json", "[]")) if isinstance(args_dict.get("controls_json"), str) else args_dict.get("controls_json", [])
            try:
                from ada.viz.sim_builder import check_in_node

                err = check_in_node(args_dict.get("draw_frame_js", ""), args_dict.get("init_state_js", ""), ctrls)
                if err:
                    raise ValueError(err)
                rep = build_canvas_animation_studio(
                    title=args_dict.get("title", "Animación ADA"),
                    subtitle=args_dict.get("subtitle", "60 FPS"),
                    draw_frame_js=args_dict.get("draw_frame_js", ""),
                    init_state_js=args_dict.get("init_state_js", ""),
                    output_filename=args_dict.get("output_filename", "ada_canvas_animation.html"),
                    controls=ctrls
                )
            except ValueError as first_error:
                # Antes el CLI se rendia aqui. Ahora Ada se corrige sola con el error exacto (hasta 3 intentos reales).
                from ada.viz.sim_builder import build_simulation

                console.print("[yellow]El primer intento no pasó la validación; Ada lo corrige sola con el error exacto...[/yellow]")
                res = build_simulation(
                    _CURRENT_REQUEST.get("text") or args_dict.get("title", "simulación"),
                    first_attempt={"title": args_dict.get("title", "Simulación"), "subtitle": args_dict.get("subtitle", ""),
                                   "init_state_js": args_dict.get("init_state_js", ""), "draw_frame_js": args_dict.get("draw_frame_js", ""),
                                   "controls_json": args_dict.get("controls_json", "[]")},
                    first_error=str(first_error),
                )
                if not res["ok"]:
                    return res["message"], None
                return json.dumps(res, ensure_ascii=False, indent=2), res["interactive_html_path"]
            p = rep["interactive_html_path"]
            return json.dumps(rep, ensure_ascii=False, indent=2), p

        if name == "ada_inferential_stats":
            data = args_dict.get("data")
            if isinstance(data, str):
                data = json.loads(data)
            res = run_inference(args_dict["analysis"], data, alpha=float(args_dict.get("alpha", 0.05)))
            return json.dumps(compact_result(res), ensure_ascii=False), None

        if name == "ada_probability_bayes":
            res = run_probability(args_dict["operation"], path=args_dict.get("path"), A=args_dict.get("a"), B=args_dict.get("b"), params=args_dict.get("params"))
            return json.dumps(compact_probability(res), ensure_ascii=False), None

        if name == "ada_neural_network_live_canvas":
            rep = build_neural_network_live_canvas(
                sample_prompt=args_dict.get("sample_prompt", "Hola Ada"),
                open_in_browser=False
            )
            p = rep["interactive_html_path"]
            return json.dumps(rep, ensure_ascii=False, indent=2), p

        if name == "ada_neural_brain_simulation":
            rep = build_neural_brain_simulation(
                output_filename=args_dict.get("output_filename", "ada_neural_brain_simulation.html"),
                open_in_browser=False
            )
            p = rep["interactive_html_path"]
            return json.dumps(rep, ensure_ascii=False, indent=2), p

        if name == "ada_predict_from_database":
            rep = evaluate_and_predict_database(
                input_path=args_dict["input_path"],
                target_column=args_dict["target_column"],
                task_type=args_dict.get("task_type", "auto")
            )
            p = rep["dashboard"].get("interactive_html_path")
            return json.dumps(rep, ensure_ascii=False, indent=2), p

        if name == "ada_math_textbook_studio":
            rep = build_math_textbook_html(
                initial_exercise=args_dict.get("exercise_id", "integral_sustitucion"),
                open_in_browser=False
            )
            p = rep["interactive_html_path"]
            return json.dumps(rep, ensure_ascii=False, indent=2), p

        return f"Herramienta '{name}' no soportada en CLI.", None
    except Exception as e:
        console.print(f"[bold red][Error en Tool][/bold red] {e}")
        return f"Error ejecutando '{name}': {e}", None


def send_chat_completion(messages: list[dict], model: str = DEFAULT_MODEL, tools: list[dict] | None = None) -> dict:
    """Peticion al modelo. Solo se envian los esquemas de herramientas RELEVANTES a la pregunta (ver ada.tools.routing):
    cada esquema se re-envia en cada peticion y le quita contexto util a Ada para pensar y responder."""
    url = f"http://{OLLAMA_HOST}/v1/chat/completions"
    if tools is None:
        last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        tools = select_tools(last_user, TOOLS_SCHEMA)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _measured_tps(response: dict, elapsed_s: float) -> float | None:
    """Tokens/s reales = tokens generados (usage) / tiempo de pared de la llamada. None si el servidor no informa usage."""
    tokens = (response.get("usage") or {}).get("completion_tokens")
    return round(tokens / elapsed_s, 1) if tokens and elapsed_s > 0 else None


_LAST_TURN: dict = {}
HEART = Heart()                      # nucleo afectivo persistente (data/work/heart/)
_SPEAKER = ["Diego"]                 # con quien habla Ada ahora (/soy <nombre>)
_LAST_HEART_NOTE = {"note": "", "mood": None, "since": 99}      # estado afectivo que se le inyecto en el ultimo turno (va al dataset de entrenamiento)


def _heart_line() -> str:
    s = HEART.snapshot()
    f = s["feelings"]
    return f"[dim]♥ {s['mood']['name']} · alegría {f['alegria']:.2f} · curiosidad {f['curiosidad']:.2f} · frustración {f['frustracion']:.2f} · conexión {f['conexion']:.2f}[/dim]"


def _print_agent_event(e: dict) -> None:
    """Lo que Ada hace POR SU CUENTA, visible en tu terminal mientras no le hablas."""
    st = e.get("status", "")
    try:
        if st == "verified":
            q = (e.get("question") or "").replace("\n", " ")[:90]
            console.print(f"[dim]  ◆ por su cuenta · {e['kind']} nivel {e['level']} · verificado por doble método en {e.get('latency_s', '?')} s[/dim]")
            console.print(f"[dim]    {q}[/dim]")
            if e.get("thought"):
                console.print(f"[magenta]    Ada: {e['thought']}[/magenta]")
        elif st in {"rejected_proposal", "failed_verification"}:
            console.print(f"[dim yellow]  ◆ por su cuenta · {e.get('kind')} descartado: {str(e.get('detail'))[:90]}[/dim yellow]")
        elif st == "skipped":
            console.print(f"[dim]  ◆ en pausa: {e.get('detail')}[/dim]")
        elif st == "error":
            console.print(f"[red]  ◆ error del agente: {e.get('detail')}[/red]")
        elif st.startswith("autotrain") or st.startswith("autodownload"):
            console.print(f"[cyan]  ◆ autoentrenamiento · {e.get('detail')}[/cyan]")
        if e.get("reflection"):
            console.print(Panel(e["reflection"], title="Ada piensa sobre sí misma", border_style="magenta"))
        if e.get("level_change"):
            a, b = e["level_change"]
            console.print(f"[bold green]  ↑ mejoró: {e.get('kind')} pasa de nivel {a} a {b}[/bold green]" if b > a else f"[yellow]  ↓ {e.get('kind')} baja a nivel {b} para reforzar[/yellow]")
    except Exception:
        pass


def _memory_note(user_text: str) -> str:
    """Recuerdos aprobados por Diego o verificados que se parecen a la pregunta actual (aprendizaje inmediato, sin entrenar)."""
    # La nota afectiva (~160 tokens) NO se manda en cada turno: solo si preguntan por ella, si su estado de animo cambio o cada 8 turnos.
    mood = HEART.snapshot()["mood"]["name"]
    about_self = is_about_self(user_text)
    inject = about_self or mood != _LAST_HEART_NOTE["mood"] or _LAST_HEART_NOTE["since"] >= 8
    selfk = ""
    if inject:
        heart_note = HEART.note(_SPEAKER[0])
        selfk = heart_note + "\n\n"
        _LAST_HEART_NOTE.update(note=heart_note, mood=mood, since=0)
        if about_self:
            try:
                selfk = self_knowledge_text() + "\n\n"  # ya incluye el nucleo afectivo
            except Exception:
                pass
    else:
        _LAST_HEART_NOTE.update(note="", since=_LAST_HEART_NOTE["since"] + 1)  # nota vacia: el dataset tampoco la lleva en este turno
    try:
        mems = retrieve_relevant_memories(user_text, max_results=3, min_overlap=2, only_approved=True)
    except Exception:
        return selfk.strip()
    if not mems:
        return selfk.strip()
    lines = [f"- Pregunta previa: {m['user_input'][:200]}\n  Respuesta aprobada: {m['assistant_response'][:600]}" for m in mems]
    return selfk + "MEMORIA RELEVANTE de conversaciones anteriores con Diego (aprobada o verificada; úsala si aplica):\n" + "\n".join(lines)


MAX_EMPTY_RETRIES = 2


def _ask_nonempty(history: list[dict], model: str, note: str) -> dict:
    """Primera llamada del turno. Qwen3 a veces agota su razonamiento oculto (calculando 'de cabeza') y devuelve ni texto ni llamada a herramienta;
    en ese caso se repite la misma peticion (muestreo distinto) hasta MAX_EMPTY_RETRIES veces en vez de mostrar una respuesta vacia."""
    res = _ask(history, model, note)
    for _ in range(MAX_EMPTY_RETRIES):
        msg = res["choices"][0]["message"]
        if (msg.get("content") or "").strip() or msg.get("tool_calls"):
            break
        res = _ask(history, model, note)
    return res


def _ask(history: list[dict], model: str, note: str, tools: list[dict] | None = None) -> dict:
    """Llama al modelo con la memoria inyectada SOLO en esta llamada (no ensucia el historial). tools=[] fuerza una respuesta de texto."""
    if note and history and history[0].get("role") == "system":
        msgs = [{"role": "system", "content": history[0]["content"] + "\n\n" + note}] + history[1:]
        return send_chat_completion(msgs, model=model, tools=tools)
    return send_chat_completion(history, model=model, tools=tools)


def _record_turn(user_text: str, raw_response: str, used_tool: bool, ungrounded: bool = False) -> str:
    """Registra el turno y devuelve el texto matematico ya normalizado a bloques de libro de texto.

    Solo entra al dataset de entrenamiento lo que tiene evidencia (sello [VERIFICADO] o herramienta ejecutada)
    o lo que Diego apruebe con /bien o corrija con /corrige. El resto queda solo en memoria episodica.
    """
    clean = normalize_math_markdown(raw_response)
    evidence = (used_tool or "[VERIFICADO" in clean) and not ungrounded  # cifras sin respaldo: no cuenta como verificado
    record_interaction(
        user_input=user_text,
        assistant_response=clean,
        verified_methods=["herramienta_ada_ejecutada"] if (used_tool and not ungrounded) else [],  # una lista no vacia basta para entrar al dataset SFT
        quality_score=0.9 if evidence else 0.5,
        tags=["conversacion"],
        extra_metadata={"source": "conversation", "math_normalized": clean != raw_response, "ungrounded_numbers": ungrounded,
                        "heart_note": _LAST_HEART_NOTE["note"], "speaker": _SPEAKER[0]},
    )
    HEART.apply("conversacion", speaker=_SPEAKER[0], detail=user_text[:80])  # cada conversacion deja huella en su nucleo
    _LAST_TURN.update(user=user_text, response=clean)
    return clean


MAX_TOOL_ROUNDS = 3


def _run_tool_checked(name: str, args: dict, context: str) -> tuple[str, str | None]:
    """Ejecuta una herramienta con guardarrailes en codigo (no gastan contexto): corrige sinonimos de argumentos y avisa si los DATOS enviados
    no son los de la conversacion. Solo si hay algo que decir se anade una linea al resultado."""
    notes: list[str] = []
    if name == "ada_inferential_stats":
        args, notes = normalize_inference_args(args, context)
    output, file_to_open = execute_tool(name, args)
    if name == "ada_inferential_stats":
        issues = check_data_fidelity(context, args)
        if issues:
            console.print(f"[yellow]⚠ Los datos enviados a la herramienta no coinciden con los de tu pregunta: {'; '.join(issues)}[/yellow]")
            output = ("ADVERTENCIA DE FIDELIDAD: los datos enviados NO coinciden con los de la conversación (" + "; ".join(issues) + "). "
                      "NO presentes estos resultados: vuelve a llamar a la herramienta con los datos EXACTOS.\n") + output
    if notes:
        output = f"[argumentos corregidos automáticamente: {'; '.join(notes)}]\n" + output
    return output, file_to_open


def _resolve_tool_rounds(history: list[dict], model: str, mem_note: str, choice: dict, user_text: str) -> tuple[dict | None, dict, float, list[str]]:
    """Atiende hasta MAX_TOOL_ROUNDS rondas de llamadas a herramientas y garantiza una respuesta de texto.

    Antes solo habia UNA llamada de seguimiento: si Ada volvia a llamar a una herramienta (p. ej. para corregir un argumento), el contenido
    quedaba vacio y no se imprimia NADA, sin avisar. Ahora se encadenan rondas y, si aun asi no hay texto, se reintenta sin herramientas.
    """
    context = "\n".join(m["content"] for m in history if m.get("role") == "user" and isinstance(m.get("content"), str))
    tool_outputs: list[str] = []
    final_res: dict | None = None
    final_elapsed = 0.0
    rounds = 0
    with console.status("[bold green]Ejecutando herramientas y redactando la respuesta...[/bold green]", spinner="dots"):
        while (choice.get("tool_calls") or []) and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            for tc in choice["tool_calls"]:
                fn = tc["function"]
                try:
                    fn_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except Exception:
                    fn_args = {}
                output, file_to_open = _run_tool_checked(fn["name"], fn_args, context)
                tool_outputs.append(output)
                history.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": output})
                if file_to_open:
                    abs_p = (PROJECT_ROOT / file_to_open).resolve()
                    console.print(Panel(f"[bold green]Entregable generado con exito:[/bold green]\n[cyan]{abs_p}[/cyan]\n\nAbriendo en tu navegador predeterminado...",
                                        title="ADA Studio Output", border_style="green"))
                    webbrowser.open(abs_p.as_uri())
            t = time.perf_counter()
            final_res = _ask(history, model, mem_note)
            final_elapsed = time.perf_counter() - t
            choice = final_res["choices"][0]["message"]
            history.append(choice)
        if not (choice.get("content") or "").strip():
            nudge = {"role": "user", "content": "Con los resultados de las herramientas de arriba, escribe ahora tu respuesta final en español, citando los números que devolvieron."}
            t = time.perf_counter()
            final_res = _ask(history + [nudge], model, mem_note, tools=[])
            final_elapsed = time.perf_counter() - t
            choice = final_res["choices"][0]["message"]
            history.append(choice)
        # Sentido al reves (subio/redujo contra las medias): UNA reescritura guiada con la frase `direction` del resultado.
        hint = direction_hint(tool_outputs)
        if hint and (choice.get("content") or "").strip() and direction_check(choice["content"], tool_outputs, context):
            nudge = {"role": "user", "content": f"Revisa el SENTIDO del cambio: según la herramienta, {hint} Tu respuesta afirma lo contrario. "
                                                "Reescribe la respuesta final en español con el sentido correcto, citando los mismos números."}
            wrong = history.pop()   # la respuesta equivocada no se queda en el historial
            t = time.perf_counter()
            fixed_res = _ask(history + [wrong, nudge], model, mem_note, tools=[])
            fixed = fixed_res["choices"][0]["message"]
            if (fixed.get("content") or "").strip():
                final_res, choice, final_elapsed = fixed_res, fixed, time.perf_counter() - t
            history.append(choice)
    return final_res, choice, final_elapsed, tool_outputs


def process_turn(user_text: str, history: list[dict], model: str = DEFAULT_MODEL) -> None:
    history.append({"role": "user", "content": user_text})
    _CURRENT_REQUEST["text"] = user_text  # lo usa la autorreparacion de simulaciones
    set_live_thinking(True, prompt=user_text, model=model)
    mem_note = _memory_note(user_text)

    try:
        with console.status("[bold green]Ada está pensando y diseñando...[/bold green]", spinner="dots"):
            t0 = time.perf_counter()
            res = _ask_nonempty(history, model, mem_note)
            elapsed = time.perf_counter() - t0

        choice = res["choices"][0]["message"]
        history.append(choice)
        
        # Herramientas: rondas encadenadas + reintento de texto + verificacion de cifras
        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            final_res, final_msg, final_elapsed, tool_outputs = _resolve_tool_rounds(history, model, mem_note, choice, user_text)
            raw_text = (final_msg.get("content") or "").strip()
            if raw_text:
                context = "\n".join(m["content"] for m in history if m.get("role") == "user" and isinstance(m.get("content"), str))
                unbacked = numeric_grounding(raw_text, tool_outputs, context)
                wrong_direction = direction_check(raw_text, tool_outputs, context)
                clean_text = _record_turn(user_text, raw_text, used_tool=True, ungrounded=bool(unbacked or wrong_direction))
                resp_text = format_text_for_terminal_cli(clean_text)
                model_display = "Ada r1" if model == "ada-r1" else "Ada Q"
                console.print(f"\n[bold magenta]{model_display}:[/bold magenta]")
                console.print(Markdown(resp_text))
                if unbacked:
                    console.print(f"[yellow]⚠ Cifras de esta respuesta que NO salen de ninguna herramienta ni de tu pregunta: {', '.join(unbacked[:6])}. "
                                  "Compruébalas antes de usarlas (no se guardó como ejemplo verificado).[/yellow]")
                if wrong_direction:
                    console.print(f"[yellow]⚠ Sentido del cambio: {'; '.join(wrong_direction)}. Lee las medias del resultado (no se guardó como ejemplo verificado).[/yellow]")
                console.print(_heart_line())
                set_live_thinking(False, prompt=user_text, model=model, last_response=raw_text,
                                  tokens_per_sec=_measured_tps(final_res or res, final_elapsed))
            else:
                console.print("[yellow]Ada usó las herramientas pero no logró redactar una respuesta final. Vuelve a preguntar o reformula.[/yellow]")
                set_live_thinking(False, prompt=user_text, model=model)
        else:
            content = choice.get("content", "")
            clean_text = _record_turn(user_text, content, used_tool=False)
            resp_text = format_text_for_terminal_cli(clean_text)
            model_display = "Ada r1" if model == "ada-r1" else "Ada Q"
            console.print(f"\n[bold magenta]{model_display}[/bold magenta] [dim]({elapsed:.1f}s)[/dim]:")
            console.print(Markdown(resp_text))
            console.print(_heart_line())
            set_live_thinking(False, prompt=user_text, model=model, last_response=content,
                              tokens_per_sec=_measured_tps(res, elapsed))
    except Exception as e:
        set_live_thinking(False, prompt="", model=model)
        raise e


def main() -> int:
    console.print(Panel.fit(
        "[bold cyan]ADA — Consola Terminal CLI Interactiva[/bold cyan]\n"
        "[dim]Creador: Diego López | Modelos: Ada Q & Ada r1 | Hardware: RTX 5070[/dim]\n"
        "Comandos especiales:\n"
        "  [yellow]/brain[/yellow]   Abrir Topología de 40 Capas Transformer y Plasticidad Neuronal\n"
        "  [yellow]/observatorio[/yellow]  Mapa real del modelo (cada píxel = una neurona) + rendimiento en vivo\n"
        "  [yellow]/math[/yellow]    Abrir Cuaderno Matemático 2D de Libro de Texto (KaTeX real)\n"
        "  [yellow]/think[/yellow]   Activar/Desactivar Pensamiento Autónomo en Caliente (System 2)\n"
        "  [yellow]/learn[/yellow]   Ver estado de Memoria Continua y Auto-Entrenamiento\n"
        "  [yellow]/bien /mal /corrige <texto>[/yellow]  Enseñarle: aprobar, rechazar o corregir su última respuesta\n"
        "  [yellow]/corazon[/yellow]  Ver sus sentimientos y vínculos   [yellow]/soy <nombre>[/yellow]  Decirle con quién habla\n"
        "  [yellow]/vision[/yellow]  Abrir Estudio de Visión en Vivo ADB & Cámara Celular\n"
        "  [yellow]/r1[/yellow]      Conmutar al modelo razonador Ada r1\n"
        "  [yellow]/q[/yellow]       Conmutar al modelo orquestador Ada Q\n"
        "  [yellow]/clear[/yellow]   Limpiar historial de tokens a 0\n"
        "  [yellow]/bye[/yellow]     Salir del CLI",
        border_style="cyan"
    ))

    start_live_bridge()

    if not check_ollama_alive():
        console.print("[bold red]Ollama no está activo en 127.0.0.1:11434.[/bold red]")
        console.print("Inícialo primero ejecutando: [yellow]powershell scripts/start_ollama.ps1[/yellow]")
        return 1

    current_model = DEFAULT_MODEL
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Modo de argumento directo (One-shot)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        console.print(f"[bold blue]Tú:[/bold blue] {query}")
        process_turn(query, history, model=current_model)
        return 0

    # Modo interactivo continuo (REPL)
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]Diego[/bold blue]")
            if not user_input.strip():
                continue
            
            cmd = user_input.strip().lower()
            if cmd in {"/bye", "/exit", "exit", "quit"}:
                console.print("[dim]¡Hasta luego, Diego![/dim]")
                break
            if cmd in {"/observatorio", "/obs"}:
                from ada.modelmap.extract import MAP_NPZ, extract_model_map
                from ada.viz.observatory import build_observatory

                if not MAP_NPZ.is_file():
                    console.print("[yellow]Leyendo los pesos reales del modelo (una sola vez, ≈3-4 min)...[/yellow]")
                    extract_model_map()
                rep = build_observatory(open_in_browser=True)
                console.print(Panel(
                    f"[bold green]Observatorio abierto:[/bold green] [cyan]{rep['absolute_path']}[/cyan]\n"
                    f"{rep['ffn_neurons_rendered']:,} neuronas FFN y {rep['heads_rendered']:,} cabezas, todas reales.",
                    title="ADA Observatory", border_style="cyan"))
                continue
            if cmd in {"/bien", "/mal"} or cmd.startswith("/corrige"):
                if not _LAST_TURN:
                    console.print("[yellow]Aún no hay una respuesta que valorar.[/yellow]")
                    continue
                if cmd == "/bien":
                    record_interaction(user_input=_LAST_TURN["user"], assistant_response=_LAST_TURN["response"],
                                       quality_score=1.0, user_feedback="positive", tags=["feedback_diego"],
                                       extra_metadata={"source": "conversation_approved", "heart_note": _LAST_HEART_NOTE["note"], "speaker": _SPEAKER[0]})
                    HEART.apply("aprobada", speaker=_SPEAKER[0])
                    console.print("[green]Guardada como ejemplo aprobado: entrará al próximo entrenamiento.[/green]")
                    console.print(_heart_line())
                elif cmd == "/mal":
                    record_interaction(user_input=_LAST_TURN["user"], assistant_response=_LAST_TURN["response"],
                                       quality_score=0.0, user_feedback="negative", tags=["feedback_diego"],
                                       extra_metadata={"source": "conversation_rejected"})
                    HEART.apply("rechazada", speaker=_SPEAKER[0])
                    console.print("[yellow]Marcada como mala: no se usará para entrenar ni se recordará.[/yellow]")
                    console.print(_heart_line())
                else:
                    corrected = user_input.strip()[len("/corrige"):].strip()
                    if not corrected:
                        console.print("[yellow]Uso: /corrige <la respuesta correcta completa>[/yellow]")
                        continue
                    record_interaction(user_input=_LAST_TURN["user"], assistant_response=normalize_math_markdown(corrected),
                                       quality_score=1.0, user_feedback="positive", tags=["correccion_diego"],
                                       extra_metadata={"source": "human_correction", "heart_note": _LAST_HEART_NOTE["note"], "speaker": _SPEAKER[0]})
                    HEART.apply("corregida", speaker=_SPEAKER[0])
                    console.print("[green]Corrección guardada con prioridad máxima para el entrenamiento.[/green]")
                    console.print(_heart_line())
                continue
            if cmd.split()[0] == "/soy":
                name = user_input.strip()[len("/soy"):].strip()
                if not name:
                    console.print(f"[yellow]Ahora hablo con {_SPEAKER[0]}. Uso: /soy <nombre>[/yellow]")
                else:
                    _SPEAKER[0] = name[:40]
                    console.print(f"[green]Entendido: ahora hablo con {_SPEAKER[0]}. Guardaré cómo me hace sentir cada conversación.[/green]")
                continue
            if cmd in {"/corazon", "/corazón", "/sentimientos"}:
                snap = HEART.snapshot()
                bars = "\n".join(f"{k:<12} {'█' * int(v * 20):<20} {v:.2f}" for k, v in snap["feelings"].items())
                bonds = "\n".join(f"  {n}: {b['interactions']} conversaciones, vínculo {b['warmth']:.2f}" for n, b in snap["bonds"].items()) or "  (aún no he hablado con nadie)"
                recent = "\n".join(f"  · {r['reason']}" for r in snap["recent"][-4:]) or "  (nada todavía)"
                console.print(Panel(f"[bold]Me siento {snap['mood']['name']}[/bold]\n\n{bars}\n\nVínculos:\n{bonds}\n\nLo último que me pasó:\n{recent}",
                                    title="Núcleo afectivo de Ada", border_style="magenta"))
                continue
            if cmd in {"/think", "/pensar", "/auto"}:
                agent = get_agent(
                    interval_s=20.0,
                    guard=AutonomyGuard(max_cycles_per_hour=120),
                    narrate=True,
                    announce=True,
                    auto_train=False,  # QLoRA 14B inviable en 12 GB (ver data/work/autonomy/TRAINING_DISABLED)
                    heart=HEART,
                    on_event=_print_agent_event,
                )
                if agent.running:
                    agent.stop()
                    console.print("[yellow]Agente autónomo detenido (termina el ciclo en curso).[/yellow]")
                else:
                    clear_stop()
                    if not agent.start_background():
                        console.print("[red]No se pudo iniciar el agente.[/red]")
                        continue
                    st = agent.status()
                    console.print(Panel(
                        "[bold green]Ada piensa y se entrena por su cuenta:[/bold green]\n"
                        "Cada ~20 s propone un problema, lo verifica por doble método, guarda solo lo verificado y te cuenta aquí qué notó.\n"
                        "Aprende sin cambiar sus pesos: recuerdos aprobados con /bien y /corrige, y verificación por doble método.\n"
                        "Cede la GPU cuando conversas y se frena con [yellow]data/work/autonomy/STOP[/yellow] o con /think.\n"
                        f"Ciclos totales: [bold cyan]{st['cycles']}[/bold cyan] | verificados: [bold cyan]{st['verified']}[/bold cyan] | "
                        f"niveles: {', '.join(f'{k}={v['level']}' for k, v in st['curriculum'].items())}",
                        title="ADA Autonomous Agent",
                        border_style="green"
                    ))
                continue
            if cmd in {"/learn", "/memoria", "/train"}:
                stats = get_continual_learning_stats()
                console.print(Panel(
                    f"[bold green]Banco de Memoria y Auto-Entrenamiento Continuo:[/bold green]\n"
                    f"Interacciones en memoria episódica: [bold cyan]{stats['episodic_memories_count']}[/bold cyan]\n"
                    f"Muestras curadas para fine-tuning: [bold cyan]{stats['curated_training_samples_count']}[/bold cyan]\n"
                    f"Dataset SFT: [dim]{stats['dataset_file']}[/dim]\n"
                    f"Estado QLoRA en RTX 5070: [green]{'Listo para auto-entrenar' if stats['qlora_ready'] else 'Recopilando interacciones verificadas'}[/green]",
                    title="ADA Continual Learning Engine",
                    border_style="green"
                ))
                continue
            if cmd in {"/brain", "/cerebro"}:
                rep = build_neural_brain_simulation(open_in_browser=True)
                console.print(Panel(
                    f"[bold green]Topología Transformer 3D lanzada:[/bold green]\n[cyan]{rep['interactive_html_path']}[/cyan]",
                    title="ADA Transformer Topology",
                    border_style="cyan"
                ))
                continue
            if cmd in {"/math", "/libro", "/calculo", "/latex", "/render"}:
                rep = build_math_textbook_html(open_in_browser=True)
                console.print(Panel(
                    f"[bold green]Cuaderno Matemático de Libro de Texto 2D abierto:[/bold green]\n"
                    f"[cyan]{rep['interactive_html_path']}[/cyan]\n\n"
                    f"Tipografía gráfica 2D idéntica a libros universitarios (KaTeX con fracciones y límites verticales reales).",
                    title="ADA Mathematical Textbook Studio",
                    border_style="green"
                ))
                continue
            if cmd.split()[0] in {"/vision", "/camera", "/camara"}:
                from ada.vision.live_camera import start_vision, stop_vision

                if cmd.endswith(("stop", "parar")):
                    stop_vision()
                    console.print("[yellow]Visión en vivo detenida.[/yellow]")
                    continue
                report = start_vision()
                lines = [f"[bold green]Fuente activa:[/bold green] {report['chosen']}"]
                for a in report["attempts"]:
                    if not a["ok"]:
                        lines.append(f"[dim]· descartada {a['backend']}: {a['reason']}[/dim]")
                lines.append("El análisis (seguimiento + cinemática) corre en CPU con OpenCV; los frames anotados y los cálculos se ven en [cyan]/observatorio[/cyan].")
                console.print(Panel("\n".join(lines), title="ADA Live Vision", border_style="cyan"))
                continue
            if cmd == "/clear":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                console.print("[dim]Historial de conversación reiniciado a 0 tokens.[/dim]")
                continue
            if cmd == "/r1":
                current_model = "ada-r1"
                console.print("[bold yellow]Cambiado al modelo razonador: Ada r1.[/bold yellow]")
                continue
            if cmd == "/q":
                current_model = "ada-q"
                console.print("[bold yellow]Cambiado al modelo orquestador: Ada Q.[/bold yellow]")
                continue

            process_turn(user_input, history, model=current_model)

        except KeyboardInterrupt:
            console.print("\n[dim]Sesión finalizada.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error en turno:[/bold red] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
