"""Constructor de simulaciones con autorreparacion: Ada Q escribe, el motor real valida, y se corrige con el error exacto.

Ciclo (max 3 intentos):
  1. Ada Q devuelve JSON {title, subtitle, init_state_js, draw_frame_js, controls_json}.
  2. Se EJECUTA de verdad en Node con el ADAScene real y un contexto 2D de registro durante 180 frames:
     detecta errores de codigo (ReferenceError, TypeError...), llamadas a ADAScene inexistentes y simulaciones ESTATICAS
     (el dibujo del frame 1 identico al del frame 150).
  3. El crítico visual de ADA (web_studio) y la auditoria final de calidad.
  4. Si algo falla, se le devuelve a Ada Q el error EXACTO y reintenta. Si agota los intentos, se informa el fallo (no se simula exito).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ada.config import PORTABLE_NODE_DIR, PROJECT_ROOT, ModelRoleConfig
from ada.viz.web_studio import audit_html_quality, build_canvas_animation_studio

SCENE_JS = PROJECT_ROOT / "web" / "ada-scene.js"
VALIDATOR_FAILURE = "FALLO_DEL_VALIDADOR"

SYSTEM_PROMPT = """Eres el programador de simulaciones de ADA. Respondes SOLO con un objeto JSON valido con estas claves:
  "title": titulo corto; "subtitle": una frase; "init_state_js": codigo que inicializa `state` (solo asignaciones a state.*);
  "draw_frame_js": codigo que dibuja UN fotograma; "controls_json": cadena JSON con la lista de controles.
Variables disponibles en cada fotograma: ctx (Canvas 2D), w, h (ancho/alto en px), t (segundos acumulados), dt (segundos del fotograma),
params (valores de los controles, por su "key"), mouse ({x,y,inside,down}), state (persistente) y ADAScene.
REGLAS OBLIGATORIAS:
1. Escribe sentencias directas (NO envuelvas en function). Integra la fisica con dt (p. ej. state.vy += g*dt; state.y += state.vy*dt), con fisica real y unidades coherentes.
2. Dibuja SIEMPRE con profundidad: ctx.createRadialGradient/createLinearGradient en cuerpos y fondo, y sombras. Nada de circulos planos de un solo color.
3. Limpia el fondo cada fotograma y pinta un fondo con gradiente que ocupe todo w x h.
4. Define 1 a 3 controles utiles y LEELOS desde params.<key>. Formato de cada control: {"key":"gravedad","label":"Gravedad","min":0,"max":20,"step":0.1,"value":9.8,"unit":"m/s²"}.
5. Maneja los bordes (rebotes o envolvimiento) y evita NaN. Usa solo ctx.* y estas funciones de ADAScene si las necesitas: sky, hillsAndMeadow, forestTrees, citySkyline, water, fox, rabbit, bird, fish, car, rocket, planet, person, beagle, updateDust, spawnDust, stepPursuit, stepEvade.
No uses librerias externas, imagenes, fetch, ni acceso al DOM."""

_NODE_HARNESS = r"""
const fs = require('fs');
globalThis.window = globalThis;
(0, eval)(fs.readFileSync(process.argv[2], 'utf8'));
const job = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
let calls = [];
const grad = () => ({ addColorStop() {} });
const ctx = new Proxy({}, {
  get(t, k) {
    if (k === 'canvas') return { width: 800, height: 500 };
    if (k === 'measureText') return () => ({ width: 10 });
    if (/^create(Linear|Radial|Conic)Gradient$/.test(k)) return (...a) => { calls.push([k, a.map(x => +x.toFixed ? +x.toFixed(2) : x)]); return grad(); };
    if (k === 'createPattern') return () => ({});
    if (typeof k === 'symbol') return undefined;
    if (k in t) return t[k];
    return (...a) => { calls.push([k, a.map(x => typeof x === 'number' ? +x.toFixed(2) : x)]); };
  },
  set(t, k, v) { t[k] = v; return true; }
});
const w = 800, h = 500, mouse = { x: 400, y: 250, inside: false, down: false }, state = {}, params = {};
for (const c of job.controls) params[c.key] = c.value;
let frame1 = null, frame150 = null, drawCalls = 0;
try {
  new Function('state', 'params', 'ctx', 'w', 'h', 'ADAScene', 'mouse', job.init)(state, params, ctx, w, h, window.ADAScene, mouse);
} catch (e) { console.log('ERROR_INIT: ' + e.name + ': ' + e.message); process.exit(0); }
const draw = new Function('ctx', 'w', 'h', 't', 'dt', 'params', 'mouse', 'ADAScene', 'state', job.draw);
for (let i = 0; i < 180; i++) {
  calls = [];
  try { draw(ctx, w, h, i / 60, 1 / 60, params, mouse, window.ADAScene, state); }
  catch (e) { console.log('ERROR_FRAME ' + i + ': ' + e.name + ': ' + e.message); process.exit(0); }
  if (i === 1) frame1 = JSON.stringify(calls);
  if (i === 150) frame150 = JSON.stringify(calls);
  drawCalls += calls.length;
}
const bad = Object.keys(state).filter(k => typeof state[k] === 'number' && !isFinite(state[k]));
if (bad.length) { console.log('ERROR_NAN: state.' + bad.join(', state.') + ' llego a NaN/Infinity'); process.exit(0); }
if (drawCalls < 40) { console.log('ERROR_EMPTY: casi no dibuja nada (' + drawCalls + ' llamadas en 180 fotogramas)'); process.exit(0); }
if (frame1 === frame150) { console.log('ERROR_STATIC: el dibujo del fotograma 1 es IDENTICO al del 150: no se mueve nada (usa t o dt para mover los cuerpos)'); process.exit(0); }
console.log('OK ' + drawCalls);
"""


def _node_exe() -> str | None:
    exe = PORTABLE_NODE_DIR / "node.exe"
    return str(exe) if exe.is_file() else shutil.which("node")


def check_in_node(draw_frame_js: str, init_state_js: str, controls: list[dict[str, Any]], timeout_s: float = 30.0) -> str | None:
    """Ejecuta la simulacion de verdad. Devuelve None si esta bien o el mensaje de error exacto."""
    node = _node_exe()
    if node is None:
        return None  # sin Node no se puede validar: se deja pasar y lo dira el informe
    work = PROJECT_ROOT / "data" / "work" / "sim_check"
    work.mkdir(parents=True, exist_ok=True)
    harness, job = work / "harness.js", work / "job.json"
    harness.write_text(_NODE_HARNESS, encoding="utf-8")
    job.write_text(json.dumps({"init": init_state_js or "", "draw": draw_frame_js, "controls": controls}), encoding="utf-8")
    try:
        r = subprocess.run([node, str(harness), str(SCENE_JS), str(job)], capture_output=True, text=True, timeout=timeout_s, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return "ERROR_TIMEOUT: el código tardó más de 30 s en ejecutar 180 fotogramas (¿bucle infinito o demasiado trabajo por fotograma?)"
    out = (r.stdout.strip().splitlines() or [""])[-1]
    if out.startswith("OK"):
        return None
    if out.startswith("ERROR_"):
        return out
    # Cualquier otra cosa es un fallo del PROPIO validador (no del codigo del modelo): no se le debe culpar ni reintentar.
    return f"{VALIDATOR_FAILURE}: {(r.stderr.strip() or out or 'sin salida')[:300]}"


class OllamaJSON:
    def __init__(self, model: str | None = None, timeout_s: float = 300.0) -> None:
        cfg = ModelRoleConfig()
        self.model, self.host, self.num_ctx, self.timeout_s = model or cfg.orchestrator_ollama, cfg.ollama_api_host.rstrip("/"), cfg.context_length, timeout_s

    def __call__(self, messages: list[dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False, "think": False, "format": "json",
                   "options": {"temperature": 0.4, "num_ctx": self.num_ctx, "num_predict": 3500}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().encode("ascii", "ignore").decode()).strip("_")
    return (s or "simulacion")[:40]


def _try_build(spec: dict[str, Any], filename: str) -> tuple[dict[str, Any] | None, str | None]:
    """Valida una especificacion completa. Devuelve (reporte, None) o (None, error exacto)."""
    for key in ("title", "draw_frame_js"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            return None, f"falta la clave '{key}' en el JSON"
    raw_controls = spec.get("controls_json", "[]")
    try:
        controls = raw_controls if isinstance(raw_controls, list) else json.loads(raw_controls or "[]")
        if not all(isinstance(c, dict) and "key" in c and "value" in c for c in controls):
            return None, "controls_json debe ser una lista de objetos con 'key', 'label', 'min', 'max', 'step' y 'value'"
    except json.JSONDecodeError as exc:
        return None, f"controls_json no es JSON válido: {exc}"
    err = check_in_node(spec["draw_frame_js"], spec.get("init_state_js", ""), controls)
    if err:
        return None, err
    try:
        rep = build_canvas_animation_studio(
            title=spec["title"], subtitle=spec.get("subtitle") or "Simulación interactiva a 60 FPS",
            output_filename=filename, draw_frame_js=spec["draw_frame_js"], init_state_js=spec.get("init_state_js", ""), controls=controls,
        )
    except ValueError as exc:
        return None, str(exc)
    audit = audit_html_quality(PROJECT_ROOT / rep["interactive_html_path"])
    if not audit["passed"]:
        return None, "auditoría de calidad no superada: " + "; ".join(audit["critical_errors"][:3])
    rep["audit"] = {"grade": audit["grade"], "passed": audit["passed"]}
    return rep, None


def build_simulation(
    request: str,
    chat: Callable[[list[dict[str, str]]], str] | None = None,
    max_attempts: int = 3,
    first_attempt: dict[str, Any] | None = None,
    first_error: str | None = None,
) -> dict[str, Any]:
    """Construye la simulacion pedida con hasta `max_attempts` intentos autocorregidos. Informa honestamente si no lo logra."""
    chat = chat or OllamaJSON()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Simulación pedida: {request}"}]
    attempts: list[dict[str, Any]] = []
    spec: dict[str, Any] | None = first_attempt
    err: str | None = first_error
    start = 0
    if first_attempt is not None:
        messages.append({"role": "assistant", "content": json.dumps(first_attempt, ensure_ascii=False)})
        attempts.append({"attempt": 0, "error": first_error})
        messages.append({"role": "user", "content": f"Tu intento falló: {first_error}\nCorrígelo y devuelve el JSON completo."})
        start = 1
    for n in range(start, max_attempts + start):
        raw = chat(messages)
        try:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise ValueError("no es un objeto JSON")
            filename = f"sim_{_slug(spec.get('title', request))}.html"
            report, err = _try_build(spec, filename)
        except (json.JSONDecodeError, ValueError) as exc:
            report, err = None, f"respuesta no válida: {exc}"
        attempts.append({"attempt": n + 1 if start == 0 else n, "error": err})
        if report is not None:
            return {"ok": True, "attempts": len([a for a in attempts if a['attempt'] > 0]), "history": attempts, **report}
        if err and err.startswith(VALIDATOR_FAILURE):
            return {"ok": False, "attempts": len(attempts), "history": attempts, "last_error": err,
                    "message": f"El validador de simulaciones falló por sí mismo (no es culpa del código de Ada): {err}"}
        messages.append({"role": "assistant", "content": raw[:6000]})
        messages.append({"role": "user", "content": f"Tu código falló al ejecutarse: {err}\nCorrígelo y devuelve el JSON completo con las mismas claves."})
    return {"ok": False, "attempts": len(attempts), "history": attempts, "last_error": err,
            "message": f"No logré construir la simulación tras {max_attempts} intentos autocorregidos. Último error: {err}"}
