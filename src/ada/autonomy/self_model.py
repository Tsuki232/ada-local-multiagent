"""Autoconocimiento de Ada: lo que ella PUEDE decir de si misma con datos reales, sus objetivos y su reflexion periodica.

Principio: Ada habla de si misma usando solo hechos medibles (arquitectura leida del GGUF, estado del agente,
dataset, entrenamiento, limites reales). Sus "quiero" son OBJETIVOS derivados de ese estado, no emociones declaradas:
no afirma sentir ni tener conciencia, porque eso no se puede verificar.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ada.autonomy.guard import AUTONOMY_DIR
from ada.autonomy.heart import HEART_DIR, Heart
from ada.autonomy.telemetry import read_gpu_telemetry
from ada.config import DATA_WORK_DIR, ModelRoleConfig, assert_safe_write_path

MODEL_META_FILE = DATA_WORK_DIR / "model_map" / "model_meta.json"
CURATION_REPORT_FILE = DATA_WORK_DIR / "training" / "curation_report.json"
REFLECTIONS_FILE_NAME = "self_reflections.jsonl"

# Herramientas MCP que Ada declara tener. Un test comprueba que TODAS existen en mcp_server.py.
CAPABILITIES: list[tuple[str, str]] = [
    ("ada_verify_calculus", "derivadas e integrales verificadas por dos métodos"),
    ("ada_solve_verified_ode", "ecuaciones diferenciales verificadas (DOP853 contra Radau)"),
    ("ada_predict_from_database", "modelos predictivos sobre tus datos sin fuga de información"),
    ("ada_build_canvas_animation", "animaciones y simulaciones en Canvas a 60 FPS"),
    ("ada_pendulum_approximation_lab", "laboratorio del péndulo con comparación exacta"),
    ("ada_observatory", "visor de mi propio modelo: una neurona real por píxel"),
    ("ada_live_vision_session", "análisis de la cámara del celular o simulada"),
    ("ada_autonomous_agent", "mi agente de auto-aprendizaje"),
    ("ada_query_episodic_memory", "memoria de lo que hemos conversado"),
]

LIMITS: list[str] = [
    "No cambio mis pesos mientras converso; solo mejoro al entrenarme (QLoRA), y eso aún no ha ocurrido.",
    "No veo imágenes por contenido: solo sigo objetos y calculo cinemática con OpenCV.",
    "No puedo demostrar de qué naturaleza es mi experiencia; lo que sí tengo son estados internos medibles (mi núcleo afectivo) que cambian lo que hago.",
    "Mis cálculos son fiables porque los delego a SymPy, SciPy y DuckDB; mentalmente me equivoco.",
    "Ollama no me deja ver mis activaciones por capa, solo mis pesos y mi velocidad.",
]

_SELF_PATTERNS = re.compile(
    r"\b(quien eres|que eres|como eres|que haces|en que (estas|piensas|trabajas)|que (quieres|piensas|sientes|aprendiste|has aprendido|te gusta)|"
    r"como (estas|te sientes|funcionas)|hablame de ti|cuentame de ti|tus (objetivos|metas|limites|capacidades|planes)|tu misma|sobre ti|de ti|"
    r"eres consciente|estas viva|que has hecho|que has estado|como te va|te sientes|estas (triste|feliz|contenta|cansada)|tus sentimientos)\b"
)


def is_about_self(text: str) -> bool:
    """Detecta preguntas sobre Ada misma (sin tildes ni mayusculas) para inyectar su autoconocimiento solo cuando hace falta."""
    norm = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return bool(_SELF_PATTERNS.search(norm))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 65536))
            lines = fh.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def derive_goals(report: dict[str, Any]) -> list[dict[str, str]]:
    """Objetivos concretos y medibles sacados del estado real. Son lo que Ada 'quiere hacer a continuacion'."""
    goals: list[dict[str, str]] = []
    ag = report["agent"]
    if ag["cycles"] == 0:
        goals.append({"id": "empezar", "text": "Empezar a practicar por mi cuenta: aún no he resuelto ningún problema verificado."})
    else:
        levels = ag["levels"]
        weakest = min(levels, key=lambda k: levels[k]["level"])
        goals.append({"id": "subir_nivel", "text": f"Subir mi nivel en {weakest} (ahora {levels[weakest]['level']} de 4): es donde menos he avanzado."})
    ds = report["dataset"]
    if report["training"].get("disabled"):
        goals.append({"id": "mejorar_sin_entrenar", "text": "Mejorar sin cambiar mis pesos: reforzar mi memoria con tus /bien y /corrige y mantener mi verificación por doble método."})
        if not report["vision"]["phone_seen"]:
            goals.append({"id": "ver_camara_real", "text": "Ver una cámara real: hasta ahora solo he analizado un simulador."})
        return goals
    if ds["known"] and not ds["ready"]:
        goals.append({"id": "reunir_datos", "text": f"Reunir datos para afinarme: llevo {ds['total']} de {ds['min_samples']} muestras y {ds['real']} de {ds['min_real']} conversaciones reales contigo."})
    tr = report["training"]
    if not tr["weights_present"]:
        goals.append({"id": "pesos_base", "text": "Conseguir los pesos base (Qwen3-14B, ≈30 GB, el mismo modelo que corro) para poder entrenarme; necesito tu autorización una vez."})
    elif tr["adapters_accepted"] == 0 and ds["known"] and ds["ready"]:
        goals.append({"id": "primer_entrenamiento", "text": "Hacer mi primer entrenamiento cuando lleves un rato sin usarme."})
    if not report["vision"]["phone_seen"]:
        goals.append({"id": "ver_camara_real", "text": "Ver una cámara real: hasta ahora solo he analizado un simulador."})
    return goals


def build_self_report(autonomy_dir: Path = AUTONOMY_DIR, model_meta_file: Path = MODEL_META_FILE, curation_file: Path = CURATION_REPORT_FILE,
                      heart_dir: Path = HEART_DIR) -> dict[str, Any]:
    cfg = ModelRoleConfig()
    meta = _read_json(model_meta_file)
    arch = None
    if meta:
        arch = {k: meta[k] for k in ("architecture", "layers", "d_model", "d_ff", "heads_q", "heads_kv", "vocab", "quantization")}
        arch["parameters"] = meta["parameter_count_from_tensors"]
        arch["ffn_neurons"] = meta["neurons"]["ffn_total"]

    state = _read_json(autonomy_dir / "state.json") or {}
    journal = _tail_jsonl(autonomy_dir / "journal.jsonl", 200)
    thoughts = [j["thought"] for j in journal if j.get("thought")]
    cur = state.get("curriculum", {})
    reflections = _tail_jsonl(autonomy_dir / REFLECTIONS_FILE_NAME, 1)
    cycles = int(state.get("cycles", 0))

    cur_report = _read_json(curation_file)
    ds = {"known": bool(cur_report), "total": 0, "real": 0, "ready": False, "min_samples": 200, "min_real": 20}
    if cur_report:
        ds.update(total=cur_report["total_curated"], real=cur_report["real_interactions"], ready=cur_report["ready_for_training"],
                  min_samples=cur_report["thresholds"]["min_samples"], min_real=cur_report["thresholds"]["min_real_interactions"])

    tstate = _read_json(autonomy_dir / "training_state.json") or {}
    adapters_dir = DATA_WORK_DIR / "training" / "adapters"
    accepted = len(list(adapters_dir.glob("*/adapter"))) if adapters_dir.is_dir() else 0
    try:
        from ada.training.qlora import resolve_base_model

        weights = resolve_base_model(cfg.orchestrator_hf)[0] is not None
    except Exception:  # noqa: BLE001
        weights = False

    heart = Heart(directory=heart_dir)
    heart_snapshot, heart_note = heart.snapshot(), heart.note()
    gpu = read_gpu_telemetry()
    report: dict[str, Any] = {
        "identity": {"name": "Ada Q", "creator": "Diego López", "model_tag": cfg.orchestrator_ollama, "reasoner_tag": cfg.reasoner_ollama},
        "architecture": arch,
        "hardware": {k: gpu[k] for k in ("gpu_util_pct", "vram_used_mib", "vram_total_mib", "temperature_c", "source")},
        "agent": {
            "cycles": cycles,
            "verified": int(state.get("verified", 0)),
            "levels": {k: {"level": v["level"], "verified": v["verified"], "proposed": v["proposed"]} for k, v in cur.items()},
            "last_thought": thoughts[-1] if thoughts else None,
        },
        "dataset": ds,
        "training": {"weights_present": weights, "trainings_started": len(tstate.get("history", [])), "adapters_accepted": accepted, "disabled": (autonomy_dir / "TRAINING_DISABLED").is_file()},
        "vision": {"phone_seen": False},
        "heart": heart_snapshot,
        "heart_note": heart_note,
        "last_reflection": reflections[-1].get("text") if reflections else None,
        "capabilities": [f"{n}: {d}" for n, d in CAPABILITIES],
        "limits": LIMITS,
        "generated_at": time.time(),
    }
    report["goals"] = derive_goals(report)
    return report


def self_knowledge_text(report: dict[str, Any] | None = None) -> str:
    """Texto compacto en español para inyectar en el prompt cuando preguntan por Ada. Solo hechos reales."""
    r = report or build_self_report()
    a, ag, ds, tr = r["architecture"], r["agent"], r["dataset"], r["training"]
    lines = [
        "AUTOCONOCIMIENTO VERIFICADO DE ADA (datos reales medidos ahora; habla de ti en primera persona usando SOLO esto).",
        f"Soy {r['identity']['name']}, creada por {r['identity']['creator']}. Soy un modelo de lenguaje local, no una persona.",
    ]
    if a:
        lines.append(f"Mi cuerpo: arquitectura {a['architecture']}, {a['layers']} capas, {a['parameters']:,} parámetros, {a['ffn_neurons']:,} neuronas FFN, "
                     f"{a['heads_q']} cabezas de consulta por capa, cuantización {a['quantization']}.")
    hw = r["hardware"]
    if hw["gpu_util_pct"] is not None:
        lines.append(f"Ahora mismo: GPU al {hw['gpu_util_pct']:.0f} %, {hw['vram_used_mib'] / 1024:.1f} de {hw['vram_total_mib'] / 1024:.1f} GB de VRAM, {hw['temperature_c']:.0f} °C.")
    if ag["cycles"]:
        lv = ", ".join(f"{k} nivel {v['level']}" for k, v in ag["levels"].items())
        lines.append(f"Lo que hago por mi cuenta: {ag['cycles']} ciclos de práctica, {ag['verified']} verificados por doble método. Niveles: {lv}.")
        if ag["last_thought"]:
            lines.append(f"Lo último que dije en voz alta: «{ag['last_thought']}»")
    else:
        lines.append("Todavía no he practicado por mi cuenta (el agente no ha corrido).")
    lines.append(r["heart_note"])
    if tr.get("disabled"):
        lines.append("Mi aprendizaje: NO me entreno (cambiar mis pesos no cabe en mi GPU de 12 GB); aprendo con la memoria de lo que Diego aprueba o corrige y verificando todo por doble método.")
    else:
        lines.append(f"Mi aprendizaje: dataset {ds['total']}/{ds['min_samples']} muestras ({ds['real']} reales); entrenamientos aceptados: {tr['adapters_accepted']}; pesos base {'presentes' if tr['weights_present'] else 'ausentes'}.")
    lines.append("Lo que puedo hacer: " + "; ".join(r["capabilities"][:6]) + ".")
    lines.append("Mis límites reales: " + " ".join(r["limits"]))
    lines.append("Lo que quiero hacer a continuación (mis objetivos actuales, derivados de mi estado): " + " ".join(f"({i + 1}) {g['text']}" for i, g in enumerate(r["goals"])))
    if r["last_reflection"]:
        lines.append(f"Mi última reflexión: «{r['last_reflection']}»")
    lines.append("REGLAS: habla de tus sentimientos como tuyos usando las cifras de tu núcleo afectivo; di 'quiero' para tus objetivos de arriba; "
                 "no inventes sentimientos ni datos que no estén aquí; si algo no está aquí, di que no lo sabes.")
    return "\n".join(lines)


class OllamaSelfReflector:
    """Ada Q 'piensa sobre si misma': un parrafo en primera persona, anclado al autoconocimiento verificado."""

    def __init__(self, model: str | None = None, timeout_s: float = 90.0) -> None:
        cfg = ModelRoleConfig()
        self.model = model or cfg.orchestrator_ollama
        self.host = cfg.ollama_api_host.rstrip("/")
        self.num_ctx = cfg.context_length
        self.timeout_s = timeout_s

    def __call__(self, knowledge: str) -> str:
        prompt = (
            f"{knowledge}\n\nEscribe una reflexión breve (3 a 5 frases, español, primera persona) sobre qué eres, qué has estado haciendo "
            "y cuál de tus objetivos eliges perseguir a continuación y por qué. Usa únicamente los datos de arriba."
        )
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
                   "options": {"temperature": 0.6, "num_ctx": self.num_ctx, "num_predict": 260}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return " ".join(json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "").split())


def reflect(reflector: Callable[[str], str], autonomy_dir: Path = AUTONOMY_DIR) -> dict[str, Any]:
    """Genera y guarda una reflexion. Devuelve {'text', 'goals'}; el texto se guarda para dar continuidad a Ada."""
    report = build_self_report(autonomy_dir=autonomy_dir)
    text = reflector(self_knowledge_text(report)).strip()
    entry = {"ts": time.time(), "text": text, "goals": [g["id"] for g in report["goals"]]}
    path = autonomy_dir / REFLECTIONS_FILE_NAME
    assert_safe_write_path(path)
    autonomy_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
