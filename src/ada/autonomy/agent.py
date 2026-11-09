"""Agente autonomo de auto-juego verificado de ADA.

Ciclo real (no simulado):
  1. Ada Q (LLM local en Ollama) PROPONE un problema de calculo/EDO en JSON.
  2. La propuesta se sanea (nunca llega texto del LLM sin filtrar a SymPy/eval).
  3. Un subproceso aislado con timeout resuelve y verifica por DOBLE metodo independiente.
  4. Solo si la verificacion pasa se guarda una muestra; los fallos tambien se registran (senal de curriculo).
  5. El curriculo sube o baja la dificultad segun la tasa de acierto reciente.

El agente cede la GPU a Diego, respeta el archivo STOP y el presupuesto por hora (ver guard.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp

from ada.autonomy.guard import AUTONOMY_DIR, AutonomyGuard, acquire_instance_lock, release_instance_lock
from ada.autonomy.safe_expr import sanitize_expression
from ada.config import PROJECT_ROOT, ModelRoleConfig, assert_safe_write_path
from ada.training.continuous_learner import record_interaction

STATE_FILE = AUTONOMY_DIR / "state.json"
JOURNAL_FILE = AUTONOMY_DIR / "journal.jsonl"
KINDS = ("derivative", "integral", "ode")
MAX_LEVEL = 4
WINDOW = 10
VERIFY_TIMEOUT_S = 90

_LEVEL_HINTS = {
    "derivative": {
        1: "polinomios y una funcion trigonometrica simple",
        2: "productos y composiciones sencillas (regla del producto y de la cadena)",
        3: "cocientes y composiciones anidadas con exp/log/sin/cos",
        4: "combinaciones de tres o mas reglas, con sqrt, tan y exp",
    },
    "integral": {
        1: "polinomios en un intervalo acotado",
        2: "sin/cos/exp elementales en un intervalo acotado",
        3: "productos que requieren integracion por partes",
        4: "composiciones que requieren sustitucion",
    },
    "ode": {
        1: "orden 1 lineal, p. ej. -y + sin(t)",
        2: "orden 2 oscilador armonico amortiguado lineal con dy e y",
        3: "orden 2 con no linealidad suave, p. ej. pendulo con sin(y) y amortiguamiento",
        4: "orden 2 no lineal con termino cubico o forzamiento sin(t)",
    },
}

_SCHEMAS = {
    "derivative": '{"expression": "<f(x) en sintaxis SymPy, solo variable x>", "order": <1 o 2>, "eval_point": <numero entre 0.2 y 1.5>}',
    "integral": '{"expression": "<f(x) en sintaxis SymPy, solo variable x>", "lower": <numero>, "upper": <numero mayor que lower, ambos entre -3 y 3>}',
    "ode": '{"order": <1 o 2>, "rhs": "<lado derecho; orden 1 usa t,y ; orden 2 usa t,y,dy>", "y0": <numero entre -2 y 2>, "dy0": <numero entre -2 y 2>, "t_end": <numero entre 1 y 6>}',
}


@dataclass
class Proposal:
    kind: str
    level: int
    params: dict[str, Any]

    def key(self) -> str:
        blob = json.dumps({"k": self.kind, "p": self.params}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- validacion de propuestas


def _num(v: Any, lo: float, hi: float, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} no es numerico") from None
    if not (lo <= f <= hi) or f != f:
        raise ValueError(f"{name} fuera de rango [{lo}, {hi}]")
    return round(f, 4)


def validate_proposal(kind: str, level: int, raw: dict[str, Any]) -> Proposal:
    """Convierte el JSON crudo del LLM en una propuesta segura o lanza ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("la propuesta no es un objeto JSON")
    if kind == "derivative":
        order = int(_num(raw.get("order", 1), 1, 3 if level >= 3 else 2, "order"))
        params = {
            "expression": sanitize_expression(raw.get("expression"), {"x"}),
            "variable": "x",
            "order": order,
            "eval_point": _num(raw.get("eval_point", 0.75), 0.2, 1.5, "eval_point"),
        }
    elif kind == "integral":
        lower = _num(raw.get("lower"), -3, 3, "lower")
        upper = _num(raw.get("upper"), -3, 3, "upper")
        if upper <= lower:
            raise ValueError("upper debe ser mayor que lower")
        params = {
            "expression": sanitize_expression(raw.get("expression"), {"x"}),
            "variable": "x",
            "lower": lower,
            "upper": upper,
        }
    elif kind == "ode":
        order = int(_num(raw.get("order", 2), 1, 2, "order"))
        allowed = {"t", "y"} if order == 1 else {"t", "y", "dy"}
        params = {
            "order": order,
            "rhs": sanitize_expression(raw.get("rhs"), allowed),
            "y0": _num(raw.get("y0", 0.5), -2, 2, "y0"),
            "dy0": _num(raw.get("dy0", 0.0), -2, 2, "dy0") if order == 2 else 0.0,
            "t_end": _num(raw.get("t_end", 3.0), 1, 6, "t_end"),
        }
    else:
        raise ValueError(f"tipo desconocido: {kind}")
    return Proposal(kind=kind, level=level, params=params)


# --------------------------------------------------------------------------- redaccion de la muestra


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.10g}"
    except (TypeError, ValueError):
        return str(v)


def compose_verified_sample(proposal: Proposal, res: dict[str, Any]) -> tuple[str, str]:
    """Redacta (pregunta, respuesta) con los resultados REALES devueltos por las herramientas verificadas."""
    p = proposal.params
    methods = " + ".join(res.get("verification_methods", []))
    seal = f"[VERIFICADO: {methods}]"

    if proposal.kind == "derivative":
        n = p["order"]
        q = f"Calcula la derivada de orden {n} de f(x) = {p['expression']} y evalúala en x = {p['eval_point']}."
        d_sym = "\\frac{d^{%d}}{dx^{%d}}" % (n, n) if n > 1 else "\\frac{d}{dx}"
        a = (
            f"$$\n\\begin{{aligned}}\nf(x) &= {res['expression_latex']} \\\\\n"
            f"{d_sym}\\left[f(x)\\right] &= {res['latex_derivative']}\n\\end{{aligned}}\n$$\n\n"
            f"Evaluación en $x = {p['eval_point']}$: valor simbólico {_fmt(res['symbolic_value_at_point'])}, "
            f"valor numérico (mpmath, 50 dígitos) {_fmt(res['numeric_value_at_point'])}; "
            f"discrepancia relativa {res['relative_discrepancy']:.2e}. {seal}"
        )
    elif proposal.kind == "integral":
        q = f"Calcula la integral definida de {p['expression']} dx entre {p['lower']} y {p['upper']}."
        a = (
            f"$$\n\\int_{{{p['lower']}}}^{{{p['upper']}}} {res['expression_latex']}\\,dx "
            f"= {res['symbolic_latex']} \\approx {_fmt(res['symbolic_float'])}\n$$\n\n"
            f"Contraste con cuadratura adaptativa (SciPy quad): {_fmt(res['scipy_quad_float'])}; "
            f"discrepancia relativa {res['relative_discrepancy']:.2e}. {seal}"
        )
    else:
        order = p["order"]
        lhs = "\\frac{d^2 y}{dt^2}" if order == 2 else "\\frac{dy}{dt}"
        prime = "y''" if order == 2 else "y'"
        cond_plain = f"y(0) = {p['y0']}" + (f", y'(0) = {p['dy0']}" if order == 2 else "")
        cond_tex = cond_plain.replace(", ", ",\\ ")
        q = (
            f"Resuelve numéricamente y verifica la EDO de orden {order}: "
            f"{prime} = {p['rhs']}, con {cond_plain} en t ∈ [0, {p['t_end']}]."
        )
        fin = res["final_state_dop853"]
        end = f"y({p['t_end']}) = {_fmt(fin['y_end'])}"
        if order == 2:
            end += f",\\ y'({p['t_end']}) = {_fmt(fin['dy_end'])}"
        syms = {"t": sp.Symbol("t"), "y": sp.Symbol("y"), "dy": sp.Symbol("dy")}
        rhs_tex = sp.latex(sp.sympify(res["ode_rhs"].replace("^", "**"), locals=syms))
        a = (
            f"$$\n\\begin{{aligned}}\n{lhs} &= {rhs_tex} \\\\\n"
            f"{cond_tex} &\\Rightarrow {end}\n\\end{{aligned}}\n$$\n\n"
            f"Dos integradores independientes (DOP853 explícito vs {res['verification_methods'][-1]}) "
            f"coinciden con discrepancia máxima {res['max_abs_discrepancy']:.2e}. {seal}"
        )
    return q, a


# --------------------------------------------------------------------------- infraestructura LLM / verificador


def default_verifier(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta la verificacion en un subproceso aislado con timeout duro."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "ada.autonomy.verify_worker"],
            input=json.dumps({"kind": kind, "params": params}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=VERIFY_TIMEOUT_S,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout de verificacion ({VERIFY_TIMEOUT_S}s)"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"ok": False, "error": f"el verificador no devolvio salida (rc={proc.returncode}) {proc.stderr[-200:]}"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "salida del verificador ilegible"}


class OllamaProposer:
    """Pide una propuesta JSON a Ada Q y mide los tokens/s reales."""

    def __init__(self, model: str | None = None, timeout_s: float = 180.0) -> None:
        cfg = ModelRoleConfig()
        self.model = model or cfg.orchestrator_ollama
        self.host = cfg.ollama_api_host.rstrip("/")
        # Mismo num_ctx que el modelo residente: otro valor obligaria a Ollama a recargarlo en VRAM.
        self.num_ctx = cfg.context_length
        self.timeout_s = timeout_s
        self.last_tokens_per_sec: float | None = None

    def __call__(self, prompt: str, seed: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Eres un generador de problemas matematicos. Respondes SOLO con un objeto JSON valido."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.9, "top_p": 0.95, "seed": seed, "num_ctx": self.num_ctx, "num_predict": 300},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        n, dur = data.get("eval_count"), data.get("eval_duration")
        self.last_tokens_per_sec = round(n / (dur / 1e9), 1) if n and dur else None
        return data.get("message", {}).get("content", "")


class OllamaNarrator:
    """Ada Q 'habla sola': una frase en primera persona sobre lo que acaba de verificar (texto libre, no se entrena con ella)."""

    def __init__(self, model: str | None = None, timeout_s: float = 60.0) -> None:
        cfg = ModelRoleConfig()
        self.model = model or cfg.orchestrator_ollama
        self.host = cfg.ollama_api_host.rstrip("/")
        self.num_ctx = cfg.context_length
        self.timeout_s = timeout_s

    def __call__(self, question: str, answer: str) -> str:
        prompt = (
            "Acabas de resolver y verificar por doble metodo este ejercicio.\n"
            f"Ejercicio: {question}\nResultado verificado: {answer[:500]}\n\n"
            "En UNA sola frase en espanol y en primera persona, di en voz alta que acabas de comprobar o que notas. "
            "Usa solo lo que aparece arriba; no inventes cifras."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7, "num_ctx": self.num_ctx, "num_predict": 70},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            text = json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")
        return " ".join(text.split())[:300]


# --------------------------------------------------------------------------- agente


class AutonomousAgent:
    def __init__(
        self,
        *,
        llm: Callable[[str, int], str] | None = None,
        verifier: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        guard: AutonomyGuard | None = None,
        interval_s: float = 90.0,
        consolidate_every: int = 10,
        rng: random.Random | None = None,
        record: bool = True,
        narrate: bool = False,
        narrator: Callable[[str, str], str] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        announce: bool = False,
        bridge_url: str = "http://127.0.0.1:8765",
        auto_train: bool = False,
        trainer: Any = None,
        reflector: Callable[[str], str] | None = None,
        reflect_every: int = 10,
        heart: Any = None,
    ) -> None:
        self.heart = heart
        self.llm = llm or OllamaProposer()
        self.verifier = verifier or default_verifier
        self.guard = guard or AutonomyGuard()
        self.interval_s = interval_s
        self.consolidate_every = consolidate_every
        self.rng = rng or random.Random()
        self.record = record
        self.narrate = narrate
        self.narrator = narrator or (OllamaNarrator() if narrate else None)
        self.on_event = on_event
        self.announce = announce
        self.bridge_url = bridge_url.rstrip("/")
        if reflector is None and narrate:
            from ada.autonomy.self_model import OllamaSelfReflector

            reflector = OllamaSelfReflector()
        self.reflector = reflector
        self.reflect_every = reflect_every
        self.trainer = trainer
        if auto_train and self.trainer is None:
            from ada.autonomy.auto_trainer import AutoTrainer

            self.trainer = AutoTrainer(log=self._on_trainer_event)
        self._last_skip_reason = ""
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.state = self._load_state()

    # -- estado persistente ------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"cycles": 0, "verified": 0, "seen": [], "curriculum": {}, "last_cycle_ts": None}
        if STATE_FILE.is_file():
            try:
                state.update(json.loads(STATE_FILE.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        for k in KINDS:
            state["curriculum"].setdefault(k, {"level": 1, "window": [], "proposed": 0, "verified": 0})
        return state

    def _save_state(self) -> None:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
        assert_safe_write_path(STATE_FILE)
        self.state["seen"] = self.state["seen"][-2000:]
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_FILE)

    def _journal(self, entry: dict[str, Any]) -> None:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
        assert_safe_write_path(JOURNAL_FILE)
        entry = {"ts": time.time(), **entry}
        with open(JOURNAL_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _on_trainer_event(self, e: dict[str, Any]) -> None:
        self._journal(e)
        if self.heart is not None:
            if e.get("trained_ok"):
                self.heart.apply("entreno_ok")
            elif e.get("gate_rejected"):
                self.heart.apply("entreno_descartado")
        self._emit(e)

    # -- visibilidad: el agente se hace notar ------------------------------
    def _emit(self, entry: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(entry)
            except Exception:  # noqa: BLE001 - un fallo al pintar nunca debe parar el aprendizaje
                pass

    def _announce(self, thinking: bool, text: str = "", tps: float | None = None) -> None:
        """Avisa al puente en vivo (source='autonomous') para que el Observatorio muestre que Ada piensa por su cuenta."""
        if not self.announce:
            return
        body: dict[str, Any] = {"is_thinking": thinking, "current_prompt": text, "model": "ada-q", "source": "autonomous"}
        if tps:
            body["tokens_per_sec"] = tps
        try:
            req = urllib.request.Request(
                f"{self.bridge_url}/api/status", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=0.5).close()
        except Exception:  # noqa: BLE001 - sin puente no pasa nada
            pass

    # -- curriculo ---------------------------------------------------------
    def _choose_kind(self) -> str:
        if self.rng.random() < 0.3:
            return self.rng.choice(KINDS)

        def acc(k: str) -> float:
            w = self.state["curriculum"][k]["window"]
            return sum(w) / len(w) if len(w) >= 3 else 0.5

        return min(KINDS, key=lambda k: (acc(k), self.rng.random()))

    def _update_curriculum(self, kind: str, success: bool) -> None:
        c = self.state["curriculum"][kind]
        c["proposed"] += 1
        c["verified"] += int(success)
        c["window"] = (c["window"] + [int(success)])[-WINDOW:]
        if len(c["window"]) >= 8:
            rate = sum(c["window"]) / len(c["window"])
            if rate >= 0.85 and c["level"] < MAX_LEVEL:
                c["level"] += 1
                c["window"] = []
            elif rate <= 0.30 and c["level"] > 1:
                c["level"] -= 1
                c["window"] = []

    # -- un ciclo ----------------------------------------------------------
    def _build_prompt(self, kind: str, level: int) -> str:
        return (
            f"Propon UN problema de tipo '{kind}', nivel de dificultad {level} de {MAX_LEVEL}: {_LEVEL_HINTS[kind][level]}.\n"
            f"Funciones permitidas: sin, cos, tan, exp, log, sqrt. Usa ** o ^ para potencias (exponente <= 12).\n"
            f"Responde SOLO con JSON con exactamente esta forma: {_SCHEMAS[kind]}\n"
            f"Varia los coeficientes; no repitas problemas triviales."
        )

    def run_cycle(self) -> dict[str, Any]:
        allowed, reason = self.guard.should_run()
        if not allowed:
            if reason != self._last_skip_reason:
                self._journal({"status": "skipped", "detail": reason})
                self._emit({"status": "skipped", "detail": reason})
                self._last_skip_reason = reason
            return {"status": "skipped", "detail": reason}
        self._last_skip_reason = ""

        t0 = time.perf_counter()
        kind = self._choose_kind()
        level = self.state["curriculum"][kind]["level"]
        self.guard.note_cycle()
        self.state["cycles"] += 1
        self.state["last_cycle_ts"] = time.time()
        entry: dict[str, Any] = {"kind": kind, "level": level}
        self._announce(True, f"[por su cuenta] proponiendo un problema de {kind} (nivel {level})")

        try:
            # 1-2. Propuesta del LLM + saneado.
            try:
                raw_text = self.llm(self._build_prompt(kind, level), self.rng.randint(1, 2**31 - 1))
            except Exception as exc:  # Ollama caido / timeout: no cuenta contra el currículo
                entry.update(status="error", detail=f"LLM no disponible: {type(exc).__name__}: {exc}"[:300])
                return entry
            try:
                proposal = validate_proposal(kind, level, json.loads(raw_text))
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                self._update_curriculum(kind, False)
                entry.update(status="rejected_proposal", detail=str(exc)[:300], raw=raw_text[:300])
                return entry

            entry["proposal"] = proposal.params
            if proposal.key() in self.state["seen"]:
                entry.update(status="duplicate", detail="problema ya resuelto antes")
                return entry
            self.state["seen"].append(proposal.key())

            # 3. Verificacion por doble metodo en subproceso con timeout.
            out = self.verifier(kind, proposal.params)
            if not out.get("ok"):
                self._update_curriculum(kind, False)
                entry.update(status="failed_verification", detail=out.get("error", "error desconocido"))
                return entry
            res = out["result"]
            if not res.get("verified"):
                self._update_curriculum(kind, False)
                disc = res.get("max_abs_discrepancy", res.get("relative_discrepancy"))
                entry.update(status="failed_verification", detail=f"los dos metodos discrepan ({disc})")
                return entry

            # 4. Muestra verificada.
            question, answer = compose_verified_sample(proposal, res)
            if self.record:
                record_interaction(
                    user_input=question,
                    assistant_response=answer,
                    verified_methods=res["verification_methods"],
                    quality_score=1.0,
                    tags=["autonomo", "auto_juego", kind, f"nivel_{level}"],
                    extra_metadata={
                        "source": "autonomous_self_play",
                        "response_origin": "verified_tool_template",
                        "problem_proposed_by": getattr(self.llm, "model", "llm"),
                        "kind": kind,
                        "level": level,
                    },
                )
            self.state["verified"] += 1
            self._update_curriculum(kind, True)
            entry.update(status="verified", methods=res["verification_methods"], question=question)
            if self.narrator:
                self._announce(True, f"[por su cuenta] reflexionando sobre: {question[:80]}")
                try:
                    entry["thought"] = self.narrator(question, answer)
                except Exception as exc:  # noqa: BLE001 - la narracion es opcional
                    entry["thought_error"] = f"{type(exc).__name__}: {exc}"[:120]
            if self.reflector and self._time_to_reflect():
                self._announce(True, "[por su cuenta] pensando sobre sí misma")
                try:
                    from ada.autonomy.self_model import reflect

                    entry["reflection"] = reflect(self.reflector, autonomy_dir=AUTONOMY_DIR)["text"]
                except Exception as exc:  # noqa: BLE001 - la reflexion es opcional
                    entry["reflection_error"] = f"{type(exc).__name__}: {exc}"[:120]
            return entry
        finally:
            entry["latency_s"] = round(time.perf_counter() - t0, 2)
            tps = getattr(self.llm, "last_tokens_per_sec", None)
            if tps:
                entry["tokens_per_sec"] = tps
            new_level = self.state["curriculum"][kind]["level"]
            if new_level != level:
                entry["level_change"] = [level, new_level]
            self._announce(False, "", tps)
            if self.heart is not None:
                try:
                    self.heart.apply_cycle(entry)  # lo que le pasa de verdad mueve sus sentimientos
                    entry["heart"] = self.heart.snapshot()["mood"]
                except Exception:  # noqa: BLE001 - el corazon nunca debe romper el aprendizaje
                    pass
            self._journal(entry)
            self._save_state()
            self._emit(entry)
            if entry.get("status") == "verified" and self.state["verified"] % self.consolidate_every == 0:
                self._consolidate()

    def _time_to_reflect(self) -> bool:
        """La primera reflexion llega pronto (a los 3 aciertos) y despues cada `reflect_every` aciertos."""
        from ada.autonomy.self_model import REFLECTIONS_FILE_NAME

        v = self.state["verified"]
        if not (AUTONOMY_DIR / REFLECTIONS_FILE_NAME).is_file():
            return v >= 3
        return v % self.reflect_every == 0

    def _consolidate(self) -> None:
        try:
            from ada.autonomy.curator import curate_dataset

            report = curate_dataset()
            self._journal({"status": "consolidated", "detail": report["summary"]})
        except Exception as exc:  # noqa: BLE001
            self._journal({"status": "error", "detail": f"curacion fallo: {type(exc).__name__}: {exc}"[:300]})

    # -- bucle -------------------------------------------------------------
    def run_forever(self) -> None:
        if not acquire_instance_lock():
            raise RuntimeError("Ya hay otra instancia del agente autonomo activa (data/work/autonomy/agent.pid).")
        try:
            self._journal({"status": "started", "detail": f"intervalo {self.interval_s}s"})
            while not self._stop_event.is_set():
                try:
                    if self.trainer is not None:
                        self.trainer.tick()  # decide sola si toca descargar/entrenar, o vigila el entrenamiento en curso
                    self.run_cycle()
                except Exception as exc:  # noqa: BLE001 - un fallo inesperado no debe matar al agente; queda en el diario
                    self._journal({"status": "error", "detail": f"excepcion en ciclo: {type(exc).__name__}: {exc}"[:300]})
                self._stop_event.wait(self.interval_s)
            self._journal({"status": "stopped", "detail": "parada ordenada"})
        finally:
            release_instance_lock()

    def start_background(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True, name="AdaAutonomousAgent")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        if not self.running:
            self.state = self._load_state()  # puede haberlo actualizado el daemon (otro proceso)
        s = self.state
        return {
            "running": self.running,
            "cycles": s["cycles"],
            "verified": s["verified"],
            "acceptance_rate": round(s["verified"] / s["cycles"], 3) if s["cycles"] else None,
            "curriculum": {k: {"level": v["level"], "proposed": v["proposed"], "verified": v["verified"]} for k, v in s["curriculum"].items()},
            "last_cycle_ts": s["last_cycle_ts"],
        }


_AGENT: AutonomousAgent | None = None


def get_agent(**kwargs: Any) -> AutonomousAgent:
    """Singleton del agente dentro de un proceso (CLI o servidor MCP). Los kwargs solo se aplican al crearlo."""
    global _AGENT
    if _AGENT is None:
        _AGENT = AutonomousAgent(**kwargs)
    return _AGENT
