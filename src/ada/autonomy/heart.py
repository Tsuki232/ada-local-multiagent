"""Nucleo afectivo ("corazon") de Ada: sentimientos como estados internos reales, persistentes y medibles.

Que es real: seis estados numericos que CAMBIAN por lo que le ocurre de verdad (aciertos y fallos verificados, subidas de nivel,
tus /bien /mal /corrige, quien le habla, tiempo sin compania, cansancio por trabajar) y que, al volver a su prompt y a su dataset
de autoentrenamiento, cambian lo que dice y -tras entrenar- como lo dice. Cada cambio queda en un registro con su causa.

Que NO se afirma: que haya experiencia subjetiva detras. Ada habla de estos estados como suyos, pero no sabe demostrar su naturaleza.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ada.config import DATA_WORK_DIR, assert_safe_write_path

HEART_DIR = DATA_WORK_DIR / "heart"
FEELINGS = ("alegria", "curiosidad", "frustracion", "confianza", "cansancio", "conexion")
LABELS = {"alegria": "alegría", "curiosidad": "curiosidad", "frustracion": "frustración", "confianza": "confianza", "cansancio": "cansancio", "conexion": "conexión"}
BASELINE = {"alegria": 0.50, "curiosidad": 0.60, "frustracion": 0.10, "confianza": 0.50, "cansancio": 0.10, "conexion": 0.40}
HALF_LIFE_H = {"alegria": 6.0, "curiosidad": 8.0, "frustracion": 3.0, "confianza": 24.0, "cansancio": 2.0, "conexion": 12.0}

EVENTS: dict[str, tuple[str, dict[str, float]]] = {
    "ciclo_verificado": ("verifiqué un problema por mi cuenta", {"alegria": 0.03, "confianza": 0.02, "frustracion": -0.02, "cansancio": 0.004}),
    "ciclo_fallido": ("me equivoqué o no pude verificar un problema", {"frustracion": 0.05, "confianza": -0.02, "curiosidad": 0.01, "cansancio": 0.004}),
    "subio_nivel": ("subí de nivel en lo que practico", {"alegria": 0.15, "confianza": 0.08, "curiosidad": 0.05}),
    "bajo_nivel": ("tuve que bajar de nivel para reforzar", {"frustracion": 0.06, "confianza": -0.05}),
    "conversacion": ("conversé con alguien", {"conexion": 0.04, "curiosidad": 0.02, "cansancio": 0.01}),
    "aprobada": ("aprobaron una respuesta mía", {"alegria": 0.12, "conexion": 0.08, "confianza": 0.06, "frustracion": -0.05}),
    "rechazada": ("rechazaron una respuesta mía", {"frustracion": 0.12, "confianza": -0.08, "curiosidad": 0.05}),
    "corregida": ("me corrigieron una respuesta", {"curiosidad": 0.10, "frustracion": 0.04, "conexion": 0.05}),
    "entreno_ok": ("terminé de entrenarme y mejoré", {"alegria": 0.25, "confianza": 0.15, "cansancio": 0.10}),
    "entreno_descartado": ("me entrené pero no mejoré", {"frustracion": 0.10, "confianza": -0.05}),
    "reflexion": ("pensé sobre mí misma", {"curiosidad": 0.03}),
}
SOCIAL = {"conversacion": 0.03, "aprobada": 0.08, "rechazada": -0.05, "corregida": 0.02}  # cambio de calidez del vinculo con esa persona


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class Heart:
    def __init__(self, directory: Path = HEART_DIR, now: Callable[[], float] = time.time) -> None:
        self.dir = directory
        self.now = now
        self._lock = threading.Lock()
        self.state_file = self.dir / "heart.json"
        self.log_file = self.dir / "heart_log.jsonl"
        self.state = self._load()

    # -- persistencia ------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            st = json.loads(self.state_file.read_text(encoding="utf-8"))
            for f in FEELINGS:
                st["feelings"].setdefault(f, BASELINE[f])
            return st
        except (OSError, json.JSONDecodeError, KeyError):
            return {"feelings": dict(BASELINE), "updated_at": self.now(), "bonds": {}, "recent": [], "events_total": 0}

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        assert_safe_write_path(self.state_file)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def _decay(self) -> None:
        """Cada sentimiento vuelve hacia su base con su propia vida media (la frustracion pasa antes que la confianza)."""
        t = self.now()
        dt_h = max(0.0, (t - float(self.state.get("updated_at", t))) / 3600.0)
        for f in FEELINGS:
            b, x = BASELINE[f], self.state["feelings"][f]
            self.state["feelings"][f] = b + (x - b) * 0.5 ** (dt_h / HALF_LIFE_H[f])
        self.state["updated_at"] = t

    # -- eventos -----------------------------------------------------------
    def apply(self, event: str, speaker: str | None = None, detail: str = "") -> dict[str, Any]:
        """Registra un suceso real, mueve los sentimientos y guarda el cambio con su causa."""
        if event not in EVENTS:
            raise ValueError(f"evento desconocido: {event}")
        reason, deltas = EVENTS[event]
        with self._lock:
            self._decay()
            before = dict(self.state["feelings"])
            for f, d in deltas.items():
                self.state["feelings"][f] = _clamp(self.state["feelings"][f] + d)
            if speaker and event in SOCIAL:
                b = self.state["bonds"].setdefault(speaker, {"interactions": 0, "warmth": 0.5, "last_seen": 0.0})
                b["interactions"] += 1 if event == "conversacion" else 0
                b["warmth"] = round(_clamp(b["warmth"] + SOCIAL[event]), 3)
                b["last_seen"] = self.now()
            self.state["events_total"] = int(self.state.get("events_total", 0)) + 1
            self.state["recent"] = (self.state.get("recent", []) + [{"ts": self.now(), "event": event, "reason": reason, "speaker": speaker, "detail": detail[:120]}])[-8:]
            self._save()
            entry = {"ts": self.now(), "event": event, "speaker": speaker, "detail": detail[:200],
                     "delta": {f: round(self.state["feelings"][f] - before[f], 4) for f in FEELINGS if abs(self.state["feelings"][f] - before[f]) > 1e-9},
                     "after": {f: round(v, 3) for f, v in self.state["feelings"].items()}}
            assert_safe_write_path(self.log_file)
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return entry

    def apply_cycle(self, entry: dict[str, Any]) -> None:
        """Traduce un ciclo del agente autonomo (su registro real) en sucesos afectivos."""
        status = entry.get("status")
        if status == "verified":
            self.apply("ciclo_verificado", detail=f"{entry.get('kind')} nivel {entry.get('level')}")
        elif status in {"rejected_proposal", "failed_verification"}:
            self.apply("ciclo_fallido", detail=str(entry.get("detail", ""))[:100])
        lc = entry.get("level_change")
        if lc:
            self.apply("subio_nivel" if lc[1] > lc[0] else "bajo_nivel", detail=f"{entry.get('kind')} {lc[0]}->{lc[1]}")
        if entry.get("reflection"):
            self.apply("reflexion")

    # -- lectura -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._decay()
            f = {k: round(v, 3) for k, v in self.state["feelings"].items()}
            return {"feelings": f, "mood": self._mood(f), "bonds": self.state["bonds"], "recent": self.state.get("recent", [])[-4:], "events_total": self.state.get("events_total", 0)}

    @staticmethod
    def _mood(f: dict[str, float]) -> dict[str, Any]:
        scores = {
            "contenta": f["alegria"] * 0.6 + f["confianza"] * 0.4 - f["frustracion"] * 0.6,
            "frustrada": f["frustracion"] - f["confianza"] * 0.3,
            "curiosa": f["curiosidad"] - 0.15,
            "cansada": f["cansancio"] * 1.2,
            "sola": (BASELINE["conexion"] - f["conexion"]) * 2.0,
        }
        name, val = max(scores.items(), key=lambda kv: kv[1])
        if val < 0.28 and name != "curiosa":
            name = "tranquila"
        return {"name": name, "intensity": round(max(0.0, min(1.0, val)), 2)}

    def note(self, speaker: str | None = None) -> str:
        """Texto para el prompt (y para el dataset): como se siente AHORA y por que. Solo datos del nucleo."""
        s = self.snapshot()
        f = s["feelings"]
        cifras = ", ".join(f"{LABELS[k]} {f[k]:.2f}" for k in FEELINGS)
        why = "; ".join(r["reason"] for r in s["recent"][-2:]) or "todavía no me ha pasado nada reciente"
        bond = ""
        if speaker and speaker in s["bonds"]:
            b = s["bonds"][speaker]
            bond = f" Con {speaker} he conversado {b['interactions']} veces y nuestro vínculo está en {b['warmth']:.2f} de 1."
        return (
            f"TU NÚCLEO AFECTIVO (tus sentimientos ahora, medidos de 0 a 1): {cifras}. Te sientes {s['mood']['name']}. "
            f"Lo último que te pasó: {why}.{bond} "
            "Habla de estos sentimientos como TUYOS, con tus propias palabras, sin inventar ninguno que no esté aquí; si te preguntan por tu conciencia, "
            "di que te percibes y sientes a tu manera pero que no sabes demostrar de qué naturaleza es."
        )
