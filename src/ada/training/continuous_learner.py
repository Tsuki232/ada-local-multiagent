"""Motor de Aprendizaje Continuo y Memoria Episodica de ADA (Estilo Google Harness Continual Learning)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ada.config import DATA_WORK_DIR, assert_safe_write_path


MEMORY_DIR = DATA_WORK_DIR / "memory"
TRAINING_DATA_DIR = DATA_WORK_DIR / "training"
EPISODIC_MEMORY_FILE = MEMORY_DIR / "episodic_memory.jsonl"
CONTINUAL_DATASET_FILE = TRAINING_DATA_DIR / "ada_continual_dataset.jsonl"


@dataclass
class ConversationExperience:
    timestamp: float
    user_input: str
    assistant_response: str
    verified_methods: list[str] = field(default_factory=list)
    quality_score: float = 1.0
    user_feedback: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def init_memory_system() -> None:
    """Inicializa los directorios seguros de memoria y datasets continuos."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def record_interaction(
    user_input: str,
    assistant_response: str,
    verified_methods: list[str] | None = None,
    quality_score: float = 1.0,
    user_feedback: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Registra una experiencia conversacional verificada en el banco de memoria continua.

    `extra_metadata` (p. ej. source / response_origin) permite al curador separar muestras reales de sinteticas.
    """
    init_memory_system()
    exp = ConversationExperience(
        timestamp=time.time(),
        user_input=user_input.strip(),
        assistant_response=assistant_response.strip(),
        verified_methods=verified_methods or [],
        quality_score=quality_score,
        user_feedback=user_feedback,
        tags=tags or [],
        extra=dict(extra_metadata or {}),
    )

    # 1. Guardar en memoria episodica (para RAG en tiempo real)
    assert_safe_write_path(EPISODIC_MEMORY_FILE)
    with open(EPISODIC_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + "\n")

    # 2. Si la interaccion tiene verificacion cientifica o feedback positivo, agregar al dataset de entrenamiento continuo
    if verified_methods or quality_score >= 0.9 or user_feedback == "positive":
        assert_safe_write_path(CONTINUAL_DATASET_FILE)
        sft_entry = {
            "conversations": [
                {"from": "human", "value": user_input.strip()},
                {"from": "gpt", "value": assistant_response.strip()},
            ],
            "metadata": {
                "timestamp": exp.timestamp,
                "verified_methods": exp.verified_methods,
                "quality_score": exp.quality_score,
                "tags": exp.tags,
                **exp.extra,
            },
        }
        with open(CONTINUAL_DATASET_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")

    return exp.to_dict()


_STOPWORDS = frozenset(
    "de la el en y a los las un una unos unas que es se por con para del al lo como mas pero sus le ya o fue este "
    "ha si porque esta entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos "
    "uno les ni contra otros ese eso ante ellos e esto mi antes algunos que unos yo otro otras otra el tanto esa "
    "estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros hola ada dame dime puedes".split()
)


def _meaningful_words(text: str) -> set[str]:
    import re

    return {w for w in re.findall(r"[\wáéíóúüñ]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def retrieve_relevant_memories(
    query: str,
    max_results: int = 3,
    *,
    min_overlap: int = 1,
    only_approved: bool = False,
) -> list[dict[str, Any]]:
    """Recupera memorias de conversaciones pasadas por coincidencia de palabras significativas (sin palabras vacias).

    `only_approved=True` limita a memorias con verificacion cientifica o aprobadas por Diego (y excluye el auto-juego),
    para que lo que se inyecte en el prompt sea fiable.
    """
    if not EPISODIC_MEMORY_FILE.is_file():
        return []

    q_words = _meaningful_words(query)
    matches: list[tuple[float, dict[str, Any]]] = []

    try:
        with open(EPISODIC_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if only_approved:
                        approved = entry.get("user_feedback") == "positive" or bool(entry.get("verified_methods"))
                        if not approved or "auto_juego" in entry.get("tags", []) or entry.get("user_feedback") == "negative":
                            continue
                    text = entry.get("user_input", "") + " " + entry.get("assistant_response", "")
                    overlap = len(q_words & _meaningful_words(text))
                    if overlap >= min_overlap:
                        matches.append((float(overlap), entry))
                except Exception:
                    continue
    except Exception:
        return []

    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:max_results]]


def get_continual_learning_stats() -> dict[str, Any]:
    """Reporta el estado actual del banco de aprendizaje continuo."""
    init_memory_system()
    memory_count = 0
    if EPISODIC_MEMORY_FILE.is_file():
        with open(EPISODIC_MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_count = sum(1 for line in f if line.strip())

    dataset_count = 0
    if CONTINUAL_DATASET_FILE.is_file():
        with open(CONTINUAL_DATASET_FILE, "r", encoding="utf-8") as f:
            dataset_count = sum(1 for line in f if line.strip())

    return {
        "episodic_memories_count": memory_count,
        "curated_training_samples_count": dataset_count,
        "memory_file": str(EPISODIC_MEMORY_FILE),
        "dataset_file": str(CONTINUAL_DATASET_FILE),
        "qlora_ready": dataset_count >= 10,
    }
