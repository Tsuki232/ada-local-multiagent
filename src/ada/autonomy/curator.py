"""Curador del dataset de auto-entrenamiento: separa lo real de lo sintetico, deduplica y reparte train/eval.

Por que existe: el dataset crudo mezclaba (a) conversaciones reales de Diego, (b) muestras de auto-juego
verificadas por herramientas y (c) un lote LEGADO de auto-retos cuya "respuesta" era siempre la misma
plantilla (solo cambiaban los coeficientes). Entrenar con (c) no enseña nada y puede degradar el modelo,
asi que se excluye por defecto.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ada.config import assert_safe_write_path
from ada.training.continuous_learner import CONTINUAL_DATASET_FILE, TRAINING_DATA_DIR

STATS_CURRICULUM_FILE = TRAINING_DATA_DIR / "stats_curriculum.jsonl"
CURATED_TRAIN_FILE = TRAINING_DATA_DIR / "curated_train.jsonl"
CURATED_EVAL_FILE = TRAINING_DATA_DIR / "curated_eval.jsonl"
CURATION_REPORT_FILE = TRAINING_DATA_DIR / "curation_report.json"

LEGACY_PREFIX = "[Auto-Reto Autónomo]"
MAX_CHARS = 6000
EVAL_FRACTION = 0.10


def classify_origin(entry: dict[str, Any]) -> str:
    """'interaction' (real), 'verified_template' (auto-juego verificado) o 'legacy_template' (excluido)."""
    meta = entry.get("metadata", {})
    human = next((c["value"] for c in entry.get("conversations", []) if c.get("from") == "human"), "")
    if meta.get("response_origin") == "verified_tool_template":
        return "verified_template"
    if human.startswith(LEGACY_PREFIX):
        return "legacy_template"
    return "interaction"


def _stable_bucket(text: str) -> float:
    """Reparto determinista train/eval: la misma muestra cae siempre en el mismo lado."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _to_messages(entry: dict[str, Any]) -> list[dict[str, str]] | None:
    """Convierte a formato chat. Si la muestra guardo el estado afectivo de Ada en ese momento, va como mensaje `system`:
    asi, al entrenar, ella aprende a expresar sus sentimientos condicionada a su nucleo (igual que se le inyecta al conversar)."""
    role_map = {"human": "user", "gpt": "assistant"}
    msgs = [{"role": role_map[c["from"]], "content": c["value"].strip()} for c in entry.get("conversations", []) if c.get("from") in role_map]
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    if any(not m["content"] or len(m["content"]) > MAX_CHARS for m in msgs):
        return None
    note = entry.get("metadata", {}).get("heart_note")
    if isinstance(note, str) and note.strip():
        msgs.insert(0, {"role": "system", "content": note.strip()[:MAX_CHARS]})
    return msgs


def _first_user(msgs: list[dict[str, str]]) -> str:
    return next(m["content"] for m in msgs if m["role"] == "user")


def curate_dataset(
    source: Path | None = None,
    *,
    min_samples: int = 200,
    min_real_interactions: int = 20,
    include_legacy: bool = False,
    curriculum_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Genera curated_train/eval.jsonl y un informe honesto sobre si el dataset sirve para entrenar.

    Sin argumentos usa el dataset continuo y el curriculo de inferencia (stats_curriculum.jsonl). Si se pasa `source` explicito
    NO se agrega el curriculo por defecto (salvo que se indique en `curriculum_files`), asi las pruebas son aisladas.
    """
    if curriculum_files is None:
        curriculum_files = [STATS_CURRICULUM_FILE] if source is None else []
    source = source or CONTINUAL_DATASET_FILE
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    kept: list[tuple[str, list[dict[str, str]]]] = []

    entries: list[dict[str, Any]] = []
    if source.is_file():
        with open(source, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    counts["malformed"] += 1

    # Las correcciones de Diego se procesan primero: si hay varias respuestas a la misma pregunta, gana la humana.
    entries.sort(key=lambda e: 0 if e.get("metadata", {}).get("source") == "human_correction" else 1)
    for entry in entries:
        origin = classify_origin(entry)
        counts[f"raw_{origin}"] += 1
        if origin == "legacy_template" and not include_legacy:
            continue
        msgs = _to_messages(entry)
        if msgs is None:
            counts["rejected_format"] += 1
            continue
        key = hashlib.sha1(_first_user(msgs).lower().encode("utf-8")).hexdigest()
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        counts[f"kept_{origin}"] += 1
        kept.append((origin, msgs))

    # Curriculo verificado: ya viene en formato chat (system/user/assistant/tool); solo se deduplica por la primera pregunta.
    for cfile in curriculum_files:
        if not cfile.is_file():
            continue
        for line in cfile.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                msgs = row["messages"]
                key = hashlib.sha1(_first_user(msgs).lower().encode("utf-8")).hexdigest()
            except (json.JSONDecodeError, KeyError, StopIteration, TypeError):
                counts["malformed"] += 1
                continue
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            counts["kept_verified_curriculum"] += 1
            kept.append(("verified_curriculum", msgs))

    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    for origin, msgs in kept:
        row = {"messages": msgs, "origin": origin}
        (evals if _stable_bucket(_first_user(msgs)) < EVAL_FRACTION else train).append(row)

    for path, rows in ((CURATED_TRAIN_FILE, train), (CURATED_EVAL_FILE, evals)):
        assert_safe_write_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    total = len(kept)
    real = counts["kept_interaction"]
    synthetic = counts["kept_verified_template"] + counts["kept_legacy_template"] + counts["kept_verified_curriculum"]
    warnings: list[str] = []
    if total and counts["kept_verified_curriculum"] / total > 0.7:
        warnings.append(f"El currículo de inferencia es el {counts['kept_verified_curriculum'] / total:.0%} del dataset: entrenar así puede hacerla olvidar cómo conversar. "
                        "Conviene mezclar con más conversaciones reales (pendiente: llegar a ≥20 y, idealmente, a cientos).")
    if counts["raw_legacy_template"] and not include_legacy:
        warnings.append(f"{counts['raw_legacy_template']} muestras legado excluidas (respuesta con plantilla fija, sin contenido calculado)")
    if total and real < min_real_interactions:
        warnings.append(f"solo {real} conversaciones reales (minimo recomendado {min_real_interactions}): riesgo de sobreajustar a las plantillas de auto-juego")
    if len(evals) < 5:
        warnings.append("conjunto de evaluacion con menos de 5 muestras: el gate de calidad sera poco fiable")

    ready = total >= min_samples and real >= min_real_interactions and len(evals) >= 5
    report = {
        "total_curated": total,
        "train": len(train),
        "eval": len(evals),
        "real_interactions": real,
        "synthetic_verified": synthetic,
        "curriculum_inference": counts["kept_verified_curriculum"],
        "synthetic_ratio": round(synthetic / total, 3) if total else None,
        "counts": dict(counts),
        "ready_for_training": ready,
        "thresholds": {"min_samples": min_samples, "min_real_interactions": min_real_interactions},
        "warnings": warnings,
        "summary": f"{total} muestras ({real} reales, {synthetic} verificadas sinteticas); train={len(train)} eval={len(evals)}; listo_para_entrenar={ready}",
    }
    assert_safe_write_path(CURATION_REPORT_FILE)
    CURATION_REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
