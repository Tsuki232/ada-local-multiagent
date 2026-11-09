"""Bitacora append-only (JSONL) con hashes SHA-256 para trazabilidad completa de ADA."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ada.config import DEFAULT_SEED, OUTPUTS_DIR, assert_safe_write_path

DEFAULT_LEDGER_PATH = OUTPUTS_DIR / "audit_ledger.jsonl"


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula el hash SHA-256 de un archivo sin modificarlo."""
    target = Path(path).resolve()
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def record_audit_event(
    action: str,
    details: dict[str, Any],
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Registra una operacion reproducible en la bitacora append-only dentro de outputs/."""
    safe_ledger = assert_safe_write_path(ledger_path)
    safe_ledger.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "seed": seed,
        "input_path": str(Path(input_path).resolve()) if input_path else None,
        "input_sha256": compute_sha256(input_path) if input_path and Path(input_path).is_file() else None,
        "output_path": str(Path(output_path).resolve()) if output_path else None,
        "output_sha256": compute_sha256(output_path) if output_path and Path(output_path).is_file() else None,
        "details": details,
    }
    with safe_ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry
