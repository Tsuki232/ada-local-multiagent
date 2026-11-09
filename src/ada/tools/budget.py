"""Medicion del presupuesto de contexto con el tokenizador REAL del modelo (Qwen3). Sin tokenizador devuelve None y los tests se omiten."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ada.config import ModelRoleConfig


@lru_cache(maxsize=1)
def _tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        from ada.training.qlora import local_base_dir, resolve_base_model

        path, _ = resolve_base_model(ModelRoleConfig().orchestrator_hf)
        return AutoTokenizer.from_pretrained(path or str(local_base_dir(ModelRoleConfig().orchestrator_hf))) if path else None
    except Exception:  # noqa: BLE001
        return None


def available() -> bool:
    return _tokenizer() is not None


def count(text: str) -> int:
    tok = _tokenizer()
    if tok is None:
        raise RuntimeError("tokenizador no disponible")
    return len(tok(text, add_special_tokens=False).input_ids)


def rendered_overhead(system_prompt: str, tools: list[dict[str, Any]] | None) -> int:
    """Tokens que el modelo recibe ANTES de la pregunta del usuario (sistema + bloque de herramientas), segun su plantilla de chat."""
    tok = _tokenizer()
    if tok is None:
        raise RuntimeError("tokenizador no disponible")
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "x"}]
    base = len(tok.apply_chat_template([{"role": "system", "content": ""}, msgs[1]], tokenize=True, return_dict=False))
    return len(tok.apply_chat_template(msgs, tools=tools or None, tokenize=True, return_dict=False)) - base


def schema_tokens(schema: dict[str, Any]) -> int:
    return count(json.dumps(schema, ensure_ascii=False))
