"""Mapa REAL del modelo Ada Q: arquitectura leida del GGUF y magnitud de cada neurona calculada de los pesos.

Todo lo que se guarda sale de los pesos desquantizados del archivo GGUF que sirve Ollama, no se inventa nada:
  * ffn_{gate,up,down}_norm[L, d_ff]  norma L2 de la fila (gate/up) o columna (down) de CADA neurona SwiGLU.
  * ffn_influence[L, d_ff]            gate*up*down: cuanta "palanca" tiene esa neurona sobre el flujo residual.
  * head_{q,k,v,o}_norm[L, H]         norma de los pesos de cada cabeza de atencion.
  * attn_norm_mean[L], ffn_norm_mean[L]  ganancia media de las RMSNorm.
OJO: son estadisticas ESTATICAS de los pesos. No son activaciones: Ollama no expone activaciones por capa.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ada.config import DATA_WORK_DIR, ModelRoleConfig, assert_safe_write_path

MAP_DIR = DATA_WORK_DIR / "model_map"
MAP_NPZ = MAP_DIR / "model_map.npz"
MAP_META = MAP_DIR / "model_meta.json"


def fetch_architecture(model: str = "ada-q") -> dict[str, Any]:
    """Lee arquitectura y lista completa de tensores desde Ollama (/api/show verbose)."""
    host = ModelRoleConfig().ollama_api_host.rstrip("/")
    req = urllib.request.Request(
        f"{host}/api/show", data=json.dumps({"model": model, "verbose": True}).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    mi = data["model_info"]
    arch = mi["general.architecture"]
    a = {k.split(".", 2)[-1] if k.startswith(arch + ".") else k: v for k, v in mi.items() if isinstance(v, (int, float, str))}
    tensors = [{"name": t["name"], "type": t["type"], "shape": [int(x) for x in t["shape"]]} for t in data["tensors"]]
    m = re.search(r"^FROM\s+(.+)$", data.get("modelfile", ""), re.M)
    return {
        "model": model,
        "architecture": arch,
        "layers": int(mi[f"{arch}.block_count"]),
        "d_model": int(mi[f"{arch}.embedding_length"]),
        "d_ff": int(mi[f"{arch}.feed_forward_length"]),
        "heads_q": int(mi[f"{arch}.attention.head_count"]),
        "heads_kv": int(mi[f"{arch}.attention.head_count_kv"]),
        "head_dim": int(mi[f"{arch}.attention.key_length"]),
        "context_native": int(mi[f"{arch}.context_length"]),
        "vocab": int(len(mi.get("tokenizer.ggml.tokens", [])) or next(t["shape"][1] for t in tensors if t["name"] == "token_embd.weight")),
        "parameter_count_reported": int(mi["general.parameter_count"]),
        "quantization": data.get("details", {}).get("quantization_level"),
        "gguf_path": m.group(1).strip() if m else None,
        "tensors": tensors,
        "extra": a,
    }


def _deq(t: Any) -> np.ndarray:
    from gguf.quants import dequantize

    if t.tensor_type.name in ("F32", "F16", "BF16"):
        return np.asarray(t.data, dtype=np.float32)
    return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)


def extract_model_map(model: str = "ada-q", progress: Callable[[int, int, float], None] | None = None) -> dict[str, Any]:
    from gguf import GGUFReader

    arch = fetch_architecture(model)
    L, H, KV, hd, dff = arch["layers"], arch["heads_q"], arch["heads_kv"], arch["head_dim"], arch["d_ff"]
    reader = GGUFReader(arch["gguf_path"])
    by = {t.name: t for t in reader.tensors}

    out = {
        "ffn_gate_norm": np.zeros((L, dff), np.float32),
        "ffn_up_norm": np.zeros((L, dff), np.float32),
        "ffn_down_norm": np.zeros((L, dff), np.float32),
        "head_q_norm": np.zeros((L, H), np.float32),
        "head_k_norm": np.zeros((L, KV), np.float32),
        "head_v_norm": np.zeros((L, KV), np.float32),
        "head_o_norm": np.zeros((L, H), np.float32),
        "attn_norm_mean": np.zeros(L, np.float32),
        "ffn_norm_mean": np.zeros(L, np.float32),
    }
    t_start = time.time()
    for i in range(L):
        p = f"blk.{i}."
        gate = _deq(by[p + "ffn_gate.weight"])  # (d_ff, d_model): fila = neurona
        out["ffn_gate_norm"][i] = np.linalg.norm(gate, axis=1)
        del gate
        up = _deq(by[p + "ffn_up.weight"])
        out["ffn_up_norm"][i] = np.linalg.norm(up, axis=1)
        del up
        down = _deq(by[p + "ffn_down.weight"])  # (d_model, d_ff): columna = neurona
        out["ffn_down_norm"][i] = np.linalg.norm(down, axis=0)
        del down
        q = _deq(by[p + "attn_q.weight"])  # (H*hd, d_model): filas agrupadas por cabeza
        out["head_q_norm"][i] = np.linalg.norm(q.reshape(H, hd * q.shape[1]), axis=1)
        del q
        k = _deq(by[p + "attn_k.weight"])
        out["head_k_norm"][i] = np.linalg.norm(k.reshape(KV, hd * k.shape[1]), axis=1)
        v = _deq(by[p + "attn_v.weight"])
        out["head_v_norm"][i] = np.linalg.norm(v.reshape(KV, hd * v.shape[1]), axis=1)
        o = _deq(by[p + "attn_output.weight"])  # (d_model, H*hd): columnas agrupadas por cabeza
        out["head_o_norm"][i] = np.linalg.norm(o.reshape(o.shape[0], H, hd).transpose(1, 0, 2).reshape(H, -1), axis=1)
        out["attn_norm_mean"][i] = float(_deq(by[p + "attn_norm.weight"]).mean())
        out["ffn_norm_mean"][i] = float(_deq(by[p + "ffn_norm.weight"]).mean())
        if progress:
            progress(i + 1, L, time.time() - t_start)

    out["ffn_influence"] = out["ffn_gate_norm"] * out["ffn_up_norm"] * out["ffn_down_norm"]

    # Contabilidad exacta de parametros por tipo de tensor: valida que "mapeamos todo el modelo".
    total = 0
    per_kind: dict[str, int] = {}
    per_layer = [0] * L
    for t in arch["tensors"]:
        n = int(np.prod(t["shape"], dtype=np.int64))
        total += n
        m = re.match(r"blk\.(\d+)\.(\w+)", t["name"])
        if m:
            per_layer[int(m.group(1))] += n
            per_kind[m.group(2)] = per_kind.get(m.group(2), 0) + n
        else:
            per_kind[t["name"]] = per_kind.get(t["name"], 0) + n

    meta = {k: v for k, v in arch.items() if k != "tensors"}
    meta.update(
        tensor_count=len(arch["tensors"]),
        parameter_count_from_tensors=total,
        parameters_match_reported=(total == arch["parameter_count_reported"]),
        params_per_layer=per_layer,
        params_by_kind=per_kind,
        neurons={"ffn_total": L * dff, "heads_q_total": L * H, "heads_kv_total": L * KV},
        generated_at=time.time(),
        extraction_seconds=round(time.time() - t_start, 1),
        note="Estadisticas estaticas de los pesos desquantizados del GGUF; no son activaciones.",
    )

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    assert_safe_write_path(MAP_NPZ)
    np.savez_compressed(MAP_NPZ, **{k: v.astype(np.float16) if v.ndim == 2 else v for k, v in out.items()})
    assert_safe_write_path(MAP_META)
    MAP_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_model_map() -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """Devuelve (meta, arrays) si el mapa ya fue extraido, o None."""
    if not (MAP_NPZ.is_file() and MAP_META.is_file()):
        return None
    meta = json.loads(MAP_META.read_text(encoding="utf-8"))
    with np.load(MAP_NPZ) as z:
        return meta, {k: z[k].astype(np.float32) for k in z.files}


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    meta = extract_model_map(progress=lambda i, n, s: print(f"  capa {i}/{n}  ({s:.0f}s)", flush=True))
    print(json.dumps({k: meta[k] for k in ("layers", "d_ff", "heads_q", "heads_kv", "tensor_count", "parameter_count_from_tensors", "parameters_match_reported", "neurons", "extraction_seconds")}, indent=2))
