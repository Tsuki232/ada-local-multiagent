"""Entrenamiento QLoRA local con compuertas de seguridad (RTX 5070, 12 GB).

Garantias de diseno:
  * Por defecto es un ENSAYO (dry-run): valida todo y no toca la GPU. Entrenar exige --execute.
  * Nunca descarga pesos solos (≈30 GB para Qwen3-14B): si faltan, imprime el comando y se detiene.
  * Gate de calidad: el adaptador SOLO se guarda si la perdida de evaluacion mejora de verdad.
  * No modifica el modelo servido en Ollama: deja un adaptador versionado en data/work/training/adapters/.
  * Reserva la GPU con un lock para que el agente autonomo no compita por VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ada.autonomy.curator import CURATED_EVAL_FILE, CURATED_TRAIN_FILE, curate_dataset
from ada.autonomy.guard import AUTONOMY_DIR, TRAINING_LOCK
from ada.autonomy.telemetry import read_gpu_telemetry
from ada.config import DATA_WORK_DIR, ModelRoleConfig, assert_safe_write_path
from ada.training.continuous_learner import TRAINING_DATA_DIR

ADAPTERS_DIR = TRAINING_DATA_DIR / "adapters"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def decide_gate(baseline_loss: float, final_loss: float, min_improvement: float = 0.01) -> dict[str, Any]:
    """Acepta el adaptador solo si la perdida de evaluacion baja al menos `min_improvement` (relativo)."""
    if not (baseline_loss == baseline_loss and final_loss == final_loss) or baseline_loss <= 0:
        return {"accepted": False, "reason": "perdidas no validas (NaN o base <= 0)", "relative_improvement": None}
    rel = (baseline_loss - final_loss) / baseline_loss
    accepted = rel >= min_improvement
    return {
        "accepted": accepted,
        "relative_improvement": round(rel, 5),
        "reason": "mejora suficiente" if accepted else f"mejora {rel:.2%} < umbral {min_improvement:.2%}: adaptador descartado",
    }


BASE_MODELS_DIR = DATA_WORK_DIR / "base_models"


def local_base_dir(repo: str) -> Path:
    return BASE_MODELS_DIR / repo.replace("/", "__")


def download_base_model(repo: str) -> str:
    """Descarga los pesos (reanudable) a data/work/base_models/. SOLO se llama con autorizacion explicita de Diego."""
    from huggingface_hub import snapshot_download

    dest = local_base_dir(repo)
    assert_safe_write_path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return snapshot_download(repo, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])


def resolve_base_model(base: str) -> tuple[str | None, str]:
    """Devuelve (ruta_local, mensaje). Ruta None si los pesos no estan en disco. NUNCA descarga."""
    p = Path(base)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p), "directorio local"
    mine = local_base_dir(base)
    if (mine / "config.json").is_file() and any(mine.glob("*.safetensors")):
        return str(mine), "descargado en data/work/base_models"
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(base, local_files_only=True), "cache de HuggingFace"
    except Exception:
        return None, f"pesos de {base} no encontrados en disco"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _token_lengths(tok: Any, rows: list[dict[str, Any]]) -> list[int]:
    """Longitud en tokens de cada ejemplo YA renderizado con la plantilla de chat real (lo que de verdad entra al modelo)."""
    return [len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)) for r in rows]


def _length_report(lengths: list[int], max_length: int) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(lengths)
    return {"n": int(a.size), "median": int(np.median(a)), "p90": int(np.percentile(a, 90)), "max": int(a.max()),
            "truncated_fraction": float((a > max_length).mean()), "max_length": max_length}


def _smoke(args: argparse.Namespace, base_path: str) -> int:
    """Prueba de VIABILIDAD: carga el modelo en 4 bits y entrena 3 pasos con los ejemplos MAS LARGOS que caben (peor caso de memoria).

    No evalua, no guarda adaptador y no toca Ollama mas alla de descargar su modelo de la VRAM. Mide el pico real de VRAM y los tokens/s.
    """
    import gc

    import psutil
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from ada.tools.ollama_manager import OllamaVRAMManager

    rows = _read_jsonl(CURATED_TRAIN_FILE)
    tok = AutoTokenizer.from_pretrained(base_path)
    lengths = _token_lengths(tok, rows)
    rep = _length_report(lengths, args.max_length)
    print(f"[longitudes] n={rep['n']} mediana={rep['median']} p90={rep['p90']} max={rep['max']} | se recorta con max_length={args.max_length}: {rep['truncated_fraction']:.1%}")
    fit = sorted(((L, r) for L, r in zip(lengths, rows) if L <= args.max_length), key=lambda t: -t[0])[:6]
    worst = [r for _, r in fit]
    print(f"[peor caso] se usan los {len(worst)} ejemplos mas largos que caben: {[L for L, _ in fit]} tokens")

    result: dict[str, Any] = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "base": args.base_model, "max_length": args.max_length, "lengths": rep,
                              "ram_available_gib": round(psutil.virtual_memory().available / 2**30, 1), "stage": "inicio", "fits": False}
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    assert_safe_write_path(TRAINING_LOCK)
    TRAINING_LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": time.time(), "smoke": True}), encoding="utf-8")
    model = trainer = None
    try:
        print("[vram] Descargando modelos de Ollama para liberar la GPU...")
        OllamaVRAMManager().unload_all()
        time.sleep(3)
        tel = read_gpu_telemetry(max_age_s=0.0)
        result["vram_used_mib_before_load"] = tel["vram_used_mib"]
        result["vram_total_mib"] = tel["vram_total_mib"]
        print(f"[gpu] tras descargar Ollama: {tel['vram_used_mib']:.0f}/{tel['vram_total_mib']:.0f} MiB usados por otros procesos (escritorio, navegador...)")

        result["stage"] = "carga del modelo"
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
        result["load_seconds"] = round(time.time() - t0, 1)
        result["vram_allocated_gib_after_load"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        print(f"[carga] {result['load_seconds']} s | VRAM asignada tras cargar: {result['vram_allocated_gib_after_load']} GiB")

        result["stage"] = "entrenamiento"
        torch.cuda.reset_peak_memory_stats()
        sft = SFTConfig(
            output_dir=str(AUTONOMY_DIR / "_smoke_work"), max_steps=3, per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-4, bf16=True,
            gradient_checkpointing=True, max_length=args.max_length, logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
            dataloader_num_workers=0, dataset_num_proc=1, seed=42,
        )
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in worst]), processing_class=tok, peft_config=lora)
        t1 = time.time()
        trainer.train()
        secs = time.time() - t1
        peak_alloc, peak_res = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
        tel2 = read_gpu_telemetry(max_age_s=0.0)
        tokens = sum(L for L, _ in fit[:3])
        result.update(stage="completado", train_seconds=round(secs, 1), peak_allocated_gib=round(peak_alloc, 2), peak_reserved_gib=round(peak_res, 2),
                      vram_used_mib_at_end=tel2["vram_used_mib"], tokens_per_second=round(tokens / secs, 1),
                      losses=[h["loss"] for h in trainer.state.log_history if "loss" in h])
        total_tokens = sum(L for L in lengths if L <= args.max_length)
        result["estimated_epoch_hours"] = round(total_tokens / (tokens / secs) / 3600, 2)
        headroom = tel2["vram_total_mib"] - tel2["vram_used_mib"]
        result["fits"] = True
        result["headroom_mib"] = round(headroom)
        print(f"[OK] CABE. Pico: {peak_alloc:.2f} GiB asignados / {peak_res:.2f} GiB reservados | VRAM en uso al final: {tel2['vram_used_mib']:.0f}/{tel2['vram_total_mib']:.0f} MiB (margen {headroom:.0f} MiB)")
        print(f"[velocidad] {result['tokens_per_second']} tokens/s -> una epoca de {len(rows)} ejemplos ~ {result['estimated_epoch_hours']} h | perdidas: {result['losses']}")
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(stage=f"SIN MEMORIA en: {result['stage']}", error=str(exc)[:300], peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        print(f"[NO CABE] Se quedo sin VRAM durante '{result['stage'].split(': ')[-1]}'. Pico reservado: {result['peak_reserved_gib']} GiB")
        return 5
    finally:
        out = DATA_WORK_DIR / "eval" / f"qlora_smoke_{result['timestamp']}.json"
        assert_safe_write_path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        TRAINING_LOCK.unlink(missing_ok=True)
        shutil.rmtree(AUTONOMY_DIR / "_smoke_work", ignore_errors=True)
        print(f"[informe] {out}")


def run(argv: list[str] | None = None) -> int:
    cfg = ModelRoleConfig()
    ap = argparse.ArgumentParser(description="QLoRA nocturno de ADA con gate de calidad")
    ap.add_argument("--execute", action="store_true", help="entrenar de verdad (sin esto solo se hace un ensayo)")
    ap.add_argument("--base-model", default=cfg.orchestrator_hf, help="ruta local o id de HuggingFace ya descargado; DEBE ser el mismo modelo que sirve Ollama (ada-q = Qwen3-14B), porque un adaptador solo encaja en su base")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048, help="ventana de entrenamiento en tokens; el curriculo tiene ejemplos largos (datos + resultado + respuesta): con 1024 se recortaria la respuesta final")
    ap.add_argument("--min-improvement", type=float, default=0.01)
    ap.add_argument("--force-small-dataset", action="store_true", help="entrenar aunque el curador diga que no esta listo")
    ap.add_argument("--smoke", action="store_true", help="PRUEBA DE VIABILIDAD: carga el modelo en 4 bits y entrena 3 pasos con los ejemplos mas largos; no guarda nada")
    ap.add_argument("--override-disabled", action="store_true", help="ignora data/work/autonomy/TRAINING_DISABLED (bajo tu responsabilidad)")
    ap.add_argument("--download", action="store_true", help="solo descargar los pesos base (lo lanza el entrenador automatico tras la autorizacion de Diego)")
    args = ap.parse_args(argv)

    flag = AUTONOMY_DIR / "TRAINING_DISABLED"
    if (args.execute or args.smoke) and flag.is_file() and not args.override_disabled:
        try:
            why = json.loads(flag.read_text(encoding="utf-8")).get("reason", "desactivado")
        except (OSError, json.JSONDecodeError):
            why = "desactivado"
        print(f"[stop] El entrenamiento esta DESACTIVADO: {why}\n       (para forzarlo: --override-disabled, bajo tu responsabilidad)")
        return 6

    if args.download:
        print(f"[descarga] {args.base_model} -> {local_base_dir(args.base_model)} (reanudable)")
        print(f"[descarga] listo: {download_base_model(args.base_model)}")
        return 0

    report = curate_dataset()
    print(f"[dataset] {report['summary']}")
    for w in report["warnings"]:
        print(f"[aviso] {w}")
    if not report["ready_for_training"] and not (args.force_small_dataset or args.smoke):
        print("[stop] El dataset aun no cumple los minimos. Usa --force-small-dataset solo si sabes lo que haces.")
        return 2

    base_path, how = resolve_base_model(args.base_model)
    tel = read_gpu_telemetry(max_age_s=0.0)
    print(f"[gpu] {tel['vram_used_mib']}/{tel['vram_total_mib']} MiB usados ({tel['source']})")
    if base_path is None:
        print(f"[stop] {how}.\n       Descarga manual (≈30 GB, requiere tu aprobación):\n"
              f"       .venv\\Scripts\\hf download {args.base_model}")
        return 3
    print(f"[base] {args.base_model} -> {base_path} ({how})")

    if args.smoke:
        return _smoke(args, base_path)

    if not args.execute:
        print(f"[ensayo] Todo listo: train={report['train']} eval={report['eval']} epochs={args.epochs}. Añade --execute para entrenar.")
        return 0

    # ------------------------------------------------------------------ entrenamiento real
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from ada.tools.ollama_manager import OllamaVRAMManager

    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    assert_safe_write_path(TRAINING_LOCK)
    TRAINING_LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}), encoding="utf-8")
    try:
        print("[vram] Descargando modelos de Ollama para liberar la GPU...")
        OllamaVRAMManager().unload_all()

        tok = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
        def _fitting(rows: list[dict[str, Any]], label: str) -> Dataset:
            """Descarta (y avisa) los ejemplos que no caben en la ventana: recortarlos ensenaria a cortar la respuesta final."""
            lens = _token_lengths(tok, rows)
            kept = [r for L, r in zip(lens, rows) if L <= args.max_length]
            if len(kept) < len(rows):
                print(f"[datos] {label}: {len(rows) - len(kept)} de {len(rows)} ejemplos superan {args.max_length} tokens y se DESCARTAN (no se recortan).")
            return Dataset.from_list([{"messages": r["messages"]} for r in kept])

        train_ds = _fitting(_read_jsonl(CURATED_TRAIN_FILE), "entrenamiento")
        eval_ds = _fitting(_read_jsonl(CURATED_EVAL_FILE), "evaluacion")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ADAPTERS_DIR / stamp
        assert_safe_write_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sft = SFTConfig(
            output_dir=str(out_dir / "_work"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            gradient_checkpointing=True,
            max_length=args.max_length,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
            dataset_num_proc=1,
            seed=42,
        )
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tok, peft_config=lora)

        baseline = float(trainer.evaluate()["eval_loss"])
        print(f"[eval] perdida base: {baseline:.4f}")
        trainer.train()
        final = float(trainer.evaluate()["eval_loss"])
        print(f"[eval] perdida final: {final:.4f}")

        gate = decide_gate(baseline, final, args.min_improvement)
        summary = {
            "timestamp": stamp,
            "base_model": args.base_model,
            "baseline_eval_loss": baseline,
            "final_eval_loss": final,
            "gate": gate,
            "dataset": {k: report[k] for k in ("train", "eval", "real_interactions", "synthetic_verified")},
            "lora": {"r": 16, "alpha": 32, "targets": LORA_TARGETS},
        }
        if gate["accepted"]:
            trainer.model.save_pretrained(out_dir / "adapter")
            tok.save_pretrained(out_dir / "adapter")
            print(f"[OK] Adaptador aceptado y guardado en {out_dir / 'adapter'}")
        else:
            print(f"[descartado] {gate['reason']}")
        (out_dir / "training_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if gate["accepted"] else 4
    finally:
        TRAINING_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(run())
