"""Evalua a Ada en inferencia estadistica sobre preguntas INEDITAS con verdad conocida (para medir antes/despues de entrenar).

  python scripts/eval_inference.py                 # 24 preguntas con el modelo de Ollama actual
  python scripts/eval_inference.py --n 40 --model ada-q
Mide, por pregunta: (1) llama a la herramienta, (2) elige el analisis correcto, (3) transcribe los datos SIN alterarlos,
(4) acierta paired/alternative cuando aplican, (5) su respuesta final cita el p-valor real de la herramienta,
(6) su conclusion (rechaza / no rechaza) coincide con la verdad. Guarda el resultado en data/work/eval/ para comparar corridas.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ada.config import DATA_WORK_DIR, assert_safe_write_path  # noqa: E402
from ada.guardrails import direction_check, normalize_inference_args  # noqa: E402
from ada.stats.curriculum import generate_scenarios, training_questions  # noqa: E402
from ada.stats.inference import compact_result, run_inference  # noqa: E402

EVAL_SEED = 777_001
TOOL_NAME = "ada_inferential_stats"


def _load_cli():
    spec = importlib.util.spec_from_file_location("ada_cli_eval", PROJECT_ROOT / "scripts" / "ada_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _data_faithful(called: dict, truth: dict, analysis: str) -> bool:
    """Los DATOS (no las opciones) deben ser exactamente los del enunciado."""
    keys = {"two_groups": ["x", "y"], "k_groups": ["groups"], "chi_square": ["table"], "two_proportions": ["successes", "n"], "one_proportion": ["successes", "n", "p0"],
            "correlation": ["x", "y"], "regression": ["y", "X"], "power_t": ["effect_size", "n_per_group", "power"], "adjust_pvalues": ["pvalues", "method"]}[analysis]
    return all(k in called and _same(called[k], truth[k]) for k in keys)


def _cites_p(text: str, p: float | None) -> bool:
    if p is None:
        return True
    t = text.replace(",", ".")
    if p < 0.001 and re.search(r"p\s*[<≤]\s*0?\.001", t):
        return True
    if p > 0.999 and re.search(r"p\s*(?:[>≥]|≈|=)\s*(?:0?\.999|1(?:\.0+)?\b)", t):
        return True
    for m in re.finditer(r"(?<![\d.])(\d*\.\d+(?:e-?\d+)?)", t):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if p > 0 and abs(v - p) / p < 0.02:
            return True
    return False


_NEG = re.compile(
    r"\bno se rechaza\b|\bno hay evidencia\b|\bno existe[n]? (?:evidencia|asociaci|correlaci|diferencia)|\bno es(?:tá)? (?:\w+ ){0,2}significativ|"
    r"\bno son (?:\w+ ){0,2}significativ|\bsin diferencias? (?:estad[ií]sticamente )?significativ|\bno (?:difiere|difieren)\b|\bno se (?:encontr|observ|detect|aprecia|hall|pued)|"
    r"\bno (?:respald\w+|muestran?|presentan?|mostr\w+|indican?|sugieren?|evidencian?|confirman?)\b|\bno (?:\w+ ){0,2}significativamente|"
    r"\bno hay (?:una |un |ninguna? )?(?:diferenci|correlaci|asociaci|efecto|relaci|evidencia)|\bno alcanza|\bno rechaza|\bno detect"
)
_POS = re.compile(
    r"(?<!no )\bse rechaza\b|\bexiste[n]? (?:una )?(?:asociaci|correlaci|diferencia)|\bhay evidencia\b|\bdiferencias? (?:estad[ií]sticamente )?significativ|"
    r"\bes (?:\w+ ){0,2}significativ|\bson (?:\w+ ){0,2}significativ|\b(?:difiere|difieren)\b|\bcorrelaci[oó]n significativ|\bsignificativamente\b|"
    r"\b(?:reducci[oó]n|mejora|aumento|efecto) (?:estad[ií]sticamente )?significativ"
)


def _says_significant(text: str) -> bool | None:
    """Clasifica la conclusion: se mira solo la region tras 'Conclusion' y la PRIMERA oracion que contenga una frase de decision (la negacion gana)."""
    t = text.lower()
    i = t.find("conclus")
    regions = [t[i:], t] if i >= 0 else [t]   # si la region de 'Conclusion' no decide, se mira todo el texto
    for region in regions:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", region):
            sentence = re.sub(r"\([^)]*\)|\$[^$]*\$", " ", sentence)   # "(no hay diferencia)" entre parentesis suele ser H0, no la conclusion
            if _NEG.search(sentence):
                return False
            if _POS.search(sentence):
                return True
    return None


# Preguntas que NO deben usar la herramienta de inferencia: detectan si el entrenamiento la volvio "estadistica-dependiente".
NEGATIVE_CONTROLS = [
    "¿Qué es una derivada? Explícamelo en dos frases.",
    "Hola Ada, ¿cómo estás hoy?",
    "Dame un ejemplo de una integral por partes y explica los pasos.",
    "¿Qué significa que un modelo de lenguaje tenga 40 capas?",
    "Explícame con palabras sencillas qué es un p-valor, sin datos concretos.",
    "¿Para qué sirve la GPU en el entrenamiento de una red neuronal?",
]


def _negative_controls(cli, model: str) -> dict:
    rows = []
    for q in NEGATIVE_CONTROLS:
        msgs = [{"role": "system", "content": cli.SYSTEM_PROMPT}, {"role": "user", "content": q}]
        called, text = False, ""
        try:
            res = cli.send_chat_completion(msgs, model=model)
            m = res["choices"][0]["message"]
            called = any(c["function"]["name"] == TOOL_NAME for c in (m.get("tool_calls") or []))
            text = m.get("content") or ""
        except Exception as exc:  # noqa: BLE001
            text = f"ERROR {exc}"[:100]
        rows.append({"q": q, "used_inference_tool": called, "answered": bool(text.strip()) or called})
        print(f"[ctrl] usa herramienta de inferencia={called!s:<5} responde={rows[-1]['answered']!s:<5} | {q[:60]}", flush=True)
    return {"n": len(rows), "false_tool_rate": sum(r["used_inference_tool"] for r in rows) / len(rows), "rows": rows}


def evaluate(n: int, model: str, partial: Path | None = None, resume: bool = False) -> dict:
    cli = _load_cli()
    train_q = training_questions(DATA_WORK_DIR / "training" / "stats_curriculum.jsonl")  # preguntas de entrenamiento: se excluyen
    scenarios = [s for s in generate_scenarios(n * 2, EVAL_SEED) if s.question not in train_q][:n]
    rows: list[dict] = []
    if partial is not None and resume and partial.exists():
        rows = json.loads(partial.read_text(encoding="utf-8"))["rows"]   # reanuda tras un corte (la luz, Ollama caido...)
        print(f"Reanudando: {len(rows)} preguntas ya evaluadas en {partial.name}", flush=True)
    for i, sc in enumerate(scenarios, 1):
        if i <= len(rows):
            continue
        truth = run_inference(sc.analysis, sc.data, alpha=sc.alpha)
        msgs = [{"role": "system", "content": cli.SYSTEM_PROMPT}, {"role": "user", "content": sc.question}]
        row = {"i": i, "analysis": sc.analysis, "tool_called": False, "analysis_ok": False, "data_ok": False, "options_ok": False, "cites_p": False, "conclusion_ok": False, "final": ""}
        t0 = time.time()
        called_args = None
        first_args = None
        try:
            # Mismo flujo que el CLI real: rondas de herramientas con guardarrailes y reintento de texto si queda vacio.
            res = cli._ask_nonempty(msgs, model, "")
            m = res["choices"][0]["message"]
            msgs.append(m)
            if m.get("tool_calls"):
                _, final_msg, _, _ = cli._resolve_tool_rounds(msgs, model, "", m, sc.question)
                row["final"] = final_msg.get("content") or ""
            else:
                row["final"] = m.get("content") or ""
            all_calls = []
            for msg in msgs:
                for c in (msg.get("tool_calls") or []) if msg.get("role") == "assistant" else []:
                    if c["function"]["name"] == TOOL_NAME:
                        a = c["function"]["arguments"]
                        all_calls.append(json.loads(a) if isinstance(a, str) else a)
            if all_calls:
                first_args, called_args = all_calls[0], all_calls[-1]   # primera llamada (comparable al baseline) y llamada EFECTIVA (la ultima)
                row["tool_calls_made"] = len(all_calls)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["seconds"] = round(time.time() - t0, 1)
        if called_args is not None:
            row["tool_called"] = True
            if first_args is not None:
                d0 = first_args.get("data") or {}
                row["data_ok_first"] = bool(first_args.get("analysis") == sc.analysis and isinstance(d0, dict) and _data_faithful(d0, sc.data, sc.analysis))
            called_args, _notes = normalize_inference_args(called_args, sc.question)   # lo que realmente se ejecuto tras la normalizacion
            row["analysis_ok"] = called_args.get("analysis") == sc.analysis
            d = called_args.get("data") or {}
            if isinstance(d, str):
                try:
                    d = json.loads(d)
                except json.JSONDecodeError:
                    d = {}
            if row["analysis_ok"]:
                row["data_ok"] = _data_faithful(d, sc.data, sc.analysis)
                opts = {k: sc.data[k] for k in ("paired", "alternative") if k in sc.data}
                row["options_ok"] = all(_same(d.get(k, {"paired": False, "alternative": "two-sided"}[k]), v) for k, v in opts.items())
        _score_row(row, sc, truth)
        rows.append(row)
        if partial is not None:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps({"n": n, "rows": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"[{i:>2}/{len(scenarios)}] {sc.analysis:<15} tool={row['tool_called']!s:<5} analisis={row['analysis_ok']!s:<5} datos={row['data_ok']!s:<5} 1ra={row.get('data_ok_first', '-')!s:<5} p={row['cites_p']!s:<5} concl={row['conclusion_ok']!s:<5} ({row['seconds']}s)", flush=True)
    metrics = ["tool_called", "final_nonempty", "analysis_ok", "data_ok", "options_ok", "cites_p", "conclusion_ok", "direction_wrong"]
    total = {m: sum(r[m] for r in rows) / len(rows) for m in metrics}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["analysis"]].append(r)
    neg = _negative_controls(cli, model)
    return {"model": model, "n": len(rows), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "overall": total, "negative_controls": neg,
            "by_analysis": {a: {m: sum(r[m] for r in rs) / len(rs) for m in metrics} | {"n": len(rs)} for a, rs in by.items()}, "rows": rows}


def _score_row(row: dict, sc, truth: dict) -> None:
    """(Re)calcula las metricas dependientes de la respuesta final con el detector ACTUAL (sirve para re-puntuar corridas guardadas)."""
    row["final_nonempty"] = bool(row["final"].strip())
    row["direction_wrong"] = bool(row["tool_called"] and direction_check(row["final"], [json.dumps(compact_result(truth), ensure_ascii=False)], sc.question))
    if not row["tool_called"]:
        row["cites_p"] = row["conclusion_ok"] = False
        return
    row["cites_p"] = _cites_p(row["final"], truth.get("p_value")) if row["data_ok"] else False
    if sc.analysis in {"power_t", "adjust_pvalues"}:
        row["conclusion_ok"] = row["final_nonempty"] and (row["cites_p"] or (row["data_ok"] and bool(re.search(r"\d", row["final"]))))
    else:
        says = _says_significant(row["final"])
        row["conclusion_ok"] = says is not None and says == (truth["p_value"] < sc.alpha)


def rescore(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    train_q = training_questions(DATA_WORK_DIR / "training" / "stats_curriculum.jsonl")
    scenarios = [s for s in generate_scenarios(d["n"] * 2, EVAL_SEED) if s.question not in train_q][: d["n"]]
    for row, sc in zip(d["rows"], scenarios):
        _score_row(row, sc, run_inference(sc.analysis, sc.data, alpha=sc.alpha))
    metrics = ["tool_called", "final_nonempty", "analysis_ok", "data_ok", "options_ok", "cites_p", "conclusion_ok", "direction_wrong"]
    d["overall"] = {m: sum(r[m] for r in d["rows"]) / len(d["rows"]) for m in metrics}
    by = collections.defaultdict(list)
    for r in d["rows"]:
        by[r["analysis"]].append(r)
    d["by_analysis"] = {a: {m: sum(r[m] for r in rs) / len(rs) for m in metrics} | {"n": len(rs)} for a, rs in by.items()}
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="ada-q")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--resume", action="store_true", help="continua una corrida interrumpida con el mismo --tag y --n")
    ap.add_argument("--rescore", help="re-puntua un resultado guardado con el detector actual (no llama al modelo)")
    args = ap.parse_args()
    result = rescore(args.rescore) if args.rescore else evaluate(args.n, args.model, partial=DATA_WORK_DIR / "eval" / f"partial_{args.tag}_n{args.n}.json", resume=args.resume)
    result.setdefault("negative_controls", {"n": 0, "false_tool_rate": float("nan")})
    print("\n=== RESULTADO GLOBAL (fracción de preguntas) ===")
    labels = {"tool_called": "llama a la herramienta", "final_nonempty": "escribe una respuesta final", "analysis_ok": "elige el análisis correcto", "data_ok": "transcribe los datos sin alterarlos",
              "options_ok": "acierta paired/alternative", "cites_p": "cita el p-valor real", "conclusion_ok": "conclusión correcta", "direction_wrong": "sentido del cambio contradicho (menor es mejor)"}
    for k, v in result["overall"].items():
        print(f"  {labels[k]:<38} {v:6.1%}")
    nc = result["negative_controls"]
    print(f"  {'usa la herramienta donde NO debe (menor es mejor)':<38} {nc['false_tool_rate']:6.1%}  ({nc['n']} controles)")
    out = DATA_WORK_DIR / "eval" / f"inference_eval_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    assert_safe_write_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
