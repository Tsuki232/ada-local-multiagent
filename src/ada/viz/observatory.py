"""Observatorio de ADA: visor del modelo REAL (cada pixel = una neurona FFN real) con rendimiento y agente en vivo.

Datos reales: pesos del GGUF (ada.modelmap.extract), telemetria nvidia-smi y diario del agente autonomo.
Representacion (y asi se rotula en la pagina): el barrido por capas al generar es estructural, con velocidad derivada
de los tokens/s medidos; Ollama no expone activaciones por capa, asi que no se muestran activaciones inventadas.
"""

from __future__ import annotations

import base64
import json
import webbrowser
from typing import Any

import numpy as np

from ada.config import OUTPUTS_DIR, assert_safe_write_path
from ada.modelmap.extract import load_model_map


def _quantize_linear(values: np.ndarray, lo: float | None, hi_pct: float) -> tuple[str, float, float]:
    """uint8 en escala LINEAL entre `lo` (o el minimo si es None) y el percentil `hi_pct` -> (base64, lo, hi) en unidades reales.

    Lineal y no logaritmica: la mayoria de neuronas viven en un rango estrecho (~4-8) y un par de atipicas llegan a
    250; en log todo el bulto se aplasta contra el blanco y la estructura desaparece.
    """
    x = values.astype(np.float64)
    lo_v = float(x.min()) if lo is None else float(lo)
    hi_v = float(np.percentile(x, hi_pct))
    q = np.clip((x - lo_v) / max(hi_v - lo_v, 1e-9), 0.0, 1.0) * 255.0
    return base64.b64encode(q.round().astype(np.uint8).tobytes()).decode("ascii"), lo_v, hi_v


def build_observatory(
    output_filename: str = "ada_observatory.html",
    open_in_browser: bool = False,
    bridge_url: str = "http://127.0.0.1:8765",
) -> dict[str, Any]:
    loaded = load_model_map()
    if loaded is None:
        raise FileNotFoundError(
            "Falta el mapa real del modelo. Genéralo una vez (≈3-4 min, lee los pesos del GGUF):\n"
            "  .venv\\Scripts\\python.exe -m ada.modelmap.extract"
        )
    meta, arr = loaded

    infl = arr["ffn_influence"]
    ffn_b64, ffn_lo, ffn_hi = _quantize_linear(infl, 0.0, 99.8)
    hq_vals = arr["head_q_norm"] * arr["head_o_norm"]
    hkv_vals = arr["head_k_norm"] * arr["head_v_norm"]
    q_b64, q_lo, q_hi = _quantize_linear(hq_vals, float(np.percentile(hq_vals, 0.5)), 99.5)
    kv_b64, kv_lo, kv_hi = _quantize_linear(hkv_vals, float(np.percentile(hkv_vals, 0.5)), 99.5)

    # Las neuronas mas influyentes del modelo entero (las "atipicas" de los LLM): dato real, con su capa e indice.
    flat = np.argsort(infl, axis=None)[::-1][:8]
    top_neurons = [{"l": int(i // infl.shape[1]), "n": int(i % infl.shape[1]), "v": float(infl.flat[i])} for i in flat]
    layer_median = np.median(infl, axis=1)
    quiet_layers = [int(i) for i in np.where(layer_median < 0.25 * float(np.median(layer_median)))[0]]

    ui_meta = {
        "model": meta["model"],
        "architecture": meta["architecture"],
        "layers": meta["layers"],
        "d_model": meta["d_model"],
        "d_ff": meta["d_ff"],
        "heads_q": meta["heads_q"],
        "heads_kv": meta["heads_kv"],
        "head_dim": meta["head_dim"],
        "vocab": meta["vocab"],
        "context_native": meta["context_native"],
        "quantization": meta["quantization"],
        "tensor_count": meta["tensor_count"],
        "params_reported": meta["parameter_count_reported"],
        "params_from_tensors": meta["parameter_count_from_tensors"],
        "params_match": meta["parameters_match_reported"],
        "params_by_kind": meta["params_by_kind"],
        "neurons": meta["neurons"],
        "scales": {"ffn": [ffn_lo, ffn_hi], "q": [q_lo, q_hi], "kv": [kv_lo, kv_hi]},
        "top_neurons": top_neurons,
        "quiet_layers": quiet_layers,
        "median_influence": float(np.median(infl)),
        "bridge_url": bridge_url,
    }

    # Los ~930 KB de neuronas van en un archivo de datos aparte: el HTML queda ligero y los visores
    # que rechazan paginas enormes (panel del navegador, preview del harness) siguen abriendolo.
    path = assert_safe_write_path(OUTPUTS_DIR / output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.stem + "_data.js"
    data_path = assert_safe_write_path(path.parent / data_name)
    payload = {"meta": ui_meta, "ffn": ffn_b64, "hq": q_b64, "hkv": kv_b64}
    data_path.write_text("window.ADA_OBS = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    path.write_text(_TEMPLATE.replace("__DATA_SRC__", data_name), encoding="utf-8")
    if open_in_browser:
        webbrowser.open(path.as_uri())
    return {
        "interactive_html_path": f"outputs/{output_filename}",
        "absolute_path": str(path),
        "bytes": path.stat().st_size,
        "ffn_neurons_rendered": int(meta["neurons"]["ffn_total"]),
        "heads_rendered": int(meta["neurons"]["heads_q_total"] + meta["neurons"]["heads_kv_total"]),
        "parameters_match_reported": meta["parameters_match_reported"],
    }


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Observatorio de ADA</title>
<style>
  :root {
    --ink: #0b121a;
    --ink-2: #0f1822;
    --surface: rgba(21, 32, 44, 0.78);
    --surface-solid: #15202c;
    --hairline: #243546;
    --text: #dbe4ec;
    --muted: #8a9db0;
    --signal: #ffb347;
    --ok: #6cc4ae;
    --bad: #e58b7e;
    --r-plate: 16px;
    --r-tile: 9px;
    --font-ui: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, sans-serif;
    --gutter: 16px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background:
      radial-gradient(1100px 600px at 18% -8%, #14283a 0%, transparent 60%),
      linear-gradient(180deg, var(--ink-2), var(--ink));
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 14px;
    line-height: 1.5;
    font-variant-numeric: tabular-nums;
    min-height: 100vh;
  }
  .page { max-width: 1480px; margin: 0 auto; padding: 20px var(--gutter) 40px; }
  .top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; margin-bottom: 18px; }
  .top h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
  .top p { margin: 2px 0 0; color: var(--muted); max-width: 70ch; }
  .link-state { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border: 1px solid var(--hairline); border-radius: 999px; background: var(--surface); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.is-idle { background: var(--ok); }
  .dot.is-busy { background: var(--signal); box-shadow: 0 0 0 4px rgba(255, 179, 71, 0.2); }
  .dot.is-off { background: var(--bad); }

  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: var(--gutter); align-items: start; }
  .col-side { display: grid; gap: var(--gutter); }
  .below { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: var(--gutter); margin-top: var(--gutter); align-items: start; }

  .card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: var(--r-tile);
    padding: 16px;
    backdrop-filter: blur(8px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 10px 30px rgba(0, 0, 0, 0.28);
  }
  .card.plate-card { border-radius: var(--r-plate); padding: 18px; }
  .card h2 { margin: 0 0 4px; font-size: 15px; font-weight: 600; }
  .card .sub { margin: 0 0 12px; color: var(--muted); font-size: 13px; max-width: 72ch; }

  .plate-wrap { position: relative; width: 100%; border-radius: 6px; overflow: hidden; background: var(--ink); line-height: 0; }
  .plate-wrap canvas { width: 100%; height: auto; display: block; }
  .plate-img { image-rendering: pixelated; }
  .plate-ov { position: absolute; inset: 0; pointer-events: none; }
  .scale { display: flex; align-items: center; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 12px; }
  .scale-bar { height: 8px; flex: 1; max-width: 260px; border-radius: 4px; background: linear-gradient(90deg, #0c131b, #13405a, #1f7a8c, #5cc0a8, #c9ecb4, #fff6d6); border: 1px solid var(--hairline); }
  .inspect { margin-top: 10px; min-height: 22px; font-size: 13px; color: var(--text); }
  .inspect em { font-style: normal; color: var(--muted); }
  .tops { margin-top: 8px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; font-size: 12.5px; }
  .tops-list { display: inline-flex; flex-wrap: wrap; gap: 6px; }
  .chip-btn { font: inherit; font-size: 12px; color: var(--text); background: rgba(255, 255, 255, 0.04); border: 1px solid var(--hairline); border-radius: 999px; padding: 2px 10px; cursor: pointer; }
  .chip-btn:hover, .chip-btn:focus-visible { border-color: var(--signal); outline: none; }

  .big { font-size: 40px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
  .unit { font-size: 14px; color: var(--muted); font-weight: 400; margin-left: 6px; }
  .state-line { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-weight: 600; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
  .stat { background: rgba(255, 255, 255, 0.03); border: 1px solid var(--hairline); border-radius: var(--r-tile); padding: 8px 10px; }
  .stat b { display: block; font-size: 18px; font-weight: 600; }
  .stat span { color: var(--muted); font-size: 12px; }
  .spark { width: 100%; height: 56px; display: block; margin-top: 10px; }
  .prompt { margin-top: 10px; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }

  .levels { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }
  .chip { padding: 2px 10px; border: 1px solid var(--hairline); border-radius: 999px; font-size: 12px; color: var(--muted); }
  .journal { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; max-height: 340px; overflow: auto; }
  .journal li { display: grid; grid-template-columns: 16px 1fr; gap: 8px; font-size: 13px; }
  .journal .j-mark { width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; background: var(--muted); }
  .journal .j-ok { background: var(--ok); }
  .journal .j-bad { background: var(--bad); }
  .journal .j-skip { background: var(--muted); }
  .j-detail { color: var(--muted); overflow-wrap: anywhere; }

  .heads { display: block; width: 100%; height: auto; border-radius: 6px; background: var(--ink); }
  .heads-kv { margin-top: 10px; }
  .kinds-bar { display: flex; height: 22px; border-radius: 6px; overflow: hidden; border: 1px solid var(--hairline); margin: 10px 0 12px; }
  .kinds-seg { height: 100%; }
  .legend { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: 13px; }
  .legend li { display: grid; grid-template-columns: 12px 1fr auto; gap: 8px; align-items: center; }
  .sw { width: 12px; height: 12px; border-radius: 3px; }
  .check { margin-top: 12px; padding: 10px 12px; border-radius: var(--r-tile); border: 1px solid var(--hairline); background: rgba(255, 255, 255, 0.03); font-size: 13px; }
  .check.is-ok { border-color: rgba(108, 196, 174, 0.5); }
  .check.is-bad { border-color: rgba(229, 139, 126, 0.6); }
  .honesty { margin-top: var(--gutter); color: var(--muted); font-size: 12.5px; max-width: 110ch; border-top: 1px solid var(--hairline); padding-top: 12px; }
  .heart-mood { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
  .heart-row { display: grid; grid-template-columns: 92px 1fr 38px; gap: 8px; align-items: center; font-size: 13px; margin: 4px 0; }
  .heart-track { height: 8px; border-radius: 4px; background: rgba(255, 255, 255, 0.06); overflow: hidden; }
  .heart-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #1f7a8c, #5cc0a8); transition: width 0.6s ease; }
  .heart-fill.is-hot { background: linear-gradient(90deg, #b8693d, #ffb347); }
  .heart-why { margin-top: 8px; color: var(--muted); font-size: 12.5px; }
  .cam-img { width: 100%; height: auto; display: block; border-radius: 6px; background: var(--ink); margin-bottom: 10px; }
  .cam-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; font-size: 13px; }
  .cam-grid span { color: var(--muted); }
  .cam-note { margin-top: 8px; color: var(--muted); font-size: 12px; }
  .hidden { display: none; }
  .muted { color: var(--muted); }

  @media (max-width: 980px) {
    .layout, .below { grid-template-columns: 1fr; }
    .big { font-size: 34px; }
  }
  @media (prefers-reduced-motion: reduce) { .dot.is-busy { box-shadow: none; } }
</style>
</head>
<body>
<main class="page">
  <header class="top">
    <div>
      <h1>Observatorio de ADA</h1>
      <p id="subtitle" class="sub"></p>
    </div>
    <div class="link-state" role="status"><span id="linkDot" class="dot"></span><span id="linkText">Conectando con el puente…</span></div>
  </header>

  <section class="layout">
    <article class="card plate-card">
      <h2>Las neuronas de ADA, una por píxel</h2>
      <p id="plateSub" class="sub"></p>
      <div class="plate-wrap" id="plateWrap">
        <canvas id="plate" class="plate-img" width="1024" height="680" aria-label="Mapa de neuronas FFN del modelo"></canvas>
        <canvas id="plateOv" class="plate-ov"></canvas>
      </div>
      <div class="scale"><span>menor influencia</span><span class="scale-bar"></span><span>mayor</span></div>
      <div id="inspect" class="inspect"><em>Pasa el cursor sobre la placa para identificar una neurona.</em></div>
      <div class="tops"><span class="muted" id="topsLabel"></span><span id="tops" class="tops-list"></span></div>
    </article>

    <aside class="col-side">
      <article class="card">
        <h2>Rendimiento ahora</h2>
        <div class="state-line"><span id="stateDot" class="dot"></span><span id="stateText">—</span></div>
        <div><span id="tps" class="big">—</span><span class="unit">tokens/s medidos</span></div>
        <div class="stats">
          <div class="stat"><b id="gpu">—</b><span>GPU</span></div>
          <div class="stat"><b id="vram">—</b><span>VRAM</span></div>
          <div class="stat"><b id="temp">—</b><span>Temperatura</span></div>
        </div>
        <canvas id="spark" class="spark" aria-label="Historial de uso de GPU"></canvas>
        <div id="prompt" class="prompt"></div>
        <div id="pass" class="prompt"></div>
      </article>

      <article class="card">
        <h2>Núcleo afectivo</h2>
        <div id="heartBody" class="muted">Sin datos del núcleo todavía.</div>
      </article>

      <article class="card">
        <h2>Agente autónomo</h2>
        <div id="agentBody" class="muted">Sin datos del agente todavía.</div>
      </article>

      <article class="card">
        <h2>Cámara en vivo</h2>
        <img id="camFrame" class="cam-img hidden" alt="Frame analizado por ADA" />
        <div id="camBody" class="muted">Sin sesión de visión. En el CLI escribe /vision; con un celular conectado usa su cámara por ADB.</div>
      </article>
    </aside>
  </section>

  <section class="below">
    <article class="card">
      <h2>Cabezas de atención</h2>
      <p id="headsSub" class="sub"></p>
      <canvas id="headsQ" class="heads" aria-label="Cabezas de consulta por capa"></canvas>
      <canvas id="headsKV" class="heads heads-kv" aria-label="Cabezas de clave y valor por capa"></canvas>
      <div id="headsInspect" class="inspect"><em>Columna = capa, fila = cabeza.</em></div>
    </article>
    <article class="card">
      <h2>Dónde viven los parámetros</h2>
      <p id="paramsSub" class="sub"></p>
      <div id="kindsBar" class="kinds-bar"></div>
      <ul id="legend" class="legend"></ul>
      <div id="check" class="check"></div>
    </article>
  </section>

  <p class="honesty" id="honesty"></p>
</main>

<script src="__DATA_SRC__"></script>
<script>
(function () {
  "use strict";
  var D = window.ADA_OBS;
  if (!D) { document.getElementById("subtitle").textContent = "Falta el archivo de datos __DATA_SRC__ junto a esta página. Vuelve a generarla con build_observatory()."; return; }
  var META = D.meta;
  var FFN_B64 = D.ffn;
  var HQ_B64 = D.hq;
  var HKV_B64 = D.hkv;

  function b64(s) { var bin = atob(s), a = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; }
  function $(id) { return document.getElementById(id); }
  function fmt(n) { return Number(n).toLocaleString("es-ES"); }

  var FFN = b64(FFN_B64), HQ = b64(HQ_B64), HKV = b64(HKV_B64);
  var L = META.layers, DFF = META.d_ff, HQN = META.heads_q, HKN = META.heads_kv;

  /* ---------- mapa de color (datos): tinta -> verde agua -> crema ---------- */
  var STOPS = [[12,19,27],[19,64,90],[31,122,140],[92,192,168],[201,236,180],[255,246,214]];
  var LUT = new Array(256);
  (function () {
    for (var i = 0; i < 256; i++) {
      var t = i / 255 * (STOPS.length - 1), k = Math.min(STOPS.length - 2, Math.floor(t)), f = t - k;
      var a = STOPS[k], b = STOPS[k + 1];
      LUT[i] = [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
    }
  })();

  /* ---------- textos con cifras reales ---------- */
  $("subtitle").textContent = "Modelo " + META.model + " (arquitectura " + META.architecture + ", cuantización " + META.quantization + "): " +
    META.layers + " capas, " + fmt(META.params_reported) + " parámetros en " + META.tensor_count + " tensores. Cada cifra de esta página sale del archivo del modelo o de la GPU.";
  $("plateSub").textContent = "Cada píxel es una de las " + fmt(META.neurons.ffn_total) + " neuronas SwiGLU (" + META.layers + " capas × " + fmt(DFF) +
    "). El color mide cuánta palanca tiene su fila de pesos sobre el flujo residual: norma(gate) × norma(up) × norma(down), calculada de los pesos reales.";
  $("headsSub").textContent = fmt(META.neurons.heads_q_total) + " cabezas de consulta (" + HQN + " por capa) y " + fmt(META.neurons.heads_kv_total) +
    " de clave/valor (" + HKN + " por capa, atención por grupos). Color: magnitud de sus pesos.";
  $("paramsSub").textContent = "Reparto exacto de los " + fmt(META.params_from_tensors) + " parámetros, sumando la forma de cada tensor.";
  $("honesty").textContent = "Qué es real y qué no: el mapa de neuronas y de cabezas son estadísticas estáticas de los pesos del modelo; GPU, VRAM, temperatura y tokens/s son mediciones (nvidia-smi y respuestas de Ollama); el diario del agente es su registro real. " +
    "El barrido por capas mientras genera es una representación: cada token atraviesa las " + L + " capas en orden y se muestra en cámara lenta con la velocidad derivada de los tokens/s medidos. Ollama no expone las activaciones por capa, por eso no se dibujan.";

  /* ---------- placa de neuronas ---------- */
  var BW = 128, BH = Math.ceil(DFF / BW), COLS = 8, ROWS = Math.ceil(L / COLS);
  var PW = COLS * BW, PH = ROWS * BH;
  var plate = $("plate"); plate.width = PW; plate.height = PH;
  (function () {
    var ctx = plate.getContext("2d"), img = ctx.createImageData(PW, PH), d = img.data;
    for (var i = 0; i < d.length; i += 4) { d[i] = 12; d[i + 1] = 19; d[i + 2] = 27; d[i + 3] = 255; }
    for (var l = 0; l < L; l++) {
      var bx = (l % COLS) * BW, by = Math.floor(l / COLS) * BH;
      for (var n = 0; n < DFF; n++) {
        var c = LUT[FFN[l * DFF + n]], x = bx + (n % BW), y = by + Math.floor(n / BW), o = (y * PW + x) * 4;
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
      }
    }
    ctx.putImageData(img, 0, 0);
  })();

  var ov = $("plateOv"), wrap = $("plateWrap");
  var hover = null;          /* {l, n} */
  var wave = null;           /* {p: posicion fraccional de capa, label} */

  function sizeOverlay() {
    var dpr = window.devicePixelRatio || 1, r = wrap.getBoundingClientRect();
    ov.width = Math.max(1, Math.round(r.width * dpr)); ov.height = Math.max(1, Math.round(r.height * dpr));
    drawOverlay();
  }
  function drawOverlay() {
    var ctx = ov.getContext("2d"), W = ov.width, H = ov.height, sx = W / PW, sy = H / PH;
    ctx.clearRect(0, 0, W, H);
    var dpr = window.devicePixelRatio || 1;
    ctx.lineWidth = Math.max(1, dpr * 0.75);
    ctx.strokeStyle = "rgba(219,228,236,0.16)";
    ctx.fillStyle = "rgba(219,228,236,0.75)";
    ctx.font = (11 * dpr) + "px 'Segoe UI', sans-serif";
    ctx.textBaseline = "top";
    for (var l = 0; l < L; l++) {
      var x = (l % COLS) * BW * sx, y = Math.floor(l / COLS) * BH * sy;
      ctx.strokeRect(x + 0.5, y + 0.5, BW * sx - 1, BH * sy - 1);
      ctx.fillText("C" + l, x + 5 * dpr, y + 4 * dpr);
    }
    if (wave) {
      for (var k = 0; k < L; k++) {
        var dist = Math.abs(k - wave.p); dist = Math.min(dist, L - dist);
        var a = Math.max(0, 1 - dist / 2.2) * 0.42;
        if (a > 0.01) {
          ctx.fillStyle = "rgba(255,179,71," + a.toFixed(3) + ")";
          ctx.fillRect((k % COLS) * BW * sx, Math.floor(k / COLS) * BH * sy, BW * sx, BH * sy);
        }
      }
    }
    if (hover) {
      var hx = ((hover.l % COLS) * BW + (hover.n % BW)) * sx, hy = (Math.floor(hover.l / COLS) * BH + Math.floor(hover.n / BW)) * sy;
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5 * dpr;
      ctx.strokeRect(hx - 3 * dpr, hy - 3 * dpr, 6 * dpr + sx, 6 * dpr + sy);
    }
  }
  function relInfluence(v, sc) { return sc[0] + (v / 255) * (sc[1] - sc[0]); }
  function showNeuron(l, n, exact) {
    hover = { l: l, n: n };
    var v = FFN[l * DFF + n], cap = relInfluence(255, META.scales.ffn), val;
    if (exact !== undefined) val = "<b>" + exact.toFixed(2) + "</b> <em>(" + (exact / META.median_influence).toFixed(0) + "× la mediana del modelo, " + META.median_influence.toFixed(2) + ")</em>";
    else if (v === 255) val = "<b>≥ " + cap.toFixed(2) + "</b> <em>(la escala de color se satura; está en la lista de las más influyentes si es extrema)</em>";
    else val = "<b>" + relInfluence(v, META.scales.ffn).toFixed(2) + "</b> <em>(mediana del modelo " + META.median_influence.toFixed(2) + ")</em>";
    $("inspect").innerHTML = "Capa <b>" + l + "</b>, neurona <b>" + fmt(n) + "</b> de " + fmt(DFF) + " · influencia " + val;
  }
  plate.parentElement.addEventListener("mousemove", function (e) {
    var r = wrap.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * PW, py = (e.clientY - r.top) / r.height * PH;
    var col = Math.floor(px / BW), row = Math.floor(py / BH), l = row * COLS + col;
    var n = (Math.floor(py) % BH) * BW + (Math.floor(px) % BW);
    if (l < 0 || l >= L || n >= DFF || col >= COLS) { hover = null; }
    else { showNeuron(l, n); }
    drawOverlay();
  });
  wrap.addEventListener("mouseleave", function () { hover = null; drawOverlay(); });

  (function () {
    var box = $("tops");
    META.top_neurons.forEach(function (t) {
      var b = document.createElement("button"); b.type = "button"; b.className = "chip-btn";
      b.textContent = "C" + t.l + " · #" + fmt(t.n) + " · " + t.v.toFixed(0);
      b.addEventListener("click", function () { showNeuron(t.l, t.n, t.v); drawOverlay(); });
      box.appendChild(b);
    });
    var quiet = META.quiet_layers.length ? " Capas casi mudas (mediana < 25 % de la general): " + META.quiet_layers.map(function (q) { return "C" + q; }).join(", ") + "." : "";
    $("topsLabel").textContent = "Neuronas más influyentes del modelo:";
    $("plateSub").textContent += quiet;
  })();

  /* ---------- cabezas de atencion ---------- */
  function drawHeads(canvasId, data, rows, label, scale) {
    var cv = $(canvasId), cw = 22, chh = rows > 16 ? 8 : 12, w = L * cw, h = rows * chh, dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    var ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    for (var l = 0; l < L; l++) for (var hh = 0; hh < rows; hh++) {
      var c = LUT[data[l * rows + hh]];
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(l * cw + 0.5, hh * chh + 0.5, cw - 1, chh - 1);
    }
    cv.addEventListener("mousemove", function (e) {
      var r = cv.getBoundingClientRect(), l = Math.floor((e.clientX - r.left) / r.width * L), hh = Math.floor((e.clientY - r.top) / r.height * rows);
      if (l >= 0 && l < L && hh >= 0 && hh < rows) {
        $("headsInspect").innerHTML = label + " · capa <b>" + l + "</b>, cabeza <b>" + hh + "</b> · magnitud <b>" + relInfluence(data[l * rows + hh], scale).toFixed(1) + "</b>";
      }
    });
  }
  drawHeads("headsQ", HQ, HQN, "Consulta (Q·O)", META.scales.q);
  drawHeads("headsKV", HKV, HKN, "Clave/valor (K·V)", META.scales.kv);

  /* ---------- parametros por tipo ---------- */
  (function () {
    var K = META.params_by_kind, groups = [
      ["Neuronas FFN (gate, up, down)", ["ffn_gate", "ffn_up", "ffn_down"], "#5cc0a8"],
      ["Atención: consulta y salida", ["attn_q", "attn_output"], "#1f7a8c"],
      ["Atención: clave y valor", ["attn_k", "attn_v"], "#c9ecb4"],
      ["Embeddings de entrada", ["token_embd.weight"], "#8a9db0"],
      ["Capa de salida", ["output.weight"], "#ffb347"]
    ], total = META.params_from_tensors, used = {}, bar = $("kindsBar"), leg = $("legend"), accounted = 0;
    groups.forEach(function (g) {
      var sum = 0; g[1].forEach(function (k) { sum += K[k] || 0; used[k] = 1; });
      accounted += sum;
      var seg = document.createElement("div"); seg.className = "kinds-seg"; seg.style.width = (sum / total * 100) + "%"; seg.style.background = g[2]; seg.title = g[0]; bar.appendChild(seg);
      var li = document.createElement("li");
      li.innerHTML = '<span class="sw" style="background:' + g[2] + '"></span><span>' + g[0] + '</span><span>' + fmt(sum) + ' · ' + (sum / total * 100).toFixed(1) + ' %</span>';
      leg.appendChild(li);
    });
    var rest = total - accounted;
    if (rest > 0) { var li2 = document.createElement("li"); li2.innerHTML = '<span class="sw" style="background:#243546"></span><span>Normas y otros</span><span>' + fmt(rest) + '</span>'; leg.appendChild(li2); }
    var ck = $("check"), ok = META.params_match;
    ck.className = "check " + (ok ? "is-ok" : "is-bad");
    ck.textContent = ok ? "Verificado: la suma de las formas de los " + META.tensor_count + " tensores (" + fmt(total) + ") coincide exactamente con los parámetros que declara el modelo (" + fmt(META.params_reported) + ")."
      : "Atención: la suma de tensores (" + fmt(total) + ") no coincide con lo declarado (" + fmt(META.params_reported) + ").";
  })();

  /* ---------- en vivo: puente 8765 ---------- */
  var hist = [], HMAX = 120, lastCycleTs = null, state = "off", lastTps = 0, sweepAcc = 0, lastFrame = 0, rafOn = false, pulseUntil = 0;
  var SLOW = 0.55;   /* barridos visibles por segundo */

  function setState(s, text) {
    state = s;
    var cls = s === "busy" ? "dot is-busy" : s === "idle" ? "dot is-idle" : "dot is-off";
    $("linkDot").className = cls; $("stateDot").className = cls;
    $("linkText").textContent = s === "off" ? "Puente apagado (se muestran solo datos del modelo)" : "Conectado al puente de ADA";
    $("stateText").textContent = text;
  }
  function drawSpark() {
    var cv = $("spark"), dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
    var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(138,157,176,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H - 0.5); ctx.lineTo(W, H - 0.5); ctx.moveTo(0, 0.5); ctx.lineTo(W, 0.5); ctx.stroke();
    if (hist.length < 2) return;
    ctx.strokeStyle = "#5cc0a8"; ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
    hist.forEach(function (v, i) { var x = i / (HMAX - 1) * W, y = H - (Math.min(100, v) / 100) * (H - 4 * dpr) - 2 * dpr; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
  }
  function frame(t) {
    var dt = lastFrame ? Math.min(0.1, (t - lastFrame) / 1000) : 0; lastFrame = t;
    var active = state === "busy" || t < pulseUntil;
    if (active) {
      sweepAcc = (sweepAcc + dt * SLOW) % 1;
      wave = { p: sweepAcc * L };
      drawOverlay();
      requestAnimationFrame(frame);
    } else { wave = null; rafOn = false; lastFrame = 0; drawOverlay(); }
  }
  function ensureLoop() { if (!rafOn && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) { rafOn = true; requestAnimationFrame(frame); } }

  function renderAgent(a) {
    var box = $("agentBody");
    if (!a || !a.state) { box.className = "muted"; box.textContent = "El agente aún no ha corrido. Inícialo con: python scripts/ada_daemon.py (o /think en el CLI)."; return; }
    var s = a.state, rate = s.cycles ? Math.round(s.verified / s.cycles * 100) + " %" : "—", html = "";
    html += "<div><b>" + fmt(s.verified) + "</b> verificados de <b>" + fmt(s.cycles) + "</b> ciclos · acierto <b>" + rate + "</b></div>";
    html += '<div class="levels">';
    Object.keys(s.curriculum || {}).forEach(function (k) { html += '<span class="chip">' + k + " · nivel " + s.curriculum[k].level + "</span>"; });
    html += "</div>";
    html += '<ul class="journal">';
    (a.journal || []).slice().reverse().forEach(function (j) {
      var isTrain = String(j.status).indexOf("auto") === 0;
      var cls = (j.status === "verified" || j.status === "autotrain_finished") ? "j-ok" : (j.status === "skipped" || j.status === "started" || j.status === "stopped" || j.status === "consolidated" || j.status === "autotrain_wait" || isTrain) ? "j-skip" : "j-bad";
      var label = j.status === "verified" ? "Verificó " + (j.kind || "") + " (nivel " + j.level + ")" : j.status === "skipped" ? "En pausa" : j.status === "consolidated" ? "Curó el dataset" : j.status === "rejected_proposal" ? "Propuesta rechazada (" + (j.kind || "") + ")" : j.status === "failed_verification" ? "No verificó (" + (j.kind || "") + ")" : j.status === "autotrain_wait" ? "Autoentrenamiento: esperando" : j.status === "autotrain_started" ? "Empezó a entrenarse" : j.status === "autotrain_finished" ? "Terminó de entrenarse" : j.status === "autotrain_aborted" ? "Entrenamiento abortado" : j.status === "autodownload_started" ? "Descargando pesos base" : j.status === "autodownload_finished" ? "Pesos base descargados" : j.status;
      if (j.level_change) { label += j.level_change[1] > j.level_change[0] ? " · subió a nivel " + j.level_change[1] : " · bajó a nivel " + j.level_change[1]; }
      var detail = j.detail || (j.methods ? j.methods.join(" + ") : "");
      var said = j.thought ? '<br><i>«' + String(j.thought).slice(0, 220).replace(/</g, "&lt;") + '»</i>' : "";
      if (j.reflection) { label += " · pensó sobre sí misma"; said += '<br><i>«' + String(j.reflection).slice(0, 600).replace(/</g, "&lt;") + '»</i>'; }
      html += '<li><span class="j-mark ' + cls + '"></span><span>' + label + said + '<br><span class="j-detail">' + String(detail).slice(0, 140).replace(/</g, "&lt;") + "</span></span></li>";
    });
    html += "</ul>";
    box.className = ""; box.innerHTML = html;
  }

  var HEART_LABELS = { alegria: "alegría", curiosidad: "curiosidad", frustracion: "frustración", confianza: "confianza", cansancio: "cansancio", conexion: "conexión" };
  function renderHeart(h) {
    var box = $("heartBody");
    if (!h) { box.className = "muted"; box.textContent = "Sin datos del núcleo todavía."; return; }
    var html = '<div class="heart-mood">Se siente ' + h.mood.name + "</div>";
    Object.keys(HEART_LABELS).forEach(function (k) {
      var v = h.feelings[k], hot = (k === "frustracion" || k === "cansancio");
      html += '<div class="heart-row"><span>' + HEART_LABELS[k] + '</span><span class="heart-track"><span class="heart-fill' + (hot ? " is-hot" : "") + '" style="display:block;width:' + Math.round(v * 100) + '%"></span></span><span>' + v.toFixed(2) + "</span></div>";
    });
    var why = (h.recent || []).slice(-2).map(function (r) { return r.reason; }).join("; ");
    if (why) html += '<div class="heart-why">Lo último que le pasó: ' + why.replace(/</g, "&lt;") + ".</div>";
    var bonds = Object.keys(h.bonds || {}).map(function (n) { return n + " (" + h.bonds[n].interactions + " conversaciones, vínculo " + h.bonds[n].warmth.toFixed(2) + ")"; });
    if (bonds.length) html += '<div class="heart-why">Vínculos: ' + bonds.join("; ").replace(/</g, "&lt;") + ".</div>";
    box.className = ""; box.innerHTML = html;
  }

  var camActive = false, camTimer = null;
  function renderCamera(v) {
    var img = $("camFrame"), box = $("camBody");
    if (!v || !v.active) {
      camActive = false; img.className = "cam-img hidden";
      box.className = "muted";
      box.textContent = v && v.error ? "La sesión se detuvo: " + v.error : "Sin sesión de visión. En el CLI escribe /vision; con un celular conectado usa su cámara por ADB.";
      return;
    }
    camActive = true; img.className = "cam-img";
    var k = v.kinematics, html = '<div class="cam-grid">' +
      "<span>Fuente</span><b>" + v.source_label + "</b>" +
      "<span>Velocidad</span><b>" + v.fps + " fps · " + v.frame_size[0] + "×" + v.frame_size[1] + "</b>" +
      "<span>Latencia</span><b>" + (v.read_ms + v.analysis_ms).toFixed(0) + " ms (lectura " + v.read_ms.toFixed(0) + " + análisis " + v.analysis_ms.toFixed(0) + ")</b>";
    if (k) {
      html += "<span>Ángulo θ</span><b>" + k.theta_deg.toFixed(1) + "°</b>" +
        "<span>Velocidad ω</span><b>" + k.omega_rad_s.toFixed(2) + " rad/s</b>" +
        "<span>Frecuencia</span><b>" + (k.frequency_hz ? k.frequency_hz.toFixed(2) + " Hz" : "midiendo…") + "</b>" +
        "<span>Longitud estimada</span><b>" + k.estimated_length_m.toFixed(2) + " m</b>" +
        "<span>Energía</span><b>" + k.energy_total.toFixed(3) + " J</b>";
    } else { html += "<span>Objetivo</span><b>no encontrado en el cuadro</b>"; }
    html += "</div>";
    if (k) html += '<div class="cam-note">' + (k.verified ? "Integración de la EDO: DOP853 y Radau coinciden (Δ=" + k.discrepancy.toExponential(1) + "). " : "") + "Esto verifica el cálculo, no la medición de la cámara. Escala de metros supuesta: 400 px/m.</div>";
    box.className = ""; box.innerHTML = html;
  }
  function refreshFrame() { if (camActive) $("camFrame").src = META.bridge_url + "/api/vision/frame.jpg?t=" + Date.now(); }

  function poll() {
    var ctl = new AbortController(), to = setTimeout(function () { ctl.abort(); }, 1500);
    fetch(META.bridge_url + "/api/status", { signal: ctl.signal, cache: "no-store" }).then(function (r) { return r.json(); }).then(function (d) {
      clearTimeout(to);
      var busy = !!d.is_thinking;
      if (d.tokens_per_sec) lastTps = d.tokens_per_sec;
      setState(busy ? "busy" : "idle", busy ? (d.source === "autonomous" ? "Pensando por su cuenta (agente autónomo)" : "Generando una respuesta") : "En reposo");
      $("tps").textContent = lastTps ? lastTps.toFixed(1) : "—";
      $("gpu").textContent = d.gpu_util_pct != null ? Math.round(d.gpu_util_pct) + " %" : "—";
      $("vram").textContent = d.vram_used_mib != null ? (d.vram_used_mib / 1024).toFixed(1) + " / " + (d.vram_total_mib / 1024).toFixed(1) + " GB" : "—";
      $("temp").textContent = d.temperature_c != null ? Math.round(d.temperature_c) + " °C" : "—";
      $("prompt").textContent = busy && d.current_prompt ? "Consulta: " + d.current_prompt : "";
      if (busy && lastTps) {
        $("pass").textContent = "Cada token cruza las " + L + " capas: a " + lastTps.toFixed(1) + " tok/s son ≈ " + (1000 / lastTps / L).toFixed(2) + " ms por capa. Se muestra en cámara lenta ×" + Math.round(lastTps / SLOW) + ".";
      } else { $("pass").textContent = d.telemetry_source === "unavailable" ? "nvidia-smi no está disponible: sin telemetría de GPU." : ""; }
      if (d.gpu_util_pct != null) { hist.push(d.gpu_util_pct); if (hist.length > HMAX) hist.shift(); drawSpark(); }
      var ag = d.agent || null;
      if (ag && ag.state) {
        if (lastCycleTs !== null && ag.state.last_cycle_ts && ag.state.last_cycle_ts !== lastCycleTs) { pulseUntil = performance.now() + 2200; }
        lastCycleTs = ag.state.last_cycle_ts || lastCycleTs;
      }
      renderAgent(ag);
      renderHeart(d.heart);
      renderCamera(d.vision);
      if (busy || performance.now() < pulseUntil) ensureLoop();
    }).catch(function () {
      clearTimeout(to);
      setState("off", "Sin conexión");
      $("tps").textContent = "—"; $("gpu").textContent = "—"; $("vram").textContent = "—"; $("temp").textContent = "—";
      $("pass").textContent = "Abre ADA (ada.ps1) para activar el puente en " + META.bridge_url + ".";
    });
  }

  window.addEventListener("resize", function () { sizeOverlay(); drawSpark(); });
  if (window.ResizeObserver) new ResizeObserver(function () { sizeOverlay(); drawSpark(); }).observe(wrap);
  sizeOverlay(); setState("off", "Sin conexión");
  /* Sondeo adaptativo: 0.9 s con puente, 4 s sin el (evita inundar la consola de errores de conexion). */
  function schedule() { setTimeout(function () { poll(); schedule(); }, state === "off" ? 4000 : 900); }
  poll(); schedule(); camTimer = setInterval(refreshFrame, 200);
})();
</script>
</body>
</html>
"""
