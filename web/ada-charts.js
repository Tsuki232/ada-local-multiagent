/* ============================================================================
   ADACharts v2.0 — Motor Científico Interactivo Retina HiDPI & 60 FPS
   100% Local / Offline / Compatible con iframe sandbox="allow-scripts"
   Creado para ADA (Arquitecto: Diego López)
   ============================================================================ */
(function (global) {
  "use strict";

  const memoryStore = {};
  const safeStorage = {
    get(key, fallback) {
      try {
        const v = global.localStorage ? global.localStorage.getItem(key) : null;
        return v !== null ? v : (key in memoryStore ? memoryStore[key] : fallback);
      } catch (_) {
        return key in memoryStore ? memoryStore[key] : fallback;
      }
    },
    set(key, val) {
      memoryStore[key] = String(val);
      try {
        if (global.localStorage) global.localStorage.setItem(key, String(val));
      } catch (_) {}
    }
  };

  const registry = [];

  function cssVar(name, fallback) {
    const val = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (val && val.trim()) || fallback;
  }

  function palette() {
    return {
      txt: cssVar("--chart-txt", "#94a3b8"),
      title: cssVar("--chart-title", "#e2e8f0"),
      grid: cssVar("--grid-stroke", "rgba(148,163,184,0.12)"),
      axis: cssVar("--axis-stroke", "rgba(148,163,184,0.28)"),
      cyan: cssVar("--accent-cyan", "#38bdf8"),
      indigo: cssVar("--accent-indigo", "#818cf8"),
      violet: cssVar("--accent-violet", "#a78bfa"),
      emerald: cssVar("--accent-emerald", "#10b981"),
      amber: cssVar("--accent-amber", "#f59e0b"),
      rose: cssVar("--accent-rose", "#f43f5e")
    };
  }

  function fmtNum(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
    const n = Number(v);
    const a = Math.abs(n);
    if (a === 0) return "0";
    if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return (n / 1e3).toFixed(1) + "k";
    if (a >= 100) return n.toFixed(1);
    if (a >= 1) return n.toFixed(2);
    if (a >= 0.01) return n.toFixed(3);
    return n.toExponential(2);
  }

  function niceBounds(minVal, maxVal, tickCount) {
    let mn = Number(minVal);
    let mx = Number(maxVal);
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) { mn = 0; mx = 1; }
    if (Math.abs(mx - mn) < 1e-9) {
      const pad = Math.abs(mn) * 0.15 || 1;
      mn -= pad;
      mx += pad;
    } else {
      const span = mx - mn;
      mn -= span * 0.06;
      mx += span * 0.08;
    }
    const count = Math.max(3, tickCount || 5);
    const rawStep = (mx - mn) / (count - 1);
    const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(rawStep) || 1)));
    const norm = rawStep / mag;
    let step = mag;
    if (norm > 5) step = 10 * mag;
    else if (norm > 2) step = 5 * mag;
    else if (norm > 1) step = 2 * mag;

    const niceMin = Math.floor(mn / step) * step;
    const niceMax = Math.ceil(mx / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.5; v += step) {
      ticks.push(Number(v.toFixed(10)));
      if (ticks.length > 14) break;
    }
    return { min: ticks[0], max: ticks[ticks.length - 1], ticks };
  }

  function setupHiDPICanvas(canvas) {
    const dpr = Math.min(global.devicePixelRatio || 1, 2.5);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(260, Math.floor(rect.width || canvas.clientWidth || 640));
    const h = Math.max(200, Math.floor(rect.height || canvas.clientHeight || 340));
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h, dpr };
  }

  function ensureTooltip(canvas) {
    const stage = canvas.parentElement;
    if (!stage) return null;
    let tip = stage.querySelector(".ada-floating-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "ada-floating-tooltip";
      stage.appendChild(tip);
    }
    return tip;
  }

  function positionTooltip(tip, stage, x, y, html) {
    if (!tip || !stage) return;
    tip.innerHTML = html;
    tip.classList.add("is-visible");
    const sw = stage.clientWidth || 600;
    const sh = stage.clientHeight || 340;
    const tw = tip.offsetWidth || 200;
    const th = tip.offsetHeight || 80;
    let left = x + 14;
    let top = y - th - 12;
    if (left + tw > sw - 10) left = x - tw - 14;
    if (left < 10) left = 10;
    if (top < 10) top = y + 16;
    if (top + th > sh - 8) top = Math.max(8, sh - th - 8);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function hideTooltip(tip) {
    if (tip) tip.classList.remove("is-visible");
  }

  function drawAxes(ctx, box, xBounds, yBounds, xLabel, yLabel, pal) {
    ctx.save();
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;

    yBounds.ticks.forEach((yv) => {
      const py = box.bottom - ((yv - yBounds.min) / (yBounds.max - yBounds.min || 1)) * box.h;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(box.left, py);
      ctx.lineTo(box.right, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.txt;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmtNum(yv), box.left - 8, py);
    });

    if (xBounds && xBounds.ticks) {
      xBounds.ticks.forEach((xv) => {
        const px = box.left + ((xv - xBounds.min) / (xBounds.max - xBounds.min || 1)) * box.w;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(px, box.top);
        ctx.lineTo(px, box.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = pal.txt;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(fmtNum(xv), px, box.bottom + 8);
      });
    }

    ctx.strokeStyle = pal.axis;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(box.left, box.top);
    ctx.lineTo(box.left, box.bottom);
    ctx.lineTo(box.right, box.bottom);
    ctx.stroke();

    if (xLabel) {
      ctx.fillStyle = pal.title;
      ctx.font = "600 11.5px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(xLabel, box.left + box.w / 2, box.bottom + 36);
    }
    if (yLabel) {
      ctx.save();
      ctx.translate(15, box.top + box.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = pal.title;
      ctx.font = "600 11.5px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  function mountInteractiveChart(canvas, renderFrameFn) {
    if (!canvas) return null;
    if (canvas.__adaCleanup) canvas.__adaCleanup();

    const state = { progress: 0, mouse: null, hoverIdx: -1, animId: 0 };
    const tip = ensureTooltip(canvas);
    const stage = canvas.parentElement;

    function draw() {
      const { ctx, w, h } = setupHiDPICanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      renderFrameFn(ctx, w, h, state, tip, stage, palette());
    }

    const t0 = performance.now();
    const duration = 560;
    function step(now) {
      const raw = Math.min(1, (now - t0) / duration);
      state.progress = 1 - Math.pow(1 - raw, 3);
      draw();
      if (raw < 1) {
        state.animId = requestAnimationFrame(step);
      }
    }
    state.animId = requestAnimationFrame(step);

    function onMove(ev) {
      const r = canvas.getBoundingClientRect();
      state.mouse = { x: ev.clientX - r.left, y: ev.clientY - r.top };
      draw();
    }
    function onLeave() {
      state.mouse = null;
      state.hoverIdx = -1;
      hideTooltip(tip);
      draw();
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    let ro = null;
    if (global.ResizeObserver && stage) {
      ro = new ResizeObserver(() => draw());
      ro.observe(stage);
    } else {
      global.addEventListener("resize", draw);
    }

    const handle = { canvas, redraw: draw };
    registry.push(handle);

    canvas.__adaCleanup = function () {
      cancelAnimationFrame(state.animId);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      if (ro) ro.disconnect();
      hideTooltip(tip);
      const idx = registry.indexOf(handle);
      if (idx >= 0) registry.splice(idx, 1);
    };
    return handle;
  }

  /* 1. Scatter Parity + 90% Prediction Confidence Band */
  function scatterParity(canvas, cfg) {
    const yt = (cfg.yTrue || []).map(Number);
    const yp = (cfg.yPred || []).map(Number);
    const band = Math.max(0, Number(cfg.q90HalfWidth || 0));
    const n = Math.min(yt.length, yp.length);
    const all = yt.slice(0, n).concat(yp.slice(0, n));
    if (band > 0 && all.length) {
      all.push(Math.min(...all) - band * 0.5, Math.max(...all) + band * 0.5);
    }
    const b = niceBounds(
      all.length ? Math.min(...all) : 0,
      all.length ? Math.max(...all) : 1,
      6
    );

    return mountInteractiveChart(canvas, (ctx, w, h, state, tip, stage, pal) => {
      const box = { left: 60, top: 22, right: w - 24, bottom: h - 44 };
      box.w = box.right - box.left;
      box.h = box.bottom - box.top;
      drawAxes(ctx, box, b, b, cfg.xLabel || "Valor Real (Test)", cfg.yLabel || "Predicción ADA", pal);

      const sx = (v) => box.left + ((v - b.min) / (b.max - b.min || 1)) * box.w;
      const sy = (v) => box.bottom - ((v - b.min) / (b.max - b.min || 1)) * box.h;

      ctx.save();
      ctx.beginPath();
      ctx.rect(box.left, box.top, box.w, box.h);
      ctx.clip();

      if (band > 0) {
        ctx.fillStyle = "rgba(56, 189, 248, " + (0.11 * state.progress).toFixed(3) + ")";
        ctx.beginPath();
        ctx.moveTo(sx(b.min), sy(b.min + band));
        ctx.lineTo(sx(b.max), sy(b.max + band));
        ctx.lineTo(sx(b.max), sy(b.max - band));
        ctx.lineTo(sx(b.min), sy(b.min - band));
        ctx.closePath();
        ctx.fill();
      }

      ctx.strokeStyle = pal.indigo;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(b.min), sy(b.min));
      ctx.lineTo(sx(b.min + (b.max - b.min) * state.progress), sy(b.min + (b.max - b.min) * state.progress));
      ctx.stroke();
      ctx.setLineDash([]);

      let bestIdx = -1;
      let bestDist = 26;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const px = sx(yt[i]);
        const pyTarget = sy(yp[i]);
        const py = box.bottom - (box.bottom - pyTarget) * state.progress;
        const err = yp[i] - yt[i];
        const inside = band <= 0 || Math.abs(err) <= band;
        pts.push({ i, px, py, yt: yt[i], yp: yp[i], err, inside });
        if (state.mouse) {
          const d = Math.hypot(state.mouse.x - px, state.mouse.y - py);
          if (d < bestDist) { bestDist = d; bestIdx = i; }
        }
      }

      pts.forEach((p) => {
        const isHover = p.i === bestIdx;
        ctx.fillStyle = p.inside ? pal.cyan : pal.amber;
        ctx.beginPath();
        ctx.arc(p.px, p.py, isHover ? 7 : 4.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isHover ? "#ffffff" : "rgba(15,23,42,0.65)";
        ctx.lineWidth = isHover ? 2.2 : 1.2;
        ctx.stroke();
      });

      if (bestIdx >= 0) {
        const p = pts[bestIdx];
        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(p.px, box.top); ctx.lineTo(p.px, box.bottom);
        ctx.moveTo(box.left, p.py); ctx.lineTo(box.right, p.py);
        ctx.stroke();
        ctx.setLineDash([]);

        positionTooltip(
          tip,
          stage,
          p.px,
          p.py,
          `<div style="font-weight:700;margin-bottom:4px;color:${pal.cyan}">Muestra Test #${p.i + 1}</div>` +
          `<div>Real: <strong class="ada-mono">${p.yt.toFixed(4)}</strong></div>` +
          `<div>Predicho: <strong class="ada-mono">${p.yp.toFixed(4)}</strong></div>` +
          `<div>Residuo: <strong class="ada-mono">${(p.yt - p.yp).toFixed(4)}</strong></div>` +
          (band > 0
            ? `<div style="margin-top:3px;color:${p.inside ? pal.emerald : pal.amber}">● ${p.inside ? "Dentro de Banda IC 90%" : "Fuera de Banda IC 90%"}</div>`
            : "")
        );
      } else {
        hideTooltip(tip);
      }
      ctx.restore();
    });
  }

  /* 2. Residuals Diagnostics Plot (y_true - y_pred vs y_pred) */
  function residuals(canvas, cfg) {
    const yt = (cfg.yTrue || []).map(Number);
    const yp = (cfg.yPred || []).map(Number);
    const band = Math.max(0, Number(cfg.q90HalfWidth || 0));
    const n = Math.min(yt.length, yp.length);
    const res = [];
    for (let i = 0; i < n; i++) res.push(yt[i] - yp[i]);
    const xB = niceBounds(yp.length ? Math.min(...yp) : 0, yp.length ? Math.max(...yp) : 1, 6);
    const maxAbs = Math.max(band * 1.15, ...(res.length ? res.map((v) => Math.abs(v)) : [1]));
    const yB = niceBounds(-maxAbs, maxAbs, 5);

    return mountInteractiveChart(canvas, (ctx, w, h, state, tip, stage, pal) => {
      const box = { left: 60, top: 22, right: w - 24, bottom: h - 44 };
      box.w = box.right - box.left;
      box.h = box.bottom - box.top;
      drawAxes(ctx, box, xB, yB, "Predicción (ŷ)", "Residuo (Real - Predicho)", pal);

      const sx = (v) => box.left + ((v - xB.min) / (xB.max - xB.min || 1)) * box.w;
      const sy = (v) => box.bottom - ((v - yB.min) / (yB.max - yB.min || 1)) * box.h;
      const yZero = sy(0);

      ctx.save();
      ctx.beginPath();
      ctx.rect(box.left, box.top, box.w, box.h);
      ctx.clip();

      if (band > 0) {
        const yTop = sy(band);
        const yBot = sy(-band);
        ctx.fillStyle = "rgba(16, 185, 129, 0.09)";
        ctx.fillRect(box.left, yTop, box.w, yBot - yTop);
      }

      ctx.strokeStyle = pal.emerald;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(box.left, yZero);
      ctx.lineTo(box.right, yZero);
      ctx.stroke();
      ctx.setLineDash([]);

      let bestIdx = -1;
      let bestDist = 26;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const px = sx(yp[i]);
        const py = yZero + (sy(res[i]) - yZero) * state.progress;
        const outlier = band > 0 && Math.abs(res[i]) > band;
        ctx.strokeStyle = outlier ? "rgba(244,63,94,0.45)" : "rgba(56,189,248,0.38)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(px, yZero);
        ctx.lineTo(px, py);
        ctx.stroke();

        pts.push({ i, px, py, r: res[i], yp: yp[i], yt: yt[i], outlier });
        if (state.mouse) {
          const d = Math.hypot(state.mouse.x - px, state.mouse.y - py);
          if (d < bestDist) { bestDist = d; bestIdx = i; }
        }
      }

      pts.forEach((p) => {
        const isHover = p.i === bestIdx;
        ctx.fillStyle = p.outlier ? pal.rose : pal.cyan;
        ctx.beginPath();
        ctx.arc(p.px, p.py, isHover ? 7 : 4.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isHover ? "#ffffff" : "rgba(15,23,42,0.65)";
        ctx.lineWidth = isHover ? 2 : 1.1;
        ctx.stroke();
      });

      if (bestIdx >= 0) {
        const p = pts[bestIdx];
        positionTooltip(
          tip,
          stage,
          p.px,
          p.py,
          `<div style="font-weight:700;margin-bottom:4px;color:${p.outlier ? pal.rose : pal.cyan}">Residuo #${p.i + 1}</div>` +
          `<div>Predicho (ŷ): <strong class="ada-mono">${p.yp.toFixed(4)}</strong></div>` +
          `<div>Real (y): <strong class="ada-mono">${p.yt.toFixed(4)}</strong></div>` +
          `<div>Error (y - ŷ): <strong class="ada-mono">${p.r.toFixed(4)}</strong></div>`
        );
      } else {
        hideTooltip(tip);
      }
      ctx.restore();
    });
  }

  /* 3. Model Arena Comparison Chart (Bars + 95% Bootstrap Whiskers) */
  function modelArena(canvas, cfg) {
    const items = cfg.items || [];
    const vals = items.map((d) => Number(d.value || 0));
    const lows = items.map((d, i) => (d.ciLow !== undefined ? Number(d.ciLow) : vals[i]));
    const highs = items.map((d, i) => (d.ciHigh !== undefined ? Number(d.ciHigh) : vals[i]));
    const all = vals.concat(lows, highs, [0]);
    const yB = niceBounds(Math.min(...all), Math.max(...all), 5);

    return mountInteractiveChart(canvas, (ctx, w, h, state, tip, stage, pal) => {
      const box = { left: 60, top: 24, right: w - 24, bottom: h - 46 };
      box.w = box.right - box.left;
      box.h = box.bottom - box.top;
      drawAxes(ctx, box, null, yB, "", cfg.metricLabel || "Métrica Test", pal);

      const sy = (v) => box.bottom - ((v - yB.min) / (yB.max - yB.min || 1)) * box.h;
      const yZero = sy(Math.max(yB.min, Math.min(0, yB.max)));
      const step = box.w / Math.max(items.length, 1);
      const bw = Math.min(64, step * 0.48);

      let hoverIdx = -1;
      const rects = [];

      items.forEach((d, idx) => {
        const cx = box.left + step * (idx + 0.5);
        const x0 = cx - bw / 2;
        const targetY = sy(Number(d.value || 0));
        const curY = yZero + (targetY - yZero) * state.progress;
        const top = Math.min(yZero, curY);
        const bh = Math.max(2, Math.abs(curY - yZero));

        if (state.mouse && state.mouse.x >= cx - step / 2 && state.mouse.x <= cx + step / 2 &&
            state.mouse.y >= box.top && state.mouse.y <= box.bottom) {
          hoverIdx = idx;
        }

        const grad = ctx.createLinearGradient(0, top, 0, top + bh);
        if (d.isWinner) {
          grad.addColorStop(0, pal.cyan);
          grad.addColorStop(1, pal.indigo);
        } else {
          grad.addColorStop(0, "rgba(148, 163, 184, 0.65)");
          grad.addColorStop(1, "rgba(100, 116, 139, 0.35)");
        }

        ctx.fillStyle = grad;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x0, top, bw, bh, 6);
        else ctx.rect(x0, top, bw, bh);
        ctx.fill();

        if (d.isWinner || hoverIdx === idx) {
          ctx.strokeStyle = d.isWinner ? pal.cyan : "#ffffff";
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }

        if (d.ciLow !== undefined && d.ciHigh !== undefined && Math.abs(d.ciHigh - d.ciLow) > 1e-6) {
          const yL = yZero + (sy(d.ciLow) - yZero) * state.progress;
          const yH = yZero + (sy(d.ciHigh) - yZero) * state.progress;
          ctx.strokeStyle = pal.amber;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(cx, yL); ctx.lineTo(cx, yH);
          ctx.moveTo(cx - 6, yL); ctx.lineTo(cx + 6, yL);
          ctx.moveTo(cx - 6, yH); ctx.lineTo(cx + 6, yH);
          ctx.stroke();
        }

        ctx.fillStyle = d.isWinner ? pal.cyan : pal.title;
        ctx.font = (d.isWinner ? "700 " : "600 ") + "11.5px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText((d.isWinner ? "★ " : "") + d.label, cx, box.bottom + 10);

        ctx.fillStyle = pal.txt;
        ctx.font = "600 11px JetBrains Mono, monospace";
        ctx.textBaseline = Number(d.value || 0) >= 0 ? "bottom" : "top";
        ctx.fillText(Number(d.value || 0).toFixed(4), cx, Number(d.value || 0) >= 0 ? top - 5 : top + bh + 5);

        rects.push({ d, cx, top });
      });

      if (hoverIdx >= 0) {
        const r = rects[hoverIdx];
        positionTooltip(
          tip,
          stage,
          r.cx,
          r.top,
          `<div style="font-weight:700;margin-bottom:4px;color:${r.d.isWinner ? pal.cyan : pal.title}">${r.d.label}${r.d.isWinner ? " ★ Mejor Modelo" : ""}</div>` +
          `<div>${cfg.metricLabel || "Métrica"}: <strong class="ada-mono">${Number(r.d.value || 0).toFixed(4)}</strong></div>` +
          (r.d.secondary ? `<div>${r.d.secondary}</div>` : "")
        );
      } else {
        hideTooltip(tip);
      }
    });
  }

  /* 4. Grouped Bars (Classification Real vs Predicho or Multi-Series Categories) */
  function groupedBars(canvas, cfg) {
    const categories = cfg.categories || [];
    const series = cfg.series || [];
    const flat = [0];
    series.forEach((s) => (s.data || []).forEach((v) => flat.push(Number(v || 0))));
    const yB = niceBounds(Math.min(...flat), Math.max(...flat), 5);

    return mountInteractiveChart(canvas, (ctx, w, h, state, tip, stage, pal) => {
      const colors = [pal.cyan, pal.emerald, pal.indigo, pal.amber, pal.rose];
      const box = { left: 58, top: 24, right: w - 24, bottom: h - 46 };
      box.w = box.right - box.left;
      box.h = box.bottom - box.top;
      drawAxes(ctx, box, null, yB, cfg.xLabel || "", cfg.yLabel || "Conteo", pal);

      const sy = (v) => box.bottom - ((v - yB.min) / (yB.max - yB.min || 1)) * box.h;
      const yZero = sy(Math.max(yB.min, Math.min(0, yB.max)));
      const groupStep = box.w / Math.max(categories.length, 1);
      const sCount = Math.max(series.length, 1);
      const bw = Math.min(34, (groupStep * 0.68) / sCount);

      let hoverCat = -1;
      categories.forEach((cat, cIdx) => {
        const gx = box.left + groupStep * (cIdx + 0.5);
        if (state.mouse && state.mouse.x >= gx - groupStep / 2 && state.mouse.x <= gx + groupStep / 2 &&
            state.mouse.y >= box.top && state.mouse.y <= box.bottom) {
          hoverCat = cIdx;
        }
        const totalW = sCount * bw + (sCount - 1) * 4;
        series.forEach((s, sIdx) => {
          const val = Number((s.data || [])[cIdx] || 0);
          const x0 = gx - totalW / 2 + sIdx * (bw + 4);
          const curY = yZero + (sy(val) - yZero) * state.progress;
          const top = Math.min(yZero, curY);
          const bh = Math.max(2, Math.abs(curY - yZero));
          ctx.fillStyle = s.color || colors[sIdx % colors.length];
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(x0, top, bw, bh, 5);
          else ctx.rect(x0, top, bw, bh);
          ctx.fill();
        });

        ctx.fillStyle = hoverCat === cIdx ? pal.cyan : pal.title;
        ctx.font = "600 11.5px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(String(cat).slice(0, 14), gx, box.bottom + 10);
      });

      if (hoverCat >= 0) {
        const gx = box.left + groupStep * (hoverCat + 0.5);
        const rows = series.map((s, sIdx) => {
          const c = s.color || colors[sIdx % colors.length];
          const v = Number((s.data || [])[hoverCat] || 0);
          return `<div><span style="color:${c}">●</span> ${s.name}: <strong class="ada-mono">${fmtNum(v)}</strong></div>`;
        }).join("");
        positionTooltip(
          tip,
          stage,
          gx,
          box.top + 30,
          `<div style="font-weight:700;margin-bottom:4px">${categories[hoverCat]}</div>` + rows
        );
      } else {
        hideTooltip(tip);
      }
    });
  }

  /* 5. Multi-Series Smooth Line & Area Chart (Time-Series, ODEs, Telemetry) */
  function lineSeries(canvas, cfg) {
    const series = cfg.series || [];
    const xs = cfg.xValues || (series[0] ? series[0].data.map((_, i) => i) : [0, 1]);
    const allY = [];
    series.forEach((s) => (s.data || []).forEach((v) => allY.push(Number(v))));
    const xB = niceBounds(Math.min(...xs), Math.max(...xs), 6);
    const yB = niceBounds(allY.length ? Math.min(...allY) : 0, allY.length ? Math.max(...allY) : 1, 5);

    return mountInteractiveChart(canvas, (ctx, w, h, state, tip, stage, pal) => {
      const colors = [pal.cyan, pal.indigo, pal.emerald, pal.amber, pal.rose];
      const box = { left: 60, top: 22, right: w - 24, bottom: h - 44 };
      box.w = box.right - box.left;
      box.h = box.bottom - box.top;
      drawAxes(ctx, box, xB, yB, cfg.xLabel || "X", cfg.yLabel || "Y", pal);

      const sx = (v) => box.left + ((v - xB.min) / (xB.max - xB.min || 1)) * box.w;
      const sy = (v) => box.bottom - ((v - yB.min) / (yB.max - yB.min || 1)) * box.h;

      ctx.save();
      ctx.beginPath();
      ctx.rect(box.left, box.top, box.w * state.progress, box.h);
      ctx.clip();

      series.forEach((s, sIdx) => {
        const col = s.color || colors[sIdx % colors.length];
        const data = s.data || [];
        if (!data.length) return;

        if (s.area !== false && sIdx === 0) {
          const grad = ctx.createLinearGradient(0, box.top, 0, box.bottom);
          grad.addColorStop(0, "rgba(56, 189, 248, 0.24)");
          grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
          ctx.fillStyle = grad;
          ctx.beginPath();
          data.forEach((yv, i) => {
            const px = sx(xs[i]);
            const py = sy(yv);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.lineTo(sx(xs[data.length - 1]), box.bottom);
          ctx.lineTo(sx(xs[0]), box.bottom);
          ctx.closePath();
          ctx.fill();
        }

        ctx.strokeStyle = col;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        data.forEach((yv, i) => {
          const px = sx(xs[i]);
          const py = sy(yv);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      });
      ctx.restore();

      if (state.mouse && xs.length && state.mouse.x >= box.left && state.mouse.x <= box.right) {
        let bestI = 0;
        let bestDx = Infinity;
        xs.forEach((xv, i) => {
          const dx = Math.abs(sx(xv) - state.mouse.x);
          if (dx < bestDx) { bestDx = dx; bestI = i; }
        });
        const hx = sx(xs[bestI]);
        ctx.strokeStyle = "rgba(148,163,184,0.4)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hx, box.top);
        ctx.lineTo(hx, box.bottom);
        ctx.stroke();
        ctx.setLineDash([]);

        const lines = series.map((s, sIdx) => {
          const col = s.color || colors[sIdx % colors.length];
          const yv = Number((s.data || [])[bestI]);
          if (Number.isFinite(yv)) {
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(hx, sy(yv), 5, 0, Math.PI * 2);
            ctx.fill();
          }
          return `<div><span style="color:${col}">●</span> ${s.name}: <strong class="ada-mono">${fmtNum(yv)}</strong></div>`;
        }).join("");

        positionTooltip(
          tip,
          stage,
          hx,
          box.top + 24,
          `<div style="font-weight:700;margin-bottom:4px">${cfg.xLabel || "X"}: ${fmtNum(xs[bestI])}</div>` + lines
        );
      } else {
        hideTooltip(tip);
      }
    });
  }

  /* Theme Toggle & Live Redraw */
  function initThemeToggle(btnId) {
    const saved = safeStorage.get("ada_studio_theme", "dark");
    document.documentElement.setAttribute("data-theme", saved);
    const btn = document.getElementById(btnId || "adaThemeBtn");
    function syncLabel() {
      if (!btn) return;
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      btn.textContent = cur === "dark" ? "☀ Modo Claro" : "☾ Modo Obsidian";
    }
    syncLabel();
    if (btn) {
      btn.addEventListener("click", () => {
        const cur = document.documentElement.getAttribute("data-theme") || "dark";
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        safeStorage.set("ada_studio_theme", next);
        syncLabel();
        registry.forEach((h) => h.redraw());
      });
    }
  }

  /* Sortable + Filterable Data Table */
  function initDataTable(tableId, searchInputId) {
    const tbl = document.getElementById(tableId);
    if (!tbl) return;
    const tbody = tbl.querySelector("tbody");
    const headers = tbl.querySelectorAll("thead th[data-sortable='true']");
    let asc = true;

    headers.forEach((th, colIdx) => {
      th.addEventListener("click", () => {
        if (!tbody) return;
        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((a, b) => {
          const av = (a.children[colIdx] ? a.children[colIdx].textContent : "").trim();
          const bv = (b.children[colIdx] ? b.children[colIdx].textContent : "").trim();
          const an = parseFloat(av.replace(/[^0-9.-]/g, ""));
          const bn = parseFloat(bv.replace(/[^0-9.-]/g, ""));
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return asc ? an - bn : bn - an;
          return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        asc = !asc;
        rows.forEach((r) => tbody.appendChild(r));
      });
    });

    const input = searchInputId ? document.getElementById(searchInputId) : null;
    if (input && tbody) {
      input.addEventListener("input", () => {
        const q = input.value.trim().toLowerCase();
        Array.from(tbody.querySelectorAll("tr")).forEach((tr) => {
          tr.style.display = !q || tr.textContent.toLowerCase().includes(q) ? "" : "none";
        });
      });
    }
  }

  function exportCanvasPNG(canvasId, filename) {
    const cv = document.getElementById(canvasId);
    if (!cv) return;
    const link = document.createElement("a");
    link.download = filename || "ada_chart.png";
    link.href = cv.toDataURL("image/png");
    link.click();
  }

  function exportRowsCSV(rows, filename) {
    if (!rows || !rows.length) return;
    const cols = Object.keys(rows[0]);
    const lines = [cols.join(",")];
    rows.forEach((r) => {
      lines.push(cols.map((c) => JSON.stringify(r[c] !== undefined && r[c] !== null ? r[c] : "")).join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "ada_export.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  global.ADACharts = {
    safeStorage,
    scatterParity,
    residuals,
    modelArena,
    groupedBars,
    lineSeries,
    initThemeToggle,
    initDataTable,
    exportCanvasPNG,
    exportRowsCSV,
    redrawAll: function () { registry.forEach((h) => h.redraw()); }
  };
})(window);
