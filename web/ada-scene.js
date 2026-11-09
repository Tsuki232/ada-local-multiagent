/* ============================================================================
   ADAScene v1.0 — Motor Procedural 2D de Biomas, Actores Articulados y Física 60 FPS
   100% Local / Offline — Diseñado para potenciar modelos locales de 14B (Ada Q)
   Creado para ADA (Arquitecto: Diego López)
   ============================================================================ */
(function (global) {
  "use strict";

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* --- 1. Biomas, Cielos y Entornos con Parallax --- */
  function sky(ctx, w, h, opts) {
    const o = opts || {};
    const timeOfDay = o.dusk !== undefined ? clamp(Number(o.dusk), 0, 1) : 0.35;
    const t = Number(o.t || 0);
    const g = ctx.createLinearGradient(0, 0, 0, h);
    if (timeOfDay < 0.45) {
      g.addColorStop(0, "#0ea5e9");
      g.addColorStop(0.58, "#7dd3fc");
      g.addColorStop(1, "#fef3c7");
    } else if (timeOfDay < 0.78) {
      g.addColorStop(0, "#0f172a");
      g.addColorStop(0.48, "#312e81");
      g.addColorStop(0.8, "#db2777");
      g.addColorStop(1, "#fb923c");
    } else {
      g.addColorStop(0, "#030712");
      g.addColorStop(0.55, "#0f172a");
      g.addColorStop(1, "#1e293b");
    }
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    // Sol o Luna con halo
    const sx = w * (o.sunX !== undefined ? o.sunX : 0.78);
    const sy = h * (0.16 + timeOfDay * 0.28);
    const halo = ctx.createRadialGradient(sx, sy, 6, sx, sy, 95);
    halo.addColorStop(0, timeOfDay > 0.75 ? "rgba(241,245,249,0.95)" : "rgba(254,240,138,0.98)");
    halo.addColorStop(0.35, timeOfDay > 0.75 ? "rgba(148,163,184,0.22)" : "rgba(251,146,60,0.34)");
    halo.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(sx, sy, 95, 0, Math.PI * 2); ctx.fill();

    // Estrellas si es atardecer/noche
    if (timeOfDay > 0.5) {
      const alpha = clamp((timeOfDay - 0.48) * 1.6, 0, 0.9);
      ctx.fillStyle = "rgba(255,255,255," + alpha.toFixed(2) + ")";
      for (let i = 0; i < 40; i++) {
        const x = (i * 137) % w;
        const y = (i * 59) % (h * 0.5);
        const r = 0.8 + (Math.sin(t * 2.6 + i) + 1) * 0.55;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Nubes suaves en movimiento
    const cloudCount = o.clouds !== undefined ? o.clouds : 4;
    const spd = (o.windSpeed !== undefined ? o.windSpeed : 18);
    for (let c = 0; c < cloudCount; c++) {
      const sc = 0.7 + (c % 3) * 0.24;
      const cx = (((c * 290 + t * spd * (0.6 + c * 0.15)) % (w + 240)) + (w + 240)) % (w + 240) - 120;
      const cy = h * (0.12 + (c % 3) * 0.08);
      ctx.fillStyle = timeOfDay > 0.65 ? "rgba(148,163,184,0.16)" : "rgba(255,255,255,0.68)";
      ctx.beginPath();
      ctx.arc(cx, cy, 22 * sc, 0, Math.PI * 2);
      ctx.arc(cx + 22 * sc, cy - 8 * sc, 28 * sc, 0, Math.PI * 2);
      ctx.arc(cx + 46 * sc, cy, 20 * sc, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function hillsAndMeadow(ctx, w, h, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const horizon = h * (o.horizon !== undefined ? o.horizon : 0.62);
    const dusk = o.dusk !== undefined ? Number(o.dusk) : 0.35;

    // Montañas lejanas
    ctx.fillStyle = dusk > 0.65 ? "#1e293b" : "#475569";
    ctx.beginPath();
    ctx.moveTo(0, horizon);
    for (let x = 0; x <= w; x += 24) {
      const y = horizon - 45 - Math.sin(x * 0.008 + 1.2) * 32 - Math.cos(x * 0.019) * 18;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fill();

    // Colinas medias
    ctx.fillStyle = dusk > 0.65 ? "#064e3b" : "#15803d";
    ctx.beginPath();
    ctx.moveTo(0, horizon + 12);
    for (let x = 0; x <= w; x += 20) {
      const y = horizon - 12 - Math.sin(x * 0.012 + 2.5) * 18;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fill();

    // Pradera principal con gradiente
    const mg = ctx.createLinearGradient(0, horizon, 0, h);
    mg.addColorStop(0, dusk > 0.65 ? "#065f46" : "#22c55e");
    mg.addColorStop(1, dusk > 0.65 ? "#022c22" : "#14532d");
    ctx.fillStyle = mg;
    ctx.fillRect(0, horizon, w, h - horizon);

    // Briznas de hierba mecidas por el viento
    const blades = o.grassCount !== undefined ? o.grassCount : 48;
    ctx.strokeStyle = dusk > 0.65 ? "rgba(16,185,129,0.35)" : "rgba(187,247,208,0.52)";
    ctx.lineWidth = 1.6;
    for (let i = 0; i < blades; i++) {
      const gx = (i * 97) % w;
      const gy = horizon + 14 + ((i * 61) % Math.max(20, h - horizon - 20));
      const sway = Math.sin(t * 3.2 + gx * 0.05) * 5;
      ctx.beginPath();
      ctx.moveTo(gx, gy);
      ctx.quadraticCurveTo(gx + sway * 0.5, gy - 8, gx + sway, gy - 14);
      ctx.stroke();
    }
    return horizon;
  }

  function forestTrees(ctx, w, h, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const count = o.count !== undefined ? o.count : 8;
    const baseY = h * (o.horizon !== undefined ? o.horizon : 0.63);
    const dusk = o.dusk !== undefined ? Number(o.dusk) : 0.35;

    for (let i = 0; i < count; i++) {
      const tx = ((i + 0.5) * (w / count)) + Math.sin(i * 4.1) * 28;
      const sc = 0.78 + (i % 3) * 0.22;
      const ty = baseY + (i % 2) * 10;
      const sway = Math.sin(t * 1.8 + i) * 3 * sc;

      // Sombra en el suelo
      ctx.fillStyle = "rgba(0,0,0,0.22)";
      ctx.beginPath(); ctx.ellipse(tx, ty + 4, 24 * sc, 6 * sc, 0, 0, Math.PI * 2); ctx.fill();

      // Tronco
      ctx.fillStyle = dusk > 0.65 ? "#3f271d" : "#78350f";
      ctx.fillRect(tx - 5 * sc, ty - 34 * sc, 10 * sc, 36 * sc);

      // Capas de pino / follaje
      const greens = dusk > 0.65 ? ["#064e3b", "#047857", "#059669"] : ["#14532d", "#15803d", "#16a34a"];
      for (let layer = 0; layer < 3; layer++) {
        const lw = (34 - layer * 7) * sc;
        const ly = ty - (28 + layer * 22) * sc;
        ctx.fillStyle = greens[layer];
        ctx.beginPath();
        ctx.moveTo(tx + sway * (layer + 1) * 0.4, ly - 32 * sc);
        ctx.lineTo(tx - lw, ly);
        ctx.lineTo(tx + lw, ly);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  function citySkyline(ctx, w, h, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const scroll = Number(o.scrollSpeed || 45) * t;
    const dusk = o.dusk !== undefined ? Number(o.dusk) : 0.65;
    const count = 14;
    const step = 120;
    const span = count * step;

    for (let i = 0; i < count; i++) {
      const bw = 74 + (i * 29) % 38;
      const bh = 150 + (i * 67) % 180;
      const bx = (((i * step - scroll) % span) + span) % span - 120;
      const by = h - bh;
      ctx.fillStyle = dusk > 0.5 ? "#0f172a" : "#1e293b";
      ctx.fillRect(bx, by, bw, bh);

      if (i % 5 === 2) {
        ctx.fillRect(bx + bw * 0.25, by - 26, bw * 0.5, 26);
        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(bx + bw * 0.5, by - 26); ctx.lineTo(bx + bw * 0.5, by - 70); ctx.stroke();
      }

      const cols = 4;
      const rows = Math.floor((bh - 20) / 18);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (((i * 13 + r * 7 + c * 5) % 9) > 3) {
            ctx.fillStyle = (r + c) % 5 === 0 ? "rgba(56,189,248,0.72)" : "rgba(253,224,71,0.78)";
            ctx.fillRect(bx + 10 + c * ((bw - 16) / cols), by + 12 + r * 17, 8, 9);
          }
        }
      }
    }
  }

  /* --- 2. Actores Vectoriales Articulados con Cinemática Real --- */

  function fox(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const spd = Number(o.speed || 1.2);
    const phase = t * 11 * spd + Number(o.phase || 0);
    const stride = Math.sin(phase);
    const strideOpp = Math.sin(phase + Math.PI);
    const bob = Math.abs(Math.cos(phase)) * 4;

    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);

    // Sombra dinámica
    ctx.fillStyle = "rgba(0,0,0,0.26)";
    ctx.beginPath(); ctx.ellipse(0, 18, 30 - bob * 0.5, 6, 0, 0, Math.PI * 2); ctx.fill();

    ctx.translate(0, -bob);

    // Cola esponjosa articulada con punta blanca
    const tailWag = Math.sin(phase * 0.8) * 0.22;
    ctx.save();
    ctx.translate(-24, -4);
    ctx.rotate(-0.25 + tailWag);
    ctx.fillStyle = "#ea580c";
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(-22, -14, -40, -4);
    ctx.quadraticCurveTo(-24, 14, 0, 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#fff7ed";
    ctx.beginPath();
    ctx.arc(-36, -3, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Patas traseras y delanteras (par lejano oscuro)
    function drawLeg(lx, ly, ang, color) {
      ctx.save();
      ctx.translate(lx, ly);
      ctx.rotate(ang);
      ctx.fillStyle = color;
      ctx.fillRect(-3, 0, 6, 15);
      ctx.fillStyle = "#1e1b18";
      ctx.fillRect(-3, 12, 7, 5);
      ctx.restore();
    }
    drawLeg(-14, 4, strideOpp * 0.55, "#9a3412");
    drawLeg(14, 4, stride * 0.55, "#9a3412");

    // Torso aerodinámico del zorro
    const bodyGrad = ctx.createLinearGradient(-24, -12, 22, 10);
    bodyGrad.addColorStop(0, "#c2410c");
    bodyGrad.addColorStop(0.55, "#f97316");
    bodyGrad.addColorStop(1, "#fb923c");
    ctx.fillStyle = bodyGrad;
    ctx.beginPath();
    ctx.ellipse(0, -1, 24, 10.5, -0.05, 0, Math.PI * 2);
    ctx.fill();

    // Pecho blanco
    ctx.fillStyle = "#ffedd5";
    ctx.beginPath();
    ctx.ellipse(14, 3, 10, 5.5, -0.2, 0, Math.PI * 2);
    ctx.fill();

    // Patas cercanas
    drawLeg(-12, 5, stride * 0.62, "#ea580c");
    drawLeg(16, 5, strideOpp * 0.62, "#ea580c");

    // Cabeza, orejas puntiagudas y hocico
    ctx.save();
    ctx.translate(23, -8);
    ctx.rotate(stride * 0.06);

    // Orejas
    ctx.fillStyle = "#c2410c";
    ctx.beginPath(); ctx.moveTo(-4, -5); ctx.lineTo(-9, -19); ctx.lineTo(1, -9); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#ea580c";
    ctx.beginPath(); ctx.moveTo(1, -5); ctx.lineTo(0, -20); ctx.lineTo(8, -8); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#1c1917";
    ctx.beginPath(); ctx.moveTo(1, -13); ctx.lineTo(0, -20); ctx.lineTo(5, -12); ctx.closePath(); ctx.fill();

    // Cráneo y hocico alargado
    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.moveTo(-6, -7);
    ctx.quadraticCurveTo(6, -10, 19, -1);
    ctx.quadraticCurveTo(8, 7, -6, 5);
    ctx.closePath();
    ctx.fill();

    // Mejilla blanca y trufa negra
    ctx.fillStyle = "#fff7ed";
    ctx.beginPath();
    ctx.moveTo(-2, 1); ctx.lineTo(18, 0); ctx.quadraticCurveTo(7, 7, -3, 5); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#0f172a";
    ctx.beginPath(); ctx.arc(19, -1, 2.2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(5, -3.5, 2.0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(5.6, -4.1, 0.7, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    ctx.restore();
  }

  function rabbit(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const spd = Number(o.speed || 1.3);
    const phase = t * 10 * spd + Number(o.phase || 0);
    const hop = Math.max(0, Math.sin(phase)) * 16;
    const stretch = Math.sin(phase) * 0.18;

    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);

    // Sombra que se encoge al saltar
    const shScale = clamp(1 - hop / 28, 0.45, 1);
    ctx.fillStyle = "rgba(0,0,0,0.22)";
    ctx.beginPath(); ctx.ellipse(0, 13, 15 * shScale, 4.5 * shScale, 0, 0, Math.PI * 2); ctx.fill();

    ctx.translate(0, -hop);
    ctx.rotate(-stretch);

    // Colita pomposa
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(-14, 1, 5.2, 0, Math.PI * 2); ctx.fill();

    // Patas traseras impulsoras
    ctx.save();
    ctx.translate(-6, 6);
    ctx.rotate(hop > 3 ? 0.55 : -0.15);
    ctx.fillStyle = o.color || "#e2e8f0";
    ctx.beginPath(); ctx.ellipse(0, 3, 7, 3.5, 0.2, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // Cuerpo redondeado
    ctx.fillStyle = o.color || "#f8fafc";
    ctx.beginPath(); ctx.ellipse(0, 0, 13.5, 9.5, -0.1, 0, Math.PI * 2); ctx.fill();

    // Patitas delanteras
    ctx.save();
    ctx.translate(9, 5);
    ctx.rotate(hop > 3 ? -0.5 : 0.1);
    ctx.fillStyle = o.color || "#f1f5f9";
    ctx.fillRect(-2, 0, 4, 8);
    ctx.restore();

    // Cabeza y orejas largas animadas
    ctx.save();
    ctx.translate(12, -6);
    const earTilt = -0.25 - (hop / 16) * 0.35;

    ctx.save();
    ctx.rotate(earTilt - 0.14);
    ctx.fillStyle = o.color || "#e2e8f0";
    ctx.beginPath(); ctx.ellipse(-2, -12, 3.2, 10, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.rotate(earTilt);
    ctx.fillStyle = o.color || "#f8fafc";
    ctx.beginPath(); ctx.ellipse(1, -12, 3.4, 10.5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#fda4af";
    ctx.beginPath(); ctx.ellipse(1, -12, 1.6, 7.2, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    ctx.fillStyle = o.color || "#f8fafc";
    ctx.beginPath(); ctx.arc(2, 0, 7.2, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#fb7185";
    ctx.beginPath(); ctx.arc(9, 0.5, 1.6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#0f172a";
    ctx.beginPath(); ctx.arc(4.2, -1.8, 1.7, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(4.7, -2.3, 0.6, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    ctx.restore();
  }

  function bird(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const flap = Math.sin(t * 11 * Number(o.speed || 1) + Number(o.phase || 0));

    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);
    ctx.rotate(Number(o.pitch || 0));

    ctx.fillStyle = o.wingBackColor || "#0284c7";
    ctx.beginPath();
    ctx.moveTo(-3, -1);
    ctx.quadraticCurveTo(-10, -24 * flap - 6, 6, -26 * flap);
    ctx.quadraticCurveTo(9, -8 * flap, 7, 0);
    ctx.fill();

    ctx.fillStyle = o.bodyColor || "#38bdf8";
    ctx.beginPath(); ctx.ellipse(0, 1, 18, 9, -0.05, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(15, -3, 7.2, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = "#fbbf24";
    ctx.beginPath(); ctx.moveTo(21, -5); ctx.lineTo(30, -2); ctx.lineTo(21, 1); ctx.closePath(); ctx.fill();

    ctx.fillStyle = "#0f172a";
    ctx.beginPath(); ctx.arc(17, -4.2, 1.8, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = o.wingColor || "#7dd3fc";
    ctx.beginPath();
    ctx.moveTo(-5, 1);
    ctx.bezierCurveTo(-15, -28 * flap, 4, -32 * flap, 11, -24 * flap);
    ctx.quadraticCurveTo(9, -6 * flap, 7, 2);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function water(ctx, w, h, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const yTop = h * (o.level !== undefined ? o.level : 0.65);
    const g = ctx.createLinearGradient(0, yTop, 0, h);
    g.addColorStop(0, "rgba(14, 165, 233, 0.82)");
    g.addColorStop(1, "rgba(12, 74, 110, 0.96)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 16) {
      const wy = yTop + Math.sin(x * 0.02 + t * 2.8) * 6 + Math.cos(x * 0.045 - t * 2.1) * 3;
      ctx.lineTo(x, wy);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();
  }

  function fish(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const swish = Math.sin(t * 9 + Number(o.phase || 0)) * 7;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);
    ctx.fillStyle = o.color || "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(-16, 0);
    ctx.lineTo(-28, -10 + swish * 0.5);
    ctx.lineTo(-24, 0);
    ctx.lineTo(-28, 10 + swish * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(0, 0, 18, 9.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#0f172a";
    ctx.beginPath(); ctx.arc(11, -2.5, 2, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function car(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const rot = t * 12 * Number(o.speed || 1);
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.beginPath(); ctx.ellipse(0, 14, 38, 5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = o.color || "#ef4444";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-36, -6, 72, 16, 6);
    else ctx.rect(-36, -6, 72, 16);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-20, -6); ctx.lineTo(-10, -19); ctx.lineTo(14, -19); ctx.lineTo(25, -6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#bae6fd";
    ctx.beginPath();
    ctx.moveTo(-16, -7); ctx.lineTo(-8, -16); ctx.lineTo(12, -16); ctx.lineTo(20, -7); ctx.closePath(); ctx.fill();
    [-21, 21].forEach((wx) => {
      ctx.save();
      ctx.translate(wx, 10);
      ctx.rotate(rot);
      ctx.fillStyle = "#0f172a";
      ctx.beginPath(); ctx.arc(0, 0, 8, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#94a3b8"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(-5, 0); ctx.lineTo(5, 0); ctx.moveTo(0, -5); ctx.lineTo(0, 5); ctx.stroke();
      ctx.restore();
    });
    ctx.restore();
  }

  function rocket(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const ang = Number(o.angle !== undefined ? o.angle : -Math.PI / 4);
    const flame = 14 + Math.sin(t * 28) * 6;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(ang);
    ctx.scale(sc, sc);
    ctx.fillStyle = "#fb923c";
    ctx.beginPath(); ctx.moveTo(-20, -6); ctx.lineTo(-20 - flame, 0); ctx.lineTo(-20, 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.moveTo(-18, -8); ctx.lineTo(-26, -16); ctx.lineTo(-10, -8); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(-18, 8); ctx.lineTo(-26, 16); ctx.lineTo(-10, 8); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath(); ctx.ellipse(0, 0, 22, 9, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath(); ctx.arc(5, 0, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function planet(ctx, x, y, radius, opts) {
    const o = opts || {};
    const r = Number(radius || 28);
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.15, x, y, r);
    g.addColorStop(0, o.lightColor || "#7dd3fc");
    g.addColorStop(1, o.color || "#1d4ed8");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    if (o.rings) {
      ctx.strokeStyle = "rgba(253, 230, 138, 0.65)";
      ctx.lineWidth = 3.5;
      ctx.beginPath(); ctx.ellipse(x, y, r * 1.85, r * 0.45, -0.25, 0, Math.PI * 2); ctx.stroke();
    }
  }

  function person(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const dir = o.dir === -1 ? -1 : 1;
    const sw = Math.sin(t * 8 * Number(o.speed || 1) + Number(o.phase || 0));
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(dir * sc, sc);
    ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 4; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(0, 2); ctx.lineTo(sw * 8, 18); ctx.moveTo(0, 2); ctx.lineTo(-sw * 8, 18); ctx.stroke();
    ctx.fillStyle = o.color || "#38bdf8";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-6, -15, 12, 18, 5);
    else ctx.rect(-6, -15, 12, 18);
    ctx.fill();
    ctx.strokeStyle = o.color || "#38bdf8"; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.moveTo(0, -12); ctx.lineTo(-sw * 8, 0); ctx.moveTo(0, -12); ctx.lineTo(sw * 8, 0); ctx.stroke();
    ctx.fillStyle = "#fed7aa";
    ctx.beginPath(); ctx.arc(0, -22, 6, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function beagle(ctx, x, y, opts) {
    const o = opts || {};
    const t = Number(o.t || 0);
    const sc = Number(o.scale || 1);
    const earWagIntensity = o.earWiggle !== undefined ? Number(o.earWiggle) : 1.0;
    const earSpeed = Number(o.earSpeed || 4.2);
    const lookX = o.lookX !== undefined ? Number(o.lookX) : (o.lookAt ? (o.lookAt.x - x) / (200 * sc) : 0);
    const lookY = o.lookY !== undefined ? Number(o.lookY) : (o.lookAt ? (o.lookAt.y - y) / (200 * sc) : 0);
    const eyeTargetX = clamp(lookX, -1, 1);
    const eyeTargetY = clamp(lookY, -0.8, 0.8);

    // Parpadeo natural y respiracion
    const blinkCycle = (t * 0.45) % 1;
    const isBlinking = blinkCycle > 0.94;
    const breath = Math.sin(t * 3.2) * 1.8;
    const headTilt = (o.headTilt !== undefined ? Number(o.headTilt) : Math.sin(t * 1.6) * 0.06) + eyeTargetX * 0.05;

    ctx.save();
    ctx.translate(x, y);
    ctx.scale(sc, sc);

    // Sombra suave en el suelo
    ctx.fillStyle = "rgba(0,0,0,0.22)";
    ctx.beginPath();
    ctx.ellipse(0, 68, 55, 14, 0, 0, Math.PI * 2);
    ctx.fill();

    // Pecho y cuerpo beagle frontal (manto blanco y negro/marron)
    ctx.save();
    ctx.translate(0, breath * 0.5);
    ctx.fillStyle = "#292524";
    ctx.beginPath();
    ctx.ellipse(0, 42, 46, 32, 0, 0, Math.PI * 2);
    ctx.fill();

    // Pecho blanco sedoso
    const chestGrad = ctx.createLinearGradient(0, 15, 0, 65);
    chestGrad.addColorStop(0, "#ffffff");
    chestGrad.addColorStop(1, "#f5f5f4");
    ctx.fillStyle = chestGrad;
    ctx.beginPath();
    ctx.ellipse(0, 44, 26, 26, 0, 0, Math.PI * 2);
    ctx.fill();

    // Patitas delanteras blancas
    function drawPaw(px, py) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.ellipse(px, py, 13, 8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#e7e5e4";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px - 4, py - 4); ctx.lineTo(px - 4, py + 4);
      ctx.moveTo(px + 4, py - 4); ctx.lineTo(px + 4, py + 4);
      ctx.stroke();
    }
    drawPaw(-26, 64);
    drawPaw(26, 64);

    // Collar rojo con placa dorada
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(0, 18, 30, 0.25 * Math.PI, 0.75 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = "#eab308";
    ctx.beginPath();
    ctx.arc(0, 46, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fef08a";
    ctx.beginPath();
    ctx.arc(-1.5, 44.5, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Cabeza articulada con inclinacion tierna
    ctx.save();
    ctx.translate(0, -6 + breath);
    ctx.rotate(headTilt);

    // OREJAS FLOPPY BEAGLE articuladas
    const earLeftAng = Math.sin(t * earSpeed) * 0.18 * earWagIntensity + 0.12;
    const earRightAng = Math.sin(t * earSpeed + 1.8) * 0.18 * earWagIntensity - 0.12;
    const earFlap = Math.cos(t * earSpeed * 1.5) * 4 * earWagIntensity;

    function drawBeagleEar(side, ang, flap) {
      ctx.save();
      ctx.translate(side * 28, -22);
      ctx.rotate(side * 0.2 + ang);

      const earGrad = ctx.createLinearGradient(0, 0, 0, 65);
      earGrad.addColorStop(0, "#854d0e");
      earGrad.addColorStop(0.4, "#a16207");
      earGrad.addColorStop(0.85, "#ca8a04");
      earGrad.addColorStop(1, "#b45309");

      ctx.fillStyle = earGrad;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(side * 18, 12, side * 24 + flap, 48, side * 8 + flap, 68);
      ctx.bezierCurveTo(side * -4 + flap, 76, side * -18, 54, side * -8, 28);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "rgba(254, 205, 211, 0.35)";
      ctx.beginPath();
      ctx.ellipse(side * 4, 34, 7, 24, side * 0.08, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    drawBeagleEar(-1, earLeftAng, earFlap);
    drawBeagleEar(1, earRightAng, -earFlap);

    // Cabeza redonda
    const headGrad = ctx.createRadialGradient(0, -6, 10, 0, 2, 42);
    headGrad.addColorStop(0, "#d97706");
    headGrad.addColorStop(0.65, "#b45309");
    headGrad.addColorStop(1, "#92400e");
    ctx.fillStyle = headGrad;
    ctx.beginPath();
    ctx.ellipse(0, -6, 36, 32, 0, 0, Math.PI * 2);
    ctx.fill();

    // Mancha blanca simetrica frontal (blaze beagle)
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(0, -36);
    ctx.bezierCurveTo(7, -26, 9, -12, 16, 4);
    ctx.bezierCurveTo(12, 18, -12, 18, -16, 4);
    ctx.bezierCurveTo(-9, -12, -7, -26, 0, -36);
    ctx.fill();

    // Hocico blanco prominente
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.ellipse(0, 8, 22, 16, 0, 0, Math.PI * 2);
    ctx.fill();

    // Mejillas tiernas con rubor sutil
    ctx.fillStyle = "rgba(251, 113, 133, 0.15)";
    ctx.beginPath();
    ctx.arc(-22, 2, 7, 0, Math.PI * 2);
    ctx.arc(22, 2, 7, 0, Math.PI * 2);
    ctx.fill();

    // Ojos grandes expresivos de beagle
    function drawEye(ex, ey) {
      ctx.save();
      ctx.translate(ex, ey);
      if (isBlinking) {
        ctx.strokeStyle = "#1c1917";
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(0, 2, 8, Math.PI * 1.1, Math.PI * 1.9);
        ctx.stroke();
      } else {
        ctx.fillStyle = "#1c1917";
        ctx.beginPath();
        ctx.ellipse(0, 0, 9.5, 9, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#78350f";
        ctx.beginPath();
        ctx.ellipse(eyeTargetX * 2, eyeTargetY * 2, 7.5, 7.5, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#0c0a09";
        ctx.beginPath();
        ctx.arc(eyeTargetX * 2.5, eyeTargetY * 2.5, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(eyeTargetX * 2 - 2.2, eyeTargetY * 2 - 2.2, 2.4, 0, Math.PI * 2);
        ctx.arc(eyeTargetX * 2 + 2, eyeTargetY * 2 + 1.6, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
    drawEye(-16, -10);
    drawEye(16, -10);

    // Cejas marrones
    const browTilt = Math.sin(t * 2.2) * 2;
    ctx.strokeStyle = "#78350f";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(-22, -21 + browTilt); ctx.lineTo(-11, -22);
    ctx.moveTo(22, -21 - browTilt); ctx.lineTo(11, -22);
    ctx.stroke();

    // Nariz negra brillante
    const noseY = 4;
    ctx.fillStyle = "#18181b";
    ctx.beginPath();
    ctx.moveTo(0, noseY + 9);
    ctx.bezierCurveTo(-11, noseY + 9, -12, noseY - 2, -7, noseY - 5);
    ctx.bezierCurveTo(-3, noseY - 7, 3, noseY - 7, 7, noseY - 5);
    ctx.bezierCurveTo(12, noseY - 2, 11, noseY + 9, 0, noseY + 9);
    ctx.fill();

    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.beginPath();
    ctx.ellipse(-2.5, noseY - 2, 3.5, 1.8, -0.2, 0, Math.PI * 2);
    ctx.fill();

    // Linea del hocico
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, noseY + 9);
    ctx.lineTo(0, noseY + 16);
    ctx.arc(-7, noseY + 16, 7, 0, Math.PI * 0.7);
    ctx.moveTo(0, noseY + 16);
    ctx.arc(7, noseY + 16, 7, Math.PI * 0.3, Math.PI);
    ctx.stroke();

    // Lenguita juguetona
    const pant = Math.sin(t * 5.5);
    if (pant > -0.1) {
      const tongueLen = 9 + pant * 4;
      ctx.fillStyle = "#fb7185";
      ctx.beginPath();
      ctx.ellipse(0, noseY + 18 + tongueLen * 0.5, 5.5, tongueLen * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#e11d48";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, noseY + 16);
      ctx.lineTo(0, noseY + 16 + tongueLen * 0.8);
      ctx.stroke();
    }

    // Puntos de bigotes
    ctx.fillStyle = "#71717a";
    function drawWhiskersDots(wx, wy) {
      for (let i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.arc(wx + (i % 2) * 4, wy + Math.floor(i / 2) * 4, 0.9, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    drawWhiskersDots(-14, 10);
    drawWhiskersDots(11, 10);

    ctx.restore();
    ctx.restore();
  }

  /* --- 3. Partículas y Cinemática de Persecución / Evasión --- */
  function updateDust(ctx, list, dt) {
    for (let i = list.length - 1; i >= 0; i--) {
      const p = list[i];
      p.life -= dt;
      if (p.life <= 0) {
        list.splice(i, 1);
        continue;
      }
      p.x += (p.vx || 0) * dt;
      p.y += (p.vy || -12) * dt;
      const a = clamp(p.life / (p.maxLife || 0.7), 0, 1) * 0.42;
      ctx.fillStyle = p.color || ("rgba(214, 211, 209, " + a.toFixed(2) + ")");
      ctx.beginPath();
      ctx.arc(p.x, p.y, (p.r || 4) * (1.4 - a), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function spawnDust(list, x, y, count) {
    const n = count || 3;
    for (let i = 0; i < n; i++) {
      if (list.length > 90) list.shift();
      list.push({
        x: x + (Math.random() - 0.5) * 10,
        y: y + (Math.random() - 0.5) * 4,
        vx: (Math.random() - 0.5) * 34,
        vy: -10 - Math.random() * 22,
        r: 3 + Math.random() * 4,
        life: 0.55 + Math.random() * 0.3,
        maxLife: 0.85
      });
    }
  }

  function stepPursuit(predator, targets, bounds, speedPx, dt) {
    if (!targets || !targets.length) return;
    let nearest = targets[0];
    let bestD = Infinity;
    targets.forEach((tg) => {
      const d = Math.hypot(tg.x - predator.x, tg.y - predator.y);
      if (d < bestD) { bestD = d; nearest = tg; }
    });
    const dx = nearest.x - predator.x;
    const dy = nearest.y - predator.y;
    const dist = Math.hypot(dx, dy) || 1;
    const vx = (dx / dist) * speedPx;
    const vy = (dy / dist) * speedPx * 0.65;
    predator.vx = (predator.vx || 0) * 0.86 + vx * 0.14;
    predator.vy = (predator.vy || 0) * 0.86 + vy * 0.14;
    predator.x = clamp(predator.x + predator.vx * dt, bounds.minX, bounds.maxX);
    predator.y = clamp(predator.y + predator.vy * dt, bounds.minY, bounds.maxY);
    predator.dir = predator.vx >= 0 ? 1 : -1;
    return { target: nearest, distance: dist };
  }

  function stepEvade(prey, predator, bounds, speedPx, dt, t) {
    const dx = prey.x - predator.x;
    const dy = prey.y - predator.y;
    const dist = Math.hypot(dx, dy) || 1;
    const panic = dist < 210 ? 1.45 : 0.72;
    let tx = (dx / dist) * speedPx * panic + Math.cos(t * 2.3 + (prey.seed || 0)) * 55;
    let ty = (dy / dist) * speedPx * 0.55 * panic + Math.sin(t * 2.9 + (prey.seed || 0)) * 38;

    // Rebote suave al acercarse a los bordes para que nunca se encierren en una esquina
    const margin = 75;
    if (prey.x < bounds.minX + margin) tx += speedPx * 1.4;
    if (prey.x > bounds.maxX - margin) tx -= speedPx * 1.4;
    if (prey.y < bounds.minY + 35) ty += speedPx * 0.9;
    if (prey.y > bounds.maxY - 35) ty -= speedPx * 0.9;

    prey.vx = (prey.vx || 0) * 0.84 + tx * 0.16;
    prey.vy = (prey.vy || 0) * 0.84 + ty * 0.16;
    prey.x = clamp(prey.x + prey.vx * dt, bounds.minX, bounds.maxX);
    prey.y = clamp(prey.y + prey.vy * dt, bounds.minY, bounds.maxY);
    prey.dir = prey.vx >= 0 ? 1 : -1;
  }

  global.ADAScene = {
    sky,
    hillsAndMeadow,
    forestTrees,
    citySkyline,
    water,
    fox,
    rabbit,
    bird,
    fish,
    car,
    rocket,
    planet,
    person,
    beagle,
    updateDust,
    spawnDust,
    stepPursuit,
    stepEvade
  };
})(window);
