"""ADA Web Design Studio v2.0 — Generador de Aplicaciones Web Interactivas Bento Glass
y Auditor Automático de Integridad DOM / CSS / Sandbox / HiDPI.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any

from ada.audit.ledger import record_audit_event
from ada.config import OUTPUTS_DIR, PROJECT_ROOT, assert_safe_write_path

STUDIO_CSS_PATH = PROJECT_ROOT / "web" / "ada-studio.css"
CHARTS_JS_PATH = PROJECT_ROOT / "web" / "ada-charts.js"
SCENE_JS_PATH = PROJECT_ROOT / "web" / "ada-scene.js"


def load_studio_assets() -> tuple[str, str]:
    """Carga los activos locales verificados del sistema de diseño ADA Studio."""
    css = STUDIO_CSS_PATH.read_text(encoding="utf-8")
    js = CHARTS_JS_PATH.read_text(encoding="utf-8")
    return css, js


def load_scene_engine_js() -> str:
    """Carga el motor procedural 2D ADAScene (biomas, actores articulados y física)."""
    return SCENE_JS_PATH.read_text(encoding="utf-8")


class _DOMInspector(HTMLParser):
    """Extrae etiquetas, clases DOM, IDs, estilos inline, scripts y enlaces relativos."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()
        self.classes: set[str] = set()
        self.ids: set[str] = set()
        self.stylesheets: list[str] = []
        self.scripts_src: list[str] = []
        self.inline_styles: list[str] = []
        self.inline_scripts: list[str] = []
        self.has_viewport: bool = False
        self.has_canvas: bool = False
        self._in_style: bool = False
        self._in_script: bool = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        self.tags.add(t)
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if t == "meta" and attr_map.get("name", "").lower() == "viewport":
            self.has_viewport = True
        if t == "canvas":
            self.has_canvas = True
        if t == "link" and "stylesheet" in attr_map.get("rel", "").lower() and attr_map.get("href"):
            self.stylesheets.append(attr_map["href"])
        if t == "script" and attr_map.get("src"):
            self.scripts_src.append(attr_map["src"])

        cls_raw = attr_map.get("class", "")
        if cls_raw and "${" not in cls_raw and "{{" not in cls_raw:
            for token in cls_raw.split():
                clean = token.strip()
                if clean and re.match(r"^[A-Za-z_-][A-Za-z0-9_-]*$", clean):
                    self.classes.add(clean)

        id_raw = attr_map.get("id", "").strip()
        if id_raw:
            self.ids.add(id_raw)

        if t == "style":
            self._in_style = True
            self._buf = []
        elif t == "script" and not attr_map.get("src"):
            self._in_script = True
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_style or self._in_script:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "style" and self._in_style:
            self.inline_styles.append("".join(self._buf))
            self._in_style = False
            self._buf = []
        elif t == "script" and self._in_script:
            self.inline_scripts.append("".join(self._buf))
            self._in_script = False
            self._buf = []


def _extract_css_defined_classes(css_text: str) -> set[str]:
    no_comments = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    # Extraer selectores antes de cada '{' evitando valores hexadecimales dentro de reglas
    defined: set[str] = set()
    for block_match in re.finditer(r"([^{}]+)\{", no_comments):
        selector_chunk = block_match.group(1)
        if selector_chunk.strip().startswith("@"):
            continue
        for cls_match in re.finditer(r"\.([A-Za-z_-][A-Za-z0-9_-]*)", selector_chunk):
            defined.add(cls_match.group(1))
    return defined


def audit_html_quality(html_path: str | Path) -> dict[str, Any]:
    """Audita estructural, visual y funcionalmente una página HTML antes de presentarla al usuario.

    Detecta:
    - Clases DOM huérfanas (usadas en HTML/JS pero borradas u omitidas en CSS, como `.grid` o `.bar-fill`).
    - Riesgos de crash en el visor `sandbox="allow-scripts"` de DeepSeek Harness (`localStorage` sin `try/catch`).
    - Antipatrones visuales tipo 'hoja de cuaderno' (ausencia de tokens CSS, grillas, profundidad visual o HiDPI en Canvas).
    """
    path = Path(html_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo HTML a auditar: {path}")

    raw_html = path.read_text(encoding="utf-8", errors="replace")
    inspector = _DOMInspector()
    inspector.feed(raw_html)

    css_chunks = list(inspector.inline_styles)
    for href in inspector.stylesheets:
        if not href.startswith(("http://", "https://", "//")):
            rel_css = (path.parent / href).resolve()
            if rel_css.exists():
                css_chunks.append(rel_css.read_text(encoding="utf-8", errors="replace"))

    js_chunks = list(inspector.inline_scripts)
    for src in inspector.scripts_src:
        if not src.startswith(("http://", "https://", "//")):
            rel_js = (path.parent / src).resolve()
            if rel_js.exists():
                js_chunks.append(rel_js.read_text(encoding="utf-8", errors="replace"))

    full_css = "\n".join(css_chunks)
    full_js = "\n".join(js_chunks)

    # También detectar clases inyectadas dinámicamente en templates JS (`class="bar-row"`)
    dynamic_classes: set[str] = set()
    for m in re.finditer(r'class=["\']([A-Za-z0-9_\-\s]+)["\']', full_js):
        for tok in m.group(1).split():
            if re.match(r"^[A-Za-z_-][A-Za-z0-9_-]*$", tok):
                dynamic_classes.add(tok)

    all_used_classes = inspector.classes | dynamic_classes
    defined_classes = _extract_css_defined_classes(full_css)

    # Ignorar estados JS opcionales puros si están documentados, pero exigir cobertura real
    orphan_classes = sorted(c for c in all_used_classes if c not in defined_classes)

    critical_errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    score = 100

    # 1. Clases DOM huérfanas (causa exacta del bug de la 'hoja de cuaderno')
    if orphan_classes:
        penalty = min(40, 10 * len(orphan_classes))
        score -= penalty
        critical_errors.append(
            f"Clases usadas en el HTML/JS sin definición en CSS ({len(orphan_classes)}): {', '.join(orphan_classes[:12])}"
        )
        recommendations.append(
            "Define todas las clases en <style> o enlaza '../web/ada-studio.css' / usa `ada_build_interactive_webapp`."
        )

    # 2. Seguridad en iframe sandbox="allow-scripts" (sin allow-same-origin)
    for idx, script_body in enumerate(inspector.inline_scripts):
        if ("localStorage" in script_body or "sessionStorage" in script_body) and "try" not in script_body and "safeStorage" not in script_body:
            score -= 25
            critical_errors.append(
                f"Script inline #{idx + 1} accede a localStorage/sessionStorage sin try/catch: esto lanza SecurityError dentro del preview sandbox de DeepSeek Harness."
            )
            recommendations.append("Envuelve localStorage en try/catch o usa `ADACharts.safeStorage`.")

    # 3. Viewport responsivo
    if not inspector.has_viewport:
        score -= 8
        warnings.append("Falta `<meta name='viewport' content='width=device-width, initial-scale=1.0'>`.")

    # 4. Tokens de diseño (CSS Custom Properties) y arquitectura moderna
    has_css_vars = "--" in full_css and "var(" in full_css
    has_modern_layout = ("display: grid" in full_css or "display:grid" in full_css or "display: flex" in full_css or "display:flex" in full_css)
    has_visual_depth = ("linear-gradient" in full_css or "radial-gradient" in full_css or "backdrop-filter" in full_css)

    if not has_css_vars:
        score -= 15
        warnings.append("El CSS no utiliza variables de sistema de diseño (`:root { --token }`), produciendo estilos planos y frágiles.")
    if not has_modern_layout:
        score -= 20
        critical_errors.append("No se detectó arquitectura CSS Grid ni Flexbox.")
    if not has_visual_depth:
        score -= 12
        warnings.append("Estética plana ('hoja de cuaderno'): faltan gradientes sutiles, superficies elevadas o glassmorphism.")

    # Detectar explícitamente el antipatrón de fondo blanco/gris plano sin variables
    if re.search(r"background(-color)?\s*:\s*#(fff|ffffff|f9f9f9|f0f4f8)\b", full_css, re.IGNORECASE) and not has_css_vars:
        score -= 20
        warnings.append("Antipatrón 'hoja de cuaderno' detectado: fondo claro plano genérico sin sistema de temas ni jerarquía Bento.")

    # 5. Escalado Retina HiDPI en Canvas
    if inspector.has_canvas and "devicePixelRatio" not in full_js:
        score -= 12
        warnings.append("Se detectó `<canvas>` sin escalado `devicePixelRatio` (se verá borroso en pantallas HiDPI y al redimensionar el panel lateral).")
        recommendations.append("Usa `ADACharts` (`../web/ada-charts.js`) o escala `canvas.width = rect.width * devicePixelRatio`.")

    score = max(0, min(100, score))
    if score >= 92 and not critical_errors:
        grade = "A+"
    elif score >= 82 and not critical_errors:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 50:
        grade = "C"
    else:
        grade = "FAIL"

    passed = score >= 82 and len(critical_errors) == 0
    return {
        "verified": passed,
        "html_path": str(path),
        "quality_score": score,
        "grade": grade,
        "passed": passed,
        "critical_errors": critical_errors,
        "warnings": warnings,
        "recommendations": recommendations,
        "orphan_classes": orphan_classes,
        "stats": {
            "html_bytes": len(raw_html.encode("utf-8")),
            "dom_tags_count": len(inspector.tags),
            "used_classes_count": len(all_used_classes),
            "defined_css_classes_count": len(defined_classes),
            "has_canvas": inspector.has_canvas,
            "has_css_variables": has_css_vars,
            "has_modern_layout": has_modern_layout,
        },
    }


def render_prediction_studio_html(
    *,
    output_html_path: Path,
    title: str,
    dataset_name: str,
    task_type: str,
    target_column: str,
    n_train: int,
    n_test: int,
    n_features: int,
    models_summary: dict[str, Any],
    best_model_name: str,
    feature_importances: list[dict[str, Any]],
    critic_report: dict[str, Any],
    chart_payload: dict[str, Any],
    sample_predictions: list[dict[str, Any]] | None = None,
    verification_badge: str = "VERIFICADO: scikit-learn + statsmodels",
) -> Path:
    """Renderiza el Estudio Web Interactivo Bento Glass v2.0 para predicciones ML y Crítico Estadístico."""
    safe_path = assert_safe_write_path(output_html_path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    studio_css, charts_js = load_studio_assets()
    scene_js = load_scene_engine_js()
    scene_js = load_scene_engine_js()

    payload = {
        "title": title,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "models": models_summary,
        "best_model": best_model_name,
        "features": feature_importances,
        "critic": critic_report,
        "chart": chart_payload,
        "samples": (sample_predictions or [])[:60],
        "verification_badge": verification_badge,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
{studio_css}
</style>
</head>
<body>
  <nav class="ada-topbar">
    <div class="ada-brand">
      <span class="ada-brand-orb"></span>
      <span>ADA ANALYTICS STUDIO</span>
      <span class="ada-pill">Creador: <strong>Diego López</strong></span>
    </div>
    <div class="ada-topbar-actions">
      <span class="ada-badge ada-badge-info" id="verifStamp"></span>
      <button type="button" class="ada-btn" id="btnExportPng">⬇ Gráfica PNG</button>
      <button type="button" class="ada-btn" id="btnExportCsv">⬇ Exportar CSV</button>
      <button type="button" class="ada-btn ada-btn-primary" id="adaThemeBtn">☀ Modo Claro</button>
    </div>
  </nav>

  <main class="ada-shell">
    <section class="ada-hero">
      <div>
        <h1 class="ada-hero-title" id="hdr"></h1>
        <p class="ada-hero-sub" id="subhdr"></p>
      </div>
      <div class="ada-pill-row" id="metaPills"></div>
    </section>

    <section class="ada-kpi-grid" id="kpiGrid"></section>

    <section class="ada-bento-grid">
      <article class="ada-card ada-col-8">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title" id="chart1Title">Laboratorio Visual en Test Set</h2>
            <p class="ada-card-subtitle" id="chart1Sub">Pasa el cursor sobre cualquier punto o barra para inspeccionar valores exactos e intervalos.</p>
          </div>
          <div class="ada-tabs" id="chartTabs"></div>
        </div>
        <div class="ada-chart-stage">
          <canvas id="canvasMain"></canvas>
        </div>
        <div class="ada-chart-legend" id="chartLegend"></div>
      </article>

      <article class="ada-card ada-col-4">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title">Crítico Estadístico de ADA</h2>
            <p class="ada-card-subtitle">Auditoría automática de supuestos y potencia</p>
          </div>
          <span id="sevBadge" class="ada-badge"></span>
        </div>
        <div class="ada-callout-stack" id="criticBox"></div>
      </article>
    </section>

    <section class="ada-bento-grid">
      <article class="ada-card ada-col-5">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title">Importancia de Variables</h2>
            <p class="ada-card-subtitle">Sensibilidad por permutación en partición Test retenida</p>
          </div>
        </div>
        <div class="ada-bar-list" id="featBox"></div>
      </article>

      <article class="ada-card ada-col-7">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title">Arena de Modelos (Split Estricto Sin Fuga)</h2>
            <p class="ada-card-subtitle">Clic en cualquier encabezado para ordenar métricas</p>
          </div>
        </div>
        <div class="ada-table-wrap">
          <table class="ada-table" id="modelsTbl"></table>
        </div>
      </article>
    </section>

    <section class="ada-bento-grid" id="samplesSection">
      <article class="ada-card ada-col-12">
        <div class="ada-table-toolbar">
          <div>
            <h2 class="ada-card-title">Explorador Interactivo de Predicciones</h2>
            <p class="ada-card-subtitle">Filtra filas en tiempo real o haz clic en las columnas para ordenar</p>
          </div>
          <input type="search" class="ada-search-input" id="sampleSearch" placeholder="Filtrar muestras..." />
        </div>
        <div class="ada-table-wrap">
          <table class="ada-table" id="samplesTbl"></table>
        </div>
      </article>
    </section>

    <footer class="ada-footer">
      <span>ADA Local Multi-Agent Intelligence · Arquitectura verificada por doble motor independiente</span>
      <span class="ada-mono" id="footerMeta"></span>
    </footer>
  </main>

<script>
{charts_js}
</script>
<script>
{scene_js}
</script>
<script>
(function () {{
  const DATA = {payload_json};
  ADACharts.initThemeToggle("adaThemeBtn");

  document.getElementById("verifStamp").textContent = "✓ " + DATA.verification_badge;
  document.getElementById("hdr").textContent = DATA.title;
  document.getElementById("subhdr").textContent =
    `Evaluación predictiva sin fuga de datos sobre '${{DATA.dataset_name}}' · Variable objetivo: '${{DATA.target_column}}' · Modelo óptimo seleccionado: ${{DATA.best_model}}`;
  document.getElementById("footerMeta").textContent =
    `Train=${{DATA.n_train}} | Test=${{DATA.n_test}} | Variables=${{DATA.n_features}}`;

  const pillsEl = document.getElementById("metaPills");
  pillsEl.innerHTML = [
    `<span class="ada-pill">Tarea: <strong>${{DATA.task_type === "regression" ? "Regresión" : "Clasificación"}}</strong></span>`,
    `<span class="ada-pill">Objetivo: <strong class="ada-mono">${{DATA.target_column}}</strong></span>`,
    `<span class="ada-pill">Partición: <strong class="ada-mono">${{DATA.n_train}} train / ${{DATA.n_test}} test</strong></span>`
  ].join("");

  const bestM = DATA.models[DATA.best_model] || {{}};
  const baseM = DATA.models["DummyBaseline"] || {{}};
  const sev = (DATA.critic && DATA.critic.severity) || "OK";

  // Render KPI Cards
  const kpiGrid = document.getElementById("kpiGrid");
  if (DATA.task_type === "regression") {{
    const impPct = baseM.rmse ? (((baseM.rmse - bestM.rmse) / baseM.rmse) * 100) : 0;
    const band = DATA.chart.q90_half_width || 0;
    kpiGrid.innerHTML = [
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-cyan)">
        <div class="ada-kpi-label">Modelo Ganador ★</div>
        <div class="ada-kpi-value">${{DATA.best_model}}</div>
        <div class="ada-kpi-meta"><span class="ada-badge ada-badge-info">${{impPct >= 0 ? "+" : ""}}${{impPct.toFixed(2)}}% vs Dummy</span></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-indigo)">
        <div class="ada-kpi-label">RMSE en Test Set</div>
        <div class="ada-kpi-value ada-mono">${{(bestM.rmse ?? 0).toFixed(4)}}</div>
        <div class="ada-kpi-meta">MAE: <strong class="ada-mono">${{(bestM.mae ?? 0).toFixed(4)}}</strong> · Banda 90%: <strong class="ada-mono">±${{band.toFixed(3)}}</strong></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-violet)">
        <div class="ada-kpi-label">R² Test & Validación Cruzada</div>
        <div class="ada-kpi-value ada-mono">${{(bestM.r2 ?? 0).toFixed(4)}}</div>
        <div class="ada-kpi-meta">CV R²: <strong class="ada-mono">${{(bestM.cv_mean ?? 0).toFixed(4)}}</strong></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: ${{sev === "OK" ? "var(--accent-emerald)" : (sev === "CAUTION" ? "var(--accent-amber)" : "var(--accent-rose)")}}">
        <div class="ada-kpi-label">Auditoría Estadística</div>
        <div class="ada-kpi-value">${{sev}}</div>
        <div class="ada-kpi-meta">${{DATA.n_features}} predictores auditados sin leakage</div>
      </div>`
    ].join("");
  }} else {{
    const impF1 = ((bestM.f1_macro ?? 0) - (baseM.f1_macro ?? 0)) * 100;
    kpiGrid.innerHTML = [
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-cyan)">
        <div class="ada-kpi-label">Clasificador Ganador ★</div>
        <div class="ada-kpi-value">${{DATA.best_model}}</div>
        <div class="ada-kpi-meta"><span class="ada-badge ada-badge-info">${{impF1 >= 0 ? "+" : ""}}${{impF1.toFixed(1)}} pts F1 vs Dummy</span></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-indigo)">
        <div class="ada-kpi-label">F1-Macro (Test)</div>
        <div class="ada-kpi-value ada-mono">${{(bestM.f1_macro ?? 0).toFixed(4)}}</div>
        <div class="ada-kpi-meta">Accuracy: <strong class="ada-mono">${{((bestM.accuracy ?? 0) * 100).toFixed(2)}}%</strong></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: var(--accent-violet)">
        <div class="ada-kpi-label">ROC-AUC / CV F1</div>
        <div class="ada-kpi-value ada-mono">${{bestM.roc_auc !== null && bestM.roc_auc !== undefined ? bestM.roc_auc.toFixed(4) : (bestM.cv_f1_mean ?? 0).toFixed(4)}}</div>
        <div class="ada-kpi-meta">CV F1-Macro: <strong class="ada-mono">${{(bestM.cv_f1_mean ?? 0).toFixed(4)}}</strong></div>
      </div>`,
      `<div class="ada-kpi-card" style="--kpi-accent: ${{sev === "OK" ? "var(--accent-emerald)" : "var(--accent-amber)"}}">
        <div class="ada-kpi-label">Auditoría Estadística</div>
        <div class="ada-kpi-value">${{sev}}</div>
        <div class="ada-kpi-meta">${{DATA.n_train + DATA.n_test}} registros evaluados</div>
      </div>`
    ].join("");
  }}

  // Critic Badge & Callouts
  const badge = document.getElementById("sevBadge");
  badge.textContent = sev;
  badge.className = "ada-badge " + (sev === "OK" ? "ada-badge-ok" : (sev === "CAUTION" ? "ada-badge-caution" : "ada-badge-risk"));

  const criticCards = [];
  ((DATA.critic && DATA.critic.warnings) || []).forEach((w) => {{
    const cls = sev === "HIGH_RISK" ? "is-risk" : "is-caution";
    criticCards.push(`<div class="ada-callout ${{cls}}"><div class="ada-callout-title">⚠ Alerta Metodológica</div><div>${{w}}</div></div>`);
  }});
  ((DATA.critic && DATA.critic.passed_checks) || []).forEach((ok) => {{
    criticCards.push(`<div class="ada-callout is-ok"><div class="ada-callout-title">✓ Control Superado</div><div>${{ok}}</div></div>`);
  }});
  if (!criticCards.length) {{
    criticCards.push(`<div class="ada-callout is-ok"><div class="ada-callout-title">✓ Sin Alertas Críticas</div><div>Todos los controles estadísticos superaron los umbrales de validación.</div></div>`);
  }}
  document.getElementById("criticBox").innerHTML = criticCards.join("");

  // Feature Importance Bars
  const featBox = document.getElementById("featBox");
  const totalPos = DATA.features.reduce((acc, f) => acc + Math.max(0, f.importance), 0) || 1e-6;
  const maxImp = Math.max(...DATA.features.map((f) => Math.max(f.importance, 0)), 1e-6);
  featBox.innerHTML = DATA.features.slice(0, 10).map((f) => {{
    const pct = Math.max(2, Math.min(100, (Math.max(0, f.importance) / maxImp) * 100));
    const share = Math.max(0, (Math.max(0, f.importance) / totalPos) * 100);
    return `<div class="ada-bar-row">
      <div class="ada-bar-label">
        <span>${{f.feature}}</span>
        <span class="ada-mono">${{f.importance.toFixed(4)}} <small style="color:var(--txt-muted)">(${{share.toFixed(1)}}%)</small></span>
      </div>
      <div class="ada-bar-track"><div class="ada-bar-fill" style="width:${{pct}}%"></div></div>
    </div>`;
  }}).join("") || `<div class="ada-callout is-info">Sin predictores disponibles.</div>`;

  // Models Table
  const tbl = document.getElementById("modelsTbl");
  if (DATA.task_type === "regression") {{
    let htmlRows = `<thead><tr>
      <th data-sortable="true">Modelo ⇅</th>
      <th data-sortable="true">RMSE Test ⇅</th>
      <th data-sortable="true">MAE Test ⇅</th>
      <th data-sortable="true">R² Test (IC 95%) ⇅</th>
      <th data-sortable="true">CV R² ⇅</th>
    </tr></thead><tbody>`;
    for (const [name, m] of Object.entries(DATA.models)) {{
      const win = name === DATA.best_model;
      const ci = m.r2_ci95 ? `[${{m.r2_ci95[0].toFixed(3)}}, ${{m.r2_ci95[1].toFixed(3)}}]` : "-";
      htmlRows += `<tr class="${{win ? "ada-row-winner" : ""}}">
        <td><strong>${{win ? "★ " : ""}}${{name}}</strong></td>
        <td class="ada-mono">${{(m.rmse ?? 0).toFixed(4)}}</td>
        <td class="ada-mono">${{(m.mae ?? 0).toFixed(4)}}</td>
        <td class="ada-mono">${{(m.r2 ?? 0).toFixed(4)}} <small style="color:var(--txt-muted)">${{ci}}</small></td>
        <td class="ada-mono">${{m.cv_mean !== undefined ? m.cv_mean.toFixed(4) : "-"}}</td>
      </tr>`;
    }}
    tbl.innerHTML = htmlRows + "</tbody>";
  }} else {{
    let htmlRows = `<thead><tr>
      <th data-sortable="true">Clasificador ⇅</th>
      <th data-sortable="true">F1-Macro (IC 95%) ⇅</th>
      <th data-sortable="true">Accuracy ⇅</th>
      <th data-sortable="true">ROC-AUC ⇅</th>
      <th data-sortable="true">CV F1 ⇅</th>
    </tr></thead><tbody>`;
    for (const [name, m] of Object.entries(DATA.models)) {{
      const win = name === DATA.best_model;
      const ci = m.f1_ci95 ? `[${{m.f1_ci95[0].toFixed(3)}}, ${{m.f1_ci95[1].toFixed(3)}}]` : "-";
      htmlRows += `<tr class="${{win ? "ada-row-winner" : ""}}">
        <td><strong>${{win ? "★ " : ""}}${{name}}</strong></td>
        <td class="ada-mono">${{(m.f1_macro ?? 0).toFixed(4)}} <small style="color:var(--txt-muted)">${{ci}}</small></td>
        <td class="ada-mono">${{((m.accuracy ?? 0) * 100).toFixed(2)}}%</td>
        <td class="ada-mono">${{m.roc_auc !== null && m.roc_auc !== undefined ? m.roc_auc.toFixed(4) : "-"}}</td>
        <td class="ada-mono">${{m.cv_f1_mean !== undefined ? m.cv_f1_mean.toFixed(4) : "-"}}</td>
      </tr>`;
    }}
    tbl.innerHTML = htmlRows + "</tbody>";
  }}
  ADACharts.initDataTable("modelsTbl");

  // Sample Predictions Explorer Table
  const samplesSec = document.getElementById("samplesSection");
  if (DATA.samples && DATA.samples.length) {{
    const cols = Object.keys(DATA.samples[0]);
    const head = "<thead><tr>" + cols.map((c) => `<th data-sortable="true">${{c}} ⇅</th>`).join("") + "</tr></thead>";
    const body = "<tbody>" + DATA.samples.map((row) => {{
      return "<tr>" + cols.map((c) => {{
        const v = row[c];
        const isNum = typeof v === "number";
        return `<td class="${{isNum ? "ada-mono" : ""}}">${{isNum ? Number(v).toFixed(4).replace(/\\.0000$/, "") : (v ?? "-")}}</td>`;
      }}).join("") + "</tr>";
    }}).join("") + "</tbody>";
    document.getElementById("samplesTbl").innerHTML = head + body;
    ADACharts.initDataTable("samplesTbl", "sampleSearch");
  }} else {{
    samplesSec.style.display = "none";
  }}

  // Interactive Multi-Tab Chart Engine
  const cv = document.getElementById("canvasMain");
  const tabsEl = document.getElementById("chartTabs");
  const legendEl = document.getElementById("chartLegend");
  const titleEl = document.getElementById("chart1Title");

  function setActiveTab(tabId) {{
    Array.from(tabsEl.querySelectorAll(".ada-tab-btn")).forEach((b) => {{
      b.classList.toggle("is-active", b.dataset.tab === tabId);
    }});

    if (DATA.task_type === "regression") {{
      if (tabId === "parity") {{
        titleEl.textContent = "Real vs Predicho + Banda de Confianza 90% (Test)";
        legendEl.innerHTML = [
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-cyan)"></span> Dentro de IC 90%</span>`,
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-amber)"></span> Fuera de IC 90%</span>`,
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-indigo)"></span> Paridad Ideal (y = x)</span>`
        ].join("");
        ADACharts.scatterParity(cv, {{
          yTrue: DATA.chart.y_true || [],
          yPred: DATA.chart.y_pred || [],
          q90HalfWidth: DATA.chart.q90_half_width || 0,
          xLabel: `Valor Real (${{DATA.target_column}})`,
          yLabel: `Predicción (${{DATA.best_model}})`
        }});
      }} else if (tabId === "residuals") {{
        titleEl.textContent = "Diagnóstico de Residuos (Error = Real - Predicho)";
        legendEl.innerHTML = [
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-emerald)"></span> Línea Error Cero & Corredor ±IC 90%</span>`,
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-rose)"></span> Residuo Atípico</span>`
        ].join("");
        ADACharts.residuals(cv, {{
          yTrue: DATA.chart.y_true || [],
          yPred: DATA.chart.y_pred || [],
          q90HalfWidth: DATA.chart.q90_half_width || 0
        }});
      }} else {{
        titleEl.textContent = "Arena Visual de Modelos (RMSE en Test — Menor es Mejor)";
        legendEl.innerHTML = `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-cyan)"></span> Modelo Ganador Seleccionado ★</span>`;
        ADACharts.modelArena(cv, {{
          metricLabel: "RMSE Test",
          items: Object.entries(DATA.models).map(([name, m]) => ({{
            label: name,
            value: m.rmse ?? 0,
            isWinner: name === DATA.best_model,
            secondary: `R² Test: ${{(m.r2 ?? 0).toFixed(4)}}`
          }}))
        }});
      }}
    }} else {{
      if (tabId === "classes") {{
        titleEl.textContent = "Distribución Real vs Predicha por Clase (Test Set)";
        legendEl.innerHTML = [
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-cyan)"></span> Conteo Real</span>`,
          `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-emerald)"></span> Conteo Predicho</span>`
        ].join("");
        ADACharts.groupedBars(cv, {{
          categories: DATA.chart.classes || [],
          yLabel: "Muestras en Test",
          series: [
            {{ name: "Real", data: DATA.chart.true_counts || [] }},
            {{ name: "Predicho", data: DATA.chart.pred_counts || [] }}
          ]
        }});
      }} else {{
        titleEl.textContent = "Arena de Clasificadores (F1-Macro + Bigotes IC 95%)";
        legendEl.innerHTML = `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-amber)"></span> Intervalo Bootstrap 95%</span>`;
        ADACharts.modelArena(cv, {{
          metricLabel: "F1-Macro",
          items: Object.entries(DATA.models).map(([name, m]) => ({{
            label: name,
            value: m.f1_macro ?? 0,
            ciLow: m.f1_ci95 ? m.f1_ci95[0] : undefined,
            ciHigh: m.f1_ci95 ? m.f1_ci95[1] : undefined,
            isWinner: name === DATA.best_model,
            secondary: `Accuracy: ${{((m.accuracy ?? 0) * 100).toFixed(1)}}%`
          }}))
        }});
      }}
    }}
  }}

  const tabDefs = DATA.task_type === "regression"
    ? [
        {{ id: "parity", label: "Paridad & IC 90%" }},
        {{ id: "residuals", label: "Residuos" }},
        {{ id: "arena", label: "Arena RMSE" }}
      ]
    : [
        {{ id: "classes", label: "Clases Real vs Pred" }},
        {{ id: "arena", label: "Arena F1-Macro" }}
      ];

  tabsEl.innerHTML = tabDefs.map((t) =>
    `<button type="button" class="ada-tab-btn" data-tab="${{t.id}}">${{t.label}}</button>`
  ).join("");
  tabsEl.querySelectorAll(".ada-tab-btn").forEach((b) => {{
    b.addEventListener("click", () => setActiveTab(b.dataset.tab));
  }});
  setActiveTab(tabDefs[0].id);

  document.getElementById("btnExportPng").addEventListener("click", () => {{
    ADACharts.exportCanvasPNG("canvasMain", `${{DATA.dataset_name}}_${{DATA.target_column}}_ada_chart.png`);
  }});
  document.getElementById("btnExportCsv").addEventListener("click", () => {{
    const rows = (DATA.samples && DATA.samples.length)
      ? DATA.samples
      : Object.entries(DATA.models).map(([model, m]) => Object.assign({{ model }}, m));
    ADACharts.exportRowsCSV(rows, `${{DATA.dataset_name}}_${{DATA.target_column}}_ada_export.csv`);
  }});
}})();
</script>
</body>
</html>
"""
    safe_path.write_text(html, encoding="utf-8")
    return safe_path


def build_interactive_webapp(
    *,
    title: str,
    subtitle: str = "",
    output_filename: str = "ada_interactive_studio.html",
    theme: str = "dark",
    badge_text: str = "ADA STUDIO · Verificado",
    kpis: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    callouts: list[dict[str, Any]] | None = None,
    table_rows: list[dict[str, Any]] | None = None,
    table_title: str = "Explorador Interactivo de Datos",
    custom_html_section: str = "",
) -> dict[str, Any]:
    """Construye una aplicación web interactiva Bento Glass con gráficas Retina HiDPI,
    pestañas dinámicas, cambio de tema Oscuro/Claro, tablas con filtro/ordenamiento
    y auditoría automática DOM/CSS/Sandbox (`quality_score >= 92`).
    """
    for ch in charts or []:
        ch_type = str(ch.get("type", "")).strip().lower()
        has_data = bool(ch.get("series") or ch.get("items") or ch.get("y") or ch.get("data") or ch.get("y_true"))
        if ch_type in {"animation", "canvas", "scene", "game", "sprite"} or (not has_data and ch_type not in {"", "line", "bars", "bar", "scatter", "parity", "residuals", "arena"}):
            raise ValueError(
                "USO INCORRECTO DE HERRAMIENTA: `ada_build_interactive_webapp` es únicamente para dashboards de datos y gráficas estadísticas (X-Y, barras, KPIs). "
                "Para crear una ANIMACIÓN VISUAL, ESCENA ARTÍSTICA o SIMULACIÓN EN CANVAS (ej. un pajarito volando sobre edificios en Nueva York, paisajes, vehículos, física o juegos), "
                "llama inmediatamente a `ada_build_canvas_animation` escribiendo el código JavaScript real de dibujo Canvas 2D (`ctx.fillRect`, `ctx.arc`, `ctx.quadraticCurveTo`, etc.) en `draw_frame_js`."
            )

    clean_name = Path(output_filename).name
    if not clean_name.endswith(".html"):
        clean_name += ".html"
    out_path = assert_safe_write_path(OUTPUTS_DIR / clean_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    studio_css, charts_js = load_studio_assets()
    scene_js = load_scene_engine_js()

    initial_theme = "light" if theme.strip().lower() == "light" else "dark"
    spec = {
        "title": title,
        "subtitle": subtitle,
        "badge_text": badge_text,
        "kpis": kpis or [],
        "charts": charts or [],
        "callouts": callouts or [],
        "table_rows": table_rows or [],
        "table_title": table_title,
    }
    spec_json = json.dumps(spec, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es" data-theme="{initial_theme}">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
{studio_css}
</style>
</head>
<body>
  <nav class="ada-topbar">
    <div class="ada-brand">
      <span class="ada-brand-orb"></span>
      <span>ADA WEB STUDIO</span>
      <span class="ada-pill">Creador: <strong>Diego López</strong></span>
    </div>
    <div class="ada-topbar-actions">
      <span class="ada-badge ada-badge-info" id="studioBadge"></span>
      <button type="button" class="ada-btn" id="btnStudioPng">⬇ PNG</button>
      <button type="button" class="ada-btn" id="btnStudioCsv">⬇ CSV</button>
      <button type="button" class="ada-btn ada-btn-primary" id="adaThemeBtn">☀ Modo Claro</button>
    </div>
  </nav>

  <main class="ada-shell">
    <section class="ada-hero">
      <div>
        <h1 class="ada-hero-title" id="studioTitle"></h1>
        <p class="ada-hero-sub" id="studioSub"></p>
      </div>
    </section>

    <section class="ada-kpi-grid" id="studioKpis"></section>

    <section class="ada-bento-grid" id="studioMainBento">
      <article class="ada-card ada-col-8" id="studioChartCard">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title" id="activeChartTitle">Visualización Interactiva</h2>
            <p class="ada-card-subtitle" id="activeChartSub">Inspección interactiva con cursor de precisión Retina HiDPI</p>
          </div>
          <div class="ada-tabs" id="studioTabs"></div>
        </div>
        <div class="ada-chart-stage">
          <canvas id="studioCanvas"></canvas>
        </div>
        <div class="ada-chart-legend" id="studioLegend"></div>
      </article>

      <article class="ada-card ada-col-4" id="studioCalloutCard">
        <div class="ada-card-header">
          <div>
            <h2 class="ada-card-title">Hallazgos & Notas Técnicas</h2>
            <p class="ada-card-subtitle">Síntesis estructurada de ADA</p>
          </div>
        </div>
        <div class="ada-callout-stack" id="studioCallouts"></div>
      </article>
    </section>

    <section class="ada-bento-grid" id="studioTableSection">
      <article class="ada-card ada-col-12">
        <div class="ada-table-toolbar">
          <div>
            <h2 class="ada-card-title" id="studioTableTitle"></h2>
            <p class="ada-card-subtitle">Búsqueda instantánea y ordenamiento por columna</p>
          </div>
          <input type="search" class="ada-search-input" id="studioSearch" placeholder="Buscar en tabla..." />
        </div>
        <div class="ada-table-wrap">
          <table class="ada-table" id="studioTable"></table>
        </div>
      </article>
    </section>

    {f'<section class="ada-bento-grid"><article class="ada-card ada-col-12">{custom_html_section}</article></section>' if custom_html_section.strip() else ''}

    <footer class="ada-footer">
      <span>ADA Web Studio v2.0 · Diseñado localmente para Diego López</span>
      <span class="ada-mono">HiDPI 60 FPS · Bento Glass Architecture</span>
    </footer>
  </main>

<script>
{charts_js}
</script>
<script>
{scene_js}
</script>
<script>
(function () {{
  const SPEC = {spec_json};
  ADACharts.initThemeToggle("adaThemeBtn");
  if ("{initial_theme}" === "light") {{
    document.documentElement.setAttribute("data-theme", "light");
  }}

  document.getElementById("studioBadge").textContent = "✓ " + SPEC.badge_text;
  document.getElementById("studioTitle").textContent = SPEC.title;
  document.getElementById("studioSub").textContent = SPEC.subtitle || "";

  const accents = ["var(--accent-cyan)", "var(--accent-indigo)", "var(--accent-emerald)", "var(--accent-amber)"];
  const kpiEl = document.getElementById("studioKpis");
  if (SPEC.kpis && SPEC.kpis.length) {{
    kpiEl.innerHTML = SPEC.kpis.map((k, idx) => `
      <div class="ada-kpi-card" style="--kpi-accent: ${{k.accent || accents[idx % accents.length]}}">
        <div class="ada-kpi-label">${{k.label || "Indicador"}}</div>
        <div class="ada-kpi-value">${{k.value ?? "-"}}</div>
        ${{k.delta || k.meta ? `<div class="ada-kpi-meta">${{k.delta ? `<span class="ada-badge ada-badge-info">${{k.delta}}</span>` : ""}} ${{k.meta || ""}}</div>` : ""}}
      </div>
    `).join("");
  }} else {{
    kpiEl.style.display = "none";
  }}

  const calloutCard = document.getElementById("studioCalloutCard");
  const chartCard = document.getElementById("studioChartCard");
  if (SPEC.callouts && SPEC.callouts.length) {{
    document.getElementById("studioCallouts").innerHTML = SPEC.callouts.map((c) => {{
      const sev = (c.severity || "info").toLowerCase();
      const cls = sev === "ok" ? "is-ok" : (sev === "caution" || sev === "warn" ? "is-caution" : (sev === "risk" || sev === "error" ? "is-risk" : "is-info"));
      return `<div class="ada-callout ${{cls}}">
        ${{c.title ? `<div class="ada-callout-title">${{c.title}}</div>` : ""}}
        <div>${{c.text || ""}}</div>
      </div>`;
    }}).join("");
  }} else {{
    calloutCard.style.display = "none";
    chartCard.className = "ada-card ada-col-12";
  }}

  const cv = document.getElementById("studioCanvas");
  const tabsEl = document.getElementById("studioTabs");
  const legendEl = document.getElementById("studioLegend");

  function renderSpecChart(idx) {{
    const ch = SPEC.charts[idx];
    if (!ch) return;
    Array.from(tabsEl.querySelectorAll(".ada-tab-btn")).forEach((b, i) => b.classList.toggle("is-active", i === idx));
    document.getElementById("activeChartTitle").textContent = ch.title || "Visualización Interactiva";
    document.getElementById("activeChartSub").textContent = ch.subtitle || "Pasa el cursor para inspeccionar coordenadas exactas";

    const kind = (ch.type || "line").toLowerCase();
    const palCols = ["var(--accent-cyan)", "var(--accent-emerald)", "var(--accent-indigo)", "var(--accent-amber)", "var(--accent-rose)"];
    if (kind === "scatter" || kind === "parity") {{
      legendEl.innerHTML = `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-cyan)"></span> Muestras</span>`;
      ADACharts.scatterParity(cv, {{
        yTrue: ch.x || ch.y_true || [],
        yPred: ch.y || ch.y_pred || [],
        q90HalfWidth: ch.band || 0,
        xLabel: ch.x_label || "X",
        yLabel: ch.y_label || "Y"
      }});
    }} else if (kind === "residuals") {{
      legendEl.innerHTML = `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:var(--accent-emerald)"></span> Referencia Error Cero</span>`;
      ADACharts.residuals(cv, {{
        yTrue: ch.y_true || [],
        yPred: ch.y_pred || [],
        q90HalfWidth: ch.band || 0
      }});
    }} else if (kind === "arena") {{
      legendEl.innerHTML = "";
      ADACharts.modelArena(cv, {{
        metricLabel: ch.y_label || "Valor",
        items: ch.items || []
      }});
    }} else if (kind === "bar" || kind === "bars" || kind === "grouped_bars") {{
      const ser = ch.series || [{{ name: ch.title || "Serie", data: ch.y || ch.data || [] }}];
      legendEl.innerHTML = ser.map((s, i) =>
        `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:${{palCols[i % palCols.length]}}"></span> ${{s.name}}</span>`
      ).join("");
      ADACharts.groupedBars(cv, {{
        categories: ch.categories || ch.x || [],
        xLabel: ch.x_label || "",
        yLabel: ch.y_label || "Valor",
        series: ser
      }});
    }} else {{
      const ser = ch.series || [{{ name: ch.title || "Serie", data: ch.y || ch.data || [] }}];
      legendEl.innerHTML = ser.map((s, i) =>
        `<span class="ada-legend-item"><span class="ada-legend-dot" style="background:${{palCols[i % palCols.length]}}"></span> ${{s.name}}</span>`
      ).join("");
      ADACharts.lineSeries(cv, {{
        xValues: ch.x,
        xLabel: ch.x_label || "X",
        yLabel: ch.y_label || "Y",
        series: ser
      }});
    }}
  }}

  if (SPEC.charts && SPEC.charts.length) {{
    tabsEl.innerHTML = SPEC.charts.map((c, i) =>
      `<button type="button" class="ada-tab-btn">${{c.tab_label || c.title || ("Vista " + (i + 1))}}</button>`
    ).join("");
    Array.from(tabsEl.querySelectorAll(".ada-tab-btn")).forEach((b, i) => {{
      b.addEventListener("click", () => renderSpecChart(i));
    }});
    renderSpecChart(0);
  }} else {{
    chartCard.style.display = "none";
    calloutCard.className = "ada-card ada-col-12";
  }}

  const tblSec = document.getElementById("studioTableSection");
  if (SPEC.table_rows && SPEC.table_rows.length) {{
    document.getElementById("studioTableTitle").textContent = SPEC.table_title || "Datos";
    const cols = Object.keys(SPEC.table_rows[0]);
    const head = "<thead><tr>" + cols.map((c) => `<th data-sortable="true">${{c}} ⇅</th>`).join("") + "</tr></thead>";
    const body = "<tbody>" + SPEC.table_rows.map((r) =>
      "<tr>" + cols.map((c) => `<td class="${{typeof r[c] === "number" ? "ada-mono" : ""}}">${{r[c] ?? "-"}}</td>`).join("") + "</tr>"
    ).join("") + "</tbody>";
    document.getElementById("studioTable").innerHTML = head + body;
    ADACharts.initDataTable("studioTable", "studioSearch");
  }} else {{
    tblSec.style.display = "none";
  }}

  document.getElementById("btnStudioPng").addEventListener("click", () => {{
    ADACharts.exportCanvasPNG("studioCanvas", "ada_studio_chart.png");
  }});
  document.getElementById("btnStudioCsv").addEventListener("click", () => {{
    if (SPEC.table_rows && SPEC.table_rows.length) {{
      ADACharts.exportRowsCSV(SPEC.table_rows, "ada_studio_data.csv");
    }}
  }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    audit = audit_html_quality(out_path)
    result = {
        "verified": audit["passed"],
        "interactive_html_path": str(out_path),
        "quality_audit": audit,
    }
    record_audit_event("build_interactive_webapp", result, output_path=out_path)
    return result


def build_canvas_animation_studio(
    *,
    title: str,
    subtitle: str = "Animación interactiva en tiempo real a 60 FPS (Retina HiDPI)",
    output_filename: str = "ada_canvas_animation.html",
    draw_frame_js: str,
    init_state_js: str = "",
    controls: list[dict[str, Any]] | None = None,
    badge_text: str = "ADA ANIMATION STUDIO · 60 FPS",
    canvas_height_px: int = 500,
) -> dict[str, Any]:
    """Construye un estudio de animación visual interactiva HTML5 Canvas a 60 FPS dentro del
    sistema Bento Glass de ADA (escalado Retina HiDPI automático, controles deslizantes en vivo,
    medidor de FPS reales, Pausa/Reanudar, Reiniciar, interacción con el mouse y captura PNG).

    Parámetros que recibe automáticamente el código `draw_frame_js` en cada fotograma:
      - `ctx`: CanvasRenderingContext2D ya escalado a Retina HiDPI
      - `w`, `h`: ancho y alto lógicos en píxeles CSS del escenario
      - `t`: tiempo acumulado en segundos (float)
      - `dt`: delta de tiempo entre fotogramas en segundos (float)
      - `params`: objeto con los valores en vivo de todos los sliders definidos en `controls`
      - `mouse`: `{ x, y, inside, clicked }` para interactividad con el cursor
      - `state`: objeto persistente entre fotogramas inicializado opcionalmente en `init_state_js`
    """
    code = (draw_frame_js or "").strip()
    if len(code) < 40 or ("ctx." not in code and "ADAScene." not in code):
        raise ValueError(
            "`draw_frame_js` debe contener código JavaScript real usando `ADAScene.*` y/o `ctx.*`."
        )

    # Quitar encabezados `function nombre(...)` para verificar que `t` o `dt` realmente animen el cuerpo
    body_without_sig = re.sub(r"function\s+[A-Za-z0-9_$]*\s*\([^)]*\)", "", code)
    has_time_motion = bool(re.search(r"\b(t|dt)\b", body_without_sig))
    has_visual_craft = (
        "ADAScene." in code
        or "createLinearGradient" in code
        or "createRadialGradient" in code
        or "quadraticCurveTo" in code
        or "bezierCurveTo" in code
    )
    if not has_time_motion or not has_visual_craft:
        failed = []
        if not has_time_motion:
            failed.append("NO USA `t` NI `dt`: nada se mueve con el tiempo; actualiza posiciones con `dt` (p. ej. state.x += state.vx * dt) o con Math.sin(t)")
        if not has_visual_craft:
            failed.append("FALTA PROFUNDIDAD VISUAL: añade ctx.createRadialGradient/createLinearGradient (brillo en esferas, fondos) o curvas; "
                          "para física con formas simples basta un degradado radial en cada cuerpo, sombras y un fondo con gradiente")
        raise ValueError(
            "CONDICIÓN QUE FALLÓ -> " + " | ".join(failed) + ".\n"
            "ANIMACIÓN RECHAZADA POR EL CRÍTICO VISUAL DE ADA: Tu código es estático (no mueve posiciones con `t` y `dt`) "
            "o dibuja figuras planas básicas ('bolitas y palitos') sin profundidad visual. "
            "Reescribe `init_state_js` y `draw_frame_js` AHORA usando movimiento real continuo (`t`, `dt`) y las primitivas "
            "articuladas a 60 FPS ya incluidas en `ADAScene`:\n"
            "- Entornos: `ADAScene.sky(ctx, w, h, { t, dusk: params.dusk ?? 0.35 })`, "
            "`ADAScene.hillsAndMeadow(ctx, w, h, { t, dusk: params.dusk ?? 0.35 })`, "
            "`ADAScene.forestTrees(ctx, w, h, { t, count: 8, dusk: params.dusk ?? 0.35 })`, "
            "`ADAScene.citySkyline(ctx, w, h, { t, scrollSpeed: 50, dusk: 0.65 })`, `ADAScene.water(ctx, w, h, { t })`\n"
            "- Actores articulados: `ADAScene.fox(ctx, x, y, { t, dir, scale: 1.1, speed })`, "
            "`ADAScene.rabbit(ctx, x, y, { t, dir, scale: 1.0, speed, phase })`, "
            "`ADAScene.bird(ctx, x, y, { t, dir, scale, speed })`, `ADAScene.fish(...)`, `ADAScene.car(...)`, `ADAScene.rocket(...)`, `ADAScene.planet(...)`, `ADAScene.person(...)`\n"
            "- Física persecución/evasión y polvo: `ADAScene.stepPursuit(state.fox, state.rabbits, bounds, 165 * speed, dt)`, "
            "`ADAScene.stepEvade(rab, state.fox, bounds, 150 * speed, dt, t)`, `ADAScene.spawnDust(state.dust, x, y, 2)`, `ADAScene.updateDust(ctx, state.dust, dt)`.\n"
            "IMPORTANTE: Escribe sentencias directas sobre `state`, `ctx` y `ADAScene` (no necesitas envolver en `function drawFrame`)."
        )

    clean_name = Path(output_filename).name
    if not clean_name.endswith(".html"):
        clean_name += ".html"
    out_path = assert_safe_write_path(OUTPUTS_DIR / clean_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    studio_css, charts_js = load_studio_assets()
    scene_js = load_scene_engine_js()
    scene_js = load_scene_engine_js()

    ctrl_list = controls or []
    controls_json = json.dumps(ctrl_list, ensure_ascii=False)
    stage_h = max(320, min(760, int(canvas_height_px)))

    html = f"""<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
{studio_css}
.ada-cinema-stage {{
  position: relative;
  width: 100%;
  height: {stage_h}px;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--border-subtle);
  background: #050814;
  box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.45);
}}
.ada-cinema-stage canvas {{
  display: block;
  width: 100%;
  height: 100%;
  cursor: pointer;
}}
.ada-hud-overlay {{
  position: absolute;
  top: 12px;
  right: 12px;
  display: flex;
  gap: 8px;
  pointer-events: none;
}}
.ada-hud-chip {{
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(6, 9, 19, 0.72);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(148, 163, 184, 0.22);
  color: #e2e8f0;
  font-size: 0.74rem;
  font-weight: 700;
}}
</style>
</head>
<body>
  <nav class="ada-topbar">
    <div class="ada-brand">
      <span class="ada-brand-orb"></span>
      <span>ADA ANIMATION STUDIO</span>
      <span class="ada-pill">Creador: <strong>Diego López</strong></span>
    </div>
    <div class="ada-topbar-actions">
      <span class="ada-badge ada-badge-info">✓ {badge_text}</span>
      <button type="button" class="ada-btn" id="btnPlayPause">⏸ Pausar</button>
      <button type="button" class="ada-btn" id="btnResetAnim">↺ Reiniciar</button>
      <button type="button" class="ada-btn" id="btnShotPng">⬇ Capturar PNG</button>
      <button type="button" class="ada-btn ada-btn-primary" id="adaThemeBtn">☀ Modo Claro</button>
    </div>
  </nav>

  <main class="ada-shell">
    <section class="ada-hero">
      <div>
        <h1 class="ada-hero-title">{title}</h1>
        <p class="ada-hero-sub">{subtitle}</p>
      </div>
      <div class="ada-pill-row">
        <span class="ada-pill">Motor: <strong>HTML5 Canvas 2D HiDPI</strong></span>
        <span class="ada-pill">Interacción: <strong>Mueve el cursor o ajusta controles en vivo</strong></span>
      </div>
    </section>

    <section class="ada-bento-grid">
      <article class="ada-card ada-col-12">
        <div class="ada-controls-row" id="animControlsBox"></div>
        <div class="ada-cinema-stage" id="cinemaStage">
          <canvas id="animCanvas"></canvas>
          <div class="ada-hud-overlay">
            <span class="ada-hud-chip ada-mono" id="fpsChip">60 FPS</span>
            <span class="ada-hud-chip ada-mono" id="timeChip">t = 0.0s</span>
          </div>
        </div>
      </article>
    </section>

    <footer class="ada-footer">
      <span>ADA Creative & Physics Canvas Engine · Renderizado local acelerado a 60 FPS</span>
      <span class="ada-mono">Retina HiDPI · Interactive Scene</span>
    </footer>
  </main>

<script>
{charts_js}
</script>
<script>
{scene_js}
</script>
<script>
(function () {{
  ADACharts.initThemeToggle("adaThemeBtn");
  const CONTROLS = {controls_json};
  const params = {{}};
  const ctrlBox = document.getElementById("animControlsBox");

  if (CONTROLS && CONTROLS.length) {{
    ctrlBox.innerHTML = CONTROLS.map((c) => {{
      const val = c.value !== undefined ? Number(c.value) : Number(c.min || 0);
      params[c.key] = val;
      return `<div class="ada-control-group">
        <label for="ctrl_${{c.key}}">
          <span>${{c.label || c.key}}</span>
          <strong class="ada-mono" id="lbl_${{c.key}}">${{val}}${{c.unit || ""}}</strong>
        </label>
        <input type="range" id="ctrl_${{c.key}}" min="${{c.min ?? 0}}" max="${{c.max ?? 100}}" step="${{c.step ?? 1}}" value="${{val}}" />
      </div>`;
    }}).join("");
    CONTROLS.forEach((c) => {{
      const el = document.getElementById("ctrl_" + c.key);
      const lbl = document.getElementById("lbl_" + c.key);
      if (el) {{
        el.addEventListener("input", () => {{
          params[c.key] = Number(el.value);
          if (lbl) lbl.textContent = el.value + (c.unit || "");
        }});
      }}
    }});
  }} else {{
    ctrlBox.style.display = "none";
  }}

  const canvas = document.getElementById("animCanvas");
  const stage = document.getElementById("cinemaStage");
  const ctx = canvas.getContext("2d");
  let logicalW = 900;
  let logicalH = {stage_h};

  function resize() {{
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const r = stage.getBoundingClientRect();
    logicalW = Math.max(300, Math.floor(r.width || 900));
    logicalH = Math.max(260, Math.floor(r.height || {stage_h}));
    canvas.width = Math.floor(logicalW * dpr);
    canvas.height = Math.floor(logicalH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
  resize();
  if (window.ResizeObserver) {{
    new ResizeObserver(resize).observe(stage);
  }} else {{
    window.addEventListener("resize", resize);
  }}

  const mouse = {{ x: logicalW * 0.5, y: logicalH * 0.35, inside: false, down: false, clicks: [] }};
  canvas.addEventListener("mousemove", (e) => {{
    const r = canvas.getBoundingClientRect();
    mouse.x = e.clientX - r.left;
    mouse.y = e.clientY - r.top;
    mouse.inside = true;
  }});
  canvas.addEventListener("mouseleave", () => {{ mouse.inside = false; mouse.down = false; }});
  canvas.addEventListener("mousedown", (e) => {{
    mouse.down = true;
    const r = canvas.getBoundingClientRect();
    mouse.clicks.push({{ x: e.clientX - r.left, y: e.clientY - r.top, t: simTime }});
    if (mouse.clicks.length > 12) mouse.clicks.shift();
  }});
  window.addEventListener("mouseup", () => {{ mouse.down = false; }});

  let state = {{}};
  function initUserState(s, w, h, params) {{
    let state = s || {{}};
    try {{
      {init_state_js}
    }} catch (e) {{
      console.error("User init error:", e);
    }}
    if (typeof initState === "function") {{
      try {{
        const ret = initState(state, w, h, params);
        if (ret && typeof ret === "object") Object.assign(state, ret);
      }} catch (e) {{
        console.error("initState() call error:", e);
      }}
    }}
    return state;
  }}
  function drawUserFrame(ctx, w, h, t, dt, params, mouse, state) {{
    {code}
    if (typeof drawFrame === "function") {{
      drawFrame(ctx, w, h, t, dt, params, mouse, state);
    }} else if (typeof render === "function") {{
      render(ctx, w, h, t, dt, params, mouse, state);
    }} else if (typeof draw === "function") {{
      draw(ctx, w, h, t, dt, params, mouse, state);
    }}
  }}

  let simTime = 0;
  let running = true;
  let lastTs = performance.now();
  let fpsFrames = 0;
  let fpsTimer = 0;
  const fpsChip = document.getElementById("fpsChip");
  const timeChip = document.getElementById("timeChip");

  try {{
    state = initUserState(state, logicalW, logicalH, params) || state;
  }} catch (err) {{
    console.error("Init error:", err);
  }}

  function loop(now) {{
    const rawDt = Math.min(0.05, Math.max(0.001, (now - lastTs) / 1000));
    lastTs = now;
    if (running) {{
      simTime += rawDt;
      fpsFrames++;
      fpsTimer += rawDt;
      if (fpsTimer >= 0.4) {{
        fpsChip.textContent = Math.round(fpsFrames / fpsTimer) + " FPS";
        fpsFrames = 0;
        fpsTimer = 0;
      }}
      timeChip.textContent = "t = " + simTime.toFixed(1) + "s";

      ctx.save();
      ctx.clearRect(0, 0, logicalW, logicalH);
      try {{
        drawUserFrame(ctx, logicalW, logicalH, simTime, rawDt, params, mouse, state);
      }} catch (err) {{
        ctx.fillStyle = "#1e1b4b";
        ctx.fillRect(0, 0, logicalW, logicalH);
        ctx.fillStyle = "#fda4af";
        ctx.font = "600 14px JetBrains Mono, monospace";
        ctx.fillText("Error en draw_frame_js: " + err.message, 24, 42);
      }}
      ctx.restore();
    }}
    requestAnimationFrame(loop);
  }}
  requestAnimationFrame(loop);

  const btnPlay = document.getElementById("btnPlayPause");
  btnPlay.addEventListener("click", () => {{
    running = !running;
    lastTs = performance.now();
    btnPlay.textContent = running ? "⏸ Pausar" : "▶ Reanudar";
  }});

  document.getElementById("btnResetAnim").addEventListener("click", () => {{
    simTime = 0;
    state = {{}};
    state = initUserState(state, logicalW, logicalH, params) || state;
  }});

  document.getElementById("btnShotPng").addEventListener("click", () => {{
    ADACharts.exportCanvasPNG("animCanvas", "{Path(clean_name).stem}.png");
  }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    audit = audit_html_quality(out_path)
    result = {
        "verified": audit["passed"],
        "interactive_html_path": str(out_path),
        "quality_audit": audit,
    }
    record_audit_event("build_canvas_animation_studio", result, output_path=out_path)
    return result
