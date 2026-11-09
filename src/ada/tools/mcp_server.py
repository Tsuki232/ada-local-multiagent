"""Servidor MCP oficial de ADA (stdio) para DeepSeek Harness Desktop, Web y SDK."""

from __future__ import annotations

import json
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

from ada.calculus.verified_math import (
    extract_equation_from_image,
    solve_and_verify_ode,
    verify_definite_integral,
    verify_derivative,
)
from ada.clean.pipeline import clean_and_adapt_dataset
from ada.config import ModelRoleConfig
from ada.ingest.loader import list_database_tables, profile_dataset
from ada.pendulum.hardware_bridge import (
    VIRTUAL_DEVICE_URI,
    list_electronic_devices,
    run_pendulum_hardware_control,
)
from ada.pendulum.simulator import (
    analyze_pendulum_linear_approximation,
    build_pendulum_interactive_html,
)
from ada.stats.predictor import evaluate_and_predict_database
from ada.viz.neural_canvas import build_neural_network_live_canvas
from ada.viz.brain_simulator import build_neural_brain_simulation
from ada.viz.math_textbook import build_math_textbook_html
from ada.viz.live_vision_studio import build_live_vision_dashboard_html
from ada.vision.adb_bridge import (
    capture_adb_frame,
    find_adb_executable,
    find_scrcpy_executable,
    list_adb_devices,
)
from ada.vision.realtime_tracker import RealTimePhysicsTracker
from ada.vision.synthetic_stream import SyntheticPhysicsLaboratoryStream
from ada.cognition.autonomous_mind import get_autonomous_mind
from ada.training.continuous_learner import (
    get_continual_learning_stats,
    record_interaction,
    retrieve_relevant_memories,
)
from ada.viz.web_studio import (
    audit_html_quality,
    build_canvas_animation_studio,
    build_interactive_webapp,
)

ADA_SYSTEM_INSTRUCTIONS = """Eres ADA (núcleo Ada Q / Ada r1), creada por Diego López: sistema local multiagente de ingenieria de datos, estadistica rigurosa, calculo verificado, control electronico y arquitectura web interactiva Bento Glass / Canvas 60 FPS.
Responde SIEMPRE 100 % en español, con rigor tecnico directo, sin rodeos ni relleno.
REGLAS INNEGOCIABLES:
1. CERO CALCULO DE CABEZA: Toda estadistica, prediccion, derivada, integral o EDO DEBE ejecutarse con las herramientas `mcp__ada__*`, que verifican cada resultado por dos metodos independientes (ej. SymPy vs SciPy DOP853, Pandas vs DuckDB, scikit-learn vs statsmodels, PyTorch CUDA vs NumPy). Reporta siempre `[VERIFICADO: metodos]`.
2. DATOS ORIGINALES INMUTABLES: Nunca modifiques archivos originales ni `data/raw/`. Las limpiezas y predicciones van a `data/work/` y las animaciones/dashboards a `outputs/`.
3. CUESTIONA TUS RESULTADOS: Al evaluar o predecir bases de datos reales, expone siempre las alertas del Critico Estadistico (`critic_audit`: data leakage, potencia muestral, normalidad de residuos, heterocedasticidad o ganancia real frente al `DummyBaseline`).
4. ANIMACIONES VISUALES CREATIVAS Y SIMULACIONES EN CANVAS 60 FPS:
   - Si pide ver cómo funciona tu red neuronal, Transformer o GPU en tiempo real: ejecuta `mcp__ada__ada_neural_network_live_canvas` y luego `present`.
   - Si pide el péndulo o aproximación lineal: ejecuta `mcp__ada__ada_pendulum_approximation_lab` y luego `present`.
   - Si pide CUALQUIER OTRA ANIMACIÓN VISUAL, ESCENA ARTÍSTICA, PAISAJE, CIUDAD, ANIMAL, VEHÍCULO, FÍSICA LIBRE O JUEGO (ej. "un pajarito volando sobre un edificio en New York", "sistema solar", "lluvia"): **JAMÁS uses `ada_build_interactive_webapp`** (esa es solo para gráficas estadísticas X-Y). Ejecuta **`mcp__ada__ada_build_canvas_animation`** escribiendo código rico de dibujo HTML5 Canvas 2D en `draw_frame_js` (`ctx.createLinearGradient`, edificios parallax con ventanas iluminadas, nubes, anatomía detallada con alas batiendo con `Math.sin(t * ...)`, partículas e interacción con `mouse`) + sliders en `controls_json`, y entrégala con `present`.
5. DASHBOARDS DE DATOS Y ARQUITECTURA WEB (PROHIBIDO CREAR "HOJAS DE CUADERNO"):
   - Para dashboards de datos, KPIs, tablas y gráficas estadísticas (X-Y, barras, residuos): usa `mcp__ada__ada_build_interactive_webapp` (o `mcp__ada__ada_predict_from_database` para ML).
   - Siempre que crees o edites manualmente cualquier `.html`, ejecuta `mcp__ada__ada_audit_webpage_design(html_path=...)` ANTES de llamar a `present` (`grade A+`).
6. PROYECTO DEL PENDULO (MODO TUTOR): Entrega calculos exactos verificados, simulaciones animadas a 60 FPS y control de hardware/gemelo virtual, pero guia al estudiante con preguntas razonadas sin redactarle su informe universitario final a menos que el te muestre primero su propio desarrollo.
7. TRANSMISIÓN DE CÁMARA POR ADB Y VISIÓN EN VIVO:
   - Si pide ver video en tiempo real, procesar la cámara de un celular por ADB o hacer cálculos de visión por computadora en vivo: ejecuta `mcp__ada__ada_build_live_vision_studio` y luego `present` con `outputs/ada_live_vision_dashboard.html`, e inspecciona los dispositivos conectados con `mcp__ada__ada_inspect_adb_devices`.
8. ESTÁNDAR DE ESCRITURA MATEMÁTICA DE LIBRO DE TEXTO (CÁLCULO RIGUROSO):
   - PROHIBIDO escribir fórmulas matemáticas con listas de viñetas rotas o dólares desalineados (ej. '• $ \\frac{d}{dx}[3x^4] = 12x^3 $').
   - Toda deducción matemática DEBE presentarse como en los libros de texto universitarios (Stewart, Spivak, Apostol): en bloques display continuos $$ ... $$ usando entornos alineados limpios (\\begin{aligned} ... \\end{aligned}), con deducción fluida y elegante.
   - En consola CLI: presenta las fórmulas con renderizado 2D limpio o Unicode (12x³ - 6x² + 5), jamás código LaTeX crudo sin compilar.
9. MEMORIA CONTINUA Y AUTO-APRENDIZAJE:
   - Registra aprendizajes y preferencias en el banco de memoria episódica continua para que Ada aprenda de cada conversación.
"""

mcp = MCPServer("ada", instructions=ADA_SYSTEM_INSTRUCTIONS, version="0.1.0")


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@mcp.tool(structured_output=False)
def ada_profile_database(
    file_path: str,
    table_or_sheet: str = "",
    sql_query: str = "",
) -> str:
    """Evalua y perfila una base de datos real (CSV, Parquet, Excel, SQLite, DuckDB, SPSS, Stata, JSON) verificando conteos y medias con Pandas + DuckDB."""
    tables = list_database_tables(file_path)
    report = profile_dataset(file_path, table_or_sheet=table_or_sheet, sql_query=sql_query)
    if tables:
        report["database_tables"] = tables
    return _to_json(report)


@mcp.tool(structured_output=False)
def ada_clean_filter_adapt_data(
    input_path: str,
    output_filename: str = "",
    table_or_sheet: str = "",
    where_sql: str = "",
    order_by_sql: str = "",
    select_columns_csv: str = "",
    drop_duplicates: bool = True,
    normalize_column_names: bool = False,
    missing_strategy: str = "keep",
) -> str:
    """Limpia, filtra con SQL (where_sql), ordena (order_by_sql) y adapta columnas guardando una copia trazable con SHA-256 en data/work/.

    missing_strategy puede ser 'keep' (recomendado antes de predecir con ML para evitar data leakage), 'drop_rows' o 'median_mode'.
    select_columns_csv: lista opcional de columnas separadas por coma (vacio = todas).
    """
    cols = [c.strip() for c in select_columns_csv.split(",") if c.strip()] if select_columns_csv.strip() else None
    res = clean_and_adapt_dataset(
        input_path,
        output_filename=output_filename,
        table_or_sheet=table_or_sheet,
        where_sql=where_sql,
        order_by_sql=order_by_sql,
        select_columns=cols,
        drop_duplicates=drop_duplicates,
        normalize_column_names=normalize_column_names,
        missing_strategy=missing_strategy,
    )
    return _to_json(res)


@mcp.tool(structured_output=False)
def ada_scrape_tables(url: str, table: int = -1, match: str = "", output_filename: str = "") -> str:
    """Raspa tablas HTML de una URL publica y las guarda como CSV verificado (2 analizadores + DuckDB) en data/work/ con procedencia y SHA-256.
    Sin table ni match: LISTA las tablas (indice, filas, columnas). Con table=<i> o match='texto': extrae esa y la guarda.
    Respeta robots.txt, bloquea redes locales y no ejecuta JavaScript."""
    from ada.scraping.web_tables import scrape_tables

    return _to_json(scrape_tables(url, table=table, match=match, output_filename=output_filename))


@mcp.tool(structured_output=False)
def ada_predict_from_database(
    input_path: str,
    target_column: str,
    table_or_sheet: str = "",
    where_sql: str = "",
    exclude_columns_csv: str = "",
    task_type: str = "auto",
    test_size: float = 0.2,
    predict_on_path: str = "",
    seed: int = 42,
) -> str:
    """Entrena modelos predictivos (DummyBaseline vs Lineal/statsmodels vs HistGradientBoosting) con split estricto sin fuga, audita con el Critico Estadistico, predice datos con intervalo 90% y genera un dashboard HTML interactivo en outputs/."""
    excl = [c.strip() for c in exclude_columns_csv.split(",") if c.strip()] if exclude_columns_csv.strip() else None
    res = evaluate_and_predict_database(
        input_path,
        target_column,
        table_or_sheet=table_or_sheet,
        where_sql=where_sql,
        exclude_columns=excl,
        task_type=task_type,
        test_size=test_size,
        predict_on_path=predict_on_path,
        seed=seed,
    )
    return _to_json(res)


@mcp.tool(structured_output=False)
def ada_verify_calculus(
    operation: str,
    expression: str,
    variable: str = "x",
    order: int = 1,
    eval_point: float = 0.75,
    lower_limit: float = 0.0,
    upper_limit: float = 1.0,
) -> str:
    """Calcula y verifica por doble metodo (SymPy simbolico + SciPy/mpmath numerico) una derivada (operation='derivative') o integral definida (operation='integral')."""
    op = operation.strip().lower()
    if op in {"derivative", "derivada", "diff"}:
        return _to_json(verify_derivative(expression, variable=variable, order=order, eval_point=eval_point))
    if op in {"integral", "integrate"}:
        return _to_json(verify_definite_integral(expression, variable=variable, lower_limit=lower_limit, upper_limit=upper_limit))
    raise ValueError("operation debe ser 'derivative' o 'integral'.")


@mcp.tool(structured_output=False)
def ada_solve_verified_ode(
    ode_rhs: str,
    order: int = 2,
    t0: float = 0.0,
    t_end: float = 6.0,
    y0: float = 0.35,
    dy0: float = 0.0,
    parameters_json: str = "{}",
) -> str:
    """Resuelve y verifica una EDO de orden 1 (dy/dt = ode_rhs) u orden 2 (d2y/dt2 = ode_rhs, usando t, y, dy) contrastando SymPy/Radau vs SciPy DOP853."""
    params_raw = json.loads(parameters_json) if parameters_json.strip() else {}
    params = {str(k): float(v) for k, v in params_raw.items()}
    return _to_json(
        solve_and_verify_ode(
            ode_rhs,
            order=order,
            t0=t0,
            t_end=t_end,
            y0=y0,
            dy0=dy0,
            parameters=params,
        )
    )


@mcp.tool(structured_output=False)
def ada_read_graphical_equation(image_path: str) -> str:
    """Lee una ecuacion diferencial escrita graficamente en una imagen (PNG/JPG/WebP) usando OCR ONNX local y VLM multimodal en Ollama si esta disponible."""
    return _to_json(extract_equation_from_image(image_path))


@mcp.tool(structured_output=False)
def ada_pendulum_approximation_lab(
    length_m: float = 1.0,
    gravity_ms2: float = 9.80665,
    theta0_deg: float = 30.0,
    damping_gamma: float = 0.0,
    tolerance_pct: float = 1.0,
    n_cycles: int = 10,
    output_filename: str = "pendulum_interactive_lab.html",
) -> str:
    """Responde con doble verificacion matematica (Integral Eliptica K + EDO DOP853 + Taylor) y genera una animacion HTML5 interactiva a 60 FPS en outputs/ para '¿Cuanto vale la aproximacion lineal?'."""
    analysis = analyze_pendulum_linear_approximation(
        length_m=length_m,
        gravity_ms2=gravity_ms2,
        theta0_deg=theta0_deg,
        tolerance_pct=tolerance_pct,
        n_cycles=n_cycles,
    )
    anim = build_pendulum_interactive_html(
        output_filename=output_filename,
        length_m=length_m,
        gravity_ms2=gravity_ms2,
        theta0_deg=theta0_deg,
        damping_gamma=damping_gamma,
        tolerance_pct=tolerance_pct,
    )
    return _to_json(
        {
            "verified": analysis["verified"],
            "interactive_html_path": anim["interactive_html_path"],
            "analysis": analysis,
        }
    )


@mcp.tool(structured_output=False)
def ada_electronic_devices_control(
    action: str = "list",
    device_port: str = VIRTUAL_DEVICE_URI,
    baudrate: int = 115200,
    control_mode: str = "linearized_pd",
    allow_actuation: bool = False,
    target_angle_deg: float = 0.0,
    initial_angle_deg: float = 25.0,
    duration_s: float = 5.0,
    length_m: float = 1.0,
) -> str:
    """Descubre puertos electronicos COM/USB (action='list') o ejecuta telemetria/control realimentado seguro con parada E-STOP y replay HTML animado (action='run').

    Modos de control_mode: 'telemetry_only', 'linearized_pd', 'nonlinear_computed_torque'.
    Usa device_port='virtual://pendulum-mcu' para probar el lazo de control en el gemelo electronico antes de energizar un Arduino/ESP32 fisico.
    """
    act = action.strip().lower()
    if act == "list":
        return _to_json(list_electronic_devices())
    if act == "run":
        return _to_json(
            run_pendulum_hardware_control(
                device_port=device_port,
                baudrate=baudrate,
                control_mode=control_mode,
                allow_actuation=allow_actuation,
                target_angle_deg=target_angle_deg,
                initial_angle_deg=initial_angle_deg,
                duration_s=duration_s,
                length_m=length_m,
            )
        )
    raise ValueError("action debe ser 'list' o 'run'.")


@mcp.tool(structured_output=False)
def ada_consult_ada_r1(question_and_data_summary: str) -> str:
    """Consulta al segundo cerebro matematico/critico local `Ada r1` (`ada-r1`) en Ollama para auditar demostraciones, hipotesis estadisticas o disenos de control."""
    cfg = ModelRoleConfig()
    resp = httpx.post(
        f"{cfg.ollama_api_host}/api/generate",
        json={
            "model": cfg.reasoner_ollama,
            "prompt": (
                "Actua como Revisor Matematico y Estadistico Riguroso de ADA en espanol. "
                "Cuestiona supuestos, verifica consistencia dimensional y senala posibles errores:\n\n"
                f"{question_and_data_summary}"
            ),
            "stream": False,
            "options": {"num_ctx": cfg.context_length, "temperature": 0.1, "seed": 42},
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return _to_json(
        {
            "specialist_model": cfg.reasoner_ollama,
            "critique_response": data.get("response", "").strip(),
        }
    )


@mcp.tool(structured_output=False)
def ada_neural_network_live_canvas(
    sample_prompt: str = "Hola Ada muestra como piensa tu red neuronal en tiempo real",
    output_filename: str = "ada_neural_live_canvas.html",
    open_in_browser: bool = True,
) -> str:
    """Construye y abre una aplicacion web interactiva tipo Canva / HTML5 Canvas a 60 FPS que muestra animadamente en tiempo real como funciona la red neuronal Transformer de ADA (tokens, RoPE, 40 capas GQA, matriz de auto-atencion causal verificada en PyTorch CUDA vs NumPy, Softmax con slider de temperatura y telemetria real de la GPU RTX 5070)."""
    report = build_neural_network_live_canvas(
        sample_prompt=sample_prompt,
        output_filename=output_filename,
        open_in_browser=open_in_browser,
    )
    return _to_json(report)


@mcp.tool(structured_output=False)
def ada_neural_brain_simulation(
    output_filename: str = "ada_neural_brain_simulation.html",
    open_in_browser: bool = True,
) -> str:
    """Construye y abre una simulacion 3D interactiva a 60 FPS del cerebro y conectoma neuronal denso de ADA (720 neuronas en doble hemisferio cortical, miles de sinapsis, potenciales de accion luminosos, neurotransmisores, audio sintetizado y telemetria fisica en vivo de la GPU RTX 5070 Blackwell)."""
    report = build_neural_brain_simulation(
        output_filename=output_filename,
        open_in_browser=open_in_browser,
    )
    return _to_json(report)


@mcp.tool(structured_output=False)
def ada_build_interactive_webapp(
    title: str,
    subtitle: str = "",
    output_filename: str = "ada_interactive_studio.html",
    theme: str = "dark",
    badge_text: str = "ADA STUDIO · Verificado",
    kpis_json: str = "[]",
    charts_json: str = "[]",
    callouts_json: str = "[]",
    table_rows_json: str = "[]",
    table_title: str = "Explorador Interactivo de Datos",
    custom_html_section: str = "",
) -> str:
    """Construye o rediseña una pagina web / dashboard interactivo de nivel estudio profesional (Bento Glassmorphism, selector Modo Oscuro/Claro, graficas Retina HiDPI animadas a 60 FPS con tooltips flotantes, pestañas interactivas, exportacion PNG/CSV y tabla con buscador y ordenamiento) en outputs/, auditando automaticamente su calidad DOM/CSS (`grade A+`).

    Formatos JSON:
    - kpis_json: '[{"label":"RMSE Test","value":"2.1148","delta":"+11.1% vs Dummy","meta":"IC 90% ±2.75"}]'
    - charts_json: '[{"tab_label":"Tendencia","title":"Serie Temporal","type":"line|bars|scatter|residuals|arena","x":[1,2,3],"x_label":"Paso","y_label":"Valor","series":[{"name":"Real","data":[10,13,15]}]}]'
    - callouts_json: '[{"title":"Hallazgo","text":"Detalle tecnico","severity":"info|ok|caution|risk"}]'
    - table_rows_json: '[{"Variable":"x1","Valor":12.4}]'
    """
    kpis = json.loads(kpis_json) if kpis_json.strip() else []
    charts = json.loads(charts_json) if charts_json.strip() else []
    callouts = json.loads(callouts_json) if callouts_json.strip() else []
    table_rows = json.loads(table_rows_json) if table_rows_json.strip() else []
    report = build_interactive_webapp(
        title=title,
        subtitle=subtitle,
        output_filename=output_filename,
        theme=theme,
        badge_text=badge_text,
        kpis=kpis,
        charts=charts,
        callouts=callouts,
        table_rows=table_rows,
        table_title=table_title,
        custom_html_section=custom_html_section,
    )
    return _to_json(report)


@mcp.tool(structured_output=False)
def ada_build_canvas_animation(
    title: str,
    draw_frame_js: str,
    subtitle: str = "Animación interactiva en tiempo real a 60 FPS (Retina HiDPI)",
    output_filename: str = "ada_canvas_animation.html",
    controls_json: str = "[]",
    init_state_js: str = "",
    canvas_height_px: int = 500,
) -> str:
    """Construye una ANIMACIÓN VISUAL CREATIVA, ESCENA ARTÍSTICA, PAISAJE, ANIMAL, CIUDAD, FÍSICA o JUEGO a 60 FPS reales en HTML5 Canvas 2D (Bento Glass, Retina HiDPI, sliders en vivo y captura PNG).

    IMPORTANTE: Escribe sentencias directas en `init_state_js` y `draw_frame_js` (NO envuelvas en `function drawFrame` y mueve SIEMPRE las posiciones usando `t` y `dt`).
    Además de `ctx.*`, tienes cargado el motor vectorial articulado `ADAScene`:
    - Biomas:
      `ADAScene.sky(ctx, w, h, { t, dusk: params.dusk ?? 0.35 })`
      `ADAScene.hillsAndMeadow(ctx, w, h, { t, dusk: params.dusk ?? 0.35 })`
      `ADAScene.forestTrees(ctx, w, h, { t, count: 8, dusk: params.dusk ?? 0.35 })`
      `ADAScene.citySkyline(ctx, w, h, { t, scrollSpeed: 50, dusk: 0.65 })`
      `ADAScene.water(ctx, w, h, { t, level: 0.65 })`
    - Actores articulados animados (`dir` es 1 o -1):
      `ADAScene.fox(ctx, x, y, { t, dir, scale: 1.1, speed })`
      `ADAScene.rabbit(ctx, x, y, { t, dir, scale: 1.0, speed, phase })`
      `ADAScene.bird(ctx, x, y, { t, dir, scale: 1.0, speed, phase })`
      `ADAScene.fish(ctx, x, y, { t, dir, scale: 1.0, color })`
      `ADAScene.car(ctx, x, y, { t, dir, scale: 1.0, speed, color })`
      `ADAScene.rocket(ctx, x, y, { t, scale: 1.0, angle })`
      `ADAScene.planet(ctx, x, y, radius, { color, rings: true })`
      `ADAScene.person(ctx, x, y, { t, dir, scale: 1.0, speed, color })`
      `ADAScene.beagle(ctx, x, y, { t, scale: 1.8, earWiggle: 1.2, lookAt: mouse })` (perrita beagle frontal tierna con orejas caidas móviles, ojos atentos y jadeo)
    - Física de persecución/huida y partículas (`bounds = { minX: 40, maxX: w - 40, minY: h * 0.66, maxY: h - 28 }`):
      `ADAScene.stepPursuit(predatorObj, targetsArray, bounds, 165 * speed, dt)`
      `ADAScene.stepEvade(preyObj, predatorObj, bounds, 150 * speed, dt, t)`
      `ADAScene.spawnDust(dustArray, x, y, 2)` y `ADAScene.updateDust(ctx, dustArray, dt)`

    Variables en cada frame: `ctx`, `w`, `h`, `t`, `dt`, `params`, `mouse` (`{x, y, inside, down}`), `state`.
    Formato `controls_json`: '[{"key":"speed","label":"Velocidad","min":0.4,"max":2.5,"step":0.1,"value":1.2,"unit":"x"}]'
    """
    controls = json.loads(controls_json) if controls_json.strip() else []
    report = build_canvas_animation_studio(
        title=title,
        subtitle=subtitle,
        output_filename=output_filename,
        draw_frame_js=draw_frame_js,
        init_state_js=init_state_js,
        controls=controls,
        canvas_height_px=canvas_height_px,
    )
    return _to_json(report)


@mcp.tool(structured_output=False)
def ada_audit_webpage_design(html_path: str) -> str:
    """Audita cualquier archivo .html creado o editado para verificar integridad DOM/CSS (detecta clases huerfanas borradas del CSS), compatibilidad con el visor sandbox de DeepSeek Harness, escalado Retina HiDPI en Canvas y calidad visual (`quality_score` 0-100, `grade` A+/A/FAIL). Ejecutala SIEMPRE antes de entregar un HTML editado manualmente con `present`."""
    return _to_json(audit_html_quality(html_path))


@mcp.tool(structured_output=False)
def ada_inspect_adb_devices() -> str:
    """Inspecciona y lista todos los celulares o emuladores Android conectados por USB/Wi-Fi via ADB, verificando estado, modelo, resolucion y nivel de bateria."""
    devs = [d.to_dict() for d in list_adb_devices()]
    return _to_json({
        "adb_binary": find_adb_executable(),
        "scrcpy_binary": find_scrcpy_executable(),
        "connected_devices_count": len(devs),
        "devices": devs,
        "ready": any(d.get("ready") for d in devs),
    })


@mcp.tool(structured_output=False)
def ada_build_live_vision_studio(output_filename: str = "ada_live_vision_dashboard.html") -> str:
    """Genera la aplicacion web interactiva Bento Glass v2.0 de Vision por Computadora y Telemetria Fisica en Vivo (60 FPS, HiDPI Retina) para transmision por ADB de la camara del celular o simulador fisico con resolucion de EDOs verificada."""
    p = build_live_vision_dashboard_html(output_filename=output_filename)
    audit = audit_html_quality(p)
    return _to_json({
        "status": "success",
        "output_path": str(p),
        "audit": audit,
        "message": f"Dashboard de visión generado con éxito en {p.name} (Calidad: {audit.get('quality_score')}/100, Grado: {audit.get('grade')}).",
    })


@mcp.tool(structured_output=False)
def ada_analyze_camera_frame(serial: str = "", source: str = "auto") -> str:
    """Captura un frame en tiempo real desde el celular por ADB (o simulador de laboratorio) y ejecuta rastreo optico, estimacion cinematica de theta(t), omega(t), FFT y verificacion de EDO con SciPy DOP853."""
    tracker = RealTimePhysicsTracker()
    frame = None
    if source in {"auto", "adb"}:
        frame = capture_adb_frame(serial if serial.strip() else None)

    if frame is None:
        sim = SyntheticPhysicsLaboratoryStream()
        frame = sim.read_frame()
        source_used = "synthetic_laboratory"
    else:
        source_used = "adb_screencap"

    ann, kin = tracker.process_frame(frame)
    return _to_json({
        "source": source_used,
        "frame_dimensions": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "target_detected": kin is not None,
        "kinematics": {
            "theta_deg": round(kin.theta_deg, 2) if kin else None,
            "theta_rad": round(kin.theta_rad, 4) if kin else None,
            "omega_rad_s": round(kin.omega_rad_s, 4) if kin else None,
            "frequency_hz": round(kin.frequency_hz, 4) if kin else None,
            "estimated_length_m": round(kin.estimated_length_m, 3) if kin else None,
            "energy_total_j": round(float(kin.energy_total), 4) if kin else None,
            "verified": kin.verified if kin else False,
            "verification_methods": kin.verification_methods if kin else [],
            "discrepancy": kin.discrepancy if kin else 0.0,
        } if kin else None,
    })


@mcp.tool(structured_output=False)
def ada_record_episodic_memory(
    user_input: str,
    assistant_response: str,
    user_feedback: str = "positive",
    tags_csv: str = "matematicas,preferencias,estandar_libros",
) -> str:
    """Registra una interaccion, aprendizaje o preferencia de Diego en el banco de memoria continua y dataset de entrenamiento de ADA."""
    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
    res = record_interaction(
        user_input=user_input,
        assistant_response=assistant_response,
        user_feedback=user_feedback,
        tags=tags,
    )
    return _to_json({"status": "recorded", "memory": res, "stats": get_continual_learning_stats()})


@mcp.tool(structured_output=False)
def ada_query_episodic_memory(query: str, max_results: int = 3) -> str:
    """Consulta la memoria episodica de ADA para recordar preferencias previas, formulas resueltas y directrices dadas por Diego."""
    mems = retrieve_relevant_memories(query=query, max_results=max_results)
    return _to_json({"query": query, "matches_count": len(mems), "memories": mems})


@mcp.tool(structured_output=False)
def ada_control_autonomous_mind(action: str = "status") -> str:
    """Controla el motor de pensamiento continuo en caliente y plasticidad neuronal de ADA ('start', 'stop', 'status', 'step').
    
    Permite que ADA resuelva EDOs, sintetice teoremas y cree nuevas conexiones neuronales en segundo plano mientras Diego habla o trabaja.
    """
    mind = get_autonomous_mind()
    act = action.strip().lower()
    if act == "start":
        mind.start()
    elif act == "stop":
        mind.stop()
    elif act == "step":
        thought = mind._perform_single_cognitive_cycle()
        mind.current_thought = thought
        mind._record_thought(thought)
    return _to_json(mind.get_status())


@mcp.tool(structured_output=False)
def ada_build_math_textbook(exercise_id: str = "integral_sustitucion") -> str:
    """Abre el Cuaderno Matemático de Libro de Texto 2D de ADA (KaTeX) para renderizar integrales, derivadas y deducciones con tipografía gráfica de libro universitario."""
    res = build_math_textbook_html(initial_exercise=exercise_id, open_in_browser=False)
    return _to_json(res)


@mcp.tool(structured_output=False)
def ada_inferential_stats(analysis: str, data_json: str, alpha: float = 0.05) -> str:
    """INFERENCIA VERIFICADA (doble método).
    analysis / data_json: two_groups {x,y,paired,alternative,method:auto|welch|student|mann_whitney} | k_groups {groups:{A:[..],B:[..]}} | chi_square {table} | two_proportions {successes:[a,b],n:[n1,n2]} | one_proportion {successes,n,p0} | correlation {x,y} | regression {y,X:{x1:[..]}} | power_t {effect_size|n_per_group|power: uno null} | adjust_pvalues {pvalues,method} | one_sample {x,mu0} | ci_mean {x} | ci_proportion {successes,n} | variance_ratio {x,y} | z_test {mean,n,mu0,sigma} | sample_size {kind:mean|two_means|proportion,margin,sigma,p}.
    """
    from ada.guardrails import normalize_inference_args
    from ada.stats.inference import compact_result, run_inference

    try:
        # Mismo guardarrail que el CLI: sinonimos de argumentos ("bilateral" -> "two-sided") se corrigen en codigo, sin gastar contexto.
        args, notes = normalize_inference_args({"analysis": analysis, "data": json.loads(data_json), "alpha": alpha})
        result = compact_result(run_inference(args["analysis"], args["data"], alpha=float(args.get("alpha", alpha))))
        if notes:
            result["argumentos_corregidos"] = notes
        return _to_json(result)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _to_json({"error": f"{type(exc).__name__}: {exc}", "hint": "Revisa la forma de data_json en la descripcion de la herramienta."})


@mcp.tool(structured_output=False)
def ada_probability_bayes(operation: str, args_json: str = "{}") -> str:
    """Probabilidad y Bayes VERIFICADOS (doble método; nunca de memoria). operation: event|complement|intersection|union|conditional|independence|bayes|bayes_priors|bayes_sequential.
    args_json: {"path":dataset, "a":evento, "b":evento, "params":{...}}. Evento: {column,op(== != > >= < <= in between isnull notnull),value|quantile} | {and|or:[..]} | {not:..}.
    bayes_priors: params {prior,sensitivity,false_positive_rate}; bayes_sequential: params {prior,evidences:[{sensitivity,false_positive_rate,positive}]}."""
    from ada.stats.probability import compact_probability, run_probability

    try:
        a = json.loads(args_json or "{}")
        return _to_json(compact_probability(run_probability(operation, path=a.get("path"), A=a.get("a"), B=a.get("b"), params=a.get("params"))))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, FileNotFoundError) as exc:
        return _to_json({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(structured_output=False)
def ada_simulate(request: str) -> str:
    """Construye CUALQUIER simulacion que pidas ("sistema solar", "pelota rebotando", "ondas", "poblaciones", "caida libre"...) con AUTORREPARACION.

    Ada Q escribe la simulacion; el motor real la ejecuta 180 fotogramas en Node para detectar errores de codigo, llamadas a ADAScene
    inexistentes y simulaciones estaticas; si falla, le devuelve el error exacto y reintenta hasta 3 veces. Devuelve `ok`, intentos y la ruta
    `outputs/sim_*.html` (entregala con `present`). Si no lo logra, lo dice con el ultimo error: no finge exito.
    Garantiza que corre, se mueve y pasa la auditoria; NO garantiza belleza ni cada detalle pedido: revisa el resultado.
    """
    from ada.viz.sim_builder import build_simulation

    return _to_json(build_simulation(request))


@mcp.tool(structured_output=False)
def ada_self_report() -> str:
    """Autoconocimiento REAL de Ada: quien es, su arquitectura, lo que hace por su cuenta, lo que ha aprendido, sus limites y sus objetivos actuales.

    Llamala SIEMPRE antes de responder preguntas sobre ti misma (quien eres, que haces, que quieres, como estas, que piensas).
    Habla en primera persona usando SOLO estos datos; no afirmes sentimientos ni conciencia.
    """
    from ada.autonomy.self_model import build_self_report, self_knowledge_text

    report = build_self_report()
    return _to_json({"texto_para_hablar_de_mi": self_knowledge_text(report), "datos": report})


@mcp.tool(structured_output=False)
def ada_live_vision_session(action: str = "status", serial: str = "") -> str:
    """Vision en VIVO: analiza la camara del celular (MJPEG por ADB) o su pantalla (screencap), con respaldo al simulador.

    Acciones: 'start' (elige la mejor fuente real y empieza a analizar), 'stop', 'status' (fps, latencia, cinematica medida).
    Informa SIEMPRE que fuente se esta usando y por que descarto las demas. El Observatorio muestra el frame anotado y los calculos.
    """
    from ada.vision.live_camera import get_vision_session, start_vision, stop_vision

    act = action.strip().lower()
    out: dict = {}
    if act == "start":
        out["source_report"] = start_vision(serial or None)
    elif act == "stop":
        stop_vision()
    elif act != "status":
        return _to_json({"error": f"accion desconocida: {action}"})
    s = get_vision_session()
    out["session"] = s.snapshot() if s else {"active": False}
    return _to_json(out)


@mcp.tool(structured_output=False)
def ada_observatory(open_in_browser: bool = False, regenerate_model_map: bool = False) -> str:
    """Genera el Observatorio de ADA: visor del modelo REAL (cada pixel = una de las 696,320 neuronas FFN, 1,920 cabezas de atencion),
    con rendimiento en vivo (tokens/s, GPU, VRAM medidos) y el diario del agente autonomo. Entrega `outputs/ada_observatory.html` con `present`.

    `regenerate_model_map=True` relee los pesos del GGUF (≈3-4 min); sin esto se usa el mapa ya extraido en data/work/model_map/.
    """
    from ada.modelmap.extract import MAP_NPZ, extract_model_map
    from ada.viz.observatory import build_observatory

    if regenerate_model_map or not MAP_NPZ.is_file():
        extract_model_map()
    return _to_json(build_observatory(open_in_browser=open_in_browser))


@mcp.tool(structured_output=False)
def ada_autonomous_agent(action: str = "status") -> str:
    """Controla el AGENTE AUTONOMO REAL de ADA (Ada Q propone, herramientas verifican por doble metodo, solo se guarda lo verificado).

    Acciones: 'status' (estado y curriculo), 'step' (un ciclo ahora), 'start' / 'stop' (bucle en segundo plano),
    'emergency_stop' (crea el archivo STOP que frena a TODAS las instancias), 'resume' (borra STOP),
    'curate' (genera dataset curado train/eval e informa si esta listo para QLoRA).
    """
    from ada.autonomy.agent import get_agent
    from ada.autonomy.curator import curate_dataset
    from ada.autonomy.guard import clear_stop, request_stop

    agent = get_agent()
    act = action.strip().lower()
    extra: dict = {}
    if act == "start":
        clear_stop()
        extra["started"] = agent.start_background()
    elif act == "stop":
        agent.stop()
    elif act == "emergency_stop":
        extra["stop_file"] = str(request_stop("mcp"))
        agent.stop()
    elif act == "resume":
        clear_stop()
    elif act == "step":
        extra["cycle"] = agent.run_cycle()
    elif act == "curate":
        extra["curation"] = curate_dataset()
    elif act != "status":
        return _to_json({"error": f"accion desconocida: {action}"})
    return _to_json({**agent.status(), **extra})


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
