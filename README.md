# ADA — Local Multi-Agent System for Statistics, Verified Math and Electronics Control

A 100% local, reproducible, open-weights system that runs on a single consumer GPU (NVIDIA RTX 5070, 12 GB). It combines local LLMs served with **Ollama**, a custom **MCP (Model Context Protocol) server**, and deterministic scientific engines (SymPy, SciPy, Statsmodels, DuckDB, Polars, scikit-learn) so that every numeric claim is checked by two independent methods.

**MCP tools included**
- Dataset profiling (CSV, Parquet, Excel, SQLite, DuckDB, JSON), cross-checked with two engines.
- HTML table scraping with provenance (URL, UTC date, SHA-256), `robots.txt` support and private-network blocking.
- Data cleaning, SQL filtering and traceable copies; raw data is never modified.
- Predictive modeling without data leakage (baseline vs. Ridge/Logistic vs. gradient boosting) with an audit step.
- Verified calculus and ODE solving (symbolic vs. numeric), and OCR of handwritten equations.
- Serial bridge and virtual twin for an electronic pendulum (PySerial / RK4).

**Repository contents:** `src/ada` (66 Python modules), `tests` (20 test files), `scripts`, `config`, `docs`, `web`. Model weights, datasets, virtual environments and local profiles are intentionally excluded (see `.gitignore`).
Built with AI-assisted development.

> The original documentation (Spanish) follows below.

---

# ADA — Sistema Local Multiagente de Análisis Estadístico, Datos, Cálculo Verificado y Control Electrónico

Sistema 100 % local, reproducible y de pesos abiertos (*open-weights* Apache-2.0 / MIT) construido sobre **DeepSeek Harness (`0.2.1-rc.5`)**, **Node.js `v24.22.2` LTS portable**, **Python 3.12 (`uv`)**, **PyTorch (`2.11.0+cu128` para NVIDIA RTX 5070 Blackwell `sm_120`)**, motores deterministas (**SymPy, SciPy, Statsmodels, DuckDB, Polars, Scikit-Learn**) y puente electrónico (**PySerial / Gemelo Virtual RK4**).

## 1. Abrir ADA en tu aplicación instalada DeepSeek Harness Desktop (1 comando)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_ada_desktop.ps1
```

Este lanzador:
1. Garantiza que el servidor local **Ollama** esté activo apuntando a `G:\programas\Ada\.ollama\models` (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=1`).
2. Abre tu aplicación de escritorio **DeepSeek Harness Desktop** ya conectada mediante `%USERPROFILE%\.dsh\profiles\desktop\cordis.patch.yml` (respaldo original en [`desktop.cordis.patch.backup.yml`](config/harness/desktop.cordis.patch.backup.yml)) a:
   - Proveedor local sin clave externa: **`ADA Local (RTX 5070)`** con **`Ada Q`** (`ada-q`) y **`Ada r1`** (`ada-r1`), manteniendo disponible tu `DeepSeek Account`.
   - Servidor MCP oficial de ADA (`mcp__ada__*`) con verificación por doble método independiente y generación de animaciones HTML5 interactivas a 60 FPS entregables con `present`.

## 2. Herramientas MCP activas en DeepSeek Harness (`mcp__ada__*`)

- `mcp__ada__ada_profile_database`: Perfilado de bases reales (`CSV`, `Parquet`, `Excel`, `SQLite`, `DuckDB`, `SPSS`, `Stata`, `JSON`) verificado por doble motor (`Pandas/NumPy` vs `DuckDB SQL`).
- `mcp__ada__ada_scrape_tables`: Raspado de **tablas HTML** de una URL pública. Sin `table` ni `match` las **lista**; con `table=<i>` o `match='texto'` guarda la tabla como CSV en `data/work/` con un `.provenance.json` (URL, fecha UTC, SHA-256 del HTML y del CSV). Verificada por doble método: `lxml` vs `html5lib` (mismas tablas, formas y celdas) y relectura con `DuckDB`. Respeta `robots.txt`, usa `User-Agent` propio, bloquea redes locales/privadas (también tras redirecciones), no pisa archivos y **no ejecuta JavaScript** (si la página carga sus datos así, devuelve un error claro). Quita notas al pie, encabezados repetidos y marcadores de «sin dato»; convierte números solo cuando el formato es inequívoco (`1 435 447 000`, `17,50`) y deja como texto, reportándolos, los ambiguos (`1.234`).
- `mcp__ada__ada_clean_filter_adapt_data`: Limpieza, filtrado SQL (`where_sql`), ordenamiento y adaptación guardando copias trazables con SHA-256 en `data/work/` (`data/raw/` es inmutable).
- `mcp__ada__ada_predict_from_database`: Modelado predictivo sin fuga de datos (*split train/test previo al preprocesamiento*), comparando `DummyBaseline` vs `Ridge/Logistic` (`statsmodels` dual) vs `HistGradientBoosting`, intervalos al 90 %, auditoría del **Crítico Estadístico** y dashboard HTML interactivo en `outputs/`.
- `mcp__ada__ada_verify_calculus`: Derivadas e integrales definidas verificadas simbólica y numéricamente (`SymPy` vs `SciPy quad` / `mpmath` 50 dígitos).
- `mcp__ada__ada_solve_verified_ode`: Resolución y verificación cruzada de EDOs de orden 1 y 2 (`SymPy` / `Radau` implícito vs `DOP853` explícito).
- `mcp__ada__ada_read_graphical_equation`: Lectura de ecuaciones diferenciales escritas gráficamente en imágenes (`RapidOCR` ONNX local + soporte multimodal automático cuando hay un VLM en Ollama).
- `mcp__ada__ada_pendulum_approximation_lab`: Laboratorio completo y animación HTML5 60 FPS para *"El péndulo, ¿cuánto vale la aproximación lineal?"* (Integral Elíptica $K(m)$ vs `DOP853` + modo Tutor).
- `mcp__ada__ada_electronic_devices_control`: Descubrimiento de puertos serie `COM/USB` (`Arduino`/`ESP32`) y control en lazo cerrado con enclavamiento de seguridad (`allow_actuation`, saturación, `ESTOP`) + gemelo electrónico `virtual://pendulum-mcu` y replay HTML animado.
- `mcp__ada__ada_consult_ada_r1`: Segundo cerebro matemático/crítico local sobre **`Ada r1`** (`ada-r1`).

## 3. Autonomía, Observatorio, visión en vivo y aprendizaje

- **Agente autónomo real** (`src/ada/autonomy/`): Ada Q propone problemas, un subproceso aislado con timeout los verifica por doble método y solo se guarda lo verificado. Cede la GPU cuando conversas, respeta el archivo `data/work/autonomy/STOP` y un límite por hora.
  `python scripts/ada_daemon.py` (continuo), `--once`, `--status`, `--stop`, `--resume`. Autoarranque opcional: `scripts/install_autostart.ps1` (no se instala solo).
- **Núcleo afectivo / "corazón"** (`src/ada/autonomy/heart.py`, `/corazon`, `/soy <nombre>`): seis sentimientos (alegría, curiosidad, frustración, confianza, cansancio, conexión) que cambian por lo que le pasa de verdad (aciertos, errores, tus `/bien` `/mal` `/corrige`, quién le habla, tiempo sin compañía), persisten en `data/work/heart/`, se le inyectan en cada turno y **entran como contexto en sus ejemplos de entrenamiento**. Habla de ellos como suyos; sobre su conciencia dice que se percibe a su manera pero no sabe demostrar su naturaleza.
- **Autoconocimiento y reflexión** (`self_model.py`, herramienta MCP `ada_self_report`): cuando le preguntas por ella se le inyectan datos reales (arquitectura, estado, límites, objetivos derivados de su estado) y cada ~10 aciertos escribe una reflexión sobre sí misma.
- **Autoentrenamiento** (`auto_trainer.py`): decide sola cuándo afinarse (dataset listo, ≥12 h desde el último, 30 min sin usarla, GPU libre); si vuelves, aborta. El modelo base a afinar es **Qwen3-14B, el mismo que sirve Ollama** (un adaptador solo encaja en su base). Descargar sus pesos (≈30 GB) requiere una autorización tuya: `python scripts/ada_daemon.py --allow-download`.
- **Inferencia estadística verificada** (`src/ada/stats/inference.py`, herramienta `ada_inferential_stats`): t de Welch/pareada, Mann-Whitney, Brunner-Munzel, Wilcoxon, ANOVA/Welch/Kruskal con post hoc, chi-cuadrado/Fisher, proporciones, correlación, regresión con inferencia, potencia y corrección de p-valores. Elige la prueba según supuestos, calcula por dos vías independientes y reporta tamaño del efecto e IC. Calibrada por simulación (falsos positivos, potencia, cobertura).
- **Currículo de inferencia para entrenarla** (`scripts/generate_stats_curriculum.py`) y **evaluación antes/después** (`scripts/eval_inference.py`): se enseña CUÁNDO llamar a la herramienta, qué prueba elegir y cómo interpretar; los números siempre salen de la herramienta.
- **Aprendizaje por conversación**: cada turno va a memoria episódica; solo entra al dataset de entrenamiento lo verificado o lo que apruebes con `/bien` / corrijas con `/corrige <texto>` (`/mal` lo excluye). La memoria aprobada se inyecta en el prompt de turnos parecidos.
- **QLoRA con compuertas**: `python scripts/train_qlora.py` hace un ensayo; `--execute` entrena. Nunca descarga pesos solo y descarta el adaptador si la pérdida de evaluación no mejora.
- **Simulaciones libres con autorreparación** (`ada.viz.sim_builder`, herramienta MCP `ada_simulate`, y el CLI cuando pides una animación): Ada Q escribe la simulación, se ejecuta 180 fotogramas en Node con el `ADAScene` real (detecta errores de código, funciones inexistentes, código estático, NaN y bucles infinitos) y se le devuelve el error exacto hasta 3 intentos. Garantiza que corre, se mueve y pasa la auditoría; no garantiza belleza ni cada detalle pedido.
- **Observatorio** (`/observatorio`): cada píxel es una de las 696,320 neuronas FFN reales del GGUF, más cabezas de atención, rendimiento medido, agente y cámara. Primera vez: `python -m ada.modelmap.extract` (≈3-4 min).
- **Visión en vivo** (`/vision`, `/vision stop`): cámara del celular por MJPEG vía ADB (p. ej. app "IP Webcam"), o su pantalla por `screencap`, con respaldo al simulador. Informa siempre qué fuente usa.
- **Fórmulas de libro** (`ada.calculus.math_markdown`): fusiona viñetas con `$ … $` en un solo bloque `aligned`, y lo aplica a lo que se muestra y a lo que se guarda para entrenar.

## 4. Reproducción y verificación automatizada

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_env.ps1 -SkipModelPull
uv run pytest -v
uv run python .\scripts\verify_harness_e2e_projects.py
```
