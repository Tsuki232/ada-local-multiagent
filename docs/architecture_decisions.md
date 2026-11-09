# Registro de Decisiones de Arquitectura (ADR) — Proyecto ADA

## ADR-001: Entorno Windows 11 Nativo, `uv` y PyTorch `cu128` para RTX 5070 (`sm_120`)
- **Contexto:** Equipo con Windows 11 Pro (`Build 26300`), GPU NVIDIA GeForce RTX 5070 (`12,227 MiB` VRAM, arquitectura Blackwell `sm_120`, Driver `610.88`, CUDA UMD `13.3`), sin WSL2 instalado.
- **Decisión:**
  1. Usar **`uv`** (`pyproject.toml` + `uv.lock`) con CPython `3.12` aislado en `.venv` para reproducir el entorno completo con un único comando.
  2. Usar **PyTorch 2.7+ compilado contra CUDA 12.8 (`cu128`)** como framework de Deep Learning y cálculo tensorial en GPU.
  3. Descartar TensorFlow nativo en Windows porque TensorFlow oficial en `pip` suspendió el soporte GPU nativo en Windows tras la versión `2.10` y carece de kernels nativos `sm_120` en Windows sin WSL2/Docker.

## ADR-002: Selección de Modelos Locales 100 % Libres (Pesos Abiertos) para los 2 Casos Clave
- **Contexto:** Se requiere soberanía total (sin dependencia de APIs externas), licencia libre para inspección, modificación y entrenamiento propio, y ajuste dentro de `12 GB` de VRAM GDDR7.
- **Decisión:** Adoptar una arquitectura **bimodal de la misma familia base (Qwen)** servida localmente por Ollama (con `OLLAMA_MODELS=./.ollama/models` en disco `G:`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` y `OLLAMA_MAX_LOADED_MODELS=1`) y respaldada por sus checkpoints `safetensors` en HuggingFace:
  1. **Caso 1 — Orquestador, Tool Calling, Limpieza de Datos y Estadística (`qwen3:14b` / `Qwen/Qwen3-14B`):**
     - **Licencia:** `Apache-2.0` (100 % libre).
     - **Por qué:** Entrenado nativamente para *tool calling* JSON multi-turno, consultas SQL/código y modo híbrido *thinking / non-thinking* soportado directamente por el adaptador `@deepseek-ai/dsh-llm-pi-ai` (`thinkingFormat: qwen`).
  2. **Caso 2 — Razonador Matemático (Cálculo/EDOs Péndulo) y Agente Crítico (`deepseek-r1:14b` / `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`):**
     - **Licencia:** `MIT` (100 % libre).
     - **Por qué:** Destilado directamente sobre arquitectura Qwen-14B con desempeño sobresaliente en deducción matemática (`93.9 %` MATH-500, `69.7 %` AIME24). Ideal para auditar supuestos estadísticos, buscar contraejemplos y revisar derivaciones simbólicas sin depender de él para aritmética mental.
- **Gestión de VRAM (12 GB):** Nunca se mantienen ambos modelos de ~9 GB residentes a la vez; `OLLAMA_MAX_LOADED_MODELS=1` y el gestor `ada.tools.ollama_manager` descargan explícitamente el modelo previo (`keep_alive=0`) antes de cargar el siguiente.

## ADR-003: Separación Estricta entre Herramientas Deterministas y Fine-Tuning QLoRA en PyTorch
- **Principio innegociable:** **Jamás entrenar un LLM para que memorice aritmética, p-valores o integrales.** Todo cálculo matemático o estadístico lo ejecutan motores deterministas (`SymPy`, `SciPy`, `mpmath`, `statsmodels`, `DuckDB`, `Polars`) con doble verificación independiente.
- **Dónde SÍ aporta valor real entrenar con PyTorch (`transformers` + `peft` + `bitsandbytes` NF4 4-bit + `trl`):**
  1. **Adaptadores QLoRA especializados (Fase 8):** Una vez construidas y probadas las herramientas MCP de Ada (Fases 2–6), generaremos datasets propios verificados de:
     - *Enrutamiento y llamadas exactas a herramientas JSON de Ada* (cero errores de esquema).
     - *Traducción Lenguaje Natural -> DuckDB SQL reproducible*.
     - *Detección de trampas metodológicas y estadísticas* (fuga de datos, p-hacking, paradoja de Simpson, violación de homocedasticidad/normalidad).
     Con `12 GB` de VRAM libres, el entrenamiento QLoRA 4-bit (`bf16` compute, `nf4`, *gradient checkpointing*, `paged_adamw_8bit`) es holgado sobre `Qwen/Qwen3-8B` (~7.2 GiB VRAM) y viable con ajuste fino de secuencia sobre `14B` (~11.2–11.8 GiB VRAM), evaluando siempre contra un conjunto *hold-out* antes de exportar el adaptador a GGUF para Ollama.
  2. **Modelos neuronales propios en PyTorch (Fase 8):**
     - **PINN (Physics-Informed Neural Network)** para la EDO no lineal del péndulo contrastada contra `DOP853` y la solución elíptica exacta de Jacobi.
     - **Autoencoder tabular** para detección multivariante de anomalías complementando IQR/MAD/Isolation Forest.


## ADR-004: Conexión Keyless de Ollama en DeepSeek Harness Desktop (`@deepseek-ai/dsh-llm-pi-ai`)
- **Contexto:** En `@deepseek-ai/dsh-llm-pi-ai@0.2.1-rc.5`, declarar `apiKeyEnv: OLLAMA_API_KEY` hace que `listModels("ollama-local")` lance `MISSING_CREDENTIAL` y devuelva una lista vacía `[]` en el selector visual de modelos de **DeepSeek Harness Desktop** si el usuario no definió esa variable en Windows o en `.credentials.yaml`.
- **Decisión:** Omitir `apiKeyEnv` en `providers.ollama-local` e inyectar `headers: { Authorization: "Bearer ollama-local" }`, junto con `compat.supportsDeveloperRole: false` (porque Ollama `0.40.10` descarta silenciosamente `role: "developer"`) y `compat.maxTokensField: max_tokens`. Así los modelos locales aparecen listos de inmediato en Desktop UI sin pedir claves API.

## ADR-005: Servidor MCP `stdio` de Esquema Plano + Animaciones HTML5 a 60 FPS vía `present`
- **Contexto:** Los modelos locales de 14B (`qwen3:14b`) maximizan su precisión de *tool calling* cuando los esquemas JSON de entrada son planos (tipos primitivos sin `anyOf` anidados), y **DeepSeek Harness Desktop** incluye `@deepseek-ai/dsh-tool-present` + `@deepseek-ai/dsh-client-ui-deliverables` en el preset `standard`.
- **Decisión:** Exponer las capacidades de ADA mediante `FastMCP("ada")` (`ada.tools.mcp_server`) montado globalmente con `@deepseek-ai/dsh-mcp-client` (`mcp__ada__*`). Todas las animaciones (péndulo no lineal vs lineal a 60 FPS, replay de telemetría electrónica y dashboards predictivos) se generan como archivos `.html` autocontenidos (100 % offline, Canvas + AGM de Gauss + RK4) en `outputs/` para entregarse con un clic mediante `present`.

## ADR-006: Control Seguro de Dispositivos Electrónicos y Visión de Ecuaciones Gráficas
- **Control Electrónico (`ada.pendulum.hardware_bridge`):** Todo acceso serie `COM/USB` (`pyserial`) opera en modo **solo telemetría** (`allow_actuation=False`) por defecto, aplica saturación de par (`|u| <= max_torque_limit`), límite angular de emergencia (`max_safe_angle_deg`) y envía `ESTOP\n` garantizado en bloque `finally`. Incluye el gemelo electrónico `virtual://pendulum-mcu` (encoder óptico de 12 bits + planta RK4) para validar controladores antes de energizar hardware físico.
- **Ecuaciones Escritas Gráficamente (`ada.calculus.verified_math`):** Se integra `rapidocr-onnxruntime` local (`< 100 MB`) con ordenamiento espacial de renglones, normalización Unicode `NFKC` y balanceo de cajas solapadas para ecuaciones impresas/claras, más detección automática de VLM multimodal en Ollama (`gemma4:12b`, ~8.0 GB, Apache-2.0, previa confirmación de descarga `> 1 GB` según Regla 3) para manuscritos complejos.
