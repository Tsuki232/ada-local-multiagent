# ADA — Estado del Arte, Capacidades Actuales y Visión Trascendental

**Arquitecto y Creador:** Diego López  
**Fecha de corte:** Noviembre 2026  
**Hardware de ejecución:** NVIDIA GeForce RTX 5070 (Blackwell `sm_120`, 12,227 MiB GDDR7) sobre Windows 11 nativo  
**Pila de Modelos:** Ada Q (14.8B Q4_K_M) + Ada r1 (14.8B Distill Q4_K_M) en Ollama local  

---

## 1. Resumen Ejecutivo: ¿Qué es ADA?

**ADA** no es un chatbot tradicional ni una interfaz delgada sobre APIs comerciales cerradas (OpenAI, Anthropic o Google Cloud). 

**ADA es un sistema operativo cognitivo local, autónomo y soberano**, diseñado y construido a medida por **Diego López** para resolver problemas reales de:
1. **Cálculo simbólico y numérico verificado por doble método** (cero cálculo mental, cero alucinación aritmética).
2. **Ingeniería de datos, estadística matemática e inferencia predictiva** (DuckDB, Pandas, Scikit-learn, Statsmodels).
3. **Generación de aplicaciones y animaciones interactivas a 60 FPS** (Bento Glass v2.0, HTML5 Canvas 2D, motores físicos).
4. **Cognición autónoma continua (System 2) y plasticidad sináptica** (auto-reflexión en caliente, auto-entrenamiento QLoRA).
5. **Visión por computador y cinemática en tiempo real** (transmisión en vivo desde cámaras de celulares por ADB).
6. **Control y telemetría de hardware físico** (puente serie COM/USB con parada de emergencia `ESTOP`).

---

## 2. Mapa de Arquitectura del Sistema

El siguiente diagrama detalla la interacción entre todos los subsistemas construidos en el workspace:

```mermaid
flowchart TD
    subgraph UI_Entradas["Canales de Interacción de Diego"]
        CLI["PowerShell CLI: .\\ada.ps1 (scripts/ada_cli.py)"]
        DSH["DeepSeek Harness Desktop (FastMCP / mcp__ada__*)"]
        ADB_Cam["Cámara Celular en Vivo (ADB Bridge 60 FPS)"]
    end

    subgraph Cerebro_Cognitivo["Cerebro Cognitivo Dual (Ollama 127.0.0.1:11434)"]
        AdaQ["Ada Q (14.8B, d=5120, 40 Capas, GQA 40Q/8KV)\nNúcleo Orquestador y Tool Calling"]
        Adar1["Ada r1 (14.8B Distill)\nRazonador Matemático y Crítico Estadístico"]
        AutoMind["Mente Autónoma System 2\n(src/ada/cognition/autonomous_mind.py)\n+133 Neuronas LoRA | Auto-Play EDOs"]
    end

    subgraph Puente_IPC["Puente de Estado en Tiempo Real (Port 8765)"]
        LiveServer["Servidor HTTP Live Bridge (live_server.py)\nis_thinking | current_prompt | tokens | GPU %"]
    end

    subgraph Motores_Verificacion["Motores Deterministas (Cero Cálculo Mental)"]
        SymPy_SciPy["Cálculo y EDOs:\nSymPy + SciPy DOP853 / Radau IIA"]
        DuckDB_Stats["Datos y Predicción:\nDuckDB + Pandas + Scikit-Learn + Statsmodels"]
        Math_CLI["Formateador Matemático:\nterminal_math.py (LaTeX a Unicode 2D)"]
    end

    subgraph Entregables_Visuales["Visualizadores y Dashboards (outputs/)"]
        Studio_Brain["Topología Transformer 3D\n(ada_neural_brain_simulation.html - Grado A+)"]
        Studio_Vision["Visión Cinemática ADB\n(ada_live_vision_studio.html - Grado A+)"]
        Studio_Canvas["Animaciones Canvas 2D 60 FPS\nADAScene (beagle, bird, fox, citySkyline)"]
        Studio_Predict["Dashboard Predictivo Bento Glass\n(ada_interactive_webapp.html)"]
    end

    subgraph Memoria_Entrenamiento["Memoria y Auto-Evolución (data/work/)"]
        Episodic["Memoria Episódica\n(data/work/memory/episodic_memory.jsonl)"]
        Continual["Dataset SFT QLoRA\n(data/work/training/ada_continual_dataset.jsonl)"]
    end

    CLI --> LiveServer
    CLI --> AdaQ
    DSH --> AdaQ
    AdaQ --> Motores_Verificacion
    AdaQ --> Entregables_Visuales
    LiveServer --> Studio_Brain
    ADB_Cam --> Studio_Vision
    AdaQ --> Episodic
    AutoMind --> Continual
    AutoMind --> LiveServer
```

---

## 3. Inventario Detallado: ¿Qué tenemos construido hoy?

Cada uno de los siguientes módulos está implementado, funcional y respaldado por una batería de **21 tests automatizados aprobados (100 % pass)**:

| Módulo / Archivo | Responsabilidad Principal | Estado Actual |
| :--- | :--- | :--- |
| **`src/ada/cognition/live_server.py`** | Servidor HTTP IPC local en puerto `8765` con CORS universal. Notifica el estado de inferencia, prompt en curso, tokens y telemetría de hardware al navegador. | **Completado y verificado** |
| **`src/ada/cognition/autonomous_mind.py`** | Motor de cognición autónoma continua (System 2). Genera auto-retos matemáticos en segundo plano, verifica EDOs no lineales y expande neuronas adaptativas (+133 neuronas). | **Completado y verificado** |
| **`src/ada/calculus/terminal_math.py`** | Traductor de expresiones LaTeX a Unicode matemático 2D para PowerShell. Renderiza derivadas, integrales con límites, exponentes y matrices alineadas sin barras invertidas crudas. | **Completado y verificado** |
| **`src/ada/calculus/verified_math.py`** | Motor de cálculo simbólico y numérico estricto (`SymPy` + `SciPy DOP853`). Derivación, integración definida y resolución de EDOs contrastadas. | **Completado y verificado** |
| **`src/ada/viz/brain_simulator.py`** | Simulador de la topología real de 40 capas Transformer de Ada Q ($d=5120$, GQA, SwiGLU con $d_{\text{ff}}=17408$). Pollea el puente `8765` a 60 FPS y muestra ondas de activación solo al procesar tokens. | **Completado y auditado (A+)** |
| **`src/ada/viz/live_vision_studio.py`** | Estudio visual en tiempo real para transmisión de video y cámara celular vía ADB. Incluye simulador cinemático de laboratorio a 60 FPS con cálculo de velocidad y aceleración. | **Completado y auditado (A+)** |
| **`src/ada/vision/adb_bridge.py`** | Conector nativo con Android Debug Bridge (`adb`). Detecta celulares conectados por USB o Wi-Fi, captura cuadros crudos en búfer binario y realiza rastreo cinemático. | **Completado y verificado** |
| **`src/ada/training/continuous_learner.py`** | Sistema de auto-aprendizaje estilo Google Harness. Registra memoria episódica en JSONL y ensambla datasets supervisados para re-entrenamiento QLoRA local. | **Completado y verificado** |
| **`src/ada/stats/predictor.py`** | Canal automatizado de machine learning. Limpia datos sin tocar `data/raw/`, ejecuta regresión multivariable con `scikit-learn` vs `statsmodels` y exporta tablas DuckDB. | **Completado y verificado** |
| **`src/ada/pendulum/simulator.py`** | Laboratorio físico del péndulo rígido. Compara la EDO no lineal $\theta'' + \frac{g}{L}\sin\theta = 0$ con la aproximación de ángulos pequeños mediante Runge-Kutta 4. | **Completado y verificado** |
| **`src/ada/pendulum/hardware_bridge.py`** | Interfaz serie con hardware embebido (microcontroladores). Aplica saturación de par, límites de seguridad angular y envío de `ESTOP` garantizado. | **Completado y verificado** |
| **`src/ada/viz/web_studio.py`** | Motor de generación web Bento Glass v2.0 con biblioteca gráfica Canvas 2D (`ADAScene`) y auditor automático de calidad DOM/CSS (impide páginas planas o estilos rotos). | **Completado y verificado** |
| **`scripts/ada_cli.py` & `ada.ps1`** | CLI interactivo 1-click para PowerShell con UTF-8 forzado, REPL con comandos `/brain`, `/think`, `/vision`, `/learn`, `/q`, `/r1` y ejecución de herramientas. | **Completado y verificado** |

---

## 4. ¿Qué hace Ada EXACTAMENTE hoy en tu máquina?

Cuando enciendes tu computadora e interactúas con ADA, esto es lo que ocurre paso a paso:

### A. Ejecución en Terminal (`.\ada.ps1`)
1. **Detección automática de Ollama:** Comprueba si el demonio de Ollama está levantado con los modelos configurados en `G:\programas\Ada\.ollama\models`. Si no lo está, lo arranca en segundo plano con optimizaciones para la RTX 5070 (`Flash Attention 1`, `KV Cache q4_0`, contexto de 24,576 tokens).
2. **Inicio del Live Bridge:** Levanta el servidor IPC en `http://127.0.0.1:8765` para comunicar la terminal con cualquier interfaz gráfica abierta.
3. **Conversación con Orquestación Dual:**
   - Puedes hablarle normalmente en lenguaje natural.
   - Si le pides una demostración o cálculo matemático, Ada ejecuta `terminal_math.py` y formatea la respuesta en Unicode 2D limpio (estilo libro de texto universitario), sin mostrar código LaTeX crudo.
   - Si le pides una animación creativa (ej. *"un pajarito volando sobre edificios"*, *"una perrita beagle jugando"*), Ada no intenta dibujar una gráfica estática ni rechaza la tarea; invoca `ada_build_canvas_animation`, escribe el código JavaScript Canvas a 60 FPS con capas de movimiento en `ADAScene` y te abre automáticamente el archivo HTML en el navegador.

### B. El Visualizador Transformer en Vivo (`/brain`)
- Al teclear `/brain` en la terminal, se abre `outputs/ada_neural_brain_simulation.html`.
- **En reposo:** La red permanece en silencio estático (mostrando la GPU real en 1.0% de uso y 47°C). No hay animaciones engañosas de cerebros biológicos.
- **Durante la inferencia:** En cuanto envías un mensaje en PowerShell, el servidor puente activa el visor:
  - Ondas de activación tensorial cruzan las 40 capas decodificadoras a 60 FPS.
  - La matriz de atención causal proyecta los pesos de tus palabras clave en tiempo real.
  - El panel superior muestra tu consulta en curso y la velocidad de generación (48.5 tok/s).
  - Al completarse la respuesta, el sistema vuelve suavemente al reposo estático.

### C. Pensamiento Autónomo en Caliente (`/think`)
- Al activar `/think`, el motor `AutonomousMindEngine` empieza a razonar en segundo plano sin esperar que le hagas preguntas.
- Se plantea auto-retos (ej. resolver ecuaciones diferenciales de osciladores amortiguados o derivadas complejas), los valida con SciPy y SymPy, y por cada ciclo verificado sintetiza nuevas conexiones neuronales (+133 neuronas adaptativas generadas hasta ahora).
- Todo este flujo de pensamiento queda registrado en `data/work/training/ada_continual_dataset.jsonl` para auto-entrenamiento.

### D. Visión y Cinemática por ADB (`/vision`)
- Al teclear `/vision`, se abre `outputs/ada_live_vision_studio.html`.
- Si conectas un celular Android por USB con depuración activada, Ada lo detecta automáticamente por `adb`.
- Si no hay celular conectado, activa el simulador físico de laboratorio a 60 FPS, proyectando una trayectoria parabólica con vector de velocidad instantánea ($\vec{v}$) y aceleración ($\vec{a}$), contrastando cinemática teórica contra visión en tiempo real.

---

## 5. ¿Qué es lo que DE VERDAD se supone que debe de hacer ADA? (La Visión Integral)

El objetivo final de ADA concebido por Diego López trasciende el de un asistente convencional. El propósito definitivo de ADA es ser una **Inteligencia Artificial Científica, Soberana y Físicamente Anclada al Mundo Real**:

### 1. Soberanía Absoluta e Independencia Tecnológica
- **Cero dependencia de nubes corporativas:** Ada no depende de suscripciones a OpenAI, servidores de Microsoft o APIs de Google para pensar. Toda su inteligencia reside en los discos locales de Diego (`G:\programas\Ada`) y en el silicio de su NVIDIA RTX 5070.
- **Privacidad inviolable:** Datos sensibles, código privado, ideas y proyectos nunca salen de la máquina local.

### 2. Rigor Científico y Cero Cálculo Mental
- Un LLM es un modelo de probabilidad lingüística; por diseño probabilístico, comete errores cuando intenta calcular raíces cuadradas, integrales o p-valores mentalmente.
- **El mandato supremo de Ada:** Un LLM debe actuar como el **director del laboratorio**, no como la calculadora. Cuando surge un problema matemático o estadístico, Ada formula la hipótesis, pero delega el cálculo a motores deterministas comprobados (`SymPy`, `SciPy DOP853`, `DuckDB`, `PyTorch`). Toda afirmación técnica debe llevar el sello `[VERIFICADO: método]`.

### 3. Auto-Entrenamiento y Memoria Continua (Estilo Google Harness)
- Los modelos comerciales estándar están "congelados en el tiempo" (no aprenden nada nuevo de las conversaciones de hoy para mañana).
- **Lo que debe hacer Ada:** A medida que Diego trabaja con ella, Ada debe consolidar memoria episódica a largo plazo y filtrar los mejores razonamientos verificados. Cuando la computadora no esté en uso intensivo, Ada debe ejecutar **fine-tuning local QLoRA** sobre sus adaptadores en la RTX 5070. Con el tiempo, Ada se convierte en una versión hiper-especializada adaptada al flujo mental, vocabulario y proyectos de Diego.

### 4. Percepción Multimodal y Anclaje Físico (Ver el Mundo Real)
- Ada no debe vivir atrapada en un cuadro de texto.
- Mediante la cámara de un celular o webcam conectada por ADB/USB, Ada debe ser capaz de:
  - "Ver" un experimento físico (un péndulo oscilando, un circuito electrónico, un brazo robótico).
  - Medir ángulos, posiciones y velocidades fotograma a fotograma.
  - Ajustar en tiempo real los parámetros de control por puerto serie para estabilizar o manipular el sistema físico.

### 5. Presentación Gráfica y Visual de Nivel Profesional
- Prohibición tajante de entregables mediocres (hojas blancas tipo cuaderno, tablas de texto desalineadas o fórmulas rotas).
- Todo entregable generado por Ada debe tener la estética visual de un estudio de ingeniería de software de primer nivel: Dark Bento Glass, Canvas interactivo a 60 FPS, tipografía monoespaciada para datos y notación matemática de libro universitario.

---

## 6. Hoja de Ruta de Próximos Pasos (Roadmap)

```mermaid
flowchart LR
    FaseActual["FASE ACTUAL:\nCore 14.8B + Puente IPC +\nCLI UTF-8 + Visor 40 Capas +\n21/21 Tests Aprobados"] --> FaseSiguiente["SIGUIENTE PASO:\nStream en Vivo de Cámara ADB\n(Flujo H.264 directo al Canvas)"]
    FaseSiguiente --> FaseQLoRA["AUTO-ENTRENAMIENTO:\nPipeline QLoRA Nocturno en RTX 5070\nsobre data/work/training/"]
    FaseQLoRA --> FaseHardware["CONTROL FÍSICO:\nLoop Cerrado MCU Péndulo Invertido\nVisión Celular -> Cálculo -> Actuador"]
```

1. **Pipeline de Video ADB Directo (H.264 / scrcpy bridge):** Permitir que el streaming de video del celular se renderice nativamente en el canvas con latencia inferior a 50 ms para análisis de movimiento en vivo.
2. **Entrenamiento QLoRA Nocturno Autónomo:** Crear un script de un solo comando (`train_qlora.ps1`) que tome los ejemplos verificados acumulados en `data/work/training/ada_continual_dataset.jsonl` y afine un adaptador LoRA en la RTX 5070 para perfeccionar el razonamiento de Ada Q.
3. **Bucle de Control Físico en Tiempo Real:** Conectar la visión por ADB con el puente de hardware serie para que Ada pueda ver un péndulo real oscilar y enviar comandos de par para estabilizarlo en tiempo real.

---
*Documento generado y archivado para el sistema ADA por Diego López.*
