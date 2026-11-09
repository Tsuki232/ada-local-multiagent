param(
    [switch]$SkipModelPull = $false,
    [switch]$SkipAgentSmoke = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Output "=== [1/6] Sincronizando entorno Python 3.12 aislado (.venv) desde uv.lock ==="
$env:UV_LINK_MODE = "copy"
uv sync --locked --python 3.12

Write-Output "=== [2/6] Verificando/instalando Node.js portable v24.22.2 LTS (SHA-256 oficial) y DeepSeek Harness ==="
$NodeVer = "v24.22.2"
$NodeSha256 = "ca1bff5d8ead13c9f2ea404a176db46bd430704d5b5113734049fa365e7799e2"
$ToolsDir = Join-Path $ProjectRoot ".tools"
$NodeDir = Join-Path $ToolsDir "node-$NodeVer-win-x64"
$ZipPath = Join-Path $ToolsDir "node-$NodeVer-win-x64.zip"
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

if (-not (Test-Path (Join-Path $NodeDir "node.exe"))) {
    Invoke-WebRequest -Uri "https://nodejs.org/dist/$NodeVer/node-$NodeVer-win-x64.zip" -OutFile $ZipPath -UseBasicParsing
    $actualHash = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLower()
    if ($actualHash -ne $NodeSha256) {
        Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
        throw "Error de integridad SHA-256 en Node.js portable: esperado $NodeSha256, obtenido $actualHash"
    }
    Expand-Archive -Path $ZipPath -DestinationPath $ToolsDir -Force
    Remove-Item -Force $ZipPath
}
$env:PATH = "$NodeDir;$env:PATH"
if (Test-Path (Join-Path $ProjectRoot "package-lock.json")) {
    & "$NodeDir\npm.cmd" ci --no-audit --no-fund
} else {
    & "$NodeDir\npm.cmd" install --no-audit --no-fund
}

Write-Output "=== [3/6] Iniciando servidor Ollama aislado en G:\ (.ollama\models) ==="
powershell -ExecutionPolicy Bypass -File .\scripts\start_ollama.ps1

if (-not $SkipModelPull) {
    $env:OLLAMA_HOST = "127.0.0.1:11434"
    $env:OLLAMA_MODELS = (Resolve-Path ".\.ollama\models").Path
    Write-Output "=== [4/6] Construyendo identidad local Ada Q y Ada r1 en Ollama ==="
    ollama pull qwen3:14b
    ollama pull deepseek-r1:14b
    ollama cp qwen3:14b ada-q
    ollama cp deepseek-r1:14b ada-r1
    ollama create ada-q -f .\config\ollama\Ada-Q.Modelfile
    ollama create ada-r1 -f .\config\ollama\Ada-r1.Modelfile
    ollama rm qwen3:14b
    ollama rm deepseek-r1:14b
} else {
    Write-Output "=== [4/6] Omitiendo pull de modelos (--SkipModelPull) ==="
}

Write-Output "=== [5/6] Ejecutando prueba de humo GPU RTX 5070 (PyTorch cu128 + QLoRA + Doble Metodo) y Pytest ==="
.\.venv\Scripts\python.exe .\scripts\smoke_test_gpu.py
.\.venv\Scripts\pytest.exe -v

if (-not $SkipAgentSmoke) {
    Write-Output "=== [6/6] Ejecutando 'Hola Mundo' real en DeepSeek Harness con modelo local ==="
    .\.venv\Scripts\python.exe .\scripts\smoke_test_agent.py
} else {
    Write-Output "=== [6/6] Omitiendo smoke test del agente (--SkipAgentSmoke) ==="
}

Write-Output "=== ENTORNO ADA REPRODUCIDO Y VERIFICADO CON EXITO ==="
