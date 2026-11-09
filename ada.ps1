# ============================================================================
# ADA — Lanzador 1-Click para Terminal CLI (PowerShell)
# Uso:
#   .\ada.ps1                              (Modo interactivo continuo REPL)
#   .\ada.ps1 "hazme una animacion..."     (Modo directo de una sola linea)
# ============================================================================
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$UserQuery
)

# Forzar codificacion UTF-8 universal en PowerShell y Python
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $PSScriptRoot).Path
$ModelsDir = Join-Path $ProjectRoot ".ollama\models"
# Modelos en el NVMe (F:) si existe la copia verificada (ruta real: un enlace no sirve para Ollama).
if (Test-Path 'F:\ollama-models\manifests-v2') { $ModelsDir = 'F:\ollama-models' }

$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_MODELS = $ModelsDir
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q4_0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_CONTEXT_LENGTH = "24576"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"

# 1. Verificar si Ollama esta activo en 127.0.0.1:11434
$alive = $false
try {
    $ver = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 1
    $alive = $true
} catch {
    $alive = $false
}

if (-not $alive) {
    Write-Host "[ADA] Iniciando servidor local Ollama en $ModelsDir..." -ForegroundColor Cyan
    Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\start_ollama.ps1`"" -WindowStyle Hidden
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $ver = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 1
            $alive = $true
            break
        } catch {}
    }
}

# 2. Ejecutar CLI de ADA
$cliScript = Join-Path $ProjectRoot "scripts\ada_cli.py"
if ($UserQuery -and $UserQuery.Count -gt 0) {
    $argStr = $UserQuery -join " "
    & uv run python $cliScript $argStr
} else {
    & uv run python $cliScript
}
