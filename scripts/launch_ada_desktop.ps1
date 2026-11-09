# ============================================================================
# ADA - Lanzador 1-Click para Ollama Aislado (Disco G:) + DeepSeek Harness App
# ============================================================================
param(
    [switch]$RestartDesktopApp
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelsDir   = Join-Path $ProjectRoot ".ollama\models"
# Modelos en el NVMe (F:) si existe la copia verificada (ruta real: un enlace no sirve para Ollama).
if (Test-Path 'F:\ollama-models\manifests-v2') { $ModelsDir = 'F:\ollama-models' }
$DesktopExe  = (Join-Path $env:LOCALAPPDATA "Programs\DeepSeek Harness\DeepSeek Harness.exe")

$env:OLLAMA_HOST           = "127.0.0.1:11434"
$env:OLLAMA_MODELS         = $ModelsDir
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE  = "q4_0"
$env:OLLAMA_NUM_PARALLEL   = "1"
$env:OLLAMA_CONTEXT_LENGTH = "24576"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE    = "30m"

# 1. Verificar si Ollama responde en 127.0.0.1:11434 con nuestros modelos de G:\programas\Ada\.ollama\models
$needStartOllama = $true
try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
    $names = @($tags.models | ForEach-Object { $_.name })
    if ($names -contains "ada-q:latest") {
        $needStartOllama = $false
        Write-Host "[ADA] Servidor Ollama activo con modelos locales en G: ($($names -join ', '))." -ForegroundColor Green
    } else {
        Write-Host "[ADA] Ollama activo pero apuntando a otro directorio de modelos; reiniciando hacia $ModelsDir..." -ForegroundColor Yellow
        Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
} catch {
    $needStartOllama = $true
}

if ($needStartOllama) {
    Write-Host "[ADA] Iniciando daemon Ollama aislado en $ModelsDir (RTX 5070, FlashAttention=1, KV=q8_0)..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
            Write-Host "[ADA] Ollama listo: $(@($tags.models | ForEach-Object { $_.name }) -join ', ')" -ForegroundColor Green
            break
        } catch {}
    }
}

# 2. Abrir o reiniciar DeepSeek Harness Desktop App
if (-not (Test-Path $DesktopExe)) {
    throw "No se encontro DeepSeek Harness Desktop en: $DesktopExe"
}

if ($RestartDesktopApp) {
    Write-Host "[ADA] Reiniciando DeepSeek Harness Desktop para recargar configuracion..." -ForegroundColor Cyan
    Get-Process -Name "DeepSeek Harness" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$runningApp = Get-Process -Name "DeepSeek Harness" -ErrorAction SilentlyContinue
if (-not $runningApp) {
    Write-Host "[ADA] Abriendo DeepSeek Harness Desktop conectado a ADA..." -ForegroundColor Green
    Start-Process -FilePath $DesktopExe -WorkingDirectory $ProjectRoot
} else {
    Write-Host "[ADA] DeepSeek Harness Desktop ya esta abierto (HMR activo sobre cordis.patch.yml)." -ForegroundColor Green
}
