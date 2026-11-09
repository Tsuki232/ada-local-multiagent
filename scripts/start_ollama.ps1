param(
    [switch]$StopExisting = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelsDir = Join-Path $ProjectRoot ".ollama\models"
# Copia verificada de los modelos en el NVMe (F:): G: (SATA) lee 14-60 MB/s lo no cacheado y cargar ada-q tardaba ~2 min en frio.
# Debe ser la RUTA REAL: Ollama no recorre una raiz que sea un enlace (junction) y veria 0 modelos.
if (Test-Path 'F:\ollama-models\manifests-v2') { $ModelsDir = 'F:\ollama-models' }
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_MODELS = $ModelsDir
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q4_0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_CONTEXT_LENGTH = "24576"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"

if ($StopExisting) {
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2
    Write-Output "Ollama ya esta activo en 127.0.0.1:11434 (version $($resp.version))."
} catch {
    Write-Output "Iniciando servidor Ollama aislado (OLLAMA_MODELS=$ModelsDir, KV=q8_0, FlashAttn=1, MaxLoaded=1)..."
    $LogOut = Join-Path $ProjectRoot ".ollama\server.out.log"
    $LogErr = Join-Path $ProjectRoot ".ollama\server.err.log"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr
    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2
            Write-Output "Ollama listo (version $($resp.version))."
            $ready = $true
            break
        } catch {}
    }
    if (-not $ready) {
        throw "No se pudo conectar a Ollama tras 10 segundos. Revisa $LogErr"
    }
}
