<#
  Registra (o quita) el agente autonomo de ADA como tarea programada al iniciar sesion.
  NO se ejecuta solo: es una decision tuya. Se ejecuta sin ventana y solo mientras tu sesion este abierta.

    powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1          # instalar
    powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1 -Remove  # quitar
  Freno de emergencia en cualquier momento:  .\.venv\Scripts\python.exe .\scripts\ada_daemon.py --stop
#>
param([switch]$Remove)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskName = "ADA-Autonomy"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Tarea '$TaskName' eliminada."
    return
}

$Pythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$Script = Join-Path $Root "scripts\ada_daemon.py"
if (-not (Test-Path $Pythonw)) { throw "No existe $Pythonw. Ejecuta scripts/reproduce_env.ps1." }

$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument "`"$Script`"" -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Agente autonomo de ADA (auto-juego verificado)" | Out-Null
Write-Output "Tarea '$TaskName' instalada: arrancara al iniciar sesion. Estado: python scripts/ada_daemon.py --status"
