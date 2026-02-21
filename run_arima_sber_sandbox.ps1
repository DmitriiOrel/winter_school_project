param(
    [string]$Figi = "BBG004730N88",
    [int]$DaysBack = 1095,
    [int]$FutureDays = 30,
    [int]$LoopMinutes = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed (exit code $LASTEXITCODE)."
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}
if (-not (Test-Path ".env")) {
    throw ".env not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = "."
$env:PYTHONDONTWRITEBYTECODE = "1"

$argsList = @(
    "-u",
    ".\tools\arima_sber_sandbox_bot.py",
    "--figi", "$Figi",
    "--days-back", "$DaysBack",
    "--future-days", "$FutureDays"
)

if ($LoopMinutes -le 0) {
    Invoke-External -Executable $pythonExe -Arguments $argsList -StepName "ARIMA forecast run"
    exit
}

Write-Host "Loop mode started. Interval: $LoopMinutes minute(s). Stop with Ctrl+C."
while ($true) {
    Invoke-External -Executable $pythonExe -Arguments $argsList -StepName "ARIMA forecast run"
    Start-Sleep -Seconds ($LoopMinutes * 60)
}
