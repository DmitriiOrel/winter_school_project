$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}
if (-not (Test-Path ".env")) {
    throw ".env not found. Run .\quickstart.ps1 -Token <YOUR_TOKEN> first."
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = "."
$env:PYTHONDONTWRITEBYTECODE = "1"

& $pythonExe -u .\tools\weekly_arima_5d_sandbox.py
if ($LASTEXITCODE -ne 0) {
    throw "Run failed with exit code $LASTEXITCODE"
}
