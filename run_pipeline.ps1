param(
    [string]$DatasetRoot = "C:\WhatsAppStudy",
    [string]$DeviceIp = "192.168.250.10",
    [string]$TestSession
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create .venv and install requirements first."
}

& $python (Join-Path $PSScriptRoot "data_handling.py") `
    $DatasetRoot `
    --device-ip $DeviceIp

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$modelArguments = @(
    (Join-Path $PSScriptRoot "models.py")
    (Join-Path $DatasetRoot "derived\features.csv")
    "--model"
    (Join-Path $PSScriptRoot "artifacts\whatsapp_random_forest.joblib")
)

if ($TestSession) {
    $modelArguments += @("--test-session", $TestSession)
}

& $python @modelArguments
exit $LASTEXITCODE
