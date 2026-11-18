$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONUNBUFFERED = "1"
$zosPython = "D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe"
$trainPython = Join-Path $root ".venv\Scripts\python.exe"
$lockPath = Join-Path $root "runs\symmetric_pilot_experiment.lock"
# A kernel-backed lock prevents a second pipeline even if PID visibility is restricted.
$lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$pidPath = Join-Path $root "runs\symmetric_pilot_experiment.pid"
$statePath = Join-Path $root "runs\symmetric_pilot_experiment.status.json"
function Write-ExperimentStatus([string]$Stage) {
  @{ stage=$Stage; pid=$PID; updated_at=(Get-Date -Format o) } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}
try {
  $PID | Set-Content -LiteralPath $pidPath -Encoding ascii
  Write-ExperimentStatus "generating"
  & $zosPython -m super_resolution_imaging.cli generate `
    --config configs\zemax_9frame_geometric.toml `
    --input data\nine_frame_pilot_sources `
    --output runs\nine_frame_pilot_32_symmetric `
    --limit 32 --patches 1 --resume
  if ($LASTEXITCODE -ne 0) { throw "Zemax generation failed: $LASTEXITCODE" }
  Write-ExperimentStatus "training_and_validation"
  & $trainPython scripts\train_symmetric_pilot_comparison.py
  if ($LASTEXITCODE -ne 0) { throw "Pilot comparison failed: $LASTEXITCODE" }
  Write-ExperimentStatus "complete"
}
catch {
  Write-ExperimentStatus "failed"
  throw
}
finally {
  $lock.Dispose()
  if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath -Force }
}
