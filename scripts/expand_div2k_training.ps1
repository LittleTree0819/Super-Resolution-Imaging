$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe"
$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot

$dataset = "runs\div2k_train_v1"
$pidFile = Join-Path $projectRoot "runs\div2k_train_v1.expansion.pid"
$snapshot = Join-Path $projectRoot "runs\div2k_train_v1\expansion_baseline.json"
$transcript = Join-Path $projectRoot "runs\div2k_train_v1.expansion.task.log"
Start-Transcript -LiteralPath $transcript -Append | Out-Null
if (Test-Path -LiteralPath $pidFile) {
  $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw)
  $generatorChild = Get-CimInstance Win32_Process -Filter "ParentProcessId=$recordedPid" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*super_resolution_imaging.cli generate*" }
  if ($recordedPid -ne $PID -and $generatorChild) {
    throw "DIV2K training expansion is already running with PID $recordedPid"
  }
}
$PID | Set-Content -LiteralPath $pidFile
try {
  if (-not (Test-Path -LiteralPath $snapshot)) {
    & $python scripts\verify_training_expansion.py snapshot --dataset $dataset --snapshot $snapshot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  & $python -m super_resolution_imaging.cli generate `
    --config configs\zemax_b.toml `
    --input data\DIV2K_train\DIV2K_train_HR `
    --output $dataset `
    --limit 800 `
    --patches 4 `
    --only-split train `
    --resume
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & $python scripts\verify_training_expansion.py check --dataset $dataset --snapshot $snapshot
  exit $LASTEXITCODE
}
finally {
  if (Test-Path -LiteralPath $pidFile) {
    $owner = [int](Get-Content -LiteralPath $pidFile -Raw)
    if ($owner -eq $PID) { Remove-Item -LiteralPath $pidFile -Force }
  }
  Stop-Transcript | Out-Null
}
