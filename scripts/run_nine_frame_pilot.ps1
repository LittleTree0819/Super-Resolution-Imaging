$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe"
$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot

$pidFile = Join-Path $projectRoot "runs\nine_frame_pilot_32.current.pid"
if (Test-Path -LiteralPath $pidFile) {
  $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw)
  if ($recordedPid -ne $PID -and (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue)) {
    throw "Nine-frame pilot is already running with PID $recordedPid"
  }
}
$PID | Set-Content -LiteralPath $pidFile

try {
  & $python -m super_resolution_imaging.cli generate `
    --config configs\zemax_9frame_pilot_refined.toml `
    --input data\nine_frame_pilot_sources `
    --output runs\nine_frame_pilot_32 `
    --limit 32 `
    --patches 1 `
    --resume
  exit $LASTEXITCODE
}
finally {
  if (Test-Path -LiteralPath $pidFile) {
    $owner = [int](Get-Content -LiteralPath $pidFile -Raw)
    if ($owner -eq $PID) { Remove-Item -LiteralPath $pidFile -Force }
  }
}
