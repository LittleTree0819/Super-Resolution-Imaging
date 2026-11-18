$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe"
$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot
$pidFile = Join-Path $projectRoot "runs\div2k_train_v1.current.pid"
if (Test-Path -LiteralPath $pidFile) {
  $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw)
  if ($recordedPid -ne $PID -and (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue)) {
    throw "DIV2K generation is already running with PID $recordedPid"
  }
}
$PID | Set-Content -LiteralPath $pidFile
try {
  & $python -c "import super_resolution_imaging, super_resolution_imaging.data.sources as s; print('generator_version=' + super_resolution_imaging.__version__); print('sources_module=' + s.__file__)"
  & $python -m super_resolution_imaging.cli generate `
    --config configs\zemax_b.toml `
    --input data\DIV2K_train\DIV2K_train_HR `
    --output runs\div2k_train_v1 `
    --limit 800 `
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
