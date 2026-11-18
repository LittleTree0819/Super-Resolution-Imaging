param([int]$GeneratorPid = 6968)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot

while (Get-Process -Id $GeneratorPid -ErrorAction SilentlyContinue) {
  Start-Sleep -Seconds 15
}
if (-not (Test-Path "runs\nine_frame_pilot_32\COMPLETE.json")) {
  throw "Nine-frame generator ended without COMPLETE.json"
}

$datasetRoot = Join-Path $projectRoot "runs\nine_frame_pilot_32"
function Get-PilotSha256([string]$Path) {
  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  $stream = [System.IO.File]::OpenRead($Path)
  try {
    return [System.BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace("-", "").ToLowerInvariant()
  }
  finally { $stream.Dispose(); $algorithm.Dispose() }
}
$records = @(Get-Content (Join-Path $datasetRoot "manifest.jsonl") | ForEach-Object { $_ | ConvertFrom-Json })
if ($records.Count -ne 32 -or @($records.sample_id | Select-Object -Unique).Count -ne 32) {
  throw "Expected 32 unique pilot records"
}
foreach ($record in $records) {
  $sample = Join-Path $datasetRoot ("samples\" + $record.sample_id)
  foreach ($group in @("a", "b")) {
    foreach ($index in 0..8) {
      $frame = Join-Path $sample ($group + "\frame_" + $index.ToString("00") + ".png")
      if (-not (Test-Path -LiteralPath $frame)) { throw "Missing frame: $frame" }
      $expected = $record.output_sha256.$group[$index]
      if ((Get-PilotSha256 $frame) -ne $expected) {
        throw "Frame hash mismatch: $frame"
      }
    }
  }
  $target = Join-Path $sample "target.png"
  if ((Get-PilotSha256 $target) -ne $record.target_sha256) {
    throw "Target hash mismatch: $target"
  }
}
"Verified 32 samples, 576 frames and 32 targets against manifest hashes"

& $python scripts\analyze_nine_frame_shifts.py `
  runs\nine_frame_pilot_32 `
  --output runs\nine_frame_pilot_32\shift_report.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts\analyze_nine_frame_pilot.py `
  --dataset runs\nine_frame_pilot_32 `
  --calibration runs\nine_frame_calibration_refined\shift_report.json `
  --output runs\nine_frame_pilot_32_analysis
exit $LASTEXITCODE
