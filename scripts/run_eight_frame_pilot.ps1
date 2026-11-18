$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$zosPython = 'D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$python = Join-Path $project '.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $project 'src'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUNBUFFERED = '1'
$config = 'configs/zemax_8frame_quarter_pilot.toml'
$geometry = 'runs/eight_frame_quarter_geometry'
$dataset = 'runs/eight_frame_quarter_pilot_32'
$analysis = 'runs/eight_frame_quarter_analysis'
$state = Join-Path $project 'runs/eight_frame_quarter_pilot_status.json'
function Save-State([string]$Stage, [string]$Status, [string]$Message = '') {
    [pscustomobject]@{stage=$Stage;status=$Status;pid=$PID;updated_at=(Get-Date -Format o);message=$Message} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
}
try {
    Save-State 'geometry' 'running'
    if (-not (Test-Path -LiteralPath "$geometry/geometry.json")) {
        & $zosPython scripts/trace_nine_frame_geometry.py --config $config --output $geometry
        if ($LASTEXITCODE -ne 0) { throw "Ray trace failed: $LASTEXITCODE" }
    }
    & $python scripts/analyze_eight_frame_pilot.py --geometry $geometry
    if ($LASTEXITCODE -ne 0) { throw 'Phase gate failed; no dataset generation started' }
    Save-State 'generation' 'running'
    & $zosPython -m super_resolution_imaging.cli generate --config $config `
        --input data/nine_frame_pilot_sources --output $dataset --limit 32 --patches 1 --resume
    if ($LASTEXITCODE -ne 0) { throw "Image generation failed: $LASTEXITCODE" }
    Save-State 'analysis' 'running'
    & $python scripts/analyze_eight_frame_pilot.py --geometry $geometry --dataset $dataset --output $analysis
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $LASTEXITCODE" }
    Save-State 'complete' 'completed'
}
catch {
    Save-State 'interrupted_or_failed' 'failed' $_.Exception.Message
    throw
}
