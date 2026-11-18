$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$env:PYTHONPATH = Join-Path $project 'src'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $project '.venv\Scripts\python.exe'
$zosPython = 'D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$status = Join-Path $project 'runs\independent_four_layout_pipeline_status.json'
function Save-State([string]$Stage, [string]$State, [string]$Message = '') {
    [pscustomobject]@{stage=$Stage;status=$State;pid=$PID;updated_at=(Get-Date -Format o);message=$Message} |
        ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
}
try {
    Save-State 'generation' 'running'
    & $zosPython scripts/generate_independent_four_layout_pilot.py
    if ($LASTEXITCODE -ne 0) {throw "Generation failed: $LASTEXITCODE"}
    Save-State 'preflight_and_training' 'running'
    & $python scripts/train_independent_four_layouts.py
    if ($LASTEXITCODE -ne 0) {throw "Training or preflight failed: $LASTEXITCODE"}
    Save-State 'complete' 'completed'
}
catch {Save-State 'failed' 'failed' $_.Exception.Message;throw}
