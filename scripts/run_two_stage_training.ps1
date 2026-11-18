param(
    [int]$Stage1Epochs = 200,
    [int]$Stage2Epochs = 100,
    [int]$BatchSize = 8,
    [int]$BaseChannels = 24
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$dataset = Join-Path $project "runs\div2k_train_v1"
$stage1 = Join-Path $project "runs\train_a_stage1"
$stage2 = Join-Path $project "runs\train_b_stage2"
$pidFile = Join-Path $project "runs\two_stage_training.pid"

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $dataset "COMPLETE.json"))) { throw "Dataset is incomplete: $dataset" }
if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "Two-stage training is already running as PID $existingPid"
    }
}

Set-Content -LiteralPath $pidFile -Value $PID -Encoding ascii
try {
    Write-Output "[$(Get-Date -Format o)] Stage 1 (A)"
    $stage1Summary = Join-Path $stage1 "summary.json"
    if (-not (Test-Path -LiteralPath $stage1Summary)) {
        $stage1Args = @(
            "-m", "super_resolution_imaging.training.cli",
            "--dataset", $dataset, "--output", $stage1, "--group", "a",
            "--epochs", $Stage1Epochs, "--batch-size", $BatchSize,
            "--base-channels", $BaseChannels, "--learning-rate", "0.0003"
        )
        $stage1Last = Join-Path $stage1 "last.pt"
        if (Test-Path -LiteralPath $stage1Last) { $stage1Args += @("--resume", $stage1Last) }
        & $python @stage1Args
        if ($LASTEXITCODE -ne 0) { throw "Stage 1 failed with exit code $LASTEXITCODE" }
    } else {
        Write-Output "Stage 1 is already complete; skipping."
    }

    $stage1Best = Join-Path $stage1 "best.pt"
    if (-not (Test-Path -LiteralPath $stage1Best)) { throw "Stage 1 best checkpoint is missing: $stage1Best" }

    Write-Output "[$(Get-Date -Format o)] Stage 2 (A to B)"
    $stage2Summary = Join-Path $stage2 "summary.json"
    if (-not (Test-Path -LiteralPath $stage2Summary)) {
        $stage2Args = @(
            "-m", "super_resolution_imaging.training.cli",
            "--dataset", $dataset, "--output", $stage2, "--group", "b",
            "--epochs", $Stage2Epochs, "--batch-size", $BatchSize,
            "--base-channels", $BaseChannels, "--learning-rate", "0.0001"
        )
        $stage2Last = Join-Path $stage2 "last.pt"
        if (Test-Path -LiteralPath $stage2Last) {
            $stage2Args += @("--resume", $stage2Last)
        } else {
            $stage2Args += @("--initialize-from", $stage1Best)
        }
        & $python @stage2Args
        if ($LASTEXITCODE -ne 0) { throw "Stage 2 failed with exit code $LASTEXITCODE" }
    } else {
        Write-Output "Stage 2 is already complete; skipping."
    }
    Write-Output "[$(Get-Date -Format o)] Two-stage training complete."
} finally {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
