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
$stage1 = Join-Path $project "runs\train_a_residual_stage1"
$stage2 = Join-Path $project "runs\train_b_residual_stage2"
$pidFile = Join-Path $project "runs\two_stage_residual_training.pid"

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $dataset "COMPLETE.json"))) { throw "Dataset is incomplete: $dataset" }
if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "Residual training is already running as PID $existingPid"
    }
}

Set-Content -LiteralPath $pidFile -Value $PID -Encoding ascii
try {
    Write-Output "[$(Get-Date -Format o)] Residual stage 1 (A)"
    if (-not (Test-Path -LiteralPath (Join-Path $stage1 "summary.json"))) {
        $arguments = @(
            "-m", "super_resolution_imaging.training.cli", "--dataset", $dataset,
            "--output", $stage1, "--group", "a", "--residual-output",
            "--epochs", $Stage1Epochs, "--batch-size", $BatchSize,
            "--base-channels", $BaseChannels, "--learning-rate", "0.0003"
        )
        $last = Join-Path $stage1 "last.pt"
        if (Test-Path -LiteralPath $last) { $arguments += @("--resume", $last) }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Residual stage 1 failed with exit code $LASTEXITCODE" }
    }

    $best = Join-Path $stage1 "best.pt"
    if (-not (Test-Path -LiteralPath $best)) { throw "Residual stage 1 best checkpoint is missing" }
    Write-Output "[$(Get-Date -Format o)] Residual stage 2 (A to B)"
    if (-not (Test-Path -LiteralPath (Join-Path $stage2 "summary.json"))) {
        $arguments = @(
            "-m", "super_resolution_imaging.training.cli", "--dataset", $dataset,
            "--output", $stage2, "--group", "b", "--residual-output",
            "--epochs", $Stage2Epochs, "--batch-size", $BatchSize,
            "--base-channels", $BaseChannels, "--learning-rate", "0.0001"
        )
        $last = Join-Path $stage2 "last.pt"
        if (Test-Path -LiteralPath $last) { $arguments += @("--resume", $last) }
        else { $arguments += @("--initialize-from", $best) }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Residual stage 2 failed with exit code $LASTEXITCODE" }
    }
    Write-Output "[$(Get-Date -Format o)] Residual two-stage training complete."
} finally {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
