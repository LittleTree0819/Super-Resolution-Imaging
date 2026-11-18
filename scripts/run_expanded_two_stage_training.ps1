param(
    [int]$Stage1Epochs = 200,
    [int]$Stage2Epochs = 100,
    [int]$BatchSize = 8,
    [int]$BaseChannels = 24,
    [double]$ComponentLossWeight = 0.2,
    [double]$SsimLossWeight = 0.2,
    [double]$SplitSigma = 4.0,
    [int]$Seed = 5400,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$dataset = Join-Path $project "runs\div2k_train_v1"
$stage1 = Join-Path $project "runs\train_a_expanded_split_residual_ssim_w020_stage1"
$stage2 = Join-Path $project "runs\train_b_expanded_split_residual_ssim_w020_stage2"
$preflight = Join-Path $project "runs\expanded_training_preflight.json"
$pidFile = Join-Path $project "runs\expanded_two_stage_training.pid"
$log = Join-Path $project "runs\expanded_two_stage_training.log"

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
foreach ($required in @("COMPLETE.json", "manifest.jsonl", "quality_exclusions.json")) {
    if (-not (Test-Path -LiteralPath (Join-Path $dataset $required))) { throw "Dataset file is missing: $required" }
}
if ($Stage1Epochs -lt 1 -or $Stage2Epochs -lt 1 -or $BatchSize -lt 1) {
    throw "Epoch and batch-size values must be positive"
}

Set-Location $project
& $python scripts\preflight_expanded_training.py `
    --dataset $dataset --output $preflight --device $Device `
    --base-channels $BaseChannels --component-loss-weight $ComponentLossWeight `
    --ssim-loss-weight $SsimLossWeight --split-sigma $SplitSigma
if ($LASTEXITCODE -ne 0) { throw "Expanded-training preflight failed with exit code $LASTEXITCODE" }
if ($PreflightOnly) {
    Write-Output "Preflight passed. No training was started."
    exit 0
}

if (Test-Path -LiteralPath $pidFile) {
    $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $trainingChild = Get-CimInstance Win32_Process -Filter "ParentProcessId=$recordedPid" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*super_resolution_imaging.training.cli*" }
    if ($trainingChild) { throw "Expanded training is already running with PID $recordedPid" }
}

$excluded = @("0552_00", "0036_01", "0449_00", "0449_01", "0449_02", "0449_03")
$PID | Set-Content -LiteralPath $pidFile -Encoding ascii
Start-Transcript -LiteralPath $log -Append | Out-Null
try {
    Write-Output "[$(Get-Date -Format o)] Expanded stage 1: A-group aberration removal"
    if (-not (Test-Path -LiteralPath (Join-Path $stage1 "summary.json"))) {
        $arguments = @(
            "-m", "super_resolution_imaging.training.cli", "--dataset", $dataset,
            "--output", $stage1, "--group", "a", "--split-residual-output",
            "--split-sigma", $SplitSigma, "--component-loss-weight", $ComponentLossWeight,
            "--ssim-loss-weight", $SsimLossWeight, "--epochs", $Stage1Epochs,
            "--batch-size", $BatchSize, "--base-channels", $BaseChannels,
            "--learning-rate", "0.0003", "--seed", $Seed, "--device", $Device,
            "--exclude"
        ) + $excluded
        $last = Join-Path $stage1 "last.pt"
        if (Test-Path -LiteralPath $last) { $arguments += @("--resume", $last) }
        elseif (Test-Path -LiteralPath $stage1) { throw "Stage 1 directory exists without a resumable last.pt: $stage1" }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Expanded stage 1 failed with exit code $LASTEXITCODE" }
    }

    $best = Join-Path $stage1 "best.pt"
    if (-not (Test-Path -LiteralPath $best)) { throw "Expanded stage 1 best checkpoint is missing" }
    Write-Output "[$(Get-Date -Format o)] Expanded stage 2: B-group multi-frame fusion"
    if (-not (Test-Path -LiteralPath (Join-Path $stage2 "summary.json"))) {
        $arguments = @(
            "-m", "super_resolution_imaging.training.cli", "--dataset", $dataset,
            "--output", $stage2, "--group", "b", "--split-residual-output",
            "--split-sigma", $SplitSigma, "--component-loss-weight", $ComponentLossWeight,
            "--ssim-loss-weight", $SsimLossWeight, "--epochs", $Stage2Epochs,
            "--batch-size", $BatchSize, "--base-channels", $BaseChannels,
            "--learning-rate", "0.0001", "--seed", $Seed, "--device", $Device,
            "--exclude"
        ) + $excluded
        $last = Join-Path $stage2 "last.pt"
        if (Test-Path -LiteralPath $last) { $arguments += @("--resume", $last) }
        elseif (Test-Path -LiteralPath $stage2) { throw "Stage 2 directory exists without a resumable last.pt: $stage2" }
        else { $arguments += @("--initialize-from", $best) }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Expanded stage 2 failed with exit code $LASTEXITCODE" }
    }
    Write-Output "[$(Get-Date -Format o)] Expanded two-stage training complete."
}
finally {
    Stop-Transcript | Out-Null
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
