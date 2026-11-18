$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$dataset = Join-Path $project "runs\div2k_train_v1"
$root = Join-Path $project "runs\expanded_training_smoke"
$stage1 = Join-Path $root "stage1_a"
$stage2 = Join-Path $root "stage2_b"
$report = Join-Path $root "smoke_report.json"
$excluded = @("0552_00", "0036_01", "0449_00", "0449_01", "0449_02", "0449_03")

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (Test-Path -LiteralPath $root) { throw "Smoke output already exists: $root" }
New-Item -ItemType Directory -Path $root | Out-Null
Set-Location $project

$common = @(
    "--dataset", $dataset, "--split-residual-output", "--split-sigma", "4.0",
    "--component-loss-weight", "0.2", "--ssim-loss-weight", "0.2",
    "--epochs", "1", "--batch-size", "2", "--base-channels", "24",
    "--train-samples", "8", "--validation-samples", "4",
    "--seed", "5400", "--device", "auto", "--exclude"
) + $excluded

try {
    Write-Output "Smoke stage 1: A group"
    & $python -m super_resolution_imaging.training.cli @common `
        --output $stage1 --group a --learning-rate 0.0003
    if ($LASTEXITCODE -ne 0) { throw "Smoke stage 1 failed with exit code $LASTEXITCODE" }
    $best = Join-Path $stage1 "best.pt"
    if (-not (Test-Path -LiteralPath $best)) { throw "Smoke stage 1 did not produce best.pt" }

    Write-Output "Smoke stage 2: B group initialized from stage 1"
    & $python -m super_resolution_imaging.training.cli @common `
        --output $stage2 --group b --learning-rate 0.0001 --initialize-from $best
    if ($LASTEXITCODE -ne 0) { throw "Smoke stage 2 failed with exit code $LASTEXITCODE" }

    & $python -c @"
import hashlib, json
from pathlib import Path
root = Path(r'$root')
a = json.loads((root / 'stage1_a' / 'summary.json').read_text())
b = json.loads((root / 'stage2_b' / 'summary.json').read_text())
assert a['group'] == 'a' and b['group'] == 'b'
assert a['train_samples'] == b['train_samples'] == 8
assert a['validation_samples'] == b['validation_samples'] == 4
assert a['ssim_loss_weight'] == b['ssim_loss_weight'] == 0.2
assert a['split_residual_output'] and b['split_residual_output']
assert b['initialized_from']['path'] == str((root / 'stage1_a' / 'best.pt').resolve())
assert b['checkpoint_reload'] == 'passed'
report = {
    'status': 'passed', 'formal_training_started': False,
    'stage1': {'best_validation_psnr_db': a['best_validation_psnr_db'], 'best_validation_l1': a['best_validation_l1']},
    'stage2': {'best_validation_psnr_db': b['best_validation_psnr_db'], 'best_validation_l1': b['best_validation_l1']},
    'train_samples': 8, 'validation_samples': 4, 'epochs_per_stage': 1,
    'device': b['device'], 'stage2_initialized_from_stage1': True,
    'checkpoint_reload': b['checkpoint_reload'],
}
(root / 'smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
"@
    if ($LASTEXITCODE -ne 0) { throw "Smoke result validation failed" }
}
catch {
    @{ status = "failed"; formal_training_started = $false; error = $_.Exception.Message } |
        ConvertTo-Json | Set-Content -LiteralPath $report -Encoding utf8
    throw
}
