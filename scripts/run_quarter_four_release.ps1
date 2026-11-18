param([switch]$AcceptKnownPhotometry)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$env:PYTHONPATH = Join-Path $project 'src'
$env:PYTHONUNBUFFERED = '1'
$zosPython = 'D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$state = Join-Path $project 'runs\quarter_four_release_status.json'
$lease=[IO.File]::Open((Join-Path $project 'runs\quarter_four_release.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$photometryArgs=@()
if ($AcceptKnownPhotometry) {$photometryArgs=@('--accept-known-photometry')}
function Read-Sha256([string]$Path) {
    $stream=[IO.File]::OpenRead($Path)
    $algorithm=[Security.Cryptography.SHA256]::Create()
    try {return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}
    finally {$algorithm.Dispose();$stream.Dispose()}
}
function Save-State([string]$Stage,[string]$Status,[string]$Message='') {
    [pscustomobject]@{pid=$PID;stage=$Stage;status=$Status;updated_at=(Get-Date -Format o);message=$Message;training_started=$false;legacy_continuation=[bool]$AcceptKnownPhotometry} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
}
try {
    if ($AcceptKnownPhotometry) {
        Save-State 'existing' 'checking' 'Verify completed samples; retain accepted photometry flags'
        & $zosPython scripts/audit_quarter_four_release.py --stage existing @photometryArgs
        if ($LASTEXITCODE -ne 0) {throw 'Existing sample integrity audit failed'}
        Save-State 'legacy-reproduction' 'checking'
        $proofPath=Join-Path $project 'runs\legacy_quarter_reproduction\latest.json'
        $reuseProof=$false
        if (Test-Path -LiteralPath $proofPath) {
            $proof=Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json
            $cfgHash=Read-Sha256 (Join-Path $project 'configs\zemax_quarter_four_frozen.toml')
            $modelHash=Read-Sha256 (Join-Path $project 'runs\div2k_quarter_four_v1\prepared_model.zmx')
            $reuseProof=$proof.passed -and $proof.completed -and $proof.config_sha256 -eq $cfgHash -and $proof.prepared_model_sha256 -eq $modelHash -and (Get-Item -LiteralPath $proofPath).LastWriteTimeUtc -gt [DateTime]::UtcNow.AddMinutes(-30)
        }
        if (-not $reuseProof) {
            & $zosPython scripts/verify_legacy_quarter_reproduction.py
            if ($LASTEXITCODE -ne 0) {throw 'Legacy optical output reproduction failed'}
        }
        else {Write-Output 'Reused recent exact four-frame legacy reproduction check'}
    }
    foreach ($stage in @('canary','train-validation','test')) {
        Save-State $stage 'generating'
        & $zosPython scripts/generate_quarter_four_release.py --stage $stage @photometryArgs
        if ($LASTEXITCODE -ne 0) {throw "Generation stopped: $stage, exit $LASTEXITCODE"}
        $qcStage = if ($stage -eq 'test') {'all'} else {$stage}
        Save-State $qcStage 'checking'
        & $zosPython scripts/audit_quarter_four_release.py --stage $qcStage @photometryArgs
        if ($LASTEXITCODE -ne 0) {throw "QC stopped: $qcStage, exit $LASTEXITCODE"}
    }
    $finalStatus=if ($AcceptKnownPhotometry) {'completed_with_known_optical_limitations'} else {'completed'}
    Save-State 'complete' $finalStatus
}
catch {Save-State 'interrupted_or_failed' 'failed' $_.Exception.Message;throw}
finally {$lease.Dispose()}
