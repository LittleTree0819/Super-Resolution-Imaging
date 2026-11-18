$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$runDir=Join-Path $projectRoot 'runs\psf_low_memory_reference\run_v1'
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$runtime='D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$env:PYTHONPATH=Join-Path $projectRoot 'src'
if (Test-Path -LiteralPath (Join-Path $runDir 'guard_status.json')) {
    $previous=Get-Content -LiteralPath (Join-Path $runDir 'guard_status.json') -Raw | ConvertFrom-Json
    if ($previous.worker_pid) {
        $existing=Get-CimInstance Win32_Process -Filter "ProcessId = $($previous.worker_pid)" -ErrorAction SilentlyContinue
        if ($existing.CommandLine -like '*run_low_memory_psf_reference.py*') {throw 'This reference batch is already running.'}
    }
}
$lease=[IO.File]::Open((Join-Path $runDir 'run.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$state=[ordered]@{state='starting';guardian_pid=$PID;worker_pid=$null;peak_working_set_bytes=0L;working_set_limit_bytes=18GB;free_memory_limit_bytes=3GB;generation_resumed=$false;training_started=$false}
$worker=$null
function Save-State {
    $state.updated_at=[DateTimeOffset]::Now.ToString('o')
    $p=Join-Path $runDir 'guard_status.json'
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath ($p+'.tmp') -Encoding utf8
    Move-Item -LiteralPath ($p+'.tmp') -Destination $p -Force
}
function Stop-OwnedWorker {
    if ($null -eq $worker) {return}
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $($worker.Id)" | Where-Object Name -eq 'OpticStudio.exe' | ForEach-Object {
        $p=Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        if ($null -ne $p) {Stop-Process -InputObject $p -Force -ErrorAction SilentlyContinue}
    }
    $worker.Refresh()
    if (-not $worker.HasExited) {Stop-Process -InputObject $worker -Force -ErrorAction SilentlyContinue}
}
try {
    Save-State
    $worker=Start-Process -FilePath $runtime -ArgumentList @('scripts/run_low_memory_psf_reference.py','--max-directions','16') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDir 'worker.stdout.log') -RedirectStandardError (Join-Path $runDir 'worker.stderr.log') -PassThru
    $state.worker_pid=$worker.Id;$state.state='running';$timer=[Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        $worker.Refresh();if ($worker.HasExited) {break}
        $state.elapsed_seconds=[int]$timer.Elapsed.TotalSeconds
        $state.free_memory_bytes=[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $($worker.Id)" | Where-Object Name -eq 'OpticStudio.exe' | ForEach-Object {
            $p=Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                $state.opticstudio_pid=$p.Id
                $state.working_set_bytes=[long]$p.WorkingSet64
                $state.peak_working_set_bytes=[Math]::Max([long]$state.peak_working_set_bytes,[long]$p.PeakWorkingSet64)
            }
        }
        if ($state.peak_working_set_bytes -gt 18GB -or $state.free_memory_bytes -lt 3GB) {
            $state.state='memory_limited';Save-State;Stop-OwnedWorker;break
        }
        if ($timer.Elapsed.TotalHours -gt 12) {
            $state.state='time_budget_reached';Save-State;Stop-OwnedWorker;break
        }
        Save-State;Start-Sleep -Seconds 2
    }
    if ($state.state -eq 'running') {
        $worker.WaitForExit();$worker.Refresh();$state.worker_exit_code=$worker.ExitCode
        if ($worker.ExitCode -ne 0) {$state.state='worker_failed'}
        else {
            $result=Get-Content -LiteralPath (Join-Path $runDir 'status.json') -Raw | ConvertFrom-Json
            $state.state=$result.state
        }
    }
}
catch {$state.state='guard_failed';$state.error=$_.Exception.Message;Stop-OwnedWorker}
finally {Save-State}
& $runtime (Join-Path $projectRoot 'scripts\write_low_memory_psf_report.py')
$state.report_exit_code=$LASTEXITCODE;Save-State
$lease.Dispose()
