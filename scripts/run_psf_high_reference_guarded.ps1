$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $projectRoot 'runs\psf_reliability_v2'
$runtime = 'D:\Program Files\ANSYS Inc\v261\commonfiles\CPython\3_10\winx64\Release\python\python.exe'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$statusPath = Join-Path $outputDir 'high_reference_guard.json'
$guard = [ordered]@{state='starting'; guardian_pid=$PID; worker_pid=$null; peak_working_set_bytes=0L; max_working_set_bytes=18GB; min_free_memory_bytes=3GB; max_elapsed_seconds=2700; generation_resumed=$false; training_started=$false}
function Save-Guard {
    $guard.updated_at = [DateTimeOffset]::Now.ToString('o')
    $temporary = $statusPath + '.tmp'
    $guard | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}
$worker = $null
function Stop-OwnedWorker {
    if ($null -eq $worker) { return }
    # Resolve ownership immediately before stopping only this batch's children.
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $($worker.Id)" | Where-Object Name -eq 'OpticStudio.exe' | ForEach-Object {
        $ownedProcess = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        if ($null -ne $ownedProcess) { Stop-Process -InputObject $ownedProcess -Force -ErrorAction SilentlyContinue }
    }
    $worker.Refresh()
    if (-not $worker.HasExited) { Stop-Process -InputObject $worker -Force -ErrorAction SilentlyContinue }
}
try {
    Save-Guard
    $worker = Start-Process -FilePath $runtime -ArgumentList @('scripts/resolve_psf_reliability.py','--suite','high_reference') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputDir 'high_reference.stdout.log') -RedirectStandardError (Join-Path $outputDir 'high_reference.stderr.log') -PassThru
    $guard.worker_pid = $worker.Id
    $guard.state = 'running'
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        $worker.Refresh()
        if ($worker.HasExited) { break }
        $guard.elapsed_seconds = [int]$timer.Elapsed.TotalSeconds
        $os = Get-CimInstance Win32_OperatingSystem
        $guard.free_memory_bytes = [long]$os.FreePhysicalMemory * 1024
        $ownedApps = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($worker.Id)" | Where-Object Name -eq 'OpticStudio.exe')
        foreach ($app in $ownedApps) {
            $p = Get-Process -Id $app.ProcessId -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                $guard.opticstudio_pid = $p.Id
                $guard.working_set_bytes = [long]$p.WorkingSet64
                $guard.peak_working_set_bytes = [Math]::Max([long]$guard.peak_working_set_bytes,[long]$p.PeakWorkingSet64)
            }
        }
        if ([long]$guard.peak_working_set_bytes -gt 18GB -or [long]$guard.free_memory_bytes -lt 3GB) {
            $guard.state = 'memory_limited'
            Save-Guard
            Stop-OwnedWorker
            break
        }
        if ($timer.Elapsed.TotalSeconds -gt 2700) {
            $guard.state = 'time_budget_reached'
            Save-Guard
            Stop-OwnedWorker
            break
        }
        Save-Guard
        Start-Sleep -Seconds 2
    }
    $worker.Refresh()
    if ($guard.state -eq 'running') {
        $guard.worker_exit_code = $worker.ExitCode
        $result = Get-Content -LiteralPath (Join-Path $outputDir 'high_reference_status.json') -Raw | ConvertFrom-Json
        $guard.state = if ($result.completed) {'completed'} else {'worker_failed'}
    }
}
catch {
    $guard.state = 'guard_failed'
    $guard.error = $_.Exception.Message
    Stop-OwnedWorker
}
finally {
    Save-Guard
}
# Always summarize completed points and failed gates; never resume production.
& $runtime (Join-Path $projectRoot 'scripts\write_psf_reliability_report.py')
$guard.report_exit_code = $LASTEXITCODE
Save-Guard
