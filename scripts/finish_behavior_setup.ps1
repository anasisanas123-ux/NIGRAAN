$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot 'venv\Scripts\python.exe'
$wheel = Join-Path $projectRoot 'data\setup_cache\torch-2.4.1+cu118-cp311-cp311-win_amd64.whl'
$installTargets = @('torch==2.4.1', 'torchvision==0.19.1')

# Refuse an incomplete wheel before stopping the current engine.
if (Test-Path -LiteralPath $wheel) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [System.IO.Compression.ZipFile]::OpenRead($wheel)
  try {
    if (-not ($archive.Entries | Where-Object { $_.FullName -eq 'torch-2.4.1+cu118.dist-info/METADATA' })) {
        throw 'The downloaded wheel is not the expected PyTorch version.'
    }
  } finally { $archive.Dispose() }
  $installTargets = @($wheel, 'torchvision==0.19.1')
}

$engineWasRunning = $false
$listener = Get-NetTCPConnection -LocalPort 5001 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $engineProcessId = $listener.OwningProcess
    $engineProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $engineProcessId"
    $homeLine = (Get-Content -LiteralPath (Join-Path $projectRoot 'venv\pyvenv.cfg') | Where-Object { $_ -like 'home = *' } | Select-Object -First 1)
    $basePython = Join-Path ($homeLine.Substring(7)) 'python.exe'
    if (($engineProcess.ExecutablePath -ine $projectPython -and $engineProcess.ExecutablePath -ine $basePython) -or
        $engineProcess.CommandLine -notmatch '(?i)(main|multi_camera)\.py') {
        throw 'Port 5001 is owned by an unexpected process. Stop the project engine manually before continuing.'
    }
    Write-Output 'Stopping the project camera engine to replace loaded PyTorch DLLs.'
    Stop-Process -Id $engineProcessId -Force
    Wait-Process -Id $engineProcessId -Timeout 10 -ErrorAction SilentlyContinue
    $engineWasRunning = $true
}

Write-Output 'Installing pinned CUDA-enabled PyTorch and matching torchvision (downloads if not already installed).'
& $projectPython -m pip install --no-cache-dir @installTargets --index-url 'https://download.pytorch.org/whl/cu118'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. The engine has not been restarted.' }

Write-Output 'Verifying the existing behavior checkpoint on a synthetic CPU clip.'
& $projectPython (Join-Path $projectRoot 'verify_behavior_model.py')
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint verification failed. Fix the reported loader error before restarting.' }
& $projectPython -m unittest discover -s (Join-Path $projectRoot 'tests') -p 'test_dashboard_regressions.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Regression checks failed.' }

if ($engineWasRunning) {
    $logDirectory = Join-Path $projectRoot 'data\logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stdout = Join-Path $logDirectory 'engine.stdout.log'
    $stderr = Join-Path $logDirectory 'engine.stderr.log'
    $engineJob = Start-Process -FilePath $projectPython -ArgumentList '-u', 'main.py' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-Output ('Restarted the project engine, launcher process ' + $engineJob.Id + '.')
    Write-Output 'Waiting for the existing model to start processing camera clips.'
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 4
        try { $result = Invoke-RestMethod -Uri 'http://127.0.0.1:5001/behavior-alerts' -TimeoutSec 3 }
        catch { continue }
        $state = $result.violence_model.state
        if ($state -eq 'load_error' -or $state -eq 'inference_error') {
            throw ('Live model failed: ' + $result.violence_model.error)
        }
        $processed = @($result.violence_model.cameras.PSObject.Properties | Where-Object { $_.Value.last_inference_at })
        if ($state -eq 'ready' -and $processed.Count -gt 0) {
            Write-Output ('Live behavior model: ' + ($result.violence_model | ConvertTo-Json -Depth 6 -Compress))
            exit 0
        }
    }
    throw 'No completed live inference yet. Check camera health and data/logs/engine.stderr.log.'
}
Write-Output 'Dependency and checkpoint checks passed. Start the project engine to use the fix.'
