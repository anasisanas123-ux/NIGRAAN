$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot 'venv\Scripts\python.exe'
$nodeExecutable = (Get-Command node).Source

# Only stop the two listeners that run this project's known entry scripts.
$processesToStop = @()
foreach ($port in @(5001, 3000)) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { continue }
    $serviceProcessId = $listener.OwningProcess
    $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $serviceProcessId"
    $expected = if ($port -eq 5001) { '(?i)(main|multi_camera)\.py' } else { '(?i)app\.js' }
    if ($serviceProcess.CommandLine -notmatch $expected) {
        throw "Port $port has an unexpected owner. Refusing to stop it."
    }
    $processesToStop += $serviceProcessId
}
foreach ($serviceProcessId in $processesToStop) {
    Stop-Process -Id $serviceProcessId -Force
    Wait-Process -Id $serviceProcessId -Timeout 10 -ErrorAction SilentlyContinue
}

$logDirectory = Join-Path $projectRoot 'data\logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Start-Process -FilePath $nodeExecutable -ArgumentList 'src/app.js' -WorkingDirectory (Join-Path $projectRoot 'dashboard') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'dashboard.stdout.log') -RedirectStandardError (Join-Path $logDirectory 'dashboard.stderr.log') | Out-Null
Start-Process -FilePath $projectPython -ArgumentList '-u', 'main.py' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'engine.stdout.log') -RedirectStandardError (Join-Path $logDirectory 'engine.stderr.log') | Out-Null
Write-Output 'Restarted the Python engine and Node dashboard.'
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 4
    try { $result = Invoke-RestMethod -Uri 'http://127.0.0.1:3000/api/cameras/behavior-alerts' -TimeoutSec 3 }
    catch { continue }
    if ($result.weapon_model.state -in @('load_error', 'inference_error')) {
        throw ('Weapon detector failed: ' + $result.weapon_model.error)
    }
    if ($result.weapon_model.state -eq 'ready' -and $result.model.state -eq 'ready') {
        Write-Output ('Live model status: ' + (@{weapon=$result.weapon_model;violence=$result.model} | ConvertTo-Json -Depth 7 -Compress))
        exit 0
    }
}
throw 'Models have not completed live inference yet. Check the logs in data/logs.'
