# One-time setup on a new Windows PC. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Safe to run again: finished steps are skipped or repeated harmlessly.
param(
    [switch]$SkipPython,  # skip the large Python package install (about 3 GB)
    [switch]$SkipNode
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host "ERROR: $text" -ForegroundColor Red; exit 1 }
function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$exe $($arguments -join ' ')' failed (exit $LASTEXITCODE)." }
}

Step 'Checking prerequisites'
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { Fail 'Python launcher (py) not found. Install Python 3.11 from python.org.' }
& py -3.11 --version
if ($LASTEXITCODE -ne 0) { Fail 'Python 3.11 is required (newer versions are not supported by the pinned packages). Install it from python.org.' }
if (-not $SkipNode -and -not (Get-Command node -ErrorAction SilentlyContinue)) { Fail 'Node.js not found. Install Node.js 20 LTS or newer from nodejs.org.' }
if (-not (Get-Service -Name MongoDB -ErrorAction SilentlyContinue)) {
    Write-Host 'WARNING: MongoDB service not found. Install MongoDB Community Server (as a service) before starting the dashboard.' -ForegroundColor Yellow
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'WARNING: No NVIDIA GPU driver found. The project will run on the CPU, with much slower face recognition.' -ForegroundColor Yellow
}

$python = Join-Path $root 'venv\Scripts\python.exe'
Step 'Python environment'
if (-not (Test-Path $python)) { Run 'py' @('-3.11', '-m', 'venv', 'venv') } else { 'venv already exists' }
if (-not $SkipPython) {
    Run $python @('-m', 'pip', 'install', '--upgrade', 'pip')
    Run $python @('-m', 'pip', 'install', '-r', 'requirements.txt')
    # insightface pulls in the CPU-only onnxruntime, which overwrites the GPU
    # build in the same folder. Keep only the GPU build.
    & $python -m pip uninstall -y onnxruntime onnxruntime-gpu
    Run $python @('-m', 'pip', 'install', '--no-deps', 'onnxruntime-gpu==1.20.2')
} else { 'Skipped (-SkipPython)' }

Step 'Model files'
Run $python @('scripts\download_models.py')

Step 'Settings files'
foreach ($pair in @(@('.env.example', '.env'), @('dashboard\.env.example', 'dashboard\.env'))) {
    if (Test-Path $pair[1]) { "$($pair[1]) already exists (kept)" }
    else { Copy-Item $pair[0] $pair[1]; "created $($pair[1])" }
}

Step 'Dashboard packages'
if (-not $SkipNode) { Push-Location dashboard; try { Run 'npm' @('ci') } finally { Pop-Location } } else { 'Skipped (-SkipNode)' }

Step 'Checking the installation'
Run $python @('-c', "import torch, onnxruntime, insightface, torchreid, ultralytics, cv2; print('Python packages OK | GPU for PyTorch:', torch.cuda.is_available())")
Run $python @('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py')

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host @'
Next:
  1. Optional: put your CCTV address in .env (see the examples there).
  2. Start the engine:     .\venv\Scripts\python.exe main.py
  3. Start the dashboard:  cd dashboard; npm start
  4. Open http://localhost:3000 and enroll your face on the Enroll page.
'@
