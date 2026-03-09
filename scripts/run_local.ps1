param(
    [string]$Config = "configs/debug.yaml"
)

if (-not $env:CONDA_PREFIX) {
    Write-Error "No active Conda environment detected. Activate your environment first."
    exit 1
}

$pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found in active Conda environment: $pythonExe"
    exit 1
}

& $pythonExe src/train.py --config $Config