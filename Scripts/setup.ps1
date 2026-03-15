# Usage: powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Creates venv, installs deps, runs the app.
# Requires: Python 3.9+

Write-Host "[1/4] Creating virtual environment at .venv"
python -m venv .venv

Write-Host "[2/4] Activating .venv"
# Activate the venv for the current process
$venvPath = ".\.venv\Scripts\Activate.ps1"
. $venvPath

Write-Host "[3/4] Upgrading pip and installing requirements"
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[4/4] Running Flask app (Phase 6 simulation)"
python app.py
