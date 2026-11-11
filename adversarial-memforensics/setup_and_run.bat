@echo off
title "Adversarial Memory Forensics - Setup & Run"
color 0A
echo.
echo ================================================================
echo   ADVERSARIAL MEMORY FORENSICS - INSTALLATION ^& SETUP
echo ================================================================
echo.

cd /d "%~dp0"

:: ── Step 1: Check Python ──────────────────────────────────────────
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)
python --version
echo       OK
echo.

:: ── Step 2: Create virtual environment ────────────────────────────
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo       Created new venv
) else (
    echo       venv already exists, reusing
)
echo.

:: ── Step 3: Activate venv and install dependencies ────────────────
echo [3/5] Installing dependencies (this may take 2-3 minutes)...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet

echo       Dependencies installed
echo.

:: ── Step 4: Create required directories ───────────────────────────
echo [4/5] Creating data directories...
if not exist "data\raw_dumps" mkdir data\raw_dumps
if not exist "data\features" mkdir data\features
if not exist "data\metadata" mkdir data\metadata
if not exist "data\timing_logs" mkdir data\timing_logs
if not exist "results\models" mkdir results\models
if not exist "results\phase0" mkdir results\phase0
if not exist "results\figures" mkdir results\figures
if not exist "logs" mkdir logs
echo       Directories ready
echo.

:: ── Step 5: Run tests to verify installation ──────────────────────
echo [5/5] Running test suite to verify installation...
echo.
python -m pytest tests/ -v --tb=short
echo.

if errorlevel 1 (
    color 0C
    echo ================================================================
    echo   [FAIL] Some tests failed. Check output above.
    echo ================================================================
) else (
    color 0A
    echo ================================================================
    echo   [SUCCESS] All tests passed! Installation verified.
    echo ================================================================
)

echo.
echo ────────────────────────────────────────────────────────────────
echo   AVAILABLE COMMANDS (run inside venv):
echo ────────────────────────────────────────────────────────────────
echo.
echo   python run.py test               Run test suite
echo   python run.py phase0 --dry-run   Phase 0 feasibility gate
echo   python run.py train              Train ensemble (synthetic)
echo   python run.py analyze --rq all   Run all RQ analyses
echo   python run.py full-experiment    Full end-to-end dry-run
echo.
echo   To activate venv later:  venv\Scripts\activate.bat
echo ────────────────────────────────────────────────────────────────
echo.
pause
