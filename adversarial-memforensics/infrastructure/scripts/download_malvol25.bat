@echo off
title MalVol-25 Dataset Setup
color 0A
echo.
echo ================================================================
echo   MalVol-25 DATASET DOWNLOAD ^& SETUP
echo ================================================================
echo.
echo   Source: IEEE DataPort (DOI: 10.21227/kg5b-nf37)
echo   Size:  ~32 GB (compressed)
echo   Paper: "MalVol-25: A Diverse, Labeled and Detailed Malware
echo           Volatile Memory Dataset for Detection and Response"
echo.
echo ================================================================
echo.

cd /d "%~dp0\.."

:: ── Step 1: Create dataset directory ──────────────────────────────
echo [1/4] Creating dataset directories...
if not exist "data\malvol25\raw" mkdir data\malvol25\raw
if not exist "data\malvol25\extracted" mkdir data\malvol25\extracted
echo       data\malvol25\raw\        -- Place raw .mem/.raw files here
echo       data\malvol25\extracted\  -- Features will be saved here
echo.

:: ── Step 2: Check if dataset already exists ───────────────────────
echo [2/4] Checking for existing dataset files...
set DUMP_COUNT=0
for %%f in (data\malvol25\raw\*.mem data\malvol25\raw\*.raw data\malvol25\raw\*.vmem) do set /a DUMP_COUNT+=1

if %DUMP_COUNT% GTR 0 (
    echo       Found %DUMP_COUNT% memory dump files. Skipping download.
    echo.
    goto :CATALOGUE
)

echo       No dump files found in data\malvol25\raw\
echo.
echo ================================================================
echo   DOWNLOAD INSTRUCTIONS
echo ================================================================
echo.
echo   MalVol-25 requires an IEEE DataPort account to download.
echo.
echo   OPTION 1 - IEEE DataPort (requires subscription):
echo     1. Go to: https://doi.org/10.21227/kg5b-nf37
echo     2. Login with your IEEE account
echo     3. Download "Dataset All Files.zip" (~32 GB)
echo     4. Extract to: data\malvol25\raw\
echo.
echo   OPTION 2 - External SharePoint link (free):
echo     1. Go to the link on the IEEE DataPort page
echo     2. Download the zip archive
echo     3. Extract all .mem/.raw files to: data\malvol25\raw\
echo.
echo   OPTION 3 - Direct extraction (if zip is downloaded):
echo     If you have the zip file, place it in data\malvol25\ and
echo     this script will extract it automatically.
echo.

:: Check for zip file
if exist "data\malvol25\*.zip" (
    echo   Found zip file! Extracting...
    for %%f in (data\malvol25\*.zip) do (
        echo   Extracting %%f ...
        powershell -Command "Expand-Archive -Path '%%f' -DestinationPath 'data\malvol25\raw' -Force"
        echo   Extraction complete.
    )
    echo.
    goto :CATALOGUE
)

echo   After downloading, place files in data\malvol25\raw\ and
echo   re-run this script.
echo.
echo   Press any key to continue anyway (will scan empty directory)...
pause >nul
echo.

:CATALOGUE
:: ── Step 3: Activate venv and scan dataset ────────────────────────
echo [3/4] Scanning and cataloguing dataset...

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo       [WARNING] Virtual environment not found. Run setup_and_run.bat first.
)

python -c "
from pipeline.malvol25_loader import MalVol25Loader
loader = MalVol25Loader('data/malvol25/raw', 'data/malvol25/extracted')
entries = loader.scan_dataset()
loader.print_summary()
"

echo.

:: ── Step 4: Feature extraction prompt ─────────────────────────────
echo [4/4] Feature extraction...
echo.
echo   To extract VolMemLyzer features (requires Volatility 3):
echo.
echo     python -c "
echo       from pipeline.malvol25_loader import MalVol25Loader
echo       loader = MalVol25Loader('data/malvol25/raw', 'data/malvol25/extracted')
echo       loader.load_catalogue()
echo       X, y = loader.extract_all_features()
echo     "
echo.
echo   Or use the CLI:
echo     python run.py malvol25-scan --dataset-dir data\malvol25\raw
echo     python run.py malvol25-extract --max-samples 50
echo.
echo ================================================================
echo   SETUP COMPLETE
echo ================================================================
echo.
pause
