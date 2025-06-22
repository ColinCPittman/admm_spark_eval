@echo off
echo HIGGS Dataset Download Script
echo =============================
echo.

echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Install from python.org
    pause
    exit /b 1
)

echo Installing kagglehub...
python -m pip install kagglehub >nul 2>&1

echo Creating download script...
(
echo import kagglehub, os, shutil, sys
echo try:
echo     path = kagglehub.dataset_download('erikbiswas/higgs-uci-dataset'^)
echo     higgs_file = None
echo     for root, dirs, files in os.walk(path^):
echo         for f in files:
echo             if 'higgs' in f.lower(^) and f.endswith('.csv'^):
echo                 higgs_file = os.path.join(root, f^)
echo                 break
echo         if higgs_file: break
echo     if higgs_file:
echo         if os.path.exists('../HIGGS.csv'^): os.remove('../HIGGS.csv'^)
echo         shutil.copy2(higgs_file, '../HIGGS.csv'^)
echo         size = os.path.getsize('../HIGGS.csv'^) / (1024*1024*1024^)
echo         print(f'SUCCESS: HIGGS.csv ready ({size:.1f} GB^)'^)
echo     else:
echo         print('ERROR: Could not find HIGGS.csv in download'^)
echo         sys.exit(1^)
echo except Exception as e:
echo     print(f'ERROR: Download failed - {e}'^)
echo     print('Try: 1. Create Kaggle account 2. Get API token 3. Save to ~/.kaggle/'^)
echo     sys.exit(1^)
) > temp.py

echo Downloading HIGGS dataset...
python temp.py
set result=%errorlevel%

del temp.py >nul 2>&1

if %result% neq 0 (
    echo.
    echo Download failed! See error above.
    pause
    exit /b 1
)

echo.
echo Download completed successfully!
pause 