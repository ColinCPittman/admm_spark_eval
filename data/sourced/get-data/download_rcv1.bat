@echo off
echo RCV1-v2 Dataset Download Script
echo ================================
echo.

set BASE_URL=http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files

echo Downloading RCV1-v2 files from JMLR...
echo.

echo 1. Downloading training data (lyrl2004_vectors_train.dat.gz)...
if exist "../lyrl2004_vectors_train.dat.gz" (
    echo Training data already exists, skipping download
) else (
    powershell -Command "Invoke-WebRequest -Uri '%BASE_URL%/lyrl2004_vectors_train.dat.gz' -OutFile '../lyrl2004_vectors_train.dat.gz'"
    if %errorlevel% neq 0 (
        echo Failed to download training data
        pause
        exit /b 1
    )
    echo Training data downloaded successfully
)
echo.

echo 2. Downloading all test data parts (pt0-pt3)...

if exist "../lyrl2004_vectors_test_pt0.dat.gz" (
    echo Test pt0 already exists, skipping download
) else (
    powershell -Command "Invoke-WebRequest -Uri '%BASE_URL%/lyrl2004_vectors_test_pt0.dat.gz' -OutFile '../lyrl2004_vectors_test_pt0.dat.gz'"
    if %errorlevel% neq 0 (
        echo Failed to download test pt0
        pause
        exit /b 1
    )
    echo Test pt0 downloaded successfully
)

if exist "../lyrl2004_vectors_test_pt1.dat.gz" (
    echo Test pt1 already exists, skipping download
) else (
    powershell -Command "Invoke-WebRequest -Uri '%BASE_URL%/lyrl2004_vectors_test_pt1.dat.gz' -OutFile '../lyrl2004_vectors_test_pt1.dat.gz'"
    if %errorlevel% neq 0 (
        echo Failed to download test pt1
        pause
        exit /b 1
    )
    echo Test pt1 downloaded successfully
)

if exist "../lyrl2004_vectors_test_pt2.dat.gz" (
    echo Test pt2 already exists, skipping download
) else (
    powershell -Command "Invoke-WebRequest -Uri '%BASE_URL%/lyrl2004_vectors_test_pt2.dat.gz' -OutFile '../lyrl2004_vectors_test_pt2.dat.gz'"
    if %errorlevel% neq 0 (
        echo Failed to download test pt2
        pause
        exit /b 1
    )
    echo Test pt2 downloaded successfully
)

if exist "../lyrl2004_vectors_test_pt3.dat.gz" (
    echo Test pt3 already exists, skipping download
) else (
    powershell -Command "Invoke-WebRequest -Uri '%BASE_URL%/lyrl2004_vectors_test_pt3.dat.gz' -OutFile '../lyrl2004_vectors_test_pt3.dat.gz'"
    if %errorlevel% neq 0 (
        echo Failed to download test pt3
        pause
        exit /b 1
    )
    echo Test pt3 downloaded successfully
)

echo.
echo All downloads completed!
echo.

echo 3. Extracting files...
echo Note: Files are gzip format. You can extract them using 7-Zip or similar.
echo After extraction, you should have:
echo - lyrl2004_vectors_train.dat
echo - lyrl2004_vectors_test_pt0.dat
echo - lyrl2004_vectors_test_pt1.dat  
echo - lyrl2004_vectors_test_pt2.dat
echo - lyrl2004_vectors_test_pt3.dat

echo.
echo Dataset download completed!
echo.
echo Files downloaded:
dir /b *.dat.gz 2>nul

echo.
echo Next steps:
echo 1. Extract .gz files using 7-Zip or similar
echo 2. Load data in Spark using MLUtils.loadLibSVMFile()

pause 