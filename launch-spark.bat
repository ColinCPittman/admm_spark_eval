@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Spark Docker Launcher
echo ========================================
echo Which Spark version?
echo [1] Spark 4.0 (modern)
echo [2] Spark 2.4 (baseline)
echo.
set /p choice="Enter 1 or 2: "

if "%choice%"=="" (
    echo ERROR: No choice entered!
    pause
    exit /b 1
)

echo.
echo You selected: %choice%
echo.

REM Check if Docker is installed
REM Test Docker and capture exit code only
docker --version >nul 2>&1
if errorlevel 1 goto install_docker

echo ✓ Docker CLI found
goto docker_ready

:install_docker
echo ⚠ Docker not found or not accessible. Installing Docker Desktop automatically...
echo This will download and install Docker Desktop (~500MB)
echo Press Ctrl+C to cancel if you want to install manually
timeout /t 5

echo Downloading Docker Desktop installer...
set "installer_path=%TEMP%\DockerDesktopInstaller.exe"
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe' -OutFile '%installer_path%'}"
if not exist "%installer_path%" (
    echo ERROR: Failed to download Docker Desktop installer.
    pause
    exit /b 1
)

echo Installing Docker Desktop...
"%installer_path%" install --quiet --accept-license
if errorlevel 1 (
    echo ERROR: Docker Desktop installation failed.
    pause
    exit /b 1
)
del "%installer_path%" >nul 2>&1
echo ✓ Docker Desktop installed successfully!

goto docker_ready

:docker_ready
echo.
echo [STEP 2] Checking if Docker Desktop is running...
docker info >nul 2>&1
set docker_running=%errorlevel%

if "%docker_running%"=="0" (
    echo ✓ Docker Desktop is already running.
    goto after_docker_wait
) else (
    echo ⚠ Docker Desktop is not running. Starting it now...
    
    REM Try multiple common locations for Docker Desktop
    set "docker_exe="
    if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        echo DEBUG: Found at C:\Program Files\Docker\Docker\Docker Desktop.exe
        set "docker_exe=C:\Program Files\Docker\Docker\Docker Desktop.exe"
    )
    if exist "%USERPROFILE%\AppData\Local\Docker\Docker Desktop.exe" (
        echo DEBUG: Found at %USERPROFILE%\AppData\Local\Docker\Docker Desktop.exe
        set "docker_exe=%USERPROFILE%\AppData\Local\Docker\Docker Desktop.exe"
    )
    if exist "C:\Users\%USERNAME%\AppData\Local\Docker\Docker Desktop.exe" (
        echo DEBUG: Found at C:\Users\%USERNAME%\AppData\Local\Docker\Docker Desktop.exe
        set "docker_exe=C:\Users\%USERNAME%\AppData\Local\Docker\Docker Desktop.exe"
    )
    echo DEBUG: docker_exe=!docker_exe!
    
    if defined docker_exe (
        echo Found Docker Desktop at: !docker_exe!
        echo Starting Docker Desktop in background...
        start /min "" "!docker_exe!"
        call :wait_for_docker
        goto after_docker_wait
    ) else (
        echo.
        echo ERROR: Could not find Docker Desktop executable.
        echo Please start Docker Desktop manually and try again.
        echo.
        echo Common locations to check:
        echo   - C:\Program Files\Docker\Docker\Docker Desktop.exe
        echo   - %USERPROFILE%\AppData\Local\Docker\Docker Desktop.exe
        echo.
        pause
        exit /b 1
    )
)

goto after_docker_wait

:wait_for_docker
setlocal enabledelayedexpansion
set wait_count=0
:wait_loop
    timeout /t 10 /nobreak >nul
    set /a wait_count+=1
    echo Checking Docker... (attempt !wait_count! of 15)
    docker info >nul 2>&1
    set docker_running=!errorlevel!
    echo DEBUG: docker_running=!docker_running! wait_count=!wait_count!
    if "!docker_running!" NEQ "0" (
        if !wait_count! lss 15 (
            goto wait_loop
        ) else (
            echo.
            echo ERROR: Docker Desktop failed to start after 2.5 minutes.
            echo Please try these steps:
            echo 1. Start Docker Desktop manually from the Start menu
            echo 2. Wait for it to show "Docker Desktop is running" in the system tray
            echo 3. Run this script again
            echo.
            pause
            exit /b 1
        )
    )
    echo ✓ Docker Desktop is ready!
    endlocal
    exit /b 0

:after_docker_wait
echo.
echo [STEP 3] Starting Spark environment...

REM Quick test that docker commands work
echo DEBUG: Testing basic docker command...
docker --version
if errorlevel 1 (
    echo ERROR: Basic docker command failed!
    pause
    exit /b 1
)

if "%choice%"=="1" (
    echo DEBUG: Entered choice 1 block
    echo Starting Spark 4.0...
    echo DEBUG: About to set COMPOSE_FILE variable
    set "COMPOSE_FILE=docker\docker-compose.yml"
    echo DEBUG: COMPOSE_FILE set to: !COMPOSE_FILE!
    echo DEBUG: Checking if file exists: !COMPOSE_FILE!
    if not exist "!COMPOSE_FILE!" (
        echo ERROR: !COMPOSE_FILE! not found!
        echo DEBUG: Current directory contents:
        dir
        pause
        exit /b 1
    )
    echo DEBUG: File exists, continuing...
    echo Using compose file: !COMPOSE_FILE!
    echo DEBUG: Current directory: %cd%
    echo DEBUG: Running command: docker compose -f "!COMPOSE_FILE!" run --rm spark40 pyspark
    echo.
    echo ========================================
    echo Launching Spark 4.0 PySpark shell...
    echo Datasets mounted at /workspace/data
    echo ========================================
    echo.
    
    REM Try docker compose first, then fallback to docker-compose
    docker compose -f "!COMPOSE_FILE!" run --rm spark40 pyspark
    if errorlevel 1 (
        echo.
        echo WARNING: 'docker compose' failed, trying 'docker-compose'...
        echo DEBUG: Running command: docker-compose -f "!COMPOSE_FILE!" run --rm spark40 pyspark
        docker-compose -f "!COMPOSE_FILE!" run --rm spark40 pyspark
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to start Spark 4.0!
            echo Possible causes:
            echo 1. Docker images need to be built (first run takes 5-10 minutes)
            echo 2. Service 'spark40' not found in compose file
            echo 3. Docker Desktop not fully ready
            echo.
            echo DEBUG: Checking if compose file is readable...
            type "!COMPOSE_FILE!"
            echo.
            pause
        )
    )
) else if "%choice%"=="2" (
    echo Starting Spark 2.4...
    set "COMPOSE_FILE=docker\docker-compose.yml"
    if not exist "%COMPOSE_FILE%" (
        echo ERROR: %COMPOSE_FILE% not found!
        pause
        exit /b 1
    )
    echo Using compose file: %COMPOSE_FILE%
    echo.
    echo ========================================
    echo Launching Spark 2.4 Scala shell...
    echo Datasets mounted at /workspace/data
    echo.
    echo NOTE: Using Scala shell due to Python 3.9 compatibility issues
    echo with Spark 2.4. For Python development, use Spark 4.0.
    echo ========================================
    echo.
    
    REM Try docker compose first, then fallback to docker-compose
    docker compose -f "%COMPOSE_FILE%" run --rm spark24 spark-shell
    if errorlevel 1 (
        echo WARNING: 'docker compose' failed, trying 'docker-compose'...
        docker-compose -f "%COMPOSE_FILE%" run --rm spark24 spark-shell
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to start Spark 2.4!
            echo Possible causes:
            echo 1. Docker images need to be built (first run takes 5-10 minutes)
            echo 2. Service 'spark24' not found in compose file
            echo 3. Docker Desktop not fully ready
            echo.
            pause
        )
    )
) else (
    echo ERROR: Invalid choice '%choice%'. Please enter 1 or 2.
    pause
    exit /b 1
)

echo.
echo PySpark session ended.
echo.
pause 