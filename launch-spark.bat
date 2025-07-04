@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Spark Docker Launcher
echo ========================================
echo Which Spark version?
echo [1] Spark 4.0 (Scala shell)
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

echo Docker CLI found
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
echo Docker Desktop installed successfully!

goto docker_ready

:docker_ready
echo.
echo [STEP 2] Checking if Docker Desktop is running...
docker info >nul 2>&1
set docker_running=%errorlevel%

if "%docker_running%"=="0" (
    echo Docker Desktop is already running.
    goto after_docker_wait
) else (
    echo Docker Desktop is not running. Starting it now...
    
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
    echo Docker Desktop is ready!
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
    echo Starting Spark 4.0 Scala shell...
    set "COMPOSE_FILE=docker\docker-compose.yml"
    if not exist "!COMPOSE_FILE!" (
        echo ERROR: !COMPOSE_FILE! not found!
        pause
        exit /b 1
    )
    echo Using compose file: !COMPOSE_FILE!
    echo.

    echo ========================================
    echo [STEP A] Starting all Spark 4.0 services...
    echo ========================================
    docker compose -f "!COMPOSE_FILE!" up -d spark-master-40 spark-worker-40 spark40
    if errorlevel 1 (
        echo ERROR: Failed to start the Spark 4.0 services. Please check Docker.
        pause
        exit /b 1
    )
    echo Spark 4.0 services are running in the background.
    echo.
    echo Waiting for Spark 4.0 master to be ready...
    timeout /t 5 /nobreak >nul
    
    REM Check if Spark master is actually running
    echo Checking Spark 4.0 master status...
    docker exec spark-master-40 jps | findstr Master
    if errorlevel 1 (
        echo WARNING: Spark 4.0 Master process not detected. Waiting longer...
        timeout /t 10 /nobreak >nul
        docker exec spark-master-40 jps | findstr Master
        if errorlevel 1 (
            echo ERROR: Spark 4.0 Master failed to start properly.
            docker logs spark-master-40
            pause
            exit /b 1
        )
    )
    echo Spark 4.0 Master is running!
    
    REM Test network connectivity
    echo Testing network connectivity...
    docker exec spark-4.0-env ping -c 1 spark-master-40
    if errorlevel 1 (
        echo ERROR: Cannot reach spark-master-40 from spark-4.0-env
        pause
        exit /b 1
    )
    echo Network connectivity OK!
    
    REM Restart worker to clear any old executors
    echo Restarting worker to clear old executors...
    docker restart spark-worker-40 >nul 2>&1
    
    REM Restart driver container to ensure no previous spark-shell is running
    echo Restarting Spark 4.0 driver container...
    docker restart spark-4.0-env >nul 2>&1
    
    timeout /t 5 /nobreak >nul
    
    echo ========================================
    echo [STEP B] Connecting to the Spark 4.0 Scala shell...
    echo ========================================
    
    REM Use docker exec for reliable connection
    docker exec -it spark-4.0-env spark-shell
    
    echo.
    echo ========================================
    echo [STEP C] Shutting down all Spark 4.0 services...
    echo ========================================
    docker compose -f "!COMPOSE_FILE!" down
) else if "%choice%"=="2" (
    echo Starting Spark 2.4...
    set "COMPOSE_FILE=docker\docker-compose.yml"
    if not exist "!COMPOSE_FILE!" (
        echo ERROR: !COMPOSE_FILE! not found!
        pause
        exit /b 1
    )
    echo Using compose file: !COMPOSE_FILE!
    echo.

    echo ========================================
    echo [STEP A] Starting all Spark services...
    echo ========================================
    docker compose -f "!COMPOSE_FILE!" up -d
    if errorlevel 1 (
        echo ERROR: Failed to start the Spark services. Please check Docker.
        pause
        exit /b 1
    )
    echo Spark services are running in the background.
    echo.
    echo Waiting for Spark master to be ready...
    timeout /t 5 /nobreak >nul
    
    REM Check if Spark master is actually running
    echo Checking Spark master status...
    docker exec spark-master jps | findstr Master
    if errorlevel 1 (
        echo WARNING: Spark Master process not detected. Waiting longer...
        timeout /t 10 /nobreak >nul
        docker exec spark-master jps | findstr Master
        if errorlevel 1 (
            echo ERROR: Spark Master failed to start properly.
            docker logs spark-master
            pause
            exit /b 1
        )
    )
    echo Spark Master is running!
    
    REM Test network connectivity
    echo Testing network connectivity...
    docker exec spark-2.4-env ping -c 1 spark-master
    if errorlevel 1 (
        echo ERROR: Cannot reach spark-master from spark-2.4-env
        pause
        exit /b 1
    )
    echo Network connectivity OK!
    
    REM Restart worker to clear any old executors
    echo Restarting worker to clear old executors...
    docker restart spark-worker >nul 2>&1
    
    REM Restart driver container to ensure no previous spark-shell is running
    echo Restarting Spark 2.4 driver container...
    docker restart spark-2.4-env >nul 2>&1
    
    timeout /t 5 /nobreak >nul
    
    echo ========================================
    echo [STEP B] Connecting to the Spark 2.4 Scala shell...
    echo ========================================
    
    REM Use docker exec for reliable connection
    docker exec -it spark-2.4-env spark-shell
    
    echo.
    echo ========================================
    echo [STEP C] Shutting down all Spark services...
    echo ========================================
    docker compose -f "!COMPOSE_FILE!" down

) else (
    echo ERROR: Invalid choice '%choice%'. Please enter 1 or 2.
    pause
    exit /b 1
)

echo.
echo Spark session ended.
echo.
pause 