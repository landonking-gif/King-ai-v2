@echo off
REM Ralph: Autonomous AI Agent Loop (Windows Batch Wrapper)
REM Calls the Python implementation for cross-platform compatibility

echo Starting Ralph Autonomous Agent Loop
echo Working directory: %CD%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if GitHub Copilot CLI is available
copilot --version >nul 2>&1
if errorlevel 1 (
    echo ❌ GitHub Copilot CLI is not installed or not in PATH
    echo Please install from https://github.com/github/copilot-cli
    pause
    exit /b 1
)

REM Check if prd.json exists
if not exist "prd.json" (
    echo ❌ prd.json not found in current directory
    echo Please run this script from the project root
    pause
    exit /b 1
)

REM Install required dependencies if missing
echo Checking dependencies...
python -c "import httpx" >nul 2>&1
if errorlevel 1 (
    echo Installing httpx...
    python -m pip install httpx
    if errorlevel 1 (
        echo Failed to install httpx
        pause
        exit /b 1
    )
)

REM Run the Python Ralph loop
echo Launching Ralph loop...
if "%~1"=="" (
    python scripts/ralph/ralph.py
) else (
    python scripts/ralph/ralph.py %1
)

pause