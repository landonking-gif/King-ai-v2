@echo off
TITLE King AI v2 - Control Center
CLS

echo Launching Imperial Control Center...

:: Try 'py' launcher (standard for Windows)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    py scripts/control.py
    pause
    exit /b
)

:: Try 'python'
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python scripts/control.py
    pause
    exit /b
)

:: Try 'python3'
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    python3 scripts/control.py
    pause
    exit /b
)

echo.
echo [ERROR] Python not found!
echo Please install Python from https://www.python.org/downloads/
echo.
pause
