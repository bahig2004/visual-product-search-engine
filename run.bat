
@echo off
setlocal

REM Move to project root (folder where this script exists)
cd /d "%~dp0"

echo ==========================================
echo Visual Product Search Engine - Quick Start
echo ==========================================

REM Pick a working Python launcher/interpreter
set "PY_CMD=py -3.13"
%PY_CMD% --version >nul 2>&1
if errorlevel 1 (
    set "PY_CMD=py"
    %PY_CMD% --version >nul 2>&1
)
if errorlevel 1 (
    echo [ERROR] Python launcher not found.
    echo Please install Python, then run this file again.
    pause
    exit /b 1
)

echo Using Python command: %PY_CMD%

echo.
echo Installing/checking dependencies...
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting backend server on http://127.0.0.1:5000
echo Opening frontend page in browser...
start "" "http://127.0.0.1:5000/"

REM Run backend in this same window
%PY_CMD% backend_api.py

echo.
echo Backend stopped.
pause

