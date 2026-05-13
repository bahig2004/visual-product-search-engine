@echo off
setlocal

cd /d "%~dp0"

echo =======================================
echo Product RAG Search Engine
echo =======================================

set "PY_CMD="
for %%P in ("py -3.13" "py -3.12" "py -3.11" "py -3.10" "py -3" "py" "python") do (
    if not defined PY_CMD (
        call :try_python %%~P
    )
)

if not defined PY_CMD (
    echo [ERROR] Could not find a usable Python with pip.
    echo Install Python 3.10+ and ensure pip is available.
    pause
    exit /b 1
)

echo Using Python command: %PY_CMD%
echo Installing dependencies for RAG app...
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)

echo Starting search backend at http://127.0.0.1:5000
start "" "http://127.0.0.1:5000/"
%PY_CMD% -m app.main

echo.
echo RAG backend stopped.
pause
exit /b 0

:try_python
set "CANDIDATE=%~1"
%CANDIDATE% --version >nul 2>&1
if errorlevel 1 exit /b 0

%CANDIDATE% -m pip --version >nul 2>&1
if errorlevel 1 (
    %CANDIDATE% -m ensurepip --upgrade >nul 2>&1
    %CANDIDATE% -m pip --version >nul 2>&1
    if errorlevel 1 exit /b 0
)

set "PY_CMD=%CANDIDATE%"
exit /b 0
