@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment .venv not found. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "src\main_gui_final.py"
if errorlevel 1 pause
