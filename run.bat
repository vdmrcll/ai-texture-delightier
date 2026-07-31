@echo off
TITLE AI Texture De-Lighter Launcher
cd /d "%~dp0"

echo [1/3] Checking Python environment...

REM Check if system Python version is 3.10, 3.11, or 3.12
py -3.12 -V >nul 2>&1
if %errorlevel%==0 (
    set PY_CMD=py -3.12
) else (
    set PY_CMD=python
)

if not exist "venv" (
    echo Creating virtual environment...
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo Error: Could not create virtual environment. Ensure Python 3.10-3.12 is installed.
        pause
        exit /b 1
    )
)

echo [2/3] Checking dependencies...
call venv\Scripts\activate.bat

REM Verify both UI packages AND CUDA PyTorch are ready
python -c "import customtkinter, tkinterdnd2, PIL, torch; assert torch.cuda.is_available()" 2>nul
if %errorlevel%==0 goto LAUNCH

echo Missing dependencies or CUDA PyTorch not detected. Installing...
call python -m pip install --upgrade pip
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
call pip install -r requirements.txt

:LAUNCH
echo [3/3] Launching application...
python gui.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
)
pause