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

REM Verify required packages locally. The CPU package is a bootstrap; it is
REM replaced with the CUDA build below when an NVIDIA GPU is available.
python -c "import customtkinter, tkinterdnd2, PIL, onnxruntime, cv2" 2>nul
if %errorlevel%==0 goto CHECK_RUNTIME

echo Missing dependencies. Installing...
call pip install -r requirements.txt
if errorlevel 1 exit /b 1

:CHECK_RUNTIME
REM Configure CUDA automatically when an NVIDIA driver is available.
REM Set DELIGHTER_FORCE_CPU=1 to explicitly use CPU mode.
nvidia-smi >nul 2>&1
if errorlevel 1 goto NO_NVIDIA
if /i "%DELIGHTER_FORCE_CPU%"=="1" goto CPU_MODE
python -c "import onnxruntime as ort; raise SystemExit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" >nul 2>&1
if errorlevel 1 (
    echo NVIDIA driver detected. Installing ONNX Runtime GPU for CUDA 11.8...
    call pip uninstall -y onnxruntime onnxruntime-gpu
    call pip install onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
    echo Installing CUDA 11.8 and cuDNN 8 runtime libraries...
    call pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cufft-cu11==10.9.0.58 nvidia-cudnn-cu11==8.9.5.29
    if errorlevel 1 echo WARNING: CUDA runtime installation failed; the application will use CPU mode.
) else (
    echo CUDA provider already available; skipping GPU package installation.
)
goto LAUNCH

:CPU_MODE
echo CPU mode forced by DELIGHTER_FORCE_CPU.
goto LAUNCH

:NO_NVIDIA
echo No NVIDIA driver detected; using CPU mode.

:LAUNCH
echo [3/3] Checking model weights...
python download_weights.py
if errorlevel 1 (
    echo Error: Could not download or verify the model weights.
    pause
    exit /b 1
)

echo Launching application...
python gui.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
)
pause
