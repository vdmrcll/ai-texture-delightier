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

REM Verify required packages. CUDA is optional; the engine supports CPU mode.
python -c "import customtkinter, tkinterdnd2, PIL, onnxruntime" 2>nul
if %errorlevel%==0 goto CHECK_RUNTIME

echo Missing dependencies. Installing...
call python -m pip install --upgrade pip
call pip install -r requirements.txt

:CHECK_RUNTIME
REM Prefer the CUDA wheel when NVIDIA's command-line driver is available.
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    python -c "from importlib.metadata import version; import onnxruntime as ort; assert version('onnxruntime-gpu').startswith('1.19.') and version('nvidia-cuda-runtime-cu11') == '11.8.89' and version('nvidia-cuda-nvrtc-cu11') == '11.8.89' and version('nvidia-cublas-cu11') == '11.11.3.6' and version('nvidia-cufft-cu11') == '10.9.0.58' and version('nvidia-cudnn-cu11') == '8.9.5.29' and 'CUDAExecutionProvider' in ort.get_available_providers()" >nul 2>&1
    if errorlevel 1 (
        echo NVIDIA driver detected. Installing ONNX Runtime GPU for CUDA 11.8...
        call pip uninstall -y onnxruntime onnxruntime-gpu
        call pip install onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
        echo Installing CUDA 11.8 and cuDNN 8 runtime libraries...
        call pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cufft-cu11==10.9.0.58 nvidia-cudnn-cu11==8.9.5.29
        if errorlevel 1 echo WARNING: CUDA runtime installation failed; the application will use CPU mode.
    )
)

:LAUNCH
echo [3/3] Launching application...
python gui.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
)
pause
