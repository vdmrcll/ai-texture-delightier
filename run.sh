#!/usr/bin/env bash
set -u

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${DELIGHTER_VENV:-venv}"
CUDA11_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. Install Python 3.10-3.12 or set PYTHON_BIN."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" || exit 1
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if "$VENV_PYTHON" -c 'import customtkinter, tkinterdnd2, PIL, onnxruntime, cv2' >/dev/null 2>&1; then
    echo "Base dependencies already installed; skipping package installation."
else
    echo "Missing base dependencies; installing..."
    "$VENV_PYTHON" -m pip install -r requirements.txt || exit 1
fi

if [ "${DELIGHTER_FORCE_CPU:-}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Configuring ONNX Runtime for CUDA 11.8..."
    if ! "$VENV_PYTHON" -c 'import onnxruntime as ort; raise SystemExit(0 if "CUDAExecutionProvider" in ort.get_available_providers() else 1)' >/dev/null 2>&1; then
        "$VENV_PYTHON" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
        "$VENV_PYTHON" -m pip install \
            onnxruntime-gpu==1.19.2 \
            --index-url "$CUDA11_INDEX" \
            --extra-index-url https://pypi.org/simple || exit 1
        "$VENV_PYTHON" -m pip install \
            nvidia-cuda-runtime-cu11==11.8.89 \
            nvidia-cuda-nvrtc-cu11==11.8.89 \
            nvidia-cublas-cu11==11.11.3.6 \
            nvidia-cufft-cu11==10.9.0.58 \
            nvidia-cudnn-cu11==8.9.5.29 || {
                echo "Warning: CUDA runtime libraries could not be installed; CPU mode will be used."
            }
    fi
elif [ "${DELIGHTER_FORCE_CPU:-}" = "1" ]; then
    echo "CPU mode forced by DELIGHTER_FORCE_CPU."
else
    echo "No NVIDIA driver detected; using CPU execution."
fi

echo "Checking model weights..."
"$VENV_PYTHON" download_weights.py || exit 1

echo "Launching application..."
exec "$VENV_PYTHON" gui.py
