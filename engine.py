import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import cv2
import numpy as np


# ONNX Runtime 1.19 does not automatically preload CUDA libraries from
# NVIDIA's Python packages. Configure both Windows DLL search and Linux
# shared-library search before importing ONNX Runtime.
_CUDA_DLL_HANDLES = []
if os.name == "nt":
    _site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for _relative_path in (
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cublas/bin",
        "nvidia/cufft/bin",
        "nvidia/cudnn/bin",
    ):
        _dll_dir = _site_packages / _relative_path
        if _dll_dir.is_dir():
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
elif sys.platform.startswith("linux"):
    _site_packages = [Path(sysconfig.get_paths().get("purelib", ""))]
    try:
        _site_packages.extend(Path(path) for path in sysconfig.get_paths().get("platlib", "").split(os.pathsep) if path)
    except AttributeError:
        pass

    _cuda_roots = []
    for _cuda_env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(_cuda_env):
            _cuda_roots.append(Path(os.environ[_cuda_env]))
    _cuda_roots.extend((Path("/usr/local/cuda"), Path("/usr/local/cuda-11.8")))
    _cuda_library_dirs = []
    for _root in _cuda_roots:
        _cuda_library_dirs.extend((_root / "lib64", _root / "lib"))
    for _site_package in _site_packages:
        for _relative_path in (
            "nvidia/cuda_runtime/lib",
            "nvidia/cuda_nvrtc/lib",
            "nvidia/cublas/lib",
            "nvidia/cufft/lib",
            "nvidia/cudnn/lib",
        ):
            _cuda_library_dirs.append(_site_package / _relative_path)
    _existing_cuda_dirs = [str(path) for path in _cuda_library_dirs if path.is_dir()]
    if _existing_cuda_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            _existing_cuda_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]
        ).strip(os.pathsep)

import onnxruntime as ort


def _hardware_name_for_provider(provider: str) -> str:
    """Return a useful hardware name without making the app platform-specific."""
    if provider == "CPUExecutionProvider":
        return _cpu_model_name()

    if provider == "CUDAExecutionProvider":
        # ONNX Runtime exposes the CUDA provider, but not the GPU model name.
        # nvidia-smi is available with standard NVIDIA driver installations.
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                gpu_name = result.stdout.strip().splitlines()[0]
                if gpu_name:
                    return gpu_name
            except (OSError, IndexError, subprocess.SubprocessError):
                pass
        return "NVIDIA GPU"

    # Keep unknown providers generic so a different ONNX Runtime build is not
    # presented as supported before the provider is explicitly wired in.
    return provider.replace("ExecutionProvider", "") + " device"


def _cpu_model_name() -> str:
    """Get a friendly CPU model name across the platforms supported by Python."""
    commands = []
    if platform.system() == "Windows":
        commands = [
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' -Name ProcessorNameString).ProcessorNameString",
            ],
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ],
            ["wmic", "cpu", "get", "name", "/value"],
        ]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
            for line in result.stdout.splitlines():
                name = line.strip()
                if name and not name.lower().startswith("name="):
                    return name
        except (OSError, subprocess.SubprocessError):
            pass

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.lower().startswith(("model name", "hardware")) and ":" in line:
                        name = line.split(":", 1)[1].strip()
                        if name:
                            return name
        except OSError:
            pass

    return platform.processor() or "CPU"

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-7, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-7, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

class DelighterInferenceEngine:
    TILE_SIZE = 1024

    def __init__(self, model_path: str = "weights/delighter_model_fp16.onnx"):
        print("\n" + "="*50)
        print("[Engine Init] Initializing ONNX Runtime engine...")
        
        available_providers = ort.get_available_providers()
        force_cpu = os.environ.get("DELIGHTER_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        if force_cpu:
            providers = ["CPUExecutionProvider"]
            print("[Engine Init] CPU fallback forced by DELIGHTER_FORCE_CPU")
        elif "CUDAExecutionProvider" in available_providers:
            # EXHAUSTIVE is ONNX Runtime's default cuDNN convolution search
            # and can take an extremely long time on older GPUs such as the
            # GTX 1080 Ti. HEURISTIC keeps the first inference responsive.
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Resolve relative model paths from the application directory, not the
        # caller's current working directory.
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            # In a PyInstaller build, keep external assets beside the .exe
            # instead of looking inside the bundled _internal directory.
            if getattr(sys, "frozen", False):
                app_dir = Path(sys.executable).resolve().parent
            else:
                app_dir = Path(__file__).resolve().parent
            resolved_model_path = app_dir / resolved_model_path
            if not resolved_model_path.is_file():
                resolved_model_path = app_dir / "weights" / Path(model_path).name
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

        try:
            self.model = ort.InferenceSession(str(resolved_model_path), providers=providers)
        except Exception as provider_error:
            if providers and providers[0] != "CPUExecutionProvider" and not force_cpu:
                failed_provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
                print(f"[Engine Init] {failed_provider} initialization failed; falling back to CPU: {provider_error}")
                providers = ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(str(resolved_model_path), providers=providers)
            else:
                raise
        active_providers = self.model.get_providers()
        self.execution_provider = next(
            (provider for provider in active_providers if provider != "CPUExecutionProvider"),
            "CPUExecutionProvider",
        )
        self.hardware_device = _hardware_name_for_provider(self.execution_provider)
        self.runtime_name = "ONNX Runtime"
        self.provider_name = {
            "CUDAExecutionProvider": "CUDA",
        }.get(self.execution_provider, "CPU")
        self.is_accelerated = self.execution_provider != "CPUExecutionProvider"
        self.device_info = f"{self.hardware_device} via {self.runtime_name} / {self.provider_name}"
        self.device_display = f"{self.runtime_name} · {self.provider_name}\n{self.hardware_device}"
        print(f"[Engine Init] Active Hardware Device: {self.device_info}")
        self.input_names = [item.name for item in self.model.get_inputs()]
        self.output_name = self.model.get_outputs()[0].name
        if len(self.input_names) != 2:
            raise ValueError(f"Expected 2 ONNX inputs, found {len(self.input_names)}: {self.input_names}")
        print(f"[Engine Init] ONNX inputs: {self.input_names}")
        print("="*50 + "\n")

    def _load_image(self, path: str, channels: int = 3) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at: {path}")

        if np.issubdtype(img.dtype, np.floating):
            # OpenCV returns EXR/HDR data as float32. These formats are
            # normally already normalized to 0..1 for this model.
            scale = 1.0
        elif img.dtype == np.uint16:
            scale = 65535.0
        elif img.dtype == np.uint8:
            scale = 255.0
        else:
            raise ValueError(f"Unsupported image data type for '{path}': {img.dtype}")

        if channels == 1:
            if len(img.shape) == 3:
                img = img[:, :, 0:1]
            elif len(img.shape) == 2:
                img = img[:, :, np.newaxis]
        elif channels == 3:
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

        img_np = img.astype(np.float32) / scale
        if not np.isfinite(img_np).all():
            raise ValueError(f"Image contains NaN or infinity values: {path}")
        img_np = np.clip(img_np, 0.0, 1.0)
        return np.transpose(img_np, (2, 0, 1)).astype(np.float32)

    def process_texture(
        self, 
        lit_path: str, 
        normal_path: str, 
        ao_path: str, 
        mask_path: str, 
        output_save_path: str,
        log_callback=None
    ):
        def log(msg):
            print(f"[Inference] {msg}")
            if log_callback:
                log_callback(msg)

        log("Loading texture maps onto GPU memory...")
        lit_t = self._load_image(lit_path, channels=3)
        normal_t = self._load_image(normal_path, channels=3)
        ao_t = self._load_image(ao_path, channels=1)
        mask_t = self._load_image(mask_path, channels=1)

        _, h, w = lit_t.shape
        inputs = {
            "normal map": normal_t,
            "ambient occlusion map": ao_t,
            "mask": mask_t,
        }
        for name, tensor in inputs.items():
            if tensor.shape[1:] != (h, w):
                raise ValueError(
                    f"{name.capitalize()} resolution {tensor.shape[2]}x{tensor.shape[1]} "
                    f"does not match lit map resolution {w}x{h}"
                )
        if (h, w) != (2048, 2048):
            log(f"WARNING: Model is trained for 2048x2048; received {w}x{h}.")
        log(f"Resolution: {w}x{h} px | Target Device: {self.device_info}")

        # 1. Color Space Conversion (sRGB -> Linear)
        log("Converting Lit RGB from sRGB to Linear space...")
        lit_linear = srgb_to_linear(lit_t)

        # 2. Geometry Stacking: Normals(3) + AO(1) + Mask(1) = 5 Channels
        log("Stacking 5-Channel Geometry Maps...")
        geometry_t = np.concatenate([normal_t, ao_t, mask_t], axis=0)

        # 3. Run the network on the 1024x1024 tiles it was trained on.
        # Edge tiles are padded and cropped back after inference.
        tile_size = self.TILE_SIZE
        tiles_y = (h + tile_size - 1) // tile_size
        tiles_x = (w + tile_size - 1) // tile_size
        total_tiles = tiles_y * tiles_x
        log(f"Processing {total_tiles} x {tile_size}x{tile_size} tiles...")
        pred_albedo_linear = np.empty((3, h, w), dtype=np.float32)
        tile_number = 0

        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                tile_number += 1
                tile_h = min(tile_size, h - y0)
                tile_w = min(tile_size, w - x0)
                lit_tile = lit_linear[:, y0:y0 + tile_h, x0:x0 + tile_w]
                geo_tile = geometry_t[:, y0:y0 + tile_h, x0:x0 + tile_w]

                pad_h = tile_size - tile_h
                pad_w = tile_size - tile_w
                if pad_h or pad_w:
                    lit_tile = np.pad(lit_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                    geo_tile = np.pad(geo_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

                log(f"Executing tile {tile_number}/{total_tiles}...")
                outputs = self.model.run(
                    [self.output_name],
                    {
                        self.input_names[0]: np.expand_dims(lit_tile, axis=0).astype(np.float32),
                        self.input_names[1]: np.expand_dims(geo_tile, axis=0).astype(np.float32),
                    },
                )
                tile_prediction = outputs[0][0, :, :tile_h, :tile_w]
                pred_albedo_linear[:, y0:y0 + tile_h, x0:x0 + tile_w] = tile_prediction

        # 5. Convert Linear -> sRGB & Apply UV Mask
        log("Converting predicted Albedo to sRGB space...")
        pred_srgb = linear_to_srgb(pred_albedo_linear)
        pred_srgb = pred_srgb * mask_t

        # 6. Save output map to disk
        log("Saving final 8-bit image to disk...")
        pred_np = np.transpose(pred_srgb, (1, 2, 0))
        pred_8bit = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_bgr = cv2.cvtColor(pred_8bit, cv2.COLOR_RGB2BGR)

        output_path = Path(output_save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output_bgr):
            raise IOError(f"Failed to write output image: {output_path}")
        log(f"SUCCESS: Saved output to '{output_path}'")

        return str(output_path), (w, h), self.device_info
