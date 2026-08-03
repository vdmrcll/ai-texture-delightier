# AI-Powered Texture De-Lighter

AI-powered texture de-lighting for **photogrammetry and scanned assets**.

This tool removes environmental and baked-in lighting from texture maps, producing a cleaner approximation of the object's underlying surface color (albedo).

![Viewport Preview](https://github.com/vdmrcll/ai-texture-delightier/blob/main/preview.jpg)

> **Status:** Early Beta — feedback and bug reports are very welcome.

## Features

* AI-based texture de-lighting
* Designed for photogrammetry and scanned assets
* Supports **FP16 and FP32** ONNX models
* NVIDIA CUDA acceleration with CPU fallback
* Automatic 2048×2048 texture tiling
* Runs locally using ONNX Runtime

## Requirements

### Operating System

* Windows 10/11 (64-bit)
* Python 3.8 or newer
* Python must be installed and available in the system `PATH`

### GPU

An NVIDIA CUDA-capable GPU is **highly recommended**.

* **Minimum:** 6 GB VRAM (e.g. GTX 1060)
* GPU execution targets **CUDA 11.8**
* Bundled GPU setup targets **cuDNN 8.9.5**
* CPU execution is supported as a fallback, but will be significantly slower

The launcher automatically installs the required CUDA/cuDNN/cuFFT runtime DLLs into the virtual environment.

## Input Maps

The network expects **four input maps**:

| Map                     | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| **Lit RGB Texture**     | Original texture containing baked-in environmental lighting |
| **Object-Space Normal** | Object-space normal map in **DirectX** format               |
| **Ambient Occlusion**   | Ambient occlusion map                                       |
| **Alpha Mask**          | Alpha/transparency mask                                     |

### Important

* All input maps should have matching dimensions.
* Use **sRGB** color space for the texture inputs.
* Use **small padding** around the texture/UV islands.
* The normal map must be **Object Space / DirectX**.

## Texture Resolution & Quality

The model was trained and optimized specifically for **2048×2048 textures**.

### Recommended

**2048×2048**

2048×2048 maps are processed as four 1024×1024 tiles, matching the network's training input size.

### Other resolutions

* **1024×1024:** Not recommended and may produce artifacts.
* **4096×4096 and larger:** Not recommended. Larger textures require substantially more VRAM and may exceed the available memory on typical GPUs.
* Processing 4K textures may require approximately **24 GB of VRAM**, depending on the configuration.

For the most predictable results, use **2048×2048 textures**.

> **Note:** Input bit depth can affect the result. Processing 8-bit or 16-bit images may produce different results.

## Models

The application supports two ONNX models:

```text
weights/
├── delighter_model_fp16.onnx
└── delighter_model_fp32.onnx
```

The GUI defaults to **FP16** and allows switching between the FP16 and FP32 models.

For packaged builds, the `weights` directory should be located beside `Delighter.exe`:

```text
Delighter/
├── Delighter.exe
└── weights/
    ├── delighter_model_fp16.onnx
    └── delighter_model_fp32.onnx
```

## Installation & Running

1. Install **Python 3.8+** and make sure it is available in your system `PATH`.
2. Clone or download this repository.
3. Make sure both ONNX model files are present in the `weights` directory.
4. Double-click:

```text
run.bat
```

The launcher will:

1. Create/configure a Python virtual environment.
2. Install the required dependencies.
3. Configure the ONNX Runtime environment.
4. Launch the application.

If CUDA is unavailable, the application will fall back to CPU execution.

## GPU / CPU Execution

The application uses **ONNX Runtime** to execute the included ONNX models.

The launcher targets:

* ONNX Runtime **1.19.2**
* NVIDIA CUDA **11.8**
* cuDNN **8.9.5**

CUDA execution is recommended for practical processing times. CPU mode is supported primarily as a fallback.

## Preview

![Viewport Preview](https://github.com/vdmrcll/ai-texture-delightier/blob/main/preview.jpg)

## Troubleshooting

### CUDA is not available

If the application falls back to CPU:

* Make sure you have an NVIDIA CUDA-capable GPU.
* Make sure your NVIDIA drivers are up to date.
* Try running `run.bat` again so the required runtime dependencies can be configured.
* Check the **Execution Log** window for errors.

### Output contains visual artifacts

Make sure:

* All four input maps are provided.
* The normal map is **Object Space / DirectX**.
* The texture is processed in **sRGB**.
* The input resolution is **2048×2048**.
* The input maps have matching dimensions.
* The source texture contains reasonable padding around UV islands.

## Beta Testing & Feedback

This project is currently in an **early beta** and your feedback is extremely valuable.

If you encounter crashes, errors, unexpected results, or visual artifacts, please send an email containing:

1. The text from the **Execution Log** window, if applicable.
2. A description and/or screenshot of the problem.
3. The format and resolution of your source textures.
4. Your GPU model, if the issue is related to performance or CUDA.

**Email:** [vdmrcll@gmail.com](mailto:vdmrcll@gmail.com)
**Subject:** `De-light beta-test feedback`

Thanks for testing and helping improve the project!
