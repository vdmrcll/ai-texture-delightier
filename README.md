# AI-Powered Texture De-Lighter

AI-powered texture de-lighting for **photogrammetry and scanned assets**.

This tool removes environmental lighting from 3D scanned assets, producing a cleaner approximation of the object's true surface color (albedo).

![Viewport Preview](https://github.com/vdmrcll/ai-texture-delightier/blob/main/preview.jpg)

> **Status:** Early Beta — feedback and bug reports are very welcome.

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

* All input maps should have matching dimensions and have 1:1 A/R.
* Use **sRGB** color space for the texture inputs.
* Use **small padding** around the texture/UV islands.
* The normal map must be **Object Space / DirectX**.

## Installation & Running

1. Install **Python 3.8+** and make sure it is available in your system `PATH`.
2. Download the app from the releases.
3. Run the installer/start script depending on your OS.

on Windows:

```text
run.bat
```

or Linux:

```text
bash run.sh
```

The launcher will:

1. Create/configure a Python virtual environment.
2. Install the required dependencies.
3. Configure the ONNX Runtime environment.
4. Launch the application.

If CUDA is unavailable, the application will fall back to CPU execution.

Thanks for testing and helping improve the project!
