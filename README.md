AI POWERED TEXTURE DE-LIGHTER (v0.1 Beta)

The application uses ONNX Runtime to run the included ONNX models.
The launcher targets ONNX Runtime 1.19.2 for NVIDIA CUDA 11.8 and
falls back to CPU execution when CUDA is unavailable.

Description:
AI based texture de-lighter as the title suggests. Its meant to remove environmental lighting from Photogrammetry assets.

Hardware Requirements:
Windows 10/11 (64-bit)
Python: Version 3.8 or higher installed and added to system PATH.
GPU: NVIDIA CUDA-capable GPU highly recommended.
  - The bundled GPU setup targets CUDA 11.8 and cuDNN 8.x.
  - The launcher installs the matching CUDA 11.8/cuDNN 8.9.5/cuFFT runtime DLLs into the virtual environment.
  - The GUI precision menu defaults to FP16 and can switch between `delighter_model_fp16.onnx` and `delighter_model_fp32.onnx`.
  - 2048x2048 maps are processed as four 1024x1024 tiles, matching the network's training input size.
  - Minimum: 6GB VRAM (e.g. GTX 1060 bare minimum!!!).
  - CPU Mode: Supported as a fallback, but processing will be significantly slower.

Input Requirements & Network Expectations:
The model processes four required input maps to accurately separate lighting from surface color:
- Lit RGB Texture (The original texture map with baked-in environmental lighting)
- Object space Normal Map (DirectX)
- Ambient Occlusion Map
- Alpha Mask

Use small padding and sRGB color space.

What to expect:
Target Resolution: The model is strictly trained and optimized for 2048x2048 textures, Loading maps with other resolutions (such as 1K or 4K+) is not recommended. 1K will probably artifact, 4k needs about 24GB of VRAM. Computing on 8-bit or 16-bit images will result in a different outcome.

How to run:
1. Ensure both model files (`delighter_model_fp16.onnx` and `delighter_model_fp32.onnx`) are placed in the `weights` directory.
   For packaged builds, keep the `weights` directory beside `Delighter.exe`.
2. Double-click the "run.bat" script. 
3. The script will automatically configure a virtual environment, install the necessary dependencies, and launch the application.

NOTE FOR TESTERS:
Since this is an early beta test, your feedback is crucial! 
If you run into any crashes, errors, or weird visual artifacts, please send an email with:
1. A copy of the text from the Execution Log window (if an error occurred).
2. A description or screenshot of the output.
3. The format and resolution of your source textures.

Send feedback to: vdmrcll@gmail.com with the subject: "De-light beta-test feedback"

Thanks everyone!
