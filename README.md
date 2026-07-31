AI POWERED TEXTURE DE-LIGHTER (v0.1 Beta)

Description:
AI based texture de-lighter as the title suggests. Its meant to remove environmental lighting from Photogrammetry assets.

Hardware Requirements:
Windows 10/11 (64-bit)
Python: Version 3.8 or higher installed and added to system PATH.
GPU: NVIDIA CUDA-capable GPU highly recommended. 
  - Minimum: 6GB+ VRAM (e.g. GTX 1080 Ti).
  - CPU Mode: Supported as a fallback, but processing will be significantly slower.

Input Requirements & Network Expectations:
The model processes four required input maps to accurately separate lighting from surface color:
- Lit RGB Texture (The original texture map with baked-in environmental lighting)
- Object space Normal Map (DirectX)
- Ambient Occlusion Map
- Alpha Mask

Use small padding and sRGB color space.
Computing on 8-bit or 16-bit images will result in a different outcome.

What to expect:
Target Resolution: The model is strictly trained and optimized for 2048x2048 textures, Loading maps with other resolutions (such as 1K or 4K+) is not recommended. 1K will probably artifact, 4k needs about 24GB of VRAM.

How to run:
1. Ensure your model weights file ("delighter_model_jit.pt") is placed in the main directory.
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