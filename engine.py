import cv2
import numpy as np
import torch

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-7, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-7, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)

class DelighterInferenceEngine:
    def __init__(self, model_path: str = "delighter_model_jit.pt"):
        print("\n" + "="*50)
        print("[Engine Init] Initializing PyTorch CUDA Engine...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.device_info = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.device_info = "CPU Mode"

        print(f"[Engine Init] Active Hardware Device: {self.device_info}")
        
        # Load compiled TorchScript model directly to GPU
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("="*50 + "\n")

    def _load_image(self, path: str, channels: int = 3) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at: {path}")

        scale = 65535.0 if img.dtype == np.uint16 else 255.0

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
        return torch.from_numpy(img_np).permute(2, 0, 1)

    @torch.no_grad()
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
        lit_t = self._load_image(lit_path, channels=3).to(self.device)
        normal_t = self._load_image(normal_path, channels=3).to(self.device)
        ao_t = self._load_image(ao_path, channels=1).to(self.device)
        mask_t = self._load_image(mask_path, channels=1).to(self.device)

        _, h, w = lit_t.shape
        log(f"Resolution: {w}x{h} px | Target Device: {self.device_info}")

        # 1. Color Space Conversion (sRGB -> Linear)
        log("Converting Lit RGB from sRGB to Linear space...")
        lit_linear = srgb_to_linear(lit_t)

        # 2. Geometry Stacking: Normals(3) + AO(1) + Mask(1) = 5 Channels
        log("Stacking 5-Channel Geometry Maps...")
        geometry_t = torch.cat([normal_t, ao_t, mask_t], dim=0)

        # 3. Add Batch Dimension -> [1, C, H, W]
        lit_input = lit_linear.unsqueeze(0)
        geo_input = geometry_t.unsqueeze(0)

        # 4. Forward Pass on GPU
        log("Executing Neural Network forward pass on GPU...")
        pred_albedo_linear = self.model(lit_input, geo_input).squeeze(0)

        # 5. Convert Linear -> sRGB & Apply UV Mask
        log("Converting predicted Albedo to sRGB space...")
        pred_srgb = linear_to_srgb(pred_albedo_linear)
        pred_srgb = pred_srgb * mask_t

        # 6. Save output map to disk
        log("Saving final 8-bit image to disk...")
        pred_np = pred_srgb.permute(1, 2, 0).cpu().numpy()
        pred_8bit = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_bgr = cv2.cvtColor(pred_8bit, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_save_path, output_bgr)
        log(f"SUCCESS: Saved output to '{output_save_path}'")

        return output_save_path, (w, h), self.device_info