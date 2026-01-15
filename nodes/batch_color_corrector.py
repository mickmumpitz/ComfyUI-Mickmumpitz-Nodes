"""
Batch Color Corrector Node for ComfyUI
Based on ComfyUI-EasyColorCorrector by regiellis
https://github.com/regiellis/ComfyUI-EasyColorCorrector

This node processes video frame sequences with AI-powered color corrections.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Check for optional advanced libraries
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image tensor to HSV. Expects input shape [B, H, W, C] with values 0-1."""
    cmax, cmax_indices = torch.max(rgb, dim=-1)
    cmin = torch.min(rgb, dim=-1)[0]
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    h[cmax_indices == 0] = (((rgb[..., 1] - rgb[..., 2]) / (delta + 1e-8)) % 6)[
        cmax_indices == 0
    ]
    h[cmax_indices == 1] = (((rgb[..., 2] - rgb[..., 0]) / (delta + 1e-8)) + 2)[
        cmax_indices == 1
    ]
    h[cmax_indices == 2] = (((rgb[..., 0] - rgb[..., 1]) / (delta + 1e-8)) + 4)[
        cmax_indices == 2
    ]

    h = h / 6.0
    h[delta == 0] = 0.0

    s = torch.where(
        cmax == 0, torch.tensor(0.0, device=rgb.device), delta / (cmax + 1e-8)
    )
    v = cmax

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Converts an HSV image tensor to RGB. Expects input shape [B, H, W, C] with values 0-1."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).floor()
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = torch.zeros_like(hsv)

    mask0, mask1, mask2 = (i % 6) == 0, (i % 6) == 1, (i % 6) == 2
    mask3, mask4, mask5 = (i % 6) == 3, (i % 6) == 4, (i % 6) == 5

    rgb[mask0] = torch.stack([v, t, p], dim=-1)[mask0]
    rgb[mask1] = torch.stack([q, v, p], dim=-1)[mask1]
    rgb[mask2] = torch.stack([p, v, t], dim=-1)[mask2]
    rgb[mask3] = torch.stack([p, q, v], dim=-1)[mask3]
    rgb[mask4] = torch.stack([t, p, v], dim=-1)[mask4]
    rgb[mask5] = torch.stack([v, p, q], dim=-1)[mask5]

    rgb[s == 0] = torch.stack([v, v, v], dim=-1)[s == 0]

    return rgb


# Color correction presets
PRESETS = {
    "Natural Portrait": {
        "warmth": 0.08,
        "vibrancy": 0.12,
        "contrast": 0.08,
        "brightness": 0.03,
    },
    "Warm Portrait": {
        "warmth": 0.18,
        "vibrancy": 0.15,
        "contrast": 0.06,
        "brightness": 0.05,
    },
    "Cool Portrait": {
        "warmth": -0.12,
        "vibrancy": 0.08,
        "contrast": 0.10,
        "brightness": 0.02,
    },
    "High Key Portrait": {
        "warmth": 0.05,
        "vibrancy": 0.08,
        "contrast": -0.05,
        "brightness": 0.20,
    },
    "Dramatic Portrait": {
        "warmth": 0.02,
        "vibrancy": 0.20,
        "contrast": 0.25,
        "brightness": -0.05,
    },
    "Epic Fantasy": {
        "warmth": 0.1,
        "vibrancy": 0.4,
        "contrast": 0.3,
        "brightness": 0.05,
    },
    "Sci-Fi Chrome": {
        "warmth": -0.2,
        "vibrancy": 0.3,
        "contrast": 0.35,
        "brightness": 0.1,
    },
    "Dark Fantasy": {
        "warmth": -0.1,
        "vibrancy": 0.25,
        "contrast": 0.4,
        "brightness": -0.15,
    },
    "Vibrant Concept": {
        "warmth": 0.05,
        "vibrancy": 0.5,
        "contrast": 0.25,
        "brightness": 0.08,
    },
    "Matte Painting": {
        "warmth": 0.08,
        "vibrancy": 0.3,
        "contrast": 0.2,
        "brightness": 0.03,
    },
    "Digital Art": {
        "warmth": 0.0,
        "vibrancy": 0.45,
        "contrast": 0.28,
        "brightness": 0.05,
    },
    "Anime Bright": {
        "warmth": 0.12,
        "vibrancy": 0.45,
        "contrast": 0.2,
        "brightness": 0.12,
    },
    "Anime Moody": {
        "warmth": -0.05,
        "vibrancy": 0.35,
        "contrast": 0.25,
        "brightness": -0.05,
    },
    "Cyberpunk": {
        "warmth": -0.15,
        "vibrancy": 0.45,
        "contrast": 0.25,
        "brightness": -0.03,
    },
    "Pastel Dreams": {
        "warmth": 0.12,
        "vibrancy": -0.08,
        "contrast": -0.08,
        "brightness": 0.12,
    },
    "Neon Nights": {
        "warmth": -0.18,
        "vibrancy": 0.40,
        "contrast": 0.20,
        "brightness": -0.05,
    },
    "Comic Book": {
        "warmth": 0.05,
        "vibrancy": 0.5,
        "contrast": 0.35,
        "brightness": 0.08,
    },
    "Cinematic": {
        "warmth": 0.12,
        "vibrancy": 0.15,
        "contrast": 0.18,
        "brightness": 0.02,
    },
    "Teal & Orange": {
        "warmth": -0.08,
        "vibrancy": 0.25,
        "contrast": 0.15,
        "brightness": 0.0,
    },
    "Film Noir": {
        "warmth": -0.05,
        "vibrancy": -0.80,
        "contrast": 0.35,
        "brightness": -0.08,
    },
    "Vintage Film": {
        "warmth": 0.15,
        "vibrancy": -0.10,
        "contrast": 0.12,
        "brightness": 0.03,
    },
    "Bleach Bypass": {
        "warmth": -0.02,
        "vibrancy": -0.25,
        "contrast": 0.30,
        "brightness": 0.05,
    },
    "Golden Hour": {
        "warmth": 0.25,
        "vibrancy": 0.18,
        "contrast": 0.08,
        "brightness": 0.08,
    },
    "Blue Hour": {
        "warmth": -0.20,
        "vibrancy": 0.15,
        "contrast": 0.12,
        "brightness": 0.02,
    },
    "Sunny Day": {
        "warmth": 0.15,
        "vibrancy": 0.20,
        "contrast": 0.10,
        "brightness": 0.08,
    },
    "Overcast": {
        "warmth": -0.08,
        "vibrancy": 0.05,
        "contrast": 0.08,
        "brightness": 0.05,
    },
    "Sepia": {
        "warmth": 0.30,
        "vibrancy": -0.35,
        "contrast": 0.08,
        "brightness": 0.03,
    },
    "Black & White": {
        "warmth": 0.0,
        "vibrancy": -1.0,
        "contrast": 0.15,
        "brightness": 0.0,
    },
    "Faded": {
        "warmth": 0.05,
        "vibrancy": -0.15,
        "contrast": -0.12,
        "brightness": 0.08,
    },
    "Moody": {
        "warmth": -0.08,
        "vibrancy": 0.12,
        "contrast": 0.20,
        "brightness": -0.08,
    },
}


class BatchColorCorrector:
    """
    Batch Color Corrector node for processing video frame sequences.
    Processes multiple frames efficiently while maintaining consistency across the sequence.
    """

    PRESETS = PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["Auto", "Preset", "Manual"], {"default": "Auto"}),
                "frames_per_batch": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Batch Size Guide:\n1-4: Best Quality but Slow\n8-16: Balanced (recommended)\n32-64: Fastest but Resource Heavy"
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "GPU: Faster but uses VRAM. CPU: Slower but uses system RAM.",
                    },
                ),
            },
            "optional": {
                "ai_analysis": ("BOOLEAN", {"default": True}),
                "preset": (
                    [
                        "Natural",
                        "Warm",
                        "Cool",
                        "High Key",
                        "Dramatic",
                        "Epic Fantasy",
                        "Sci-Fi Chrome",
                        "Dark Fantasy",
                        "Vibrant Concept",
                        "Matte Painting",
                        "Digital Art",
                        "Anime Bright",
                        "Anime Moody",
                        "Cyberpunk",
                        "Pastel Dreams",
                        "Neon Nights",
                        "Comic Book",
                        "Cinematic",
                        "Teal & Orange",
                        "Film Noir",
                        "Vintage Film",
                        "Bleach Bypass",
                        "Golden Hour",
                        "Blue Hour",
                        "Sunny Day",
                        "Overcast",
                        "Sepia",
                        "Black & White",
                        "Faded",
                        "Moody",
                    ],
                    {"default": "Natural"},
                ),
                "effect_strength": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "enhancement_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.1},
                ),
                "adjust_for_skin_tone": ("BOOLEAN", {"default": True}),
                "white_balance_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "warmth": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "vibrancy": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "brightness": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "tint": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "lift": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "noise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "extract_palette": ("BOOLEAN", {"default": False}),
                "reference_image": ("IMAGE",),
                "reference_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = (
        "images",
        "palette_data",
        "histogram",
        "palette_image",
        "frame_count",
    )
    FUNCTION = "batch_color_correct"
    CATEGORY = "image/color"
    DESCRIPTION = "Process video frame sequences with AI-powered color corrections. Supports Auto, Preset, and Manual modes."
    OUTPUT_NODE = True

    def batch_color_correct(
        self,
        images,
        mode="Auto",
        frames_per_batch=16,
        use_gpu=True,
        ai_analysis=True,
        preset="Natural",
        effect_strength=0.4,
        enhancement_strength=0.8,
        adjust_for_skin_tone=True,
        white_balance_strength=0.6,
        warmth=0.0,
        vibrancy=0.0,
        brightness=0.0,
        contrast=0.0,
        tint=0.0,
        lift=0.0,
        gamma=0.0,
        gain=0.0,
        noise=0.0,
        extract_palette=False,
        reference_image=None,
        reference_strength=0.5,
        mask=None,
    ):
        """
        GPU-optimized batch processing for video frame sequences.
        """
        total_frames = images.shape[0]
        frame_height = images.shape[1]
        frame_width = images.shape[2]
        device = images.device

        # Handle GPU processing based on user choice
        if use_gpu and torch.cuda.is_available():
            if not str(device).startswith("cuda"):
                images = images.cuda()
                device = images.device
                if mask is not None:
                    mask = mask.cuda()
        elif use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA not available - falling back to CPU")

        print(f"Batch Color Corrector: Processing {total_frames} frames ({frame_width}x{frame_height}) on {device}")

        # Pre-allocate output tensor
        _, height, width, channels = images.shape
        final_images = torch.zeros((total_frames, height, width, channels), device=device, dtype=images.dtype)

        all_palette_data = []
        all_histograms = []
        all_palette_images = []
        processed_count = 0

        try:
            for batch_start in range(0, total_frames, frames_per_batch):
                # Check for interruption
                try:
                    import comfy.model_management as model_management
                    if model_management.interrupt_processing:
                        print("Batch processing interrupted by user")
                        if processed_count > 0:
                            return (
                                final_images[:processed_count].cpu(),
                                "",
                                torch.zeros((1, 512, 768, 3)),
                                torch.zeros((1, 120, 600, 3)),
                                processed_count,
                            )
                        else:
                            return (
                                images.cpu(),
                                "",
                                torch.zeros((1, 512, 768, 3)),
                                torch.zeros((1, 120, 600, 3)),
                                0,
                            )
                except:
                    pass

                batch_end = min(batch_start + frames_per_batch, total_frames)
                batch_frames = images[batch_start:batch_end]
                batch_masks = mask[batch_start:batch_end] if mask is not None else None

                # Process batch
                batch_processed = self._process_batch_gpu(
                    batch_frames=batch_frames,
                    batch_masks=batch_masks,
                    mode=mode,
                    ai_analysis=ai_analysis,
                    preset=preset,
                    effect_strength=effect_strength,
                    enhancement_strength=enhancement_strength,
                    adjust_for_skin_tone=adjust_for_skin_tone,
                    white_balance_strength=white_balance_strength,
                    warmth=warmth,
                    vibrancy=vibrancy,
                    brightness=brightness,
                    contrast=contrast,
                    tint=tint,
                    lift=lift,
                    gamma=gamma,
                    gain=gain,
                    noise=noise,
                    device=device,
                )

                final_images[batch_start:batch_end] = batch_processed
                processed_count = batch_end

                # Extract palette from middle frame
                if extract_palette and batch_start <= total_frames // 2 < batch_end:
                    all_palette_data.append("GPU_BATCH_MODE")
                    all_histograms.append(torch.zeros((1, 512, 768, 3), device=device))
                    all_palette_images.append(torch.zeros((1, 120, 600, 3), device=device))

        except KeyboardInterrupt:
            print("Batch processing interrupted")
            if processed_count > 0:
                return (
                    final_images[:processed_count].cpu(),
                    "",
                    torch.zeros((1, 512, 768, 3)),
                    torch.zeros((1, 120, 600, 3)),
                    processed_count,
                )
            else:
                return (
                    images.cpu(),
                    "",
                    torch.zeros((1, 512, 768, 3)),
                    torch.zeros((1, 120, 600, 3)),
                    0,
                )
        except Exception as e:
            print(f"Error during batch processing: {e}")
            if processed_count > 0:
                return (
                    final_images[:processed_count].cpu(),
                    "",
                    torch.zeros((1, 512, 768, 3)),
                    torch.zeros((1, 120, 600, 3)),
                    processed_count,
                )
            else:
                return (
                    images.cpu(),
                    "",
                    torch.zeros((1, 512, 768, 3)),
                    torch.zeros((1, 120, 600, 3)),
                    0,
                )

        if processed_count == total_frames:
            representative_palette = all_palette_data[0] if all_palette_data else ""
            representative_histogram = (
                all_histograms[0].cpu()
                if all_histograms
                else torch.zeros((1, 512, 768, 3))
            )
            representative_palette_img = (
                all_palette_images[0].cpu()
                if all_palette_images
                else torch.zeros((1, 120, 600, 3))
            )

            # Memory cleanup
            if use_gpu and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Batch processing complete: {total_frames} frames processed")

            return (
                final_images.cpu(),
                representative_palette,
                representative_histogram,
                representative_palette_img,
                total_frames,
            )
        else:
            return (
                images.cpu(),
                "",
                torch.zeros((1, 512, 768, 3)),
                torch.zeros((1, 120, 600, 3)),
                0,
            )

    def _process_batch_gpu(
        self,
        batch_frames,
        batch_masks,
        mode,
        ai_analysis,
        preset,
        effect_strength,
        enhancement_strength,
        adjust_for_skin_tone,
        white_balance_strength,
        warmth,
        vibrancy,
        brightness,
        contrast,
        tint,
        lift,
        gamma,
        gain,
        noise,
        device,
    ):
        """
        GPU-optimized batch processing that processes multiple frames simultaneously.
        """
        batch_size = batch_frames.shape[0]

        if str(batch_frames.device) != str(device):
            batch_frames = batch_frames.to(device)
            if batch_masks is not None:
                batch_masks = batch_masks.to(device)

        original_batch = batch_frames.clone()
        processed_batch = batch_frames.clone()

        # AI Analysis on first frame
        analysis = None
        if ai_analysis:
            analysis = self._analyze_image_gpu(batch_frames[0], device)

        # Apply preset modifications if in Preset mode
        if mode == "Preset":
            preset_mapping = {
                "Natural": "Natural Portrait",
                "Warm": "Warm Portrait",
                "Cool": "Cool Portrait",
                "High Key": "High Key Portrait",
                "Dramatic": "Dramatic Portrait",
            }

            full_preset_name = preset_mapping.get(preset, preset)
            if full_preset_name in self.PRESETS:
                preset_values = self.PRESETS[full_preset_name]
                warmth += preset_values.get("warmth", 0.0)
                vibrancy += preset_values.get("vibrancy", 0.0)
                brightness += preset_values.get("brightness", 0.0)
                contrast += preset_values.get("contrast", 0.0)

            if analysis and analysis["scene_type"] in ["concept_art", "anime", "stylized_art"]:
                vibrancy *= 1.4
                contrast *= 1.25

        # Auto mode processing
        if mode == "Auto":
            # White balance
            if white_balance_strength > 0.0:
                B, H, W, C = processed_batch.shape
                flat_batch = processed_batch.view(B, -1, C)
                percentile_40 = torch.quantile(flat_batch, 0.40, dim=1, keepdim=True)
                percentile_60 = torch.quantile(flat_batch, 0.60, dim=1, keepdim=True)
                midtone_mean = (percentile_40 + percentile_60) / 2.0
                avg_gray = torch.mean(midtone_mean, dim=-1, keepdim=True)
                scale = avg_gray / (midtone_mean + 1e-6)
                scale = torch.lerp(torch.ones_like(scale), scale, white_balance_strength)
                scale = scale.view(B, 1, 1, C)
                processed_batch = processed_batch * scale
                processed_batch = torch.clamp(processed_batch, 0.0, 1.0)

            # Enhancement based on scene analysis
            if enhancement_strength > 0.2:
                hsv_temp = rgb_to_hsv(processed_batch)
                h_temp, s_temp, v_temp = hsv_temp[..., 0], hsv_temp[..., 1], hsv_temp[..., 2]

                scene_type = "general"
                lighting = "auto"

                if analysis:
                    scene_type = analysis["scene_type"]
                    lighting = analysis["lighting_condition"]

                    if scene_type == "anime":
                        contrast_boost = 0.18 * enhancement_strength
                        saturation_boost = 0.55 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)
                    elif scene_type == "concept_art":
                        contrast_boost = 0.25 * enhancement_strength
                        saturation_boost = 0.40 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)
                    elif scene_type == "portrait":
                        warmth += 0.05 * enhancement_strength
                        contrast_boost = 0.12 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                    else:
                        contrast_boost = 0.15 * enhancement_strength
                        saturation_boost = 0.20 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)

                    if lighting == "low_light":
                        brightness += 0.1 * enhancement_strength
                        contrast += 0.4 * enhancement_strength
                    elif lighting == "bright":
                        brightness -= 0.05 * enhancement_strength
                    elif lighting == "flat":
                        contrast += 0.5 * enhancement_strength

                s_temp = torch.clamp(s_temp, 0.0, 1.0)
                v_temp = torch.clamp(v_temp, 0.0, 1.0)
                processed_batch = hsv_to_rgb(torch.stack([h_temp, s_temp, v_temp], dim=-1))

        # Convert to HSV for color corrections
        hsv_batch = rgb_to_hsv(processed_batch)
        h, s, v = hsv_batch[..., 0], hsv_batch[..., 1], hsv_batch[..., 2]

        # Manual mode corrections
        if mode == "Manual":
            if warmth != 0.0:
                h = (h + warmth * 0.1) % 1.0

            if vibrancy != 0.0:
                saturation_mask = 1.0 - s
                s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)

            if brightness != 0.0:
                v = v + brightness * (1.0 - v * 0.5)

            if contrast != 0.0:
                v = 0.5 + (v - 0.5) * (1.0 + contrast)

            # Tint adjustment using LAB color space
            if tint != 0.0 and OPENCV_AVAILABLE:
                try:
                    temp_hsv = torch.stack([h, s, v], dim=-1)
                    temp_rgb = hsv_to_rgb(temp_hsv)

                    for i in range(temp_rgb.shape[0]):
                        image_np = (temp_rgb[i].cpu().numpy() * 255).astype(np.uint8)
                        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                        tint_shift = tint * 30
                        lab[:, :, 1] = np.clip(lab[:, :, 1] + tint_shift, 0, 255)
                        corrected_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        temp_rgb[i] = torch.from_numpy(corrected_rgb.astype(np.float32) / 255.0).to(device)

                    corrected_hsv = rgb_to_hsv(temp_rgb)
                    h, s, v = corrected_hsv[..., 0], corrected_hsv[..., 1], corrected_hsv[..., 2]
                except Exception as e:
                    print(f"Tint processing failed: {e}")

            # 3-way color correction
            shadows_mask = 1.0 - torch.clamp(v * 3.0, 0.0, 1.0)
            midtones_mask = 1.0 - torch.abs(v - 0.5) * 2.0
            highlights_mask = torch.clamp((v - 0.66) * 3.0, 0.0, 1.0)

            if lift != 0.0:
                v = v + (lift * 0.8 * shadows_mask)

            if gamma != 0.0:
                gamma_exp = 1.0 / (1.0 + gamma * 1.2)
                v_gamma = torch.pow(torch.clamp(v, 0.001, 1.0), gamma_exp)
                v = torch.lerp(v, v_gamma, midtones_mask)

            if gain != 0.0:
                v = v + (gain * 0.8 * highlights_mask)

            # Add noise
            if noise > 0.0:
                mono_noise = torch.randn(
                    (batch_size, processed_batch.shape[1], processed_batch.shape[2], 1),
                    device=device,
                )
                luminance_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)

                rgb_temp = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
                rgb_temp += mono_noise * noise * 0.15 * luminance_mask
                rgb_temp = torch.clamp(rgb_temp, 0.0, 1.0)

                hsv_temp = rgb_to_hsv(rgb_temp)
                h, s, v = hsv_temp[..., 0], hsv_temp[..., 1], hsv_temp[..., 2]

        # Convert back to RGB
        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)
        processed_hsv = torch.stack([h, s, v], dim=-1)
        processed_batch = hsv_to_rgb(processed_hsv)

        # Apply effect strength
        if mode in ["Auto", "Preset"]:
            processed_batch = torch.lerp(original_batch, processed_batch, effect_strength)

        # Apply masks
        if batch_masks is not None:
            if batch_masks.shape[1:] != (processed_batch.shape[1], processed_batch.shape[2]):
                batch_masks = F.interpolate(
                    batch_masks.unsqueeze(1),
                    size=(processed_batch.shape[1], processed_batch.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            batch_masks = batch_masks.unsqueeze(-1)
            processed_batch = torch.lerp(original_batch, processed_batch, batch_masks)

        processed_batch = torch.clamp(processed_batch, 0.0, 1.0)

        return processed_batch

    def _analyze_image_gpu(self, image_tensor, device):
        """
        GPU-based image analysis without CPU bottlenecks.
        """
        hsv = rgb_to_hsv(image_tensor.unsqueeze(0))[0]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        brightness_mean = torch.mean(v).item()
        brightness_std = torch.std(v).item()

        if brightness_mean < 0.3:
            lighting_condition = "low_light"
        elif brightness_mean > 0.8:
            lighting_condition = "bright"
        elif brightness_std < 0.15:
            lighting_condition = "flat"
        else:
            lighting_condition = "optimal"

        saturation_mean = torch.mean(s).item()
        saturation_std = torch.std(s).item()

        if saturation_mean > 0.6 and saturation_std > 0.25:
            scene_type = "concept_art"
        elif saturation_mean > 0.5:
            scene_type = "stylized_art"
        elif saturation_mean < 0.3:
            scene_type = "portrait"
        else:
            scene_type = "realistic_photo"

        # Edge detection using Sobel filters
        gray = torch.mean(image_tensor, dim=-1, keepdim=True)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)

        gray_padded = F.pad(
            gray.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
        )
        edges_x = F.conv2d(gray_padded, sobel_x)
        edges_y = F.conv2d(gray_padded, sobel_y)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        edge_density = torch.mean(edges).item()

        return {
            "scene_type": scene_type,
            "lighting_condition": lighting_condition,
            "brightness_mean": brightness_mean,
            "saturation_mean": saturation_mean,
            "edge_density": edge_density,
            "has_faces": False,
            "skin_tone_areas": [],
        }


NODE_CLASS_MAPPINGS = {
    "BatchColorCorrector": BatchColorCorrector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchColorCorrector": "Batch Color Corrector",
}
