"""
Control Crossfade Node for ComfyUI
===================================
Eliminates the "flash" at iteration boundaries in iterative Wan2.1 SkyReels workflows.

Blending layout (0-indexed, num_start_frames=14, blend_length=8):
  - Frames 0..5   → pure start_frames
  - Frames 6..13  → cross-fade start_frames → control_frames
  - Frames 14+    → pure control_frames

Optional mask output follows the same zones:
  - Frames 0..5   → black (0.0)
  - Frames 6..13  → cross-fade black → mask
  - Frames 14+    → mask as-is
"""

import torch


def _apply_curve(t, curve):
    """Apply easing curve to a 0→1 linear ramp."""
    if curve == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    elif curve == "ease_in":
        return t * t
    elif curve == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    return t  # linear


class ControlCrossfadeIterationFix:
    """
    Blends start_frames into control_frames to prevent flash artifacts at
    iteration boundaries.

    Zones (0-indexed):
      [0, blend_start)             → pure start_frames
      [blend_start, num_start_frames) → cross-fade
      [num_start_frames, ...)       → pure control_frames

    Where blend_start = num_start_frames - blend_length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_frames": ("IMAGE",),
                "control_frames": ("IMAGE",),
                "num_start_frames": ("INT", {
                    "default": 14,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Frame index where control takes over fully (blend target)"
                }),
                "blend_length": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Number of frames the cross-fade spans before num_start_frames"
                }),
                "blend_curve": (["linear", "ease_in_out", "ease_in", "ease_out"],),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match color/brightness of control frames to start frames at the transition"
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
            "hidden": {
                "iteration": "INT",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_control", "blended_mask")
    FUNCTION = "apply_crossfade"
    CATEGORY = "image/transform"
    DESCRIPTION = "Eliminates flash artifacts at iteration boundaries by cross-fading start frames into control frames"

    def apply_crossfade(self, start_frames, control_frames, num_start_frames,
                        blend_length, blend_curve, color_match,
                        mask=None, iteration=0):
        B, H, W, C = control_frames.shape
        device = control_frames.device

        if mask is not None:
            mask_B = mask.shape[0]
        else:
            mask_B = 0

        # First iteration: pass through unchanged
        if iteration == 0:
            if mask is not None:
                mask_out = mask
            else:
                mask_out = torch.ones(B, H, W, device=device)
            return (control_frames, mask_out)

        # --- Zone boundaries (0-indexed) ---
        blend_start = max(0, num_start_frames - blend_length)
        blend_end = min(B, num_start_frames)  # exclusive
        actual_blend = blend_end - blend_start

        # --- Color matching offset ---
        color_offset = torch.zeros(1, 1, 1, C, device=device)
        if color_match and actual_blend > 0:
            # Compare at the boundary where blending begins
            ref_idx = min(blend_start, B - 1)
            start_ref_idx = min(blend_start, start_frames.shape[0] - 1)
            start_mean = start_frames[start_ref_idx].mean(dim=(0, 1), keepdim=True)
            ctrl_mean = control_frames[ref_idx].mean(dim=(0, 1), keepdim=True)
            color_offset = (start_mean - ctrl_mean).unsqueeze(0)  # (1,1,1,C)

        # --- Blend weights for the transition zone ---
        if actual_blend > 1:
            t = torch.linspace(0.0, 1.0, actual_blend, device=device)
            weights = _apply_curve(t, blend_curve)
        elif actual_blend == 1:
            weights = torch.tensor([1.0], device=device)
        else:
            weights = torch.zeros(0, device=device)

        # --- Build image output ---
        result = control_frames.clone()
        B_start = start_frames.shape[0]

        # Pure start zone
        for i in range(min(blend_start, B)):
            si = min(i, B_start - 1)
            result[i] = start_frames[si]

        # Blend zone
        for i in range(actual_blend):
            frame_idx = blend_start + i
            w = weights[i]

            si = min(frame_idx, B_start - 1)
            s_frame = start_frames[si]
            c_frame = control_frames[frame_idx]

            # Color correction fades out as we approach pure control
            if color_match:
                correction = 1.0 - w
                c_frame = (c_frame + color_offset[0, 0] * correction).clamp(0.0, 1.0)

            result[frame_idx] = (1.0 - w) * s_frame + w * c_frame

        # Frames >= blend_end are already control_frames from the clone

        # --- Build mask output ---
        if mask is not None:
            mask_out = torch.zeros(B, H, W, device=device)

            # Pure start zone → black (already zeros)

            # Blend zone: black → mask
            for i in range(actual_blend):
                frame_idx = blend_start + i
                w = weights[i]
                mi = min(frame_idx, mask_B - 1)
                mask_out[frame_idx] = w * mask[mi]

            # Pure control zone → mask as-is
            for i in range(blend_end, B):
                mi = min(i, mask_B - 1)
                mask_out[i] = mask[mi]
        else:
            mask_out = torch.ones(B, H, W, device=device)

        return (result, mask_out)


class ControlCrossfadeSimple:
    """
    Simplified version: just blends two image batches at a specified transition frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_A": ("IMAGE",),
                "images_B": ("IMAGE",),
                "transition_frame": ("INT", {
                    "default": 14,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                }),
                "blend_frames": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 40,
                    "step": 1,
                }),
                "curve": (["ease_in_out", "linear", "ease_in", "ease_out"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "blend"
    CATEGORY = "image/transform"
    DESCRIPTION = "Cross-fade between two image sequences at a given frame"

    def blend(self, images_A, images_B, transition_frame, blend_frames, curve):
        B = max(images_A.shape[0], images_B.shape[0])
        H, W, C = images_A.shape[1], images_A.shape[2], images_A.shape[3]

        result = torch.zeros(B, H, W, C, device=images_A.device)

        half = blend_frames // 2
        blend_start = max(0, transition_frame - half)
        blend_end = min(B, transition_frame + half)

        for i in range(B):
            idx_a = min(i, images_A.shape[0] - 1)
            idx_b = min(i, images_B.shape[0] - 1)

            if i < blend_start:
                result[i] = images_A[idx_a]
            elif i >= blend_end:
                result[i] = images_B[idx_b]
            else:
                t = (i - blend_start) / max(1, blend_end - blend_start - 1)
                w = _apply_curve(torch.tensor(t), curve).item()
                result[i] = (1.0 - w) * images_A[idx_a] + w * images_B[idx_b]

        return (result,)


NODE_CLASS_MAPPINGS = {
    "ControlCrossfadeIterationFix": ControlCrossfadeIterationFix,
    "ControlCrossfadeSimple": ControlCrossfadeSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlCrossfadeIterationFix": "Control Crossfade (Iteration Fix)",
    "ControlCrossfadeSimple": "Control Crossfade (Simple)",
}
