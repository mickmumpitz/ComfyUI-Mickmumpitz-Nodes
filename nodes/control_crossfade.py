"""
Control Crossfade Node for ComfyUI
===================================
Eliminates the "flash" at iteration boundaries in iterative Wan2.1 SkyReels workflows.

Problem: When iteration N+1 starts, the first ~14 frames come from the previous iteration's
generated output (start images). At frame 15, the control input hard-switches to the raw
control video. Because the generated output has drifted in color/lighting from the original
control footage, this abrupt switch creates a visible flash.

Solution: This node creates a smooth cross-fade zone in the control images around the
transition point. Instead of a hard cut, the control gradually blends from the previous
iteration's generated frames to the actual control video over a configurable number of frames.
"""

import torch


class ControlCrossfadeIterationFix:
    """
    Blends control images at iteration boundaries to prevent flash artifacts.

    For iteration N+1:
      - Frames 0 to (num_start_frames - blend_length): use prev_gen_frames as control
      - Frames (num_start_frames - blend_length) to (num_start_frames + blend_length): cross-fade
      - Frames (num_start_frames + blend_length) onward: use raw control_frames
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_frames": ("IMAGE",),
                "prev_gen_frames": ("IMAGE",),
                "num_start_frames": ("INT", {
                    "default": 14,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of start/overlap frames from the previous iteration"
                }),
                "blend_length": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Number of frames over which to cross-fade (centered on the switch point)"
                }),
                "blend_curve": (["linear", "ease_in_out", "ease_in", "ease_out"],),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match color/brightness of control frames to prev_gen at the transition point"
                }),
            },
            "optional": {
                "iteration_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Current iteration index. Set to 0 or -1 to skip blending (first iteration needs no fix)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_control",)
    FUNCTION = "apply_crossfade"
    CATEGORY = "image/transform"
    DESCRIPTION = "Eliminates flash artifacts at iteration boundaries by cross-fading control images"

    def apply_crossfade(self, control_frames, prev_gen_frames, num_start_frames,
                        blend_length, blend_curve, color_match, iteration_index=-1):
        # First iteration: no previous output to blend with, pass through unchanged
        if iteration_index == 0 or iteration_index == -1:
            return (control_frames,)

        B, H, W, C = control_frames.shape
        B_prev = prev_gen_frames.shape[0]

        # Clone control frames to avoid modifying the original
        result = control_frames.clone()

        # Calculate blend boundaries
        half_blend = blend_length // 2
        blend_start = max(0, num_start_frames - half_blend)
        blend_end = min(B, num_start_frames + half_blend)
        actual_blend_length = blend_end - blend_start

        if actual_blend_length < 2:
            return (control_frames,)

        # Optional: Color match the control frames to the previous generation
        color_offset = torch.zeros(1, 1, 1, C, device=control_frames.device)
        if color_match and B_prev > 0:
            switch_idx = min(num_start_frames, B - 1)
            prev_ref_idx = min(B_prev - 1, num_start_frames - 1)

            prev_mean = prev_gen_frames[prev_ref_idx].mean(dim=(0, 1), keepdim=True)
            ctrl_mean = control_frames[switch_idx].mean(dim=(0, 1), keepdim=True)
            color_offset = (prev_mean - ctrl_mean).unsqueeze(0)

        # Generate blend weights based on the chosen curve
        t = torch.linspace(0.0, 1.0, actual_blend_length, device=control_frames.device)

        if blend_curve == "ease_in_out":
            weights = t * t * (3.0 - 2.0 * t)
        elif blend_curve == "ease_in":
            weights = t * t
        elif blend_curve == "ease_out":
            weights = 1.0 - (1.0 - t) ** 2
        else:  # linear
            weights = t

        # weights: 0.0 = fully prev_gen, 1.0 = fully control
        weights = weights.view(-1, 1, 1, 1)

        # Apply the cross-fade in the blend zone
        for i in range(actual_blend_length):
            frame_idx = blend_start + i
            w = weights[i]

            prev_idx = min(frame_idx, B_prev - 1)
            prev_frame = prev_gen_frames[prev_idx]
            ctrl_frame = control_frames[frame_idx]

            if color_match:
                correction_strength = 1.0 - weights[i]
                corrected_ctrl = ctrl_frame + (color_offset[0, 0] * correction_strength).squeeze(0)
                corrected_ctrl = corrected_ctrl.clamp(0.0, 1.0)
            else:
                corrected_ctrl = ctrl_frame

            result[frame_idx] = (1.0 - w) * prev_frame + w * corrected_ctrl

        # For frames BEFORE the blend zone, replace control with prev_gen
        for i in range(blend_start):
            prev_idx = min(i, B_prev - 1)
            result[i] = prev_gen_frames[prev_idx]

        return (result,)


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
                if curve == "ease_in_out":
                    w = t * t * (3.0 - 2.0 * t)
                elif curve == "ease_in":
                    w = t * t
                elif curve == "ease_out":
                    w = 1.0 - (1.0 - t) ** 2
                else:
                    w = t
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
