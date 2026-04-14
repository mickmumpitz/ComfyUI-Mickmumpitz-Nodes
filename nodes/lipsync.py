"""
Lipsync helper nodes for LTX-Video audio-driven workflows:

  - AudioTimestepOverride   — opens the A2V cross-attention gate for
                               IC-LoRA + input-audio workflows
  - PoseKeypointToMask      — 68-point face landmark → region mask
  - MaskDirectionalExtend   — extend a mask in chosen directions
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn.functional as F


_REGION_CONTOURS = {
    "mouth":           list(range(48, 60)),
    "mouth_inner":     list(range(60, 68)),
    "lips":            list(range(48, 68)),
    "chin":            list(range(5, 12)) + [57],
    "lower_face":      list(range(0, 17)) + [33],
    "full_face":       list(range(0, 17)) + list(range(26, 16, -1)),
    "nose":            [27, 31, 32, 33, 34, 35],
    "mouth_and_nose":  list(range(27, 36)) + list(range(48, 60)),
    "right_eye":       list(range(36, 42)),
    "left_eye":        list(range(42, 48)),
    "both_eyes":       list(range(36, 48)),
    "right_eyebrow":   list(range(17, 22)),
    "left_eyebrow":    list(range(22, 27)),
    "both_eyebrows":   list(range(17, 27)),
    "eyes_and_brows":  list(range(17, 48)),
    "upper_face":      list(range(17, 27)) + list(range(36, 48)),
}


def _is_normalized(keypoints: np.ndarray) -> bool:
    return bool(np.all((0 <= np.abs(keypoints)) & (np.abs(keypoints) <= 1)))


def _extract_face_mask(face_kps_flat, region, canvas_hw):
    h, w = canvas_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_kps_flat is None or len(face_kps_flat) < 68 * 3:
        return mask

    kps = np.array(face_kps_flat, dtype=np.float64).reshape(-1, 3)[:, :2]
    if _is_normalized(kps):
        kps[:, 0] *= w
        kps[:, 1] *= h

    contour = kps[_REGION_CONTOURS[region]].astype(np.int32)
    cv2.fillPoly(mask, [contour], 255)
    return mask


class AudioTimestepOverride:
    """Override audio denoise mask to open the A2V cross-attention gate.

    When input audio uses SetLatentNoiseMask(0), the AV model's
    process_timestep sets a_timestep = 0 for all audio tokens. The A2V
    cross-attention gate (computed from a_timestep.max()) shuts, blocking
    audio from influencing video — the mouth freezes.

    This node installs a model_function_wrapper that replaces the
    audio_denoise_mask with a uniform non-zero value so the model sees
    a_timestep = scale * sigma. The sampler's inpainting blend still uses
    the original noise_mask = 0 (preserving 100 % of the audio signal).

    Recommended value: ~0.93.
    """

    CATEGORY = "Mickmumpitz/Lipsync"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "audio_timestep_scale": ("FLOAT", {
                "default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": (
                    "Uniform scale for the audio denoise mask seen by the "
                    "model. 1.0 = model sees audio at same noise as video. "
                    "0.0 = original behaviour (gate closed)."
                ),
            }),
        }}

    def apply(self, model, audio_timestep_scale):
        if audio_timestep_scale <= 0:
            return (model,)

        scale = float(audio_timestep_scale)
        m = model.clone()

        def unet_wrapper(apply_model, args):
            c = args["c"]
            if "audio_denoise_mask" in c:
                orig = c["audio_denoise_mask"]
                c["audio_denoise_mask"] = torch.ones_like(orig) * scale
            return apply_model(args["input"], args["timestep"], **c)

        m.set_model_unet_function_wrapper(unet_wrapper)
        return (m,)


class PoseKeypointToMask:
    """Extract a facial-region mask from DWPose 68-point face keypoints."""

    CATEGORY = "Mickmumpitz/Lipsync"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "extract_mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_kps": ("POSE_KEYPOINT",),
            "region": (list(_REGION_CONTOURS.keys()), {
                "default": "mouth",
                "tooltip": "Which facial region to mask.",
            }),
            "dilate_px": ("INT", {
                "default": 8, "min": 0, "max": 128, "step": 1,
                "tooltip": "Expand the mask outward by this many pixels.",
            }),
            "blur_px": ("INT", {
                "default": 6, "min": 0, "max": 64, "step": 1,
                "tooltip": "Feather the mask edges with a box blur.",
            }),
            "person_index": ("INT", {
                "default": 0, "min": 0, "max": 10, "step": 1,
                "tooltip": "Which detected person to use (0 = first).",
            }),
        }}

    def extract_mask(self, pose_kps, region, dilate_px, blur_px, person_index):
        masks = []
        for frame_data in pose_kps:
            h = frame_data["canvas_height"]
            w = frame_data["canvas_width"]

            people = frame_data.get("people", [])
            if person_index < len(people):
                face_kps_flat = people[person_index].get(
                    "face_keypoints_2d", []
                )
                mask = _extract_face_mask(face_kps_flat, region, (h, w))
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

            masks.append(mask)

        mask_np = np.stack(masks, axis=0).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np)

        if dilate_px > 0:
            k = dilate_px * 2 + 1
            mask_t = F.max_pool2d(
                mask_t.unsqueeze(1), kernel_size=k, stride=1, padding=dilate_px,
            ).squeeze(1)

        if blur_px > 0:
            k = blur_px * 2 + 1
            m4 = mask_t.unsqueeze(1)
            m4 = F.avg_pool2d(m4, kernel_size=k, stride=1, padding=blur_px)
            m4 = F.avg_pool2d(m4, kernel_size=k, stride=1, padding=blur_px)
            mask_t = m4.squeeze(1)

        return (mask_t.clamp(0.0, 1.0),)


class MaskDirectionalExtend:
    """Extend a mask independently in each direction."""

    CATEGORY = "Mickmumpitz/Lipsync"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "extend"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "left": ("INT", {
                "default": 0, "min": 0, "max": 512, "step": 1,
                "tooltip": "Extend mask leftward by this many pixels.",
            }),
            "right": ("INT", {
                "default": 0, "min": 0, "max": 512, "step": 1,
                "tooltip": "Extend mask rightward by this many pixels.",
            }),
            "down": ("INT", {
                "default": 0, "min": 0, "max": 512, "step": 1,
                "tooltip": "Extend mask downward by this many pixels.",
            }),
            "up": ("INT", {
                "default": 0, "min": 0, "max": 512, "step": 1,
                "tooltip": "Extend mask upward by this many pixels.",
            }),
            "blur_px": ("INT", {
                "default": 0, "min": 0, "max": 64, "step": 1,
                "tooltip": "Feather the extended edges with a box blur.",
            }),
        }}

    def extend(self, mask, left, right, down, up, blur_px):
        if mask.ndim == 2:
            m = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[1] == 1:
            m = mask.squeeze(1)
        else:
            m = mask

        m = m.float().clone()

        if left > 0 or right > 0:
            k_w = left + right + 1
            padded = F.pad(m.unsqueeze(1), (left, right, 0, 0), mode="constant", value=0)
            m = F.max_pool2d(padded, kernel_size=(1, k_w), stride=1).squeeze(1)

        if up > 0 or down > 0:
            k_h = up + down + 1
            padded = F.pad(m.unsqueeze(1), (0, 0, down, up), mode="constant", value=0)
            m = F.max_pool2d(padded, kernel_size=(k_h, 1), stride=1).squeeze(1)

        if blur_px > 0:
            k = blur_px * 2 + 1
            m4 = m.unsqueeze(1)
            m4 = F.avg_pool2d(m4, kernel_size=k, stride=1, padding=blur_px)
            m4 = F.avg_pool2d(m4, kernel_size=k, stride=1, padding=blur_px)
            m = m4.squeeze(1)

        return (m.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "AudioTimestepOverride": AudioTimestepOverride,
    "PoseKeypointToMask": PoseKeypointToMask,
    "MaskDirectionalExtend": MaskDirectionalExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTimestepOverride": "Audio Timestep Override",
    "PoseKeypointToMask": "Pose Keypoint to Mask",
    "MaskDirectionalExtend": "Mask Directional Extend",
}
