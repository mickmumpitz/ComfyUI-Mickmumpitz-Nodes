"""
ComfyUI nodes: select an OpenPose body region, crop it for a detailer pass,
and stitch the result back into the original image.

Nodes
-----
* OpenPose Image To Keypoints  : rendered OpenPose image  -> POSE_KEYPOINT
* OpenPose Part Mask           : POSE_KEYPOINT + IMAGE     -> MASK (+ bbox)
* OpenPose Part Crop           : POSE_KEYPOINT + IMAGE     -> STITCH + crop + mask
* OpenPose Part Stitch         : STITCH + detailed IMAGE   -> IMAGE
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import regions
from . import image_extract

# ComfyUI tooltip text wants left/right to be unambiguous.
_LR_NOTE = ("Left/right are anatomical (the subject's own side). 'right_hand' is "
            "the person's right hand, which usually appears on the LEFT of the image. "
            "Enable mirror_left_right if you prefer image-space sides.")

_MIRROR = {
    "left_hand": "right_hand", "right_hand": "left_hand",
    "left_foot": "right_foot", "right_foot": "left_foot",
    "left_arm": "right_arm", "right_arm": "left_arm",
    "left_leg": "right_leg", "right_leg": "left_leg",
}

_INTERP = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic", "area": "area"}

_PERSON_ORDER = ["detection", "left_right", "right_left", "top_bottom"]
_PERSON_ORDER_NOTE = (
    "How person_index is ordered when there are multiple characters. "
    "'detection' = the detector's raw order (unstable between frames). "
    "'left_right' = index 0 is the leftmost character, 1 the next, etc. "
    "Use left_right for a stable 2-character setup.")

_KEYPOINTS_NOTE = (
    "Crop a box enclosing these BODY keypoint numbers instead of the named "
    "region. Leave empty to use 'region'. Examples: '4' | '2,3,4' | '2-4'. "
    "BODY_18: 0 nose, 1 neck, 2 R-shoulder, 3 R-elbow, 4 R-wrist, "
    "5 L-shoulder, 6 L-elbow, 7 L-wrist, 8 R-hip, 9 R-knee, 10 R-ankle, "
    "11 L-hip, 12 L-knee, 13 L-ankle, 14 R-eye, 15 L-eye, 16 R-ear, 17 L-ear. "
    "(R/L are the subject's anatomical sides.)")


# ---------------------------------------------------------------------------
# Tensor helpers (IMAGE = [B,H,W,C] float 0..1, MASK = [B,H,W] float 0..1)
# ---------------------------------------------------------------------------
def _resize_image(img, w, h, mode="bicubic"):
    if img.shape[1] == h and img.shape[2] == w:
        return img
    x = img.permute(0, 3, 1, 2)  # BCHW
    kwargs = {} if mode in ("nearest", "area") else {"align_corners": False}
    x = F.interpolate(x, size=(h, w), mode=_INTERP.get(mode, "bicubic"), **kwargs)
    x = x.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    return x


def _resize_mask(mask, w, h, mode="bilinear"):
    if mask.shape[1] == h and mask.shape[2] == w:
        return mask
    x = mask.unsqueeze(1)  # B1HW
    kwargs = {} if mode in ("nearest", "area") else {"align_corners": False}
    x = F.interpolate(x, size=(h, w), mode=_INTERP.get(mode, "bilinear"), **kwargs)
    return x.squeeze(1).clamp(0.0, 1.0)


def _gaussian_blur_mask(mask, sigma):
    if sigma <= 0:
        return mask
    radius = max(1, int(round(sigma * 3)))
    ksize = radius * 2 + 1
    coords = torch.arange(ksize, dtype=torch.float32, device=mask.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum())
    x = mask.unsqueeze(1)  # B1HW
    kx = g.view(1, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky)
    return x.squeeze(1).clamp(0.0, 1.0)


def _feathered_rect(b, h, w, feather, device):
    """A [b,h,w] mask: 1 inside, feathered toward the rectangle edges."""
    m = torch.ones((b, h, w), dtype=torch.float32, device=device)
    if feather > 0:
        inset = min(int(feather), max(0, min(h, w) // 2 - 1))
        if inset > 0:
            m.zero_()
            m[:, inset:h - inset, inset:w - inset] = 1.0
        m = _gaussian_blur_mask(m, max(feather / 2.0, 1.0))
    return m


def _img_to_np_uint8(img_tensor_single):
    arr = (img_tensor_single.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr  # HxWxC


def _resolve_region(region, mirror):
    if mirror and region in _MIRROR:
        return _MIRROR[region]
    return region


def _compute_box(image, pose_keypoint, region, person_index, batch_index,
                 mirror, padding, make_square, min_size, multiple_of, conf,
                 person_order="detection", keypoint_indices=""):
    """Shared: returns (box, info) where box is (x,y,w,h) clamped to the image
    or None, and info is a diagnostic dict describing what was found."""
    _, H, W, _ = image.shape
    info = {"n_people": 0, "region_used": None, "body_joints": [],
            "hand_left": False, "hand_right": False, "raw_box": None, "box": None,
            "reason": ""}
    frame = regions.get_frame(pose_keypoint, batch_index)
    info["n_people"] = regions.num_people(frame)
    person = regions.get_person(frame, person_index, order=person_order)
    if person is None:
        info["reason"] = "no people in pose_keypoint for this frame"
        return None, info
    parsed = regions.parse_person(frame, person, W, H, conf_threshold=conf)
    info["body_joints"] = [i for i, p in enumerate(parsed["body"]) if p is not None]
    info["hand_left"] = any(p is not None for p in parsed["hand_left"])
    info["hand_right"] = any(p is not None for p in parsed["hand_right"])
    indices = regions.parse_indices(keypoint_indices)
    if indices:
        # Explicit bone numbers override the named region.
        raw = regions.keypoint_bbox(parsed, indices, group="body")
        info["region_used"] = "keypoints " + ",".join(str(i) for i in indices)
    else:
        region = _resolve_region(region, mirror)
        raw = regions.region_bbox(parsed, region)
        info["region_used"] = region
    info["raw_box"] = raw
    box = regions.finalize_box(raw, W, H, padding=padding, make_square=make_square,
                               min_size=min_size, multiple_of=multiple_of)
    info["box"] = box
    if raw is None:
        info["reason"] = "region/keypoints not found for the selected person"
    return box, info


def _format_diag(prefix, info, person_index, person_order):
    msg = (f"[{prefix}] people={info['n_people']} "
           f"person_index={person_index} order={person_order} "
           f"region='{info['region_used']}' "
           f"body_joints_found={info['body_joints']} "
           f"hand_left={info['hand_left']} hand_right={info['hand_right']} "
           f"box={info['box']}")
    if info["box"] is None:
        msg += "  -> NO BOX: " + (info["reason"] or "unknown") + \
               "  (falling back to WHOLE IMAGE — output will look uncropped)"
    return msg


# ===========================================================================
# Node 1: rendered OpenPose image -> POSE_KEYPOINT
# ===========================================================================
class OpenPoseImageToKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openpose_image": ("IMAGE", {"tooltip": "A rendered OpenPose / DWPose skeleton image."}),
            },
            "optional": {
                "color_tolerance": ("INT", {"default": 40, "min": 0, "max": 128,
                                            "tooltip": "Per-channel color match tolerance for joint detection."}),
                "min_dot_area": ("INT", {"default": 4, "min": 1, "max": 200,
                                         "tooltip": "Ignore color blobs smaller than this (px)."}),
                "reconstruct_hands_face": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Also recover hand (blue) and face (white) dot clusters."}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE", "STRING")
    RETURN_NAMES = ("pose_keypoint", "debug_overlay", "info")
    FUNCTION = "extract"
    CATEGORY = "Mickmumpitz/OpenPosePartCropper"
    DESCRIPTION = ("Recover OpenPose keypoints from an already-rendered skeleton image "
                   "(when you don't have the original POSE_KEYPOINT data).")

    def extract(self, openpose_image, color_tolerance=40, min_dot_area=4, reconstruct_hands_face=True):
        frames = []
        overlays = []
        total_people = 0
        for b in range(openpose_image.shape[0]):
            np_img = _img_to_np_uint8(openpose_image[b])
            frame = image_extract.extract_pose(
                np_img[..., :3], color_tol=color_tolerance, min_area=min_dot_area,
                reconstruct_hands_face=reconstruct_hands_face,
            )
            frames.append(frame)
            total_people += len(frame.get("people", []))
            overlays.append(self._overlay(np_img[..., :3], frame))

        overlay_tensor = torch.from_numpy(
            np.stack(overlays, axis=0).astype(np.float32) / 255.0
        )
        info = f"Frames: {len(frames)} | people detected: {total_people}"
        return (frames, overlay_tensor, info)

    @staticmethod
    def _overlay(np_img_rgb, frame):
        canvas = np_img_rgb.copy()
        H, W, _ = canvas.shape
        try:
            import cv2
            for person in frame.get("people", []):
                body = person.get("pose_keypoints_2d") or []
                for i in range(0, len(body) - 2, 3):
                    x, y, c = body[i], body[i + 1], body[i + 2]
                    if c <= 0:
                        continue
                    px, py = int(x * W), int(y * H)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1)
                    cv2.putText(canvas, str(i // 3), (px + 4, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        return canvas


# ===========================================================================
# Node 2: POSE_KEYPOINT + IMAGE -> MASK (for use with any crop/inpaint node)
# ===========================================================================
class OpenPosePartMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_keypoint": ("POSE_KEYPOINT",),
                "region": (regions.REGION_LIST, {"default": "right_hand", "tooltip": _LR_NOTE}),
            },
            "optional": {
                "person_index": ("INT", {"default": 0, "min": 0, "max": 64}),
                "padding": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 4.0, "step": 0.05,
                                      "tooltip": "Expand the box around the region (1.0 = tight)."}),
                "make_square": ("BOOLEAN", {"default": False}),
                "min_size": ("INT", {"default": 64, "min": 8, "max": 4096}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 256,
                                      "tooltip": "Feather the mask edges."}),
                "mirror_left_right": ("BOOLEAN", {"default": False, "tooltip": _LR_NOTE}),
                "confidence": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "person_order": (_PERSON_ORDER, {"default": "detection", "tooltip": _PERSON_ORDER_NOTE}),
                "keypoint_indices": ("STRING", {"default": "", "tooltip": _KEYPOINTS_NOTE}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "x", "y", "width", "height")
    FUNCTION = "make_mask"
    CATEGORY = "Mickmumpitz/OpenPosePartCropper"
    DESCRIPTION = ("Build a rectangular mask around an OpenPose region. Feed it into "
                   "InpaintCrop (crop-and-stitch) or any inpaint/detailer node.")

    def make_mask(self, image, pose_keypoint, region, person_index=0, padding=1.3,
                  make_square=False, min_size=64, mask_blur=0, mirror_left_right=False,
                  confidence=0.05, person_order="detection", keypoint_indices=""):
        B, H, W, _ = image.shape
        mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
        last = (0, 0, 0, 0)
        for b in range(B):
            box, info = _compute_box(image, pose_keypoint, region, person_index, b,
                                     mirror_left_right, padding, make_square, min_size,
                                     multiple_of=1, conf=confidence,
                                     person_order=person_order,
                                     keypoint_indices=keypoint_indices)
            if b == 0:
                print(_format_diag("OpenPose Part Mask", info, person_index, person_order))
            if box is None:
                continue
            x, y, w, h = box
            mask[b, y:y + h, x:x + w] = 1.0
            last = box
        if mask_blur > 0:
            mask = _gaussian_blur_mask(mask, mask_blur / 2.0)
        x, y, w, h = last
        return (mask, x, y, w, h)


# ===========================================================================
# Node 3: POSE_KEYPOINT + IMAGE -> STITCH + cropped image + cropped mask
# ===========================================================================
class OpenPosePartCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_keypoint": ("POSE_KEYPOINT",),
                "region": (regions.REGION_LIST, {"default": "right_hand", "tooltip": _LR_NOTE}),
            },
            "optional": {
                "person_index": ("INT", {"default": 0, "min": 0, "max": 64}),
                "padding": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 4.0, "step": 0.05}),
                "make_square": ("BOOLEAN", {"default": True,
                                            "tooltip": "Square crops are convenient for detailers."}),
                "min_size": ("INT", {"default": 128, "min": 8, "max": 4096}),
                "output_resize": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8,
                                          "tooltip": "Resize the crop so its long side = this (0 = keep native size)."}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 64,
                                        "tooltip": "Force crop/output dims to a multiple of this (VAE-friendly)."}),
                "mask_feather": ("INT", {"default": 16, "min": 0, "max": 512,
                                         "tooltip": "Feather used when stitching the crop back."}),
                "upscale_method": (list(_INTERP.keys()), {"default": "bicubic"}),
                "downscale_method": (list(_INTERP.keys()), {"default": "area"}),
                "mirror_left_right": ("BOOLEAN", {"default": False, "tooltip": _LR_NOTE}),
                "confidence": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "person_order": (_PERSON_ORDER, {"default": "detection", "tooltip": _PERSON_ORDER_NOTE}),
                "keypoint_indices": ("STRING", {"default": "", "tooltip": _KEYPOINTS_NOTE}),
            },
        }

    RETURN_TYPES = ("OPP_STITCH", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask", "info")
    FUNCTION = "crop"
    CATEGORY = "Mickmumpitz/OpenPosePartCropper"
    DESCRIPTION = ("Crop an OpenPose region for a detailer pass. Returns a stitch handle, "
                   "the cropped image, and a feathered mask. Pair with OpenPose Part Stitch.")

    def crop(self, image, pose_keypoint, region, person_index=0, padding=1.4,
             make_square=True, min_size=128, output_resize=1024, multiple_of=8,
             mask_feather=16, upscale_method="bicubic", downscale_method="area",
             mirror_left_right=False, confidence=0.05, person_order="detection",
             keypoint_indices=""):
        B, H, W, _ = image.shape
        device = image.device

        crops = []
        masks = []
        regions_px = []  # (x, y, w, h) native region per batch item
        out_w = out_h = None
        info_lines = []

        for b in range(B):
            box, info = _compute_box(image, pose_keypoint, region, person_index, b,
                                     mirror_left_right, padding, make_square, min_size,
                                     multiple_of=multiple_of, conf=confidence,
                                     person_order=person_order,
                                     keypoint_indices=keypoint_indices)
            diag = _format_diag("OpenPose Part Crop", info, person_index, person_order)
            if b == 0 or B <= 4:
                print(diag)
            info_lines.append(f"frame {b}: {diag}")
            if box is None:
                # No detection: fall back to the whole image so the graph still runs.
                box = (0, 0, W, H)
            x, y, w, h = box
            regions_px.append(box)

            sub = image[b:b + 1, y:y + h, x:x + w, :]

            # Decide output resolution.
            tw, th = w, h
            if output_resize and output_resize > 0:
                scale = output_resize / float(max(w, h))
                tw = max(multiple_of, int(round(w * scale)))
                th = max(multiple_of, int(round(h * scale)))
                if multiple_of > 1:
                    tw = int(round(tw / multiple_of) * multiple_of)
                    th = int(round(th / multiple_of) * multiple_of)
            method = upscale_method if (tw > w or th > h) else downscale_method
            sub = _resize_image(sub, tw, th, method)

            # Keep all batch crops the same size (stack-friendly); use first crop's size.
            if out_w is None:
                out_w, out_h = tw, th
            elif (tw, th) != (out_w, out_h):
                sub = _resize_image(sub, out_w, out_h, method)

            crops.append(sub)
            masks.append(_feathered_rect(1, sub.shape[1], sub.shape[2], mask_feather, device))

        cropped_image = torch.cat(crops, dim=0)
        cropped_mask = torch.cat(masks, dim=0)

        stitch = {
            "original_image": image,
            "regions": regions_px,           # native (x,y,w,h) per batch item
            "feather": mask_feather,
            "upscale_method": upscale_method,
            "downscale_method": downscale_method,
        }
        return (stitch, cropped_image, cropped_mask, "\n".join(info_lines))


# ===========================================================================
# Node 4: STITCH + detailed image -> final IMAGE
# ===========================================================================
class OpenPosePartStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch": ("OPP_STITCH",),
                "image": ("IMAGE", {"tooltip": "The detailed/processed crop to paste back."}),
            },
            "optional": {
                "blend_mask": ("MASK", {"tooltip": "Optional override mask (same size as the processed crop)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "Mickmumpitz/OpenPosePartCropper"
    DESCRIPTION = "Paste the detailed crop back into the original image with feathered blending."

    def stitch(self, stitch, image, blend_mask=None):
        original = stitch["original_image"]
        regions_px = stitch["regions"]
        feather = stitch.get("feather", 16)
        up = stitch.get("upscale_method", "bicubic")
        down = stitch.get("downscale_method", "area")

        out = original.clone()
        device = out.device
        image = image.to(device)
        B = out.shape[0]

        for b in range(B):
            x, y, w, h = regions_px[min(b, len(regions_px) - 1)]
            proc = image[min(b, image.shape[0] - 1):min(b, image.shape[0] - 1) + 1]
            method = up if (w > proc.shape[2] or h > proc.shape[1]) else down
            proc = _resize_image(proc, w, h, method)

            if blend_mask is not None:
                m = blend_mask[min(b, blend_mask.shape[0] - 1):min(b, blend_mask.shape[0] - 1) + 1]
                m = _resize_mask(m, w, h)
            else:
                m = _feathered_rect(1, h, w, feather, device)
            m = m.unsqueeze(-1)  # 1,h,w,1

            region = out[b:b + 1, y:y + h, x:x + w, :]
            out[b:b + 1, y:y + h, x:x + w, :] = m * proc + (1.0 - m) * region

        return (out,)
