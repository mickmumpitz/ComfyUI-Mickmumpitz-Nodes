"""
Compose + Match (per-frame) Node for ComfyUI
============================================
Composites a *source* over a *destination* through a *mask*, with an optional
per-frame correction so the inserted region blends into the surrounding plate.

Two correction modes:

1. Grade match (surround)  [default, recommended for VFX inserts]
   ------------------------------------------------------------------
   Use case: you regenerate a whole frame with a video model (e.g. WAN) to add
   an element (lava on the floor) but only want to keep the masked region and
   composite it back over the *original* footage. The VAE round-trip shifts the
   whole regenerated frame's grading (luminance / contrast / white balance), so
   a straight composite shows a seam at the mask border.

   Outside the mask, the source (regen) and destination (original) are the SAME
   scene content, so the difference between them there is purely that grading
   drift -- true paired samples. We fit a per-channel transform
   ``original ~= a * generated + b`` on those surround pixels and apply it to the
   *source* inside the mask. The element keeps its own colors (lava stays lava);
   only its grading is shifted to match the plate, so the border lines up.

   This needs the source to be a full-frame regeneration of the SAME shot as the
   destination (aligned content outside the mask). It does NOT need color-matcher.

2. Color match (palette)
   ------------------------------------------------------------------
   The original distribution-based approach (Kijai `color-matcher`): recolor one
   image so its color *distribution* matches the other. Good for matching the
   look of two different plates; NOT suitable for the VFX-insert case above
   (it matches content statistics, so it would tint the insert toward the
   surrounding scene -- e.g. lava turning green).

Compositing is always "source over destination": out = dest*(1-mask) + src*mask.

Wiring for grade match: source = the regenerated frame (with the new element),
destination = the original footage, mask = the region to keep from the source.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

try:
    from comfy.utils import common_upscale
except Exception:  # pragma: no cover - fallback if comfy isn't importable at import time
    common_upscale = None


# Minimum number of sampled pixels before a region selection is trusted; below
# this we fall back (whole frame for color match, untouched target for grade
# match) to avoid degenerate statistics (e.g. a 1px sliver of surround).
_MIN_REGION_PIXELS = 16

_METHODS = ["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"]


def _resize_bhwc(img, height, width):
    """Resize a (B,H,W,C) tensor to (B,height,width,C)."""
    if img.shape[1] == height and img.shape[2] == width:
        return img
    chw = img.movedim(-1, 1)
    if common_upscale is not None:
        chw = common_upscale(chw, width, height, "bilinear", "center")
    else:
        chw = F.interpolate(chw, size=(height, width), mode="bilinear", align_corners=False)
    return chw.movedim(1, -1)


def _pick(t, i):
    """Per-frame index with broadcast(1)->all and clamp-to-last for mismatches."""
    return t[min(i, t.shape[0] - 1)]


def _dilate(binary_hw, band):
    """Morphological dilation of a (H,W) {0,1} float mask by ``band`` px."""
    k = 2 * band + 1
    d = F.max_pool2d(binary_hw[None, None], kernel_size=k, stride=1, padding=band)
    return d[0, 0] > 0.5


def _fit_channel(g, o, fit):
    """Fit ``o ~= a*g + b`` for one channel (1-D tensors) with a robust refine.
    Returns (a, b) as scalar tensors."""
    gm = g.mean()
    om = o.mean()
    if fit == "Offset only":
        return torch.tensor(1.0), om - gm

    var = ((g - gm) ** 2).mean()
    if var < 1e-8:  # flat surround -> can't estimate gain, fall back to offset
        return torch.tensor(1.0), om - gm
    a = ((g - gm) * (o - om)).mean() / var
    b = om - a * gm

    # Robust refine: drop residual outliers (moving content / occlusions in the
    # surround that aren't part of the static grading drift), then refit.
    resid = o - (a * g + b)
    sd = resid.std()
    if sd > 1e-8:
        keep = resid.abs() <= 2.5 * sd
        if int(keep.sum()) >= _MIN_REGION_PIXELS:
            g2, o2 = g[keep], o[keep]
            gm2, om2 = g2.mean(), o2.mean()
            var2 = ((g2 - gm2) ** 2).mean()
            if var2 >= 1e-8:
                a = ((g2 - gm2) * (o2 - om2)).mean() / var2
                b = om2 - a * gm2

    return a.clamp(0.2, 5.0), b


class ComposeColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "mask": ("MASK",),
                "correction": (
                    ["Grade match (surround)", "Color match (palette)", "Off"],
                    {"default": "Grade match (surround)"},
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                # --- Grade match (surround) settings ---
                "grade_fit": (["Gain + offset", "Offset only"], {"default": "Gain + offset"}),
                "surround_band": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                # --- Color match (palette) settings ---
                "colormatch_reference": (["Source", "Destination"], {"default": "Destination"}),
                "method": (_METHODS, {"default": "mkl"}),
                "match_region": (
                    ["Whole frame", "Masked region", "Outside masked region"],
                    {"default": "Whole frame"},
                ),
                "recolor_region": (["Full frame", "Source", "Destination"], {"default": "Source"}),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose"
    CATEGORY = "Mickmumpitz/Image"
    DESCRIPTION = (
        "Composite source over destination via a mask, with a per-frame correction "
        "so the inserted region matches the plate. 'Grade match (surround)' estimates "
        "the VAE/grading drift from the unchanged area outside the mask (where source "
        "& destination share content) and applies it to the source inside the mask -- "
        "fixes seam luminance/contrast without changing the element's colors. "
        "'Color match (palette)' is the distribution-based color-matcher approach."
    )

    def compose(
        self,
        destination,
        source,
        mask,
        correction,
        strength,
        grade_fit,
        surround_band,
        colormatch_reference,
        method,
        match_region,
        recolor_region,
        multithread,
    ):
        # Canvas is the destination size; source & mask are resized to match.
        dest = destination[..., :3].cpu().float()
        _, height, width, _ = dest.shape

        src = source[..., :3].cpu().float()
        src = _resize_bhwc(src, height, width)

        # MASK -> (B,H,W,1) on the destination canvas.
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        m = mask.cpu().float()
        m = _resize_bhwc(m.unsqueeze(-1), height, width)  # (B,H,W,1)

        active = correction != "Off" and strength != 0
        if active and correction == "Color match (palette)":
            try:
                from color_matcher import ColorMatcher  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "Can't import color-matcher. Install with: pip install color-matcher"
                ) from e

        n = max(dest.shape[0], src.shape[0], m.shape[0])
        if len({dest.shape[0], src.shape[0], m.shape[0]} - {1}) > 1:
            logging.warning(
                "ComposeColorMatch: mismatched frame counts "
                f"(destination={dest.shape[0]}, source={src.shape[0]}, mask={m.shape[0]}); "
                f"producing {n} frames (clamping shorter inputs to their last frame)."
            )

        def process(i):
            d = _pick(dest, i)  # (H,W,3) destination / original plate
            s = _pick(src, i)   # (H,W,3) source / inserted element
            mi = _pick(m, i)    # (H,W,1)

            if active:
                if correction == "Grade match (surround)":
                    # Correct the source's grading to the plate using the
                    # paired surround, then composite its masked region.
                    s = self._grade_match(s, d, mi, grade_fit, surround_band, strength)
                elif colormatch_reference == "Source":
                    # recolor destination to match the source frame
                    recolored = self._match_frame(d, s, mi, method, strength, match_region)
                    d = self._apply_region(d, recolored, mi, recolor_region)
                else:  # "Destination": recolor source to match the destination frame
                    recolored = self._match_frame(s, d, mi, method, strength, match_region)
                    s = self._apply_region(s, recolored, mi, recolor_region)

            # source over destination
            return d * (1.0 - mi) + s * mi

        if multithread and n > 1:
            max_threads = min(os.cpu_count() or 1, n)
            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                out = list(ex.map(process, range(n)))
        else:
            out = [process(i) for i in range(n)]

        result = torch.stack(out, dim=0).to(torch.float32).clamp_(0.0, 1.0)
        return (result,)

    # ------------------------------------------------------------------ #
    # Grade match (surround) -- paired differential grading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _grade_match(target, reference, mask_hw1, fit, band, strength):
        """Shift ``target`` (H,W,3) grading to match ``reference`` (H,W,3),
        estimating a per-channel affine from pixels OUTSIDE the mask (where the
        two images share content). ``band`` > 0 restricts the estimate to a ring
        of that width around the mask; 0 uses the whole outside region."""
        m2 = mask_hw1[..., 0]  # (H,W)
        outside = m2 <= 0.5
        if band > 0:
            ring = _dilate((m2 > 0.5).float(), band) & outside
            sel = ring if int(ring.sum()) >= _MIN_REGION_PIXELS else outside
        else:
            sel = outside

        if int(sel.sum()) < _MIN_REGION_PIXELS:
            return target  # not enough surround to estimate the drift

        g = target[sel]     # (K,3) generated / source
        o = reference[sel]  # (K,3) original / destination

        a = torch.ones(3)
        b = torch.zeros(3)
        for c in range(3):
            a[c], b[c] = _fit_channel(g[:, c], o[:, c], fit)

        corrected = target * a + b
        if strength != 1.0:
            corrected = target + strength * (corrected - target)
        return corrected

    # ------------------------------------------------------------------ #
    # Color match (palette) -- distribution transfer via color-matcher
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_region(original, recolored, mask_hw1, recolor_region):
        """Blend ``recolored`` back onto ``original`` (both H,W,3) limited to the
        chosen region, feathered by the soft mask (H,W,1)."""
        if recolor_region == "Full frame":
            return recolored
        # "Source" writes inside the mask; "Destination" writes outside it.
        w = mask_hw1 if recolor_region == "Source" else (1.0 - mask_hw1)
        return original * (1.0 - w) + recolored * w

    @staticmethod
    def _match_frame(target, reference, mask_hw1, method, strength, match_region):
        """Recolor ``target`` (H,W,3) so its colors match ``reference`` (H,W,3),
        sampling the reference palette according to ``match_region``."""
        from color_matcher import ColorMatcher

        target_np = target.numpy()
        ref_np = reference.numpy()

        if match_region == "Whole frame":
            region_ref = ref_np
        else:
            mask2d = mask_hw1[..., 0].numpy()
            if match_region == "Masked region":
                sel = mask2d > 0.5
            else:  # "Outside masked region"
                sel = mask2d <= 0.5
            pixels = ref_np[sel]  # (K,3)
            if pixels.shape[0] < _MIN_REGION_PIXELS:
                region_ref = ref_np  # fall back to whole frame on degenerate selection
            else:
                region_ref = pixels.reshape(-1, 1, 3)

        try:
            cm = ColorMatcher()
            matched = cm.transfer(src=target_np, ref=region_ref, method=method)
            if strength != 1.0:
                matched = target_np + strength * (matched - target_np)
            return torch.from_numpy(matched).to(torch.float32)
        except Exception as e:  # mirror KJNodes: fall back to the untouched target
            logging.warning(f"ComposeColorMatch color transfer failed: {e}")
            return target


NODE_CLASS_MAPPINGS = {
    "ComposeColorMatch": ComposeColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComposeColorMatch": "Compose + Match (per-frame)",
}
