"""
Compose + Color Match (per-frame) Node for ComfyUI
==================================================
A fused node that composites a *source* over a *destination* using a *mask*,
while optionally color-matching the two **frame by frame** before compositing.

Why a single fused node instead of stacking ImageComposite + ColorMatch?
Stacking a stand-alone color-match node matches a whole batch against a single
reference (or whole-batch statistics). Here, destination frame ``i`` is matched
to the *corresponding* source frame ``i``, which is what you want when both
inputs are video and you're testing whether per-frame matching gives a cleaner
composite.

Color matching reuses Kijai's `color-matcher` based ColorMatch behaviour
(same method list, ``strength`` and ``multithread`` semantics).

Attributes
----------
colormatch_reference : Source | Destination | Off
    Names the *color reference*; the OTHER image is the one that gets recolored.
      - Source      -> recolor the DESTINATION to match the source frame.
      - Destination -> recolor the SOURCE to match the destination frame.
      - Off         -> no color matching, just composite.
method : color-matcher transfer method (mkl, hm, reinhard, mvgd, ...).
strength : 0..10 blend between the original target and the matched result
    (same as KJNodes: ``out = target + strength * (matched - target)``).
match_region : Whole frame | Masked region | Outside masked region
    Selects which pixels of the *reference* image define the target palette.
      - Whole frame          -> use the full reference frame (KJNodes behaviour).
      - Masked region        -> only reference pixels where mask > 0.5.
      - Outside masked region-> only reference pixels where mask <= 0.5.
recolor_region : Full frame | Source | Destination
    Selects *where* the recolor is written onto the target frame (feathered by
    the soft mask), before compositing.
      - Full frame  -> recolor the whole target frame.
      - Source      -> only the masked region (where the source ends up).
      - Destination -> only the outside-mask region (where the destination shows).
multithread : process frames concurrently with a thread pool.

Compositing is "source over destination": result = dest*(1-mask) + source*mask.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import torch

try:
    from comfy.utils import common_upscale
except Exception:  # pragma: no cover - fallback if comfy isn't importable at import time
    common_upscale = None


# Minimum number of sampled reference pixels before a region selection is
# trusted; below this we fall back to the whole frame to avoid degenerate
# colour statistics (e.g. a 1px sliver of mask).
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
        chw = torch.nn.functional.interpolate(
            chw, size=(height, width), mode="bilinear", align_corners=False
        )
    return chw.movedim(1, -1)


def _pick(t, i):
    """Per-frame index with broadcast(1)->all and clamp-to-last for mismatches."""
    return t[min(i, t.shape[0] - 1)]


class ComposeColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "mask": ("MASK",),
                "colormatch_reference": (["Source", "Destination", "Off"], {"default": "Source"}),
                "method": (_METHODS, {"default": "mkl"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "match_region": (
                    ["Whole frame", "Masked region", "Outside masked region"],
                    {"default": "Whole frame"},
                ),
                "recolor_region": (
                    ["Full frame", "Source", "Destination"],
                    {"default": "Source"},
                ),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose"
    CATEGORY = "Mickmumpitz/Image"
    DESCRIPTION = (
        "Composite source over destination via a mask, while optionally "
        "color-matching the two FRAME BY FRAME (destination frame i matched to "
        "source frame i) before compositing. Color matching reuses color-matcher "
        "(same methods/strength as KJNodes ColorMatch)."
    )

    def compose(
        self,
        destination,
        source,
        mask,
        colormatch_reference,
        method,
        strength,
        match_region,
        recolor_region,
        multithread,
    ):
        # Canvas is the destination size; source & mask are resized to match.
        dest = destination[..., :3].cpu().float()
        b, height, width, _ = dest.shape

        src = source[..., :3].cpu().float()
        src = _resize_bhwc(src, height, width)

        # MASK -> (B,H,W,1) on the destination canvas.
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        m = mask.cpu().float()
        m = _resize_bhwc(m.unsqueeze(-1), height, width)  # (B,H,W,1)

        do_match = colormatch_reference != "Off" and strength != 0
        if do_match:
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
            d = _pick(dest, i)  # (H,W,3)
            s = _pick(src, i)
            mi = _pick(m, i)  # (H,W,1)

            if do_match:
                if colormatch_reference == "Source":
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
    "ComposeColorMatch": "Compose + Color Match (per-frame)",
}
