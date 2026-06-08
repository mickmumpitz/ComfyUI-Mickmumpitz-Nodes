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

Element trims (saturation / black_point)
----------------------------------------
A VAE round-trip tends to slightly desaturate and lift the blacks ("milky")
of the regenerated content. Grade match fixes luminance/contrast/white-balance
from the surround, but the surround is usually less saturated than the inserted
element, so a linear per-channel fit under-restores the element's saturation.
Two optional trims run on the corrected source before compositing:
  - saturation : luma-preserving saturation scale (1.0 = unchanged).
  - black_point: pull lifted blacks back down to de-milk (0.0 = unchanged).
  - edge_falloff: feather the trims inward from the mask edge (px). At the very
    border the trims are 0 (so it matches the plate seamlessly) and ramp to full
    strength ``edge_falloff`` px inside. 0 = uniform (no feather).
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


def _inner_ramp(m_bhw1, falloff):
    """Inner-distance ramp for a (B,H,W,1) soft mask: 0 at the mask edge, rising
    linearly to 1 at ``falloff`` px inside. Built by iterative 3x3 erosion, so a
    pixel's value is its (clamped) depth from the border / falloff. Returns
    (B,H,W,1)."""
    x = (m_bhw1[..., 0] > 0.5).float()[:, None]  # (B,1,H,W)
    acc = torch.zeros_like(x)
    cur = x
    for _ in range(falloff):
        cur = -F.max_pool2d(-cur, kernel_size=3, stride=1, padding=1)  # erosion
        acc = acc + cur
    return (acc / falloff).clamp_(0.0, 1.0).movedim(1, -1)  # (B,H,W,1)


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
                # --- Element trims (applied to the corrected source) ---
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "black_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.001}),
                "edge_falloff": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                # --- Grade match (surround) settings ---
                "grade_fit": (["Gain + offset", "Offset only"], {"default": "Gain + offset"}),
                "surround_band": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                # --- Color match (palette) settings ---
                "colormatch_reference": (["Source", "Destination"], {"default": "Destination"}),
                "method": (_METHODS, {"default": "mkl"}),
                "reference_region": (
                    ["Outside mask", "Full frame", "Inside mask"],
                    {"default": "Outside mask"},
                ),
                "multithread": ("BOOLEAN", {"default": True}),
                # --- Temporal anchor (iteration continuity) ---
                "temporal_anchor": ("BOOLEAN", {"default": False}),
                "anchor_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "anchor_frames": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
            },
            "optional": {
                # Previous iteration's corrected tail (e.g. IterVideoRouter
                # current_start) -- the temporal-anchor reference.
                "previous_frames": ("IMAGE",),
            },
            "hidden": {
                "iteration": "INT",
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
        "'Color match (palette)' is the distribution-based color-matcher approach. "
        "'temporal_anchor' (iterative video) re-matches each iteration's element to "
        "the previous iteration's corrected tail so VAE drift can't compound."
    )

    def compose(
        self,
        destination,
        source,
        mask,
        correction,
        strength,
        saturation,
        black_point,
        edge_falloff,
        grade_fit,
        surround_band,
        colormatch_reference,
        method,
        reference_region,
        multithread,
        temporal_anchor=False,
        anchor_strength=1.0,
        anchor_frames=0,
        previous_frames=None,
        iteration=0,
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

        trims_on = saturation != 1.0 or black_point > 0.0
        # Inner-distance ramp so the trims fade to 0 at the mask border (no seam).
        edge_w = _inner_ramp(m, edge_falloff) if (trims_on and edge_falloff > 0) else None

        # Temporal anchor: re-match this iteration's corrected element to the
        # previous iteration's corrected tail so VAE drift (paleness / lifted
        # blacks) can't compound across iterations. Only on iteration > 0 with a
        # reference wired in (e.g. IterVideoRouter's current_start).
        anchor_active = (
            temporal_anchor and anchor_strength != 0
            and previous_frames is not None and iteration > 0
        )
        ref = None
        if anchor_active:
            ref = previous_frames[..., :3].cpu().float()
            ref = _resize_bhwc(ref, height, width)
            if ref.shape[0] == 0:
                anchor_active = False

        n = max(dest.shape[0], src.shape[0], m.shape[0])
        if len({dest.shape[0], src.shape[0], m.shape[0]} - {1}) > 1:
            logging.warning(
                "ComposeColorMatch: mismatched frame counts "
                f"(destination={dest.shape[0]}, source={src.shape[0]}, mask={m.shape[0]}); "
                f"producing {n} frames (clamping shorter inputs to their last frame)."
            )

        def compute_element(i):
            """Correct one frame's source down to (d, element, mask) -- everything
            up to but not including the final composite, so the temporal anchor can
            be fit on the corrected elements before they're composited."""
            d = _pick(dest, i)  # (H,W,3) destination / original plate
            s = _pick(src, i)   # (H,W,3) source / inserted element
            mi = _pick(m, i)    # (H,W,1)

            if active:
                if correction == "Grade match (surround)":
                    # Correct the source's grading to the plate using the paired surround.
                    s = self._grade_match(s, d, mi, grade_fit, surround_band, strength)
                elif colormatch_reference == "Source":
                    # reference = source -> recolor the destination (the "other" image)
                    d = self._match_frame(d, s, mi, method, strength, reference_region)
                else:  # reference = destination -> recolor the source
                    s = self._match_frame(s, d, mi, method, strength, reference_region)

            # Polish the inserted element (restore saturation / de-milk) before comp.
            if trims_on:
                ew = _pick(edge_w, i) if edge_w is not None else None
                s = self._trim(s, saturation, black_point, ew)

            return d, s, mi

        if multithread and n > 1:
            max_threads = min(os.cpu_count() or 1, n)
            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                elems = list(ex.map(compute_element, range(n)))
        else:
            elems = [compute_element(i) for i in range(n)]

        # Re-anchor the whole iteration's element to the previous tail (one affine,
        # fit on the content-paired overlap, applied to every frame).
        if anchor_active:
            a, b = self._anchor_fit(elems, ref, anchor_frames)
            if a is not None:
                for i in range(n):
                    d, s, mi = elems[i]
                    corrected = s * a + b
                    if anchor_strength != 1.0:
                        corrected = s + anchor_strength * (corrected - s)
                    elems[i] = (d, corrected, mi)

        # source over destination
        out = [d * (1.0 - mi) + s * mi for (d, s, mi) in elems]
        result = torch.stack(out, dim=0).to(torch.float32).clamp_(0.0, 1.0)
        return (result,)

    # ------------------------------------------------------------------ #
    # Element trims -- counteract VAE desaturation / lifted blacks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trim(img, saturation, black_point, edge_w=None):
        """Polish (H,W,3): lift the black point (de-milk), then scale saturation
        about luma (hue/grading preserving). ``edge_w`` (H,W,1 or None) feathers
        both trims toward identity where it is 0. Output is clamped downstream."""
        w = 1.0 if edge_w is None else edge_w
        out = img
        if black_point > 0.0:
            bp = black_point * w  # per-pixel black point (0 at the mask edge)
            out = (out - bp) / (1.0 - bp)
        if saturation != 1.0:
            luma_w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=out.dtype)
            luma = (out * luma_w).sum(dim=-1, keepdim=True)
            sat = 1.0 + (saturation - 1.0) * w  # 1.0 (no change) at the mask edge
            out = luma + sat * (out - luma)
        return out

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
    # Temporal anchor -- iteration-to-iteration continuity
    # ------------------------------------------------------------------ #
    @staticmethod
    def _anchor_fit(elems, ref, anchor_frames):
        """Match the current iteration's corrected element distribution to the
        previous iteration's corrected tail, sampled INSIDE the mask.

        Unlike grade match (which regresses paired pixels of the SAME plate), the
        anchor compares two INDEPENDENT generations of the element -- the current
        head vs the previous tail. WAN doesn't reproduce the start frames
        pixel-for-pixel, so per-pixel correspondence is unreliable; a covariance
        fit would collapse and wash the element out. Instead we match per-channel
        mean and std (distribution transfer), which is alignment-free and restores
        saturation/contrast via the std ratio.

        ``elems`` is the list of (d, element, mask) tuples for this iteration;
        ``ref`` is (R,H,W,3), the previous corrected tail. Element pixels are
        pooled across the overlap frames (the last ``ov`` of ``ref``, the first
        ``ov`` of the iteration). Returns (a, b) as (3,) tensors mapping
        ``a*current + b`` toward the reference, or (None, None) when there aren't
        enough masked pixels to trust."""
        r = ref.shape[0]
        n = len(elems)
        ov = anchor_frames if anchor_frames > 0 else r
        ov = min(ov, r, n)
        if ov < 1:
            return None, None

        g_parts, o_parts = [], []
        for k in range(ov):
            _, s_k, m_k = elems[k]
            o_k = ref[r - ov + k]            # paired previous-tail frame
            sel = m_k[..., 0] > 0.5          # the element (only region in the output)
            if int(sel.sum()) == 0:
                continue
            g_parts.append(s_k[sel])         # current corrected element pixels
            o_parts.append(o_k[sel])         # previous corrected element pixels (target)

        if not g_parts:
            return None, None
        g = torch.cat(g_parts, dim=0)
        o = torch.cat(o_parts, dim=0)
        if g.shape[0] < _MIN_REGION_PIXELS:
            return None, None

        # Per-channel distribution (mean/std) transfer -- robust to the spatial
        # mismatch between two independent generations of the same element.
        gm, om = g.mean(dim=0), o.mean(dim=0)
        gs, os_ = g.std(dim=0), o.std(dim=0)
        a = torch.ones(3)
        b = torch.zeros(3)
        for c in range(3):
            if gs[c] > 1e-5:
                a[c] = (os_[c] / gs[c]).clamp(0.2, 5.0)
            b[c] = om[c] - a[c] * gm[c]
        return a, b

    # ------------------------------------------------------------------ #
    # Color match (palette) -- distribution transfer via color-matcher
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_frame(target, reference, mask_hw1, method, strength, reference_region):
        """Recolor ``target`` (H,W,3) so its colors match ``reference`` (H,W,3),
        sampling the reference palette according to ``reference_region``."""
        from color_matcher import ColorMatcher

        target_np = target.numpy()
        ref_np = reference.numpy()

        if reference_region == "Full frame":
            region_ref = ref_np
        else:
            mask2d = mask_hw1[..., 0].numpy()
            if reference_region == "Inside mask":
                sel = mask2d > 0.5
            else:  # "Outside mask"
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
