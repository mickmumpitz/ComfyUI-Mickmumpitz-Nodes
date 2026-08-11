"""MiniMax H3 Resolution Node for ComfyUI

H3 does not letterbox. There is no canvas it falls back to: the sampled frame
is whatever `width` / `height` say, while every reference image and video
carries its *own* spatial grid into the same packed sequence. Both grids are
area-normalised - each axis covers `(dim / sqrt(w * h)) * 32` units of RoPE
space, centred on 16. Two consequences run this node:

* **Aspect ratio decides framing.** Target and reference register only when
  their canvases have the same ratio. A wider target reaches past the
  reference's grid, a narrower one sees a centre crop. That is the zoom people
  report when they type a resolution H3 did not pick.
* **Absolute resolution moves nothing.** The extent stays `ratio * 32` whatever
  the pixel count; a bigger canvas samples the same field of view more densely.
  Nothing extrapolates, so the canvas is free to scale - as long as the ratio
  is held.

So: ratio from the clay pass, size from a multiplier, and never the two
confused. 1.0x is `adapt_canvas`, the size H3 was trained on and the size it
gives the reference; anything else is a deliberate step away from it.

The ratio is taken from the reference's *encoded* canvas, not its file size. A
pass below the native pixel budget is not upscaled by H3, only rounded to 32,
and rounding a small pass moves its ratio: 640x360 encodes as 640x352, ratio
1.818, where `adapt_canvas` would have asked for 1.75. That gap is a 4 %
reframe nobody ordered.
"""

import math

# comfy_extras/nodes_minimax_h3.py
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

MIN_UNITS = 4                     # 128 px, below this nothing survives the VAE
MAX_UNITS = 160                   # 5120 px per axis, a guard rail not a target

SCALES = {
    "0.5x  draft": 0.5,
    "0.75x  preview": 0.75,
    "1.0x  native (trained)": 1.0,
    "1.25x": 1.25,
    "1.5x": 1.5,
    "2.0x": 2.0,
    "Custom": None,
}
SCALE_LIST = list(SCALES.keys())

# Ratios go through adapt_canvas before they are used, so a preset lands on the
# canvas H3 itself would pick for that format - 16:9 becomes 1.75, not 1.7778.
ASPECTS = {
    "From reference": None,
    "16:9": (16, 9),
    "9:16 (Vertical)": (9, 16),
    "1:1 (Square)": (1, 1),
    "4:3": (4, 3),
    "3:4 (Vertical)": (3, 4),
    "2.39:1 (Scope)": (239, 100),
    "Custom": None,
}
ASPECT_LIST = list(ASPECTS.keys())


def adapt_canvas(width, height):
    """H3's own canvas rule: 768 short edge, 768*1344 area cap, round to 32.

    Reimplemented rather than imported so the node still loads on an
    installation without the H3 extra. Checked against
    `comfy_extras.nodes_minimax_h3.adapt_canvas`.
    """
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def nominal_pixels(ratio):
    """H3's pixel budget at a ratio, before any rounding.

    `adapt_canvas` rounds to 32 at the end, so feeding its own output back into
    it does not return that output - 1568x640 comes back as 1600x640. Taking
    the budget unrounded and rounding once, in `fit_ratio`, keeps 1.0x equal to
    the canvas H3 picks instead of one rounding step away from it.
    """
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    return min(nom_w * nom_h, float(MAX_PIXELS))


def reference_canvas(width, height):
    """The canvas H3 gives a reference *video*: adapt_canvas, but never upwards.

    `MiniMaxH3ReferenceToVideo` falls back to the plain rounded source size
    whenever the source has fewer pixels than the adapted canvas would.
    """
    cw, ch = adapt_canvas(width, height)
    if width * height < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return cw, ch


def _around(units):
    # Two units of slack per axis, deliberately narrow. A wider window would
    # sometimes find a mathematically exact ratio, but only by giving up 15 %
    # of the pixels for a reframe of half a percent - which is a few pixels
    # across the whole frame and cannot be seen.
    lo = max(MIN_UNITS, int(math.floor(units)) - 2)
    hi = min(MAX_UNITS, int(math.ceil(units)) + 2)
    return range(lo, max(lo, hi) + 1)


def fit_ratio(ratio, target_pixels):
    """Closest 32-px canvas to `target_pixels` at `ratio`. Ratio first.

    Both axes stay multiples of 32, which keeps the latent dimensions even and
    the DiT's 2x2 patchify free of padding. Rounding each axis on its own would
    bend the ratio, and the ratio is the one thing that must not bend - so the
    pair is searched instead. Nearest ratio wins, nearest area breaks the tie.
    """
    ideal_h = math.sqrt(target_pixels / ratio) / CANVAS_MULTIPLE
    ideal_w = ideal_h * ratio
    best = None
    for b in _around(ideal_h):
        for a in _around(ideal_w):
            err_ratio = abs(math.log((a / b) / ratio))
            err_area = abs(a * b * CANVAS_MULTIPLE ** 2 - target_pixels)
            key = (round(err_ratio, 6), err_area)
            if best is None or key < best[0]:
                best = (key, a, b)
    return best[1] * CANVAS_MULTIPLE, best[2] * CANVAS_MULTIPLE


class MinimaxH3Resolution:
    """Canvas for MiniMax H3: ratio from the clay pass, size from a multiplier."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_LIST, {"default": "From reference"}),
                "scale": (SCALE_LIST, {"default": "1.0x  native (trained)"}),
                "custom_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0,
                                           "step": 0.05}),
                "custom_aspect_width": ("INT", {"default": 16, "min": 1, "max": 10000}),
                "custom_aspect_height": ("INT", {"default": 9, "min": 1, "max": 10000}),
            },
            "optional": {
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info")
    FUNCTION = "get_resolution"
    CATEGORY = "Mickmumpitz/utils"
    DESCRIPTION = ("Canvas for MiniMax H3. Aspect ratio from the clay pass, size "
                   "from a multiplier on H3's own canvas. 1.0x is what the model "
                   "was trained on; H3 reframes rather than letterboxes, so the "
                   "ratio is held whatever the multiplier.")

    def get_resolution(self, aspect_ratio, scale, custom_scale,
                       custom_aspect_width, custom_aspect_height, reference=None):
        lines = []
        ref_canvas = None

        if reference is not None:
            src_h, src_w = int(reference.shape[1]), int(reference.shape[2])
            ref_canvas = reference_canvas(src_w, src_h)
            lines.append(f"reference   {src_w}x{src_h}  ->  H3 encodes it as "
                         f"{ref_canvas[0]}x{ref_canvas[1]}")
            if ref_canvas != adapt_canvas(src_w, src_h):
                lines.append(f"NOTE  the pass is below H3's pixel budget, so it is "
                             f"only rounded to 32 and its ratio shifts to "
                             f"{ref_canvas[0] / ref_canvas[1]:.4f}. Feed at least "
                             f"{MAX_PIXELS // 1000} k pixels to avoid that.")

        if aspect_ratio == "From reference" and ref_canvas is not None:
            anchor = ref_canvas
        else:
            if aspect_ratio == "From reference":
                lines.append("no reference connected, falling back to the custom ratio")
            if aspect_ratio in ("From reference", "Custom"):
                aw, ah = custom_aspect_width, custom_aspect_height
            else:
                aw, ah = ASPECTS[aspect_ratio]
            anchor = adapt_canvas(aw, ah)

        ratio = anchor[0] / anchor[1]
        budget = nominal_pixels(ratio)
        native = fit_ratio(ratio, budget)
        factor = custom_scale if scale == "Custom" else SCALES[scale]
        width, height = fit_ratio(ratio, budget * factor * factor)

        lines.append(f"native 1.0x  {native[0]}x{native[1]}")
        lines.append(f"canvas       {width}x{height}   {width * height / 1e6:.2f} MP"
                     f"   ratio {width / height:.4f}   {factor:g}x")

        if ref_canvas is not None:
            off = abs(math.log((width / height) / (ref_canvas[0] / ref_canvas[1])))
            pct = (math.exp(off) - 1) * 100
            # Under a percent is less than H3's own rounding does to a 16:9 pass
            # on the way to 1.75, so it is reported but not flagged.
            lines.append(f"reframe vs reference   {pct:.2f} %")
            if pct > 1.0:
                lines.append("WARNING  H3 reframes, it does not letterbox. "
                             "Expect a crop or a zoom, not black bars.")
        if factor > 1.0:
            lines.append(f"above the trained canvas: {factor * factor:.2f}x the pixels, "
                         f"and attention grows with the square of that")

        return (width, height, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "MinimaxH3Resolution": MinimaxH3Resolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3Resolution": "MiniMax H3 Resolution",
}
