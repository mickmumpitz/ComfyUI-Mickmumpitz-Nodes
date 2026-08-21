"""Panorama tools: turn one ordinary photo into a partial equirectangular (360) canvas and
clean up the result.

Six nodes:

  PerspToErpWarp     -- photo -> ERP canvas + validity mask + inpaint mask + a technical
                        equirect reference chart.
  EstimateFOV        -- recover hFOV + pitch from image geometry (vanishing points, no EXIF),
                        to feed the warp above.
  SeamRoll           -- roll a pano by half its width so the left/right wrap seam moves to
                        the centre, with a centred seam mask for inpainting. Its own inverse.
  StageSwitch        -- lazy on/off bypass: pick one of two images; the unselected branch is
                        pruned from the graph instead of computed and discarded.
  HarmonizeBoundary  -- bend the generated surroundings to meet the placed photo's colour /
                        brightness with no visible step (harmonic / Poisson-style solve).
  UnfilledMask       -- mask the near-black regions an outpaint pass left unfilled (poles) so
                        a second pass can finish them.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import fov_estimate as fe

CATEGORY = "Mickmumpitz/Panorama"


def _parse_color(s, default=(0.0, 0.0, 0.0)):
    """'#RRGGBB' / '#RGB' / 'r,g,b' (0-255) -> (r,g,b) floats in [0,1]. Unparseable input
    falls back to `default` with a warning rather than failing a queued render."""
    t = str(s).strip()
    if not t:
        return default
    if "," in t:
        try:
            parts = [float(p) for p in t.split(",")]
        except ValueError:
            parts = []
        if len(parts) == 3:
            return tuple(min(max(p, 0.0), 255.0) / 255.0 for p in parts)
    h = t.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) == 6:
        try:
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            pass
    print(f"[Mickmumpitz/Pano] unparseable fill_color {s!r} -> using default")
    return default


def _erp_chart(W, H, valid_np, grid_deg=10, floor=True):
    """Render a technical equirectangular reference chart (no source image):
    lat/lon grid + degree labels, floor+ceiling perspective grids (how a horizontal
    plane's grid curves into equirect, converging at the horizon), and the dotted FOV
    footprint outline (taken from the warp's ``valid`` mask so it matches exactly).
    Returns an (H,W,3) float32 image in [0,1]. Feeds a generator as a pure geometry hint.
    """
    img = np.full((H, W, 3), 255, np.uint8)

    def px(lon, lat):
        return (lon + math.pi) / (2 * math.pi) * W, (math.pi / 2 - lat) / math.pi * H

    def dir2px(d):
        n = d / (np.linalg.norm(d) + 1e-9)
        lon = math.atan2(n[0], n[2])
        lat = math.asin(max(-1.0, min(1.0, n[1])))
        return px(lon, lat)

    def draw_wrapped(pts, color, thick=1, dotted=False):
        seg, prev = [], None
        def flush(s):
            if len(s) < 2:
                return
            arr = np.array(s, np.int32)
            if dotted:
                for i in range(0, len(arr) - 1, 2):
                    cv2.line(img, tuple(arr[i]), tuple(arr[i + 1]), color, thick, cv2.LINE_AA)
            else:
                cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)
        for (x, y) in pts:
            if prev is not None and abs(x - prev) > W * 0.5:   # crossed the +/-180 wrap
                flush(seg)
                seg = []
            seg.append((int(round(x)), int(round(y))))
            prev = x
        flush(seg)

    light, mid, dark, lab = (218, 218, 218), (165, 165, 165), (95, 95, 95), (120, 120, 120)

    # --- lat/lon grid ---
    for deg in range(-180, 181, grid_deg):
        x = int((deg + 180) / 360.0 * W)
        cv2.line(img, (x, 0), (x, H - 1), dark if deg in (-180, -90, 0, 90, 180) else light, 1, cv2.LINE_AA)
    for deg in range(-90, 91, grid_deg):
        y = int((90 - deg) / 180.0 * H)
        cv2.line(img, (0, y), (W - 1, y), dark if deg == 0 else light, 1, cv2.LINE_AA)

    # --- floor (+ ceiling) perspective grids: horizontal plane at Y=-1 (and +1) ---
    if floor:
        ext, cell, nsamp = 20.0, 2.0, 260
        ks = np.arange(-ext, ext + 0.001, cell)
        zs = np.linspace(-ext, ext, nsamp)
        for sign in (-1.0, 1.0):
            for k in ks:
                draw_wrapped([dir2px(np.array([k, sign, z])) for z in zs], mid, 1)
                draw_wrapped([dir2px(np.array([x0, sign, k])) for x0 in zs], mid, 1)

    # --- FOV footprint outline (dotted) from the warp valid mask ---
    if valid_np is not None and valid_np.any():
        cnts, _ = cv2.findContours(valid_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 8:
                continue
            step = max(1, len(c) // 400)
            draw_wrapped([(p[0][0], p[0][1]) for p in c[::step]], (35, 35, 35), 2, dotted=True)

    # --- degree labels ---
    for deg in range(-180, 181, 30):
        x = int((deg + 180) / 360.0 * W)
        cv2.putText(img, str(deg), (min(x + 3, W - 26), H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lab, 1, cv2.LINE_AA)
    for deg in range(-80, 81, 20):
        y = int((90 - deg) / 180.0 * H)
        cv2.putText(img, str(deg), (5, max(y - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lab, 1, cv2.LINE_AA)

    return img[..., ::-1].astype(np.float32) / 255.0   # BGR(cv2)->RGB


class PerspToErpWarp:
    """Start image -> perspective->ERP projected canvas + validity mask (N-frame batch).

    Geometrically correct pinhole->equirectangular forward warp: the rectilinear start
    image is placed on a 2:1 ERP canvas (unknown region = fill_color, black by default)
    with the proper equirect distortion (straight lines curve, angles compress toward the
    edges), as a real 360 capture of that view would look. Centered on the forward
    direction (+Z = center column), +X right, +Y up. Mask convention: white = known,
    black = hole to generate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1408, "min": 64, "max": 8192, "step": 16,
                    "tooltip": "ERP canvas width. Keep 2:1 (width = 2*height) for a full pano."}),
                "height": ("INT", {"default": 704, "min": 32, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 1, "min": 1, "max": 257, "step": 4,
                    "tooltip": "Frames in the still batch (N = 1+4k). 1 = single ERP still "
                               "(fastest); 5 gives a video model a little temporal extent."}),
                "h_fov_deg": ("FLOAT", {"default": 70.0, "min": 10.0, "max": 170.0, "step": 0.5,
                    "tooltip": "HORIZONTAL field of view of the source photo, in degrees. Photos "
                               "do not carry this, so it is set here. Too small places the photo "
                               "too tightly, too large smears it across the canvas. Match your "
                               "camera: phone main ~70, wide ~90, ultrawide ~110-120."}),
            },
            "optional": {
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Where on the sphere the photo points. 0 is canvas centre, "
                               "180 is the wrap seam."}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Camera tilt. 0 for a level shot. If the horizon comes out "
                               "wrong, this is the value to change first."}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Feather (px) on the inpaint_mask edge, ramping INTO the known "
                               "side only. Keep at 0 for models trained on hard edges; a ramp "
                               "there is content mixed towards the fill colour."}),
                "supersample": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5,
                    "tooltip": "Anti-alias quality. The source is heavily minified into the ERP "
                               "footprint; this area-prefilters it to footprint x this factor "
                               "before a bicubic warp, killing aliasing/moire. 2 is a good default, "
                               "3-4 for very high-res sources, 1 = off (old bilinear-ish)."}),
                "grid_spacing_deg": ("INT", {"default": 10, "min": 5, "max": 90, "step": 5,
                    "tooltip": "Lat/lon grid spacing (deg) on the 'erp_guide' chart. The guide is a "
                               "TECHNICAL equirect reference (no photo): lat/lon grid + degree "
                               "labels, floor/ceiling perspective curves, and a dotted FOV box "
                               "marking exactly where the image sits. Feed it to a generator as a "
                               "pure geometry hint for the 360 layout."}),
                "guide_opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = draw only the lat/lon grid + FOV box; >0 = also draw the "
                               "floor + ceiling perspective curves (how a ground/ceiling plane "
                               "distorts into equirect)."}),
                "fill_color": ("STRING", {"default": "#000000",
                    "tooltip": "Colour of the UNKNOWN region on the ERP canvas (everything "
                               "outside the photo's footprint). Hex '#RRGGBB' / '#RGB', or "
                               "'r,g,b' with 0-255 components. Default black. This only tints "
                               "control_video -- the masks are unchanged, so an inpainting "
                               "sampler still regenerates the region regardless. Handy when a "
                               "model reacts badly to black init (try '#808080' mid-grey)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("control_video", "control_mask", "inpaint_mask", "erp_guide")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, image, width, height, length, h_fov_deg, yaw_deg=0.0, pitch_deg=0.0,
              mask_feather=0, supersample=2.0, grid_spacing_deg=10, guide_opacity=0.55,
              fill_color="#000000"):
        W, H = int(width), int(height)
        src = image[0].permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)  # [1,3,sh,sw]
        sh, sw = int(src.shape[2]), int(src.shape[3])

        # Pinhole intrinsics from the horizontal FOV (square pixels -> same f vertically,
        # so the vertical FOV is implied by the source aspect ratio).
        f = (sw / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
        cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0

        # Per-ERP-pixel viewing ray. Center column -> lon 0 = +Z forward; top row -> +Y up.
        xs = torch.arange(W, dtype=torch.float32)
        ys = torch.arange(H, dtype=torch.float32)
        lon = (xs + 0.5) / W * (2 * math.pi) - math.pi           # [-pi, pi)
        lat = (math.pi / 2.0) - (ys + 0.5) / H * math.pi         # [+pi/2, -pi/2]
        lat_g, lon_g = torch.meshgrid(lat, lon, indexing="ij")   # [H,W]
        cl = torch.cos(lat_g)
        dx = cl * torch.sin(lon_g)
        dy = torch.sin(lat_g)
        dz = cl * torch.cos(lon_g)

        # Rotate the rays by -(yaw,pitch) so the camera looks at (yaw,pitch) instead of +Z.
        ay, ap = math.radians(yaw_deg), math.radians(pitch_deg)
        if ay:
            ca, sa = math.cos(ay), math.sin(ay)
            dx, dz = ca * dx - sa * dz, sa * dx + ca * dz        # yaw about +Y
        if ap:
            cp, sp = math.cos(ap), math.sin(ap)
            dy, dz = cp * dy - sp * dz, sp * dy + cp * dz        # pitch about +X

        # Project rays in front of the camera into source pixels (+Y up -> pixel y down).
        front = dz > 1e-6
        dzc = torch.where(front, dz, torch.ones_like(dz))
        xf = f * (dx / dzc) + cx
        yf = cy - f * (dy / dzc)
        valid = front & (xf >= 0) & (xf <= sw - 1) & (yf >= 0) & (yf <= sh - 1)

        # Normalized grid (align_corners=True matches the (size-1) normalization). This is
        # resolution-independent, so it stays valid after the source is prefiltered below.
        gx = 2.0 * xf / max(sw - 1, 1) - 1.0
        gy = 2.0 * yf / max(sh - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)        # [1,H,W,2]

        # Anti-alias prefilter: the source spans h_fov_deg across only ~width*hfov/360 ERP
        # columns, so it is minified by ~sw/that -> a plain bilinear tap aliases badly.
        # Area-downscale the source to the on-ERP footprint (x supersample) FIRST, so the
        # warp resamples near 1:1. Then bicubic for crisp interpolation, border padding so
        # bicubic taps at the frustum edge don't ring into black.
        foot_w = max(1.0, width * (h_fov_deg / 360.0))
        tgt_w = int(max(1, round(foot_w * max(1.0, float(supersample)))))
        src_s = src
        if tgt_w < sw:
            tgt_h = int(max(1, round(tgt_w * sh / sw)))
            src_s = F.interpolate(src, size=(tgt_h, tgt_w), mode="area")
        sampled = F.grid_sample(src_s, grid, mode="bicubic",
                                padding_mode="border", align_corners=True)[0].clamp(0, 1)  # [3,H,W]

        validf = valid.float()
        # Composite the warped view over the fill colour: known pixels keep the photo,
        # everything outside the frustum takes fill_color (black by default).
        fill = torch.tensor(_parse_color(fill_color), dtype=torch.float32)  # [3]
        a = validf.unsqueeze(-1)
        canvas = sampled.permute(1, 2, 0) * a + fill.view(1, 1, 3) * (1.0 - a)
        mask3 = validf.unsqueeze(-1).repeat(1, 1, 3)                     # white = known (IMAGE)

        # Inpaint mask (MASK): white = HOLE to generate = inverse of known. The feather
        # must only ramp INTO the known region -- every true hole stays fully 1 so the
        # sampler never blends the black init latent back in (that caused a dark ring).
        hole_bin = 1.0 - validf                                        # [H,W] {0,1}
        hole = hole_bin
        r = int(mask_feather)
        if r > 0:
            k = 2 * r + 1
            m = hole_bin.view(1, 1, H, W)
            kh = torch.ones(1, 1, 1, k) / k
            kv = torch.ones(1, 1, k, 1) / k
            m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), kh)
            m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), kv)
            hole = torch.maximum(m.view(H, W).clamp(0, 1), hole_bin)   # holes stay 1
        inpaint_mask = hole.unsqueeze(0)                                # [1,H,W] MASK

        # --- ERP guide: a technical equirect chart (NO source image) -- full-sphere lat/lon
        # grid + labels, floor/ceiling perspective curves, and the dotted FOV footprint
        # (from `valid`, so it marks exactly where the image sits). A pure geometry hint. ---
        valid_np = valid.detach().cpu().numpy().astype("uint8")
        guide_np = _erp_chart(W, H, valid_np, grid_deg=max(5, int(grid_spacing_deg)),
                              floor=(float(guide_opacity) > 0.0))
        erp_guide = torch.from_numpy(np.ascontiguousarray(guide_np)).float().unsqueeze(0)  # [1,H,W,3]

        n = max(1, int(length))
        control_video = canvas.unsqueeze(0).repeat(n, 1, 1, 1)   # [N,H,W,3]
        control_mask = mask3.unsqueeze(0).repeat(n, 1, 1, 1)     # [N,H,W,3]
        cov = float(validf.mean().item()) * 100.0
        print(f"[Mickmumpitz/Pano] {n}x {W}x{H} ERP, perspective->equirect warp: "
              f"hFOV={h_fov_deg:.0f} src {sw}x{sh} -> {cov:.1f}% of pano known "
              f"(yaw={yaw_deg:.0f}, pitch={pitch_deg:.0f}), feather={r}px, "
              f"fill={fill_color}; erp_guide chart drawn")
        return (control_video, control_mask, inpaint_mask, erp_guide)


class EstimateFOV:
    """Estimate horizontal FOV + pitch from image geometry (vanishing points, no EXIF).

    DEPRECATED: in practice the estimator misplaces level shots into the nadir and the
    hFOV error can be huge. Prefer typing the FOV into the warp node by hand (phone main
    ~70, wide ~90). Kept only so old workflows still load.

    Wire the estimated h_fov_deg / pitch_deg into the ERP Warp node. Honest fallback: on
    a near-1-point-perspective scene (focal under-determined) it returns fallback_fov and
    says so in `status`. Check the `debug` overlay (detected lines + the two VP circles).

    Accuracy, measured against pinhole views cut from real panos at known FOV/pitch:
    pitch is reliable (mean error ~4 deg), but h_fov_deg is only a STARTING ESTIMATE --
    mean error ~18 deg, and it can be far off (60 -> 121) when the orthogonal VP pair is
    poorly conditioned. Sanity-check it against your camera (phone main ~70, wide ~90)
    and just type the number into the warp node if it looks wrong.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fallback_fov": ("FLOAT", {"default": 65.0, "min": 20.0, "max": 170.0, "step": 1.0,
                    "tooltip": "Returned when the geometry can't pin down the focal length "
                               "(degenerate / 1-point perspective). ~65 suits a 16:9 phone frame."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("h_fov_deg", "pitch_deg", "status", "debug")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, fallback_fov=65.0):
        rgb = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        bgr = rgb[..., ::-1].copy()
        hfov, pitch, status, overlay = fe.estimate_fov_pitch(bgr, fallback_fov=float(fallback_fov))
        print(f"[Mickmumpitz/Pano/FOV] {status}")
        ov = torch.from_numpy(overlay.astype("float32") / 255.0).unsqueeze(0)
        return (float(hfov), float(pitch), status, ov)


class SeamRoll:
    """Equirect WRAP-SEAM killer prep/unroll: roll a pano by half its width so the left/right
    wrap seam moves to the CENTER, and emit a centered vertical seam mask for inpainting.

    Usage: apply once to PREP -> (rolled image, seam_mask); inpaint the masked centre strip;
    then apply this node AGAIN on the result (ignore the mask output) to roll back. Rolling by
    W/2 is its own inverse, so the same node both prepares and unrolls. Only the former wrap
    seam (now the centre strip) is a discontinuity; the new left/right edges are former-interior
    content and are restored exactly by the roll-back.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seam_width": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Width of the centre seam mask as a fraction of the pano width. "
                               "0.06 ~= a 6% strip. Wide enough to cover the wrap discontinuity."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Horizontal feather (px) on the seam mask. Keep 0 for a hard binary "
                               "mask; raise for a soft-mask sampler."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rolled_image", "seam_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, seam_width, feather):
        x = image if image.ndim == 4 else image.unsqueeze(0)   # [B,H,W,C]
        B, H, W, C = x.shape
        rolled = torch.roll(x, shifts=W // 2, dims=2)          # wrap seam -> centre

        mw = max(1, int(round(W * float(seam_width))))
        c = W // 2
        x0 = max(0, c - mw // 2)
        x1 = min(W, x0 + mw)
        m = torch.zeros((H, W), dtype=torch.float32)
        m[:, x0:x1] = 1.0
        r = int(feather)
        if r > 0:
            k = 2 * r + 1
            mm = m.view(1, 1, H, W)
            kh = torch.ones(1, 1, 1, k) / k
            mm = F.conv2d(F.pad(mm, (r, r, 0, 0), mode="replicate"), kh)
            m = mm.view(H, W).clamp(0, 1)
        print(f"[Mickmumpitz/Pano] wrap-seam roll {W//2}px; seam mask {mw}px @ centre col {c}, feather {r}")
        return (rolled, m.unsqueeze(0))


class StageSwitch:
    """Toggle a pipeline stage on/off. ``enabled`` picks the processed branch (on_image)
    or the bypass branch (off_image), so downstream always gets a valid image. Inputs are
    LAZY: when a stage is OFF, its processed branch is never requested, so those nodes are
    pruned and not computed (real deactivation, not just hidden). Wire the stage's output
    into on_image and the stage's INPUT (pre-stage image) into off_image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "STAGE ON",
                                        "label_off": "STAGE OFF (bypass)"}),
                "on_image": ("IMAGE", {"lazy": True}),
                "off_image": ("IMAGE", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, enabled, on_image=None, off_image=None):
        want = on_image if enabled else off_image
        if want is None:
            return ["on_image" if enabled else "off_image"]
        return []

    def run(self, enabled, on_image=None, off_image=None):
        return (on_image if enabled else off_image,)


class HarmonizeBoundary:
    """Bend the generated surroundings to meet the placed photo, with no visible step.

    Solves a smooth (harmonic / Poisson-style) correction field over the generated region:
    laplacian(c)=0 inside the hole, c = plate - image on the known plate boundary, and
    returns image + c. Because c is harmonic it is smooth, so the generated content's own
    gradients (texture, detail) are untouched -- only the low-frequency level is bent to
    meet the plate exactly at the boundary. The plate is never modified.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The outpainted panorama (generated surroundings)."}),
                "plate": ("IMAGE", {"tooltip": "The ERP canvas from the Warp node (control_video) "
                                               "- your photo placed on the sphere."}),
                "inpaint_mask": ("MASK", {"tooltip": "inpaint_mask from the Warp node. "
                                                     "White = hole to generate, black = your photo."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "1.0 = generated side meets the plate exactly at the boundary. "
                               "Lower keeps more of the generated look at the cost of a "
                               "visible step."}),
            },
            "optional": {
                "solve_scale": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Downsample factor for the solve. The correction field is smooth, "
                               "so 8 is plenty and fast. Lower = slower, no visible gain."}),
                "iterations": ("INT", {"default": 800, "min": 50, "max": 20000, "step": 50,
                    "tooltip": "Relaxation steps. More = the field spreads further from the "
                               "boundary. 800 at scale 8 covers a 2048x1024 pano."}),
                "wrap_horizontal": ("BOOLEAN", {"default": True,
                    "tooltip": "Equirect panoramas wrap left/right - keep this on so the "
                               "correction is continuous across the seam."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "correction_field")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, plate, inpaint_mask, strength,
            solve_scale=8, iterations=800, wrap_horizontal=True):
        img = image[0].float()                       # [H,W,3] 0..1
        plt = plate[0].float()
        if plt.shape[:2] != img.shape[:2]:
            plt = F.interpolate(plt.permute(2, 0, 1)[None], size=img.shape[:2],
                                mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        m = inpaint_mask[0].float() if inpaint_mask.ndim == 3 else inpaint_mask.float()
        if m.shape != img.shape[:2]:
            m = F.interpolate(m[None, None], size=img.shape[:2],
                              mode="bilinear", align_corners=False)[0, 0]

        H, W, _ = img.shape
        known = (m < 0.5).float()                    # 1 where the plate is real
        if known.sum() < 16:
            print("[Mickmumpitz/Pano] HarmonizeBoundary: no known region, passing through")
            return (image, torch.zeros_like(image))

        diff = (plt - img) * known.unsqueeze(-1)     # desired correction on the plate side

        s = max(1, int(solve_scale))
        h, w = max(4, H // s), max(8, W // s)
        # area-average down: fixed weight + the weighted correction, so partially-known
        # cells carry the mean of their real pixels only
        kd = F.interpolate(known[None, None], size=(h, w), mode="area")[0, 0]
        dd = F.interpolate(diff.permute(2, 0, 1)[None], size=(h, w), mode="area")[0]
        fixed = (kd > 0.5)
        vals = torch.where(fixed.unsqueeze(0), dd / kd.clamp(min=1e-6), torch.zeros_like(dd))

        c = torch.zeros_like(vals)                   # [3,h,w]
        c = torch.where(fixed.unsqueeze(0), vals, c)
        fx = fixed.unsqueeze(0)
        for _ in range(int(iterations)):
            if wrap_horizontal:
                left = torch.roll(c, 1, dims=2)
                right = torch.roll(c, -1, dims=2)
            else:
                left = torch.cat([c[:, :, :1], c[:, :, :-1]], dim=2)
                right = torch.cat([c[:, :, 1:], c[:, :, -1:]], dim=2)
            up = torch.cat([c[:, :1, :], c[:, :-1, :]], dim=1)
            down = torch.cat([c[:, 1:, :], c[:, -1:, :]], dim=1)
            c = 0.25 * (left + right + up + down)
            c = torch.where(fx, vals, c)             # re-impose the boundary condition

        cf = F.interpolate(c[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        cf = cf.permute(1, 2, 0)                     # [H,W,3]
        out = (img + cf * float(strength)).clamp(0, 1)

        rng = cf.abs().max().item()
        print(f"[Mickmumpitz/Pano] HarmonizeBoundary: solved {h}x{w} x {iterations} it, "
              f"max correction {rng:.3f} ({rng*255:.1f}/255), strength {strength}")
        vis = (cf * 0.5 / max(rng, 1e-6) + 0.5).clamp(0, 1)
        return (out.unsqueeze(0), vis.unsqueeze(0))


class UnfilledMask:
    """Mask the regions an outpaint pass left unfilled (near-black), so a second pass can
    finish them.

    An ERP-outpaint model sometimes gives up on the zenith/nadir when the source photo is
    wide but shallow -- it leaves them at canvas black. That is a hole, not a geometry
    failure, so the fix is simply to inpaint it. Restricted to the poles by default, so
    genuinely dark content near the horizon (shadow, night) is never touched.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The outpainted panorama to inspect."}),
                "threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Luminance below this counts as unfilled. 0.10 catches canvas "
                               "black without touching deep shadow."}),
                "limit_to_poles": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Only consider this fraction of the height at the top and at the "
                               "bottom. 0 = whole image (risky: masks real dark content)."}),
            },
            "optional": {
                "grow": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Expand the mask into the good content so the fill has context "
                               "to blend with."}),
                "blur": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Soften the mask edge."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT")
    RETURN_NAMES = ("mask", "mask_preview", "unfilled_rows")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, threshold, limit_to_poles, grow=24, blur=12.0):
        img = image[0].float()
        H, W, _ = img.shape
        lum = img.mean(dim=2)
        m = (lum < float(threshold)).float()

        if limit_to_poles > 0:
            band = max(1, int(round(H * float(limit_to_poles))))
            keep = torch.zeros((H, 1), dtype=m.dtype)
            keep[:band] = 1.0
            keep[H - band:] = 1.0
            m = m * keep

        rows = int((m.mean(dim=1) > 0.5).sum().item())

        if grow > 0:
            k = 2 * int(grow) + 1
            m = F.max_pool2d(m[None, None], kernel_size=k, stride=1, padding=int(grow))[0, 0]
        if blur > 0:
            r = int(blur)
            k = 2 * r + 1
            mm = m[None, None]
            kh = torch.ones(1, 1, 1, k) / k
            kv = torch.ones(1, 1, k, 1) / k
            mm = F.conv2d(F.pad(mm, (r, r, 0, 0), mode="circular"), kh)   # equirect wrap
            mm = F.conv2d(F.pad(mm, (0, 0, r, r), mode="replicate"), kv)
            m = mm[0, 0].clamp(0, 1)

        print(f"[Mickmumpitz/Pano] UnfilledMask: {rows} unfilled rows, "
              f"{m.mean().item()*100:.1f}% of the pano masked for a fill pass")
        vis = m.unsqueeze(-1).repeat(1, 1, 3)
        return (m.unsqueeze(0), vis.unsqueeze(0), rows)


class Krea2FullResReference:
    """Replace the reference latents of a Krea-2-edit conditioning with full-resolution ones.

    WHY THIS NODE EXISTS
    --------------------
    TextEncodeKrea2OstrisEdit caps references at 1 MP (REF_LATENT_MAX_PIXELS). And
    pack_ref_latents in ai-toolkit/ComfyUI assigns the reference's RoPE positions as
    arange(h) / arange(w) FROM ZERO. If the ref grid is smaller than the target grid,
    the reference therefore sits in the TOP-LEFT corner instead of covering the frame:

        target 2048x1024 -> target grid 128x64
        reference gets capped to 1456x720 -> ref grid 91x45
        => the reference covers 50% of the area, anchored top-left

    Visible as the original photo stuck in the upper-left corner. At 1408x704 and
    1440x720 it goes unnoticed because both are under 1 MP, so the cap is a no-op.
    Anything above that falls apart.

    For pixel-exact ERP outpainting the ref grid must match the target grid exactly.
    This node encodes the canvas uncapped and sets it as reference_latents, overriding
    whatever the encode attached.

    COST: reference tokens grow quadratically. 1408x704 is 3872 tokens, 2048x1024 is
    8192. With kv_cache on the model patch they are computed once and reused across all
    steps, so only the extra attention keys remain.

    NOTE: ModelSamplingFlux must be set to the same resolution -- Krea's mu is
    resolution-dependent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "max_megapixels": ("FLOAT", {"default": 8.0, "min": 0.25, "max": 32.0,
                                             "step": 0.25,
                                             "tooltip": "Safety cap against running out of "
                                                        "memory. 8 lets 2048x1024 through "
                                                        "uncapped."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Sets the Krea-2 reference at full resolution so the ref and target "
                   "grids line up. Without it, above 1 MP the reference sits in the "
                   "top-left corner.")

    SNAP = 16   # VAE f8 * patch 2

    def run(self, conditioning, vae, image, max_megapixels=8.0):
        import node_helpers
        import comfy.utils

        s = image.movedim(-1, 1)
        h, w = int(s.shape[2]), int(s.shape[3])
        cap = int(float(max_megapixels) * 1024 * 1024)
        scale = min(1.0, math.sqrt(cap / float(w * h)))
        nw = max(int(round(w * scale / self.SNAP)) * self.SNAP, self.SNAP)
        nh = max(int(round(h * scale / self.SNAP)) * self.SNAP, self.SNAP)
        if (nh, nw) != (h, w):
            s = comfy.utils.common_upscale(s, nw, nh, "area", "disabled")

        latent = vae.encode(s.movedim(1, -1)[:, :, :, :3])
        out = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": [latent]})

        gw, gh = nw // self.SNAP, nh // self.SNAP
        note = "" if (nh, nw) == (h, w) else f"  (capped to {max_megapixels} MP!)"
        print(f"[Mickmumpitz/Pano] Krea2FullResReference: reference {w}x{h} -> {nw}x{nh}, "
              f"ref grid {gw}x{gh} = {gw * gh} tokens{note}")
        return (out,)


class PanoRollHorizontal:
    """Roll a panorama horizontally so the wrap seam moves to the image centre.

    Replaces the previous chain of two ImageCrop plus ImageStitch, on both sides of the
    seam pass. The advantage is not the node count: the crop chain needed half the image
    width as a NUMBER in the graph, so it had to be re-entered whenever the canvas
    changed. Here the width comes from the image itself; there is nothing to keep in sync.

    Applied twice with the same settings it returns the original (canvas widths are
    divisible by 16, hence even).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fraction": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Fraction of the image width. 0.5 pushes "
                                                  "the seam exactly to the centre."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Horizontal wrap-around roll. Brings the wrap seam to the image centre and back."

    def run(self, image, fraction=0.5):
        W = int(image.shape[2])
        shift = int(round(W * float(fraction))) % W if W else 0
        if shift == 0:
            return (image,)
        return (torch.roll(image, shifts=shift, dims=2),)


class PanoSeamMask:
    """Mask for the seam strip in the image centre, with soft flanks.

    Replaces SolidMask + SolidMask + MaskComposite + FeatherMask. The x position is
    (width - strip width) / 2 and is computed here instead of living as a number in the
    graph. Width and height come straight from the two size inputs; the graph needs
    nothing else. Unlike a blur-based feather, the ramp stays INSIDE the strip, so the
    mask never grows past the intended width.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1408, "min": 256, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 704, "min": 128, "max": 4096, "step": 16}),
                "seam_width": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 16,
                                       "tooltip": "Width of the strip that gets repainted."}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                                    "tooltip": "Soft flank left and right, in pixels."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Centred seam strip as a mask; position and flanks are computed from "
                   "width and strip width.")

    def run(self, width, height, seam_width, feather):
        W, H = int(width), int(height)
        s = max(min(int(seam_width), W), 1)
        f = max(int(feather), 0)
        x0 = (W - s) // 2

        col = torch.zeros(W, dtype=torch.float32)
        col[x0:x0 + s] = 1.0
        if f > 0:
            ramp = torch.linspace(0.0, 1.0, f + 2, dtype=torch.float32)[1:-1]
            n = min(f, s // 2)
            if n > 0:
                col[x0:x0 + n] = ramp[:n]
                col[x0 + s - n:x0 + s] = ramp[:n].flip(0)
        mask = col.view(1, 1, W).expand(1, H, W).clone()
        print(f"[Mickmumpitz/Pano] SeamMask: {W}x{H}, strip {s} at x={x0}, flank {f}")
        return (mask,)
