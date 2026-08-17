"""Vanishing-point FOV / pitch estimator (no EXIF needed) for the image->pano front-end.

Detects line segments (OpenCV LSD), extracts dominant vanishing points by a length-
weighted RANSAC, and recovers the focal length from an ORTHOGONAL pair of FINITE
vanishing points via the standard right-angle constraint:

    (v1 - c) . (v2 - c) + f^2 = 0        (c = principal point = image center)

-> f = sqrt( -(v1 - c).(v2 - c) )  when the dot product is negative.

hFOV = 2*atan(W / (2f)). Pitch is read from the forward vanishing point.

IMPORTANT / honest limitation: a near-1-point-perspective scene (e.g. a path running
straight away, all strong lines sharing ONE vanishing point + verticals at infinity) is
mathematically UNDER-DETERMINED for focal length. The estimator detects that (no valid
finite orthogonal pair) and returns the caller's fallback FOV with a status saying so,
rather than inventing a number.
"""
import math
import numpy as np
import cv2


def _detect_segments(gray, min_len):
    """LSD line segments -> (N,4) float array [x1,y1,x2,y2], filtered by min length."""
    try:
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
    except Exception:
        lines = None
    if lines is None or len(lines) == 0:
        # Fallback: Canny + probabilistic Hough.
        edges = cv2.Canny(gray, 60, 180)
        hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=int(min_len), maxLineGap=10)
        if hl is None:
            return np.zeros((0, 4), np.float32)
        segs = hl.reshape(-1, 4).astype(np.float32)
    else:
        segs = lines.reshape(-1, 4).astype(np.float32)
    d = segs[:, 2:4] - segs[:, 0:2]
    lens = np.hypot(d[:, 0], d[:, 1])
    return segs[lens >= min_len]


def _ransac_vp(segs, lens, iters, thresh_sin, rng):
    """One length-weighted RANSAC vanishing point. Returns (vp_xy or None, inlier_mask)."""
    n = len(segs)
    if n < 2:
        return None, np.zeros(n, bool)
    p1 = np.concatenate([segs[:, 0:2], np.ones((n, 1), np.float32)], 1)
    p2 = np.concatenate([segs[:, 2:4], np.ones((n, 1), np.float32)], 1)
    lines = np.cross(p1, p2)                         # homogeneous line per segment
    mids = (segs[:, 0:2] + segs[:, 2:4]) * 0.5
    dirs = segs[:, 2:4] - segs[:, 0:2]
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    w = lens / lens.sum()

    best_score, best_mask = -1.0, np.zeros(n, bool)
    idx = np.arange(n)
    for _ in range(iters):
        i, j = rng.choice(idx, size=2, replace=False, p=w)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-6:
            continue
        vp = vp[:2] / vp[2]
        to_vp = vp[None, :] - mids
        nrm = np.linalg.norm(to_vp, axis=1) + 1e-9
        sin_ang = np.abs(dirs[:, 0] * to_vp[:, 1] - dirs[:, 1] * to_vp[:, 0]) / nrm
        mask = sin_ang < thresh_sin
        score = lens[mask].sum()
        if score > best_score:
            best_score, best_mask = score, mask
    if best_score <= 0:
        return None, np.zeros(n, bool)
    # Refit the VP to all its inliers (least squares intersection of inlier lines).
    L = lines[best_mask]
    A = L[:, :2]
    b = -L[:, 2]
    try:
        vp = np.linalg.lstsq(A, b, rcond=None)[0]
    except Exception:
        return None, best_mask
    return vp, best_mask


def estimate_fov_pitch(bgr, fallback_fov=65.0, fov_min=35.0, fov_max=150.0):
    """Estimate (hFOV_deg, pitch_deg, status, overlay_rgb) for a rectilinear image."""
    H, W = bgr.shape[:2]
    diag = math.hypot(W, H)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    min_len = 0.02 * diag
    segs = _detect_segments(gray, min_len)
    overlay = bgr.copy()

    if len(segs) < 6:
        status = f"only {len(segs)} usable lines -> fallback {fallback_fov:.0f} deg"
        return float(fallback_fov), 0.0, status, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    d = segs[:, 2:4] - segs[:, 0:2]
    lens = np.hypot(d[:, 0], d[:, 1])
    rng = np.random.default_rng(0)               # deterministic

    # Extract up to 3 dominant VPs, peeling off inliers each round. Each VP also records
    # whether its supporting segments are predominantly VERTICAL in the image -- that
    # identifies the zenith/nadir VP, which is what the pitch is read from below.
    pool = np.ones(len(segs), bool)
    vps = []                                     # (vp_xy, support_len, n_inliers, is_vertical)
    for _ in range(3):
        if pool.sum() < 6:
            break
        s = segs[pool]
        l = lens[pool]
        vp, mask = _ransac_vp(s, l, iters=1500, thresh_sin=math.sin(math.radians(1.5)), rng=rng)
        if vp is None or mask.sum() < 4:
            break
        dv = s[mask][:, 2:4] - s[mask][:, 0:2]
        dv /= (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
        is_vert = float(np.abs(dv[:, 1]).mean()) > float(np.abs(dv[:, 0]).mean())
        vps.append((vp, float(l[mask].sum()), int(mask.sum()), is_vert))
        # remove those inliers from the global pool
        pool_idx = np.flatnonzero(pool)
        pool[pool_idx[mask]] = False

    for (x1, y1, x2, y2) in segs.astype(int):
        cv2.line(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)

    c = np.array([W / 2.0, H / 2.0])
    finite = [v for v in vps if np.linalg.norm(v[0] - c) < 50 * diag and np.all(np.isfinite(v[0]))]

    # Best orthogonal FINITE pair by the right-angle constraint.
    best = None                                  # (f, supp, va, vb)
    for a in range(len(finite)):
        for b in range(a + 1, len(finite)):
            ua, ub = finite[a][0] - c, finite[b][0] - c
            f2 = -float(ua @ ub)
            if f2 > (0.05 * diag) ** 2:           # reject near-degenerate / tiny f
                f = math.sqrt(f2)
                supp = min(finite[a][2], finite[b][2])
                if best is None or supp > best[1]:
                    best = (f, supp, finite[a][0], finite[b][0])

    if best is None:
        why = ("1-point perspective (all lines share one vanishing point) -- focal is "
               "under-determined" if len(finite) < 2 else
               "no orthogonal finite vanishing-point pair found")
        status = f"{len(vps)} VP(s), {len(finite)} finite: {why} -> fallback {fallback_fov:.0f} deg"
        return float(fallback_fov), 0.0, status, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    f, supp, va, vb = best
    hfov = 2.0 * math.degrees(math.atan(W / (2.0 * f)))
    hfov_c = float(min(max(hfov, fov_min), fov_max))

    # Pitch from the VERTICAL (zenith/nadir) vanishing point. Do NOT pick "the finite VP
    # nearest the centre column" as a forward VP -- the vertical VP normally sits ON that
    # column (world verticals stay vertical in image), so such a pick lands on it by
    # accident and misreads its polar angle as pitch (it returned +/-60..88 deg for every
    # input). Instead: the ray to the vertical VP IS the world vertical in camera coords
    # (x right, y down, z forward); flip it to point world-DOWN, and the camera's elevation
    # follows from its z component. Level camera -> down = (0,1,0), z-comp 0 -> pitch 0.
    vert = [v for v in finite if v[3]]
    pitch, pitch_src = 0.0, "no vertical VP -> pitch 0"
    if vert:
        vv = max(vert, key=lambda v: v[2])[0]                  # best-supported vertical VP
        d = np.array([vv[0] - c[0], vv[1] - c[1], f], float)
        d /= (np.linalg.norm(d) + 1e-9)
        if abs(d[1]) > 1e-6:
            down = d if d[1] > 0 else -d                       # world DOWN in camera frame
            pitch = -math.degrees(math.asin(float(np.clip(down[2], -1.0, 1.0))))
            pitch_src = f"pitch {pitch:+.0f} deg"

    for v, col in ((va, (0, 200, 0)), (vb, (0, 160, 255))):
        vx, vy = int(np.clip(v[0], -1e4, 1e4)), int(np.clip(v[1], -1e4, 1e4))
        cv2.circle(overlay, (max(0, min(W - 1, vx)), max(0, min(H - 1, vy))), 8, col, 2)
    clamp_note = "" if abs(hfov - hfov_c) < 0.5 else f" (raw {hfov:.0f} clamped)"
    status = (f"{len(finite)} finite VPs, support {supp} lines -> "
              f"hFOV {hfov_c:.0f} deg, {pitch_src}{clamp_note}")
    return hfov_c, float(pitch), status, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
