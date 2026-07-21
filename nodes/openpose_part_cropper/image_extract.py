"""
Reconstruct OpenPose keypoints from an already-rendered OpenPose skeleton image.

How it works
------------
comfyui_controlnet_aux draws each of the 18 body joints as a SOLID, full-color
filled circle (radius 4), drawn *after* the limb sticks, which are dimmed to 60%
intensity. The 18 joint colors are unique, so we can recover each joint's pixel
position by isolating its exact color and taking the centroid of the resulting
blob(s) -- one blob per person.

Hands are drawn as clusters of blue dots (0,0,255) and the face as clusters of
small white dots (255,255,255). We optionally detect those dense clusters and
attach them to the nearest wrist / nose so hand & face crops are accurate.

The output matches the standard POSE_KEYPOINT structure, so it drops into the
Part Mask / Part Crop nodes (or any node that consumes POSE_KEYPOINT).
"""

import math
import numpy as np

# Body joint palette, index == BODY_18 keypoint index (open_pose/util.py).
BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
]
HAND_COLOR = [0, 0, 255]    # same as body index 12 (R_HIP) -> disambiguated by clustering
FACE_COLOR = [255, 255, 255]

# Indices used as person anchors when grouping multi-person detections.
from .regions import (NOSE, NECK, R_SHOULDER, L_SHOULDER, R_ELBOW, L_ELBOW,
                      R_WRIST, L_WRIST, L_KNEE)


# ---------------------------------------------------------------------------
# Connected-component centroids (cv2 preferred, scipy fallback, then numpy).
# ---------------------------------------------------------------------------
def _components(mask, min_area):
    """Return list of (cx, cy, area) for connected True regions in a 2D mask."""
    try:
        import cv2
        num, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        out = []
        for i in range(1, num):  # 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out.append((float(centroids[i][0]), float(centroids[i][1]), int(area)))
        return out
    except Exception:
        pass
    try:
        from scipy import ndimage
        labels, num = ndimage.label(mask)
        out = []
        for i in range(1, num + 1):
            ys, xs = np.where(labels == i)
            if len(xs) >= min_area:
                out.append((float(xs.mean()), float(ys.mean()), int(len(xs))))
        return out
    except Exception:
        pass
    # Last resort: a single centroid over all matching pixels.
    ys, xs = np.where(mask)
    if len(xs) >= min_area:
        return [(float(xs.mean()), float(ys.mean()), int(len(xs)))]
    return []


def _color_mask(img, color, tol):
    diff = np.abs(img.astype(np.int16) - np.array(color, dtype=np.int16))
    return np.all(diff <= tol, axis=-1)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_pose(img_rgb_uint8, color_tol=40, min_area=4, reconstruct_hands_face=True):
    """
    img_rgb_uint8 : HxWx3 uint8 RGB array (a rendered OpenPose image)
    Returns a single-frame pose dict (normalized 0..1 coordinates).
    """
    H, W, _ = img_rgb_uint8.shape

    # 1) Detect every body joint color -> list of component centroids.
    detections = {}  # joint_index -> [(cx, cy, area), ...]
    for idx, color in enumerate(BODY_COLORS):
        mask = _color_mask(img_rgb_uint8, color, color_tol)
        comps = _components(mask, min_area)
        detections[idx] = comps

    # Body joint 12 (Left Knee) is drawn in the SAME blue [0,0,255] as the hand
    # dots, so its color match also picks up hands. Drop dense (hand) clusters so
    # the knee joint itself stays clean.
    detections[L_KNEE] = _drop_clusters(detections[L_KNEE])

    # 2) Determine person count & anchors from the most stable single joints.
    anchor_candidates = [NECK, NOSE, R_SHOULDER, L_SHOULDER]
    anchors = []
    best_count = 0
    for ai in anchor_candidates:
        comps = detections.get(ai, [])
        if len(comps) > best_count:
            best_count = len(comps)
            anchors = [(c[0], c[1]) for c in comps]
    if not anchors:
        # Fall back to whichever joint has the most detections.
        for idx, comps in detections.items():
            if len(comps) > len(anchors):
                anchors = [(c[0], c[1]) for c in comps]
    if not anchors:
        return {"people": [], "canvas_width": W, "canvas_height": H}

    # Merge anchors that sit almost on top of each other. Color-match noise
    # (limbs drawn at 60% intensity, high color_tolerance) and junk detections
    # can spawn several near-duplicate anchors for a SINGLE person, which would
    # otherwise split that person's joints across phantom people.
    anchors = _dedupe_anchors(anchors, radius=0.05 * max(W, H))

    num_persons = len(anchors)

    # 3) Assign each joint's detections to the nearest person anchor.
    #    people_body[p][joint_idx] = (cx, cy) or None
    people_body = [[None] * 18 for _ in range(num_persons)]
    for joint_idx, comps in detections.items():
        # Sort biggest first so the dominant blob wins each person.
        comps_sorted = sorted(comps, key=lambda c: -c[2])
        used = set()
        for (cx, cy, _area) in comps_sorted:
            p = _nearest_anchor(cx, cy, anchors)
            if p in used:
                continue  # one detection of this joint per person
            people_body[p][joint_idx] = (cx, cy)
            used.add(p)

    # 4) Optionally reconstruct hands (blue clusters) and face (white clusters).
    hand_clusters = []
    face_points_per_person = [None] * num_persons
    if reconstruct_hands_face:
        hand_clusters = _detect_clusters(img_rgb_uint8, HAND_COLOR, color_tol, min_area)
        # The blue Left-Knee joint is the same color as hand dots and shows up
        # here as a lone tiny "cluster". Reject clusters that (a) coincide with a
        # detected knee joint, or (b) are too small to be a real hand (a hand is
        # ~21 dots; a single isolated dot is a body joint, not a hand).
        knee_pts = [people_body[i][L_KNEE] for i in range(num_persons)
                    if people_body[i][L_KNEE] is not None]
        excl_r = max(10.0, 0.012 * max(W, H))
        hand_clusters = [hc for hc in hand_clusters
                         if not _is_body_joint(hc, knee_pts, excl_r)]
        face_clusters = _detect_clusters(img_rgb_uint8, FACE_COLOR, min(color_tol, 30), max(min_area, 2))
        # Attach each face cluster to nearest nose.
        for fc in face_clusters:
            p = _nearest_anchor(fc["cx"], fc["cy"],
                                [(_b(people_body[i], NOSE) or anchors[i]) for i in range(num_persons)])
            face_points_per_person[p] = fc["points"]

    # 4b) Assign hand clusters to a person + a side, and synthesize any missing
    #     wrist joint from its hand cluster (so hand/arm regions still get a box).
    hands_per_person = ([{"left": None, "right": None} for _ in range(num_persons)]
                        if not reconstruct_hands_face
                        else _assign_all_hands(people_body, hand_clusters))

    # 5) Build the frame dict (normalized).
    people = []
    for p in range(num_persons):
        body = people_body[p]
        person = {
            "pose_keypoints_2d": _flatten_body(body, W, H),
            "face_keypoints_2d": None,
            "hand_left_keypoints_2d": None,
            "hand_right_keypoints_2d": None,
        }
        if reconstruct_hands_face:
            if face_points_per_person[p]:
                person["face_keypoints_2d"] = _flatten_points(face_points_per_person[p], W, H)
            if hands_per_person[p]["left"]:
                person["hand_left_keypoints_2d"] = _flatten_points(hands_per_person[p]["left"], W, H)
            if hands_per_person[p]["right"]:
                person["hand_right_keypoints_2d"] = _flatten_points(hands_per_person[p]["right"], W, H)
        people.append(person)

    return {"people": people, "canvas_width": W, "canvas_height": H}


def _assign_all_hands(people_body, hand_clusters):
    """Assign each blue hand cluster to a person and an anatomical side.

    Returns a list (one per person) of {"left": points|None, "right": points|None}.
    Also mutates people_body to *synthesize* a wrist joint from the hand cluster
    whenever that wrist wasn't detected on its own -- this is the common failure
    where the wrist dot is swallowed by the dense hand cluster, which previously
    left the hand region with no box at all.
    """
    n = len(people_body)
    result = [{"left": None, "right": None} for _ in range(n)]
    clusters = [hc for hc in (hand_clusters or []) if not hc.get("_used")]
    if not clusters or n == 0:
        return result

    def refs(pb):
        pts = [pb[i] for i in (L_WRIST, R_WRIST) if pb[i] is not None]
        if pts:
            return pts
        for i in (NECK, NOSE, R_SHOULDER, L_SHOULDER):
            if pb[i] is not None:
                return [pb[i]]
        return []

    person_refs = [refs(people_body[p]) for p in range(n)]

    # 1) Owner = the person whose wrist/anchor is nearest this cluster.
    owner_of = []
    for hc in clusters:
        cx, cy = hc["cx"], hc["cy"]
        best_p, best_d = 0, float("inf")
        for p in range(n):
            for (rx, ry) in person_refs[p]:
                d = (cx - rx) ** 2 + (cy - ry) ** 2
                if d < best_d:
                    best_d, best_p = d, p
        owner_of.append(best_p)

    # 2) Within each owner, label sides and fill missing wrists.
    for p in range(n):
        pb = people_body[p]
        mine = [clusters[i] for i in range(len(clusters)) if owner_of[i] == p]
        if not mine:
            continue
        # Reference each side by its best available arm joint. Using the elbow /
        # shoulder as a fallback matters when the wrist dot was swallowed by the
        # hand cluster -- otherwise a lone known wrist wins BOTH hands.
        left_ref = _first_joint(pb, (L_WRIST, L_ELBOW, L_SHOULDER))
        right_ref = _first_joint(pb, (R_WRIST, R_ELBOW, R_SHOULDER))

        def known_side(hc):
            cx, cy = hc["cx"], hc["cy"]
            dl = _d2(cx, cy, left_ref) if left_ref else None
            dr = _d2(cx, cy, right_ref) if right_ref else None
            if dl is not None and dr is not None:
                return "left" if dl <= dr else "right"
            if dl is not None:
                return "left"
            if dr is not None:
                return "right"
            return None

        taken = {}
        leftovers = []
        for hc in mine:
            s = known_side(hc)
            if s and s not in taken:
                taken[s] = hc
            else:
                leftovers.append(hc)

        if not taken and leftovers:
            # No wrist known at all: fall back to image x (leftmost -> left side).
            leftovers.sort(key=lambda h: h["cx"])
            for s, hc in zip(("left", "right"), leftovers):
                taken[s] = hc
            leftovers = leftovers[2:]
        else:
            for hc in leftovers:
                free = [s for s in ("left", "right") if s not in taken]
                if not free:
                    break
                taken[free[0]] = hc

        for s, hc in taken.items():
            result[p][s] = hc["points"]
            widx = L_WRIST if s == "left" else R_WRIST
            if pb[widx] is None:
                pb[widx] = _cluster_wrist_point(hc, pb, widx)

    return result


def _first_joint(pb, indices):
    for i in indices:
        if pb[i] is not None:
            return pb[i]
    return None


def _cluster_wrist_point(hc, pb, widx):
    """Best guess for the wrist inside a hand cluster: the cluster point closest
    to the elbow (the hand attaches to the arm there), else the cluster center."""
    elbow = pb[L_ELBOW] if widx == L_WRIST else pb[R_ELBOW]
    pts = hc.get("points") or []
    if elbow and pts:
        return min(pts, key=lambda pt: (pt[0] - elbow[0]) ** 2 + (pt[1] - elbow[1]) ** 2)
    return (hc["cx"], hc["cy"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b(body, idx):
    return body[idx] if idx < len(body) else None


def _dedupe_anchors(anchors, radius):
    """Greedily merge anchors within `radius` px of an already-kept anchor."""
    kept = []
    r2 = radius * radius
    for (ax, ay) in anchors:
        dup = False
        for (kx, ky) in kept:
            if (ax - kx) ** 2 + (ay - ky) ** 2 <= r2:
                dup = True
                break
        if not dup:
            kept.append((ax, ay))
    return kept


def _nearest_anchor(cx, cy, anchors):
    best = 0
    best_d = float("inf")
    for i, (ax, ay) in enumerate(anchors):
        d = (cx - ax) ** 2 + (cy - ay) ** 2
        if d < best_d:
            best_d = d
            best = i
    return best


def _drop_clusters(comps, neighbor_radius_factor=6.0, min_neighbors=3):
    """Remove components that sit inside a dense cluster (i.e. hand dots)."""
    if len(comps) <= 1:
        return comps
    kept = []
    for i, (cx, cy, area) in enumerate(comps):
        radius = neighbor_radius_factor * math.sqrt(max(area, 1))
        neighbors = 0
        for j, (ox, oy, _oa) in enumerate(comps):
            if i == j:
                continue
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= radius * radius:
                neighbors += 1
        if neighbors < min_neighbors:
            kept.append((cx, cy, area))
    return kept if kept else comps


def _detect_clusters(img, color, tol, min_area, link_factor=5.0):
    """
    Group nearby same-color dot components into clusters.
    Returns [{cx, cy, points: [(x,y),...], bbox}], one per cluster.
    """
    mask = _color_mask(img, color, tol)
    comps = _components(mask, min_area)
    if not comps:
        return []
    # Single-link grouping by proximity.
    parent = list(range(len(comps)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ri = link_factor * math.sqrt(max(comps[i][2], 1))
            rj = link_factor * math.sqrt(max(comps[j][2], 1))
            r = max(ri, rj)
            if (comps[i][0] - comps[j][0]) ** 2 + (comps[i][1] - comps[j][1]) ** 2 <= r * r:
                union(i, j)

    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(c)

    clusters = []
    for members in groups.values():
        pts = [(m[0], m[1]) for m in members]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        clusters.append({
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
            "points": pts,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        })
    return clusters


def _cluster_span(hc):
    x1, y1, x2, y2 = hc["bbox"]
    return max(x2 - x1, y2 - y1)


def _is_body_joint(hc, joint_pts, excl_r):
    """
    True if a blue cluster is really a body joint (the Left Knee shares the hand
    color) rather than a hand: it coincides with a detected knee, or it's a lone
    tiny dot too small to be a 21-point hand.
    """
    cx, cy = hc["cx"], hc["cy"]
    for (jx, jy) in joint_pts:
        if (cx - jx) ** 2 + (cy - jy) ** 2 <= excl_r * excl_r:
            return True
    if len(hc["points"]) <= 1 and _cluster_span(hc) < 12.0:
        return True
    return False


def _d2(cx, cy, pt):
    return (cx - pt[0]) ** 2 + (cy - pt[1]) ** 2


def _flatten_body(body, W, H):
    flat = []
    for pt in body:
        if pt is None:
            flat.extend([0.0, 0.0, 0.0])
        else:
            flat.extend([pt[0] / W, pt[1] / H, 1.0])
    return flat


def _flatten_points(points, W, H):
    flat = []
    for (x, y) in points:
        flat.extend([x / W, y / H, 1.0])
    return flat
