"""
Shared region -> bounding-box logic for the OpenPose Part Cropper nodes.

This module understands the standard OpenPose / DWPose POSE_KEYPOINT structure
produced by comfyui_controlnet_aux:

    [                                   # one entry per frame / image
        {
            "people": [
                {
                    "pose_keypoints_2d":       [x,y,c, x,y,c, ... 18 pts],
                    "face_keypoints_2d":       [x,y,c, ... 68-70 pts] | None,
                    "hand_left_keypoints_2d":  [x,y,c, ... 21 pts] | None,
                    "hand_right_keypoints_2d": [x,y,c, ... 21 pts] | None,
                },
                ...
            ],
            "canvas_width":  int,
            "canvas_height": int,
        },
        ...
    ]

Coordinates are normally normalized to 0..1, but some producers emit pixel
coordinates. We auto-detect and normalize, then scale to the real image size.
"""

import math

# ---------------------------------------------------------------------------
# BODY_18 keypoint indices (OpenPose / COCO ordering)
# Left / right are anatomical (the subject's own left / right).
# ---------------------------------------------------------------------------
NOSE = 0
NECK = 1
R_SHOULDER = 2
R_ELBOW = 3
R_WRIST = 4
L_SHOULDER = 5
L_ELBOW = 6
L_WRIST = 7
R_HIP = 8
R_KNEE = 9
R_ANKLE = 10
L_HIP = 11
L_KNEE = 12
L_ANKLE = 13
R_EYE = 14
L_EYE = 15
R_EAR = 16
L_EAR = 17

# Region identifiers shown to the user (anatomical left/right).
REGION_LIST = [
    "head",
    "face",
    "left_hand",
    "right_hand",
    "both_hands",
    "left_foot",
    "right_foot",
    "both_feet",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "torso",
    "full_body",
]


# ---------------------------------------------------------------------------
# Keypoint parsing
# ---------------------------------------------------------------------------
def _flat_to_triples(flat):
    """Turn a flat [x,y,c,x,y,c,...] list into [(x,y,c), ...]."""
    if flat is None:
        return []
    triples = []
    for i in range(0, len(flat) - 2, 3):
        triples.append((float(flat[i]), float(flat[i + 1]), float(flat[i + 2])))
    return triples


def get_frame(pose_keypoint, index):
    """Return the pose dict for a given image index (clamped)."""
    if pose_keypoint is None:
        return None
    # A single dict (not wrapped in a list).
    if isinstance(pose_keypoint, dict):
        return pose_keypoint
    if isinstance(pose_keypoint, (list, tuple)) and len(pose_keypoint) > 0:
        idx = min(index, len(pose_keypoint) - 1)
        return pose_keypoint[idx]
    return None


def _person_center(person, axis):
    """Mean x (axis='x') or y (axis='y') of a person's valid body keypoints.

    Used only for *ordering* people, so the coordinate scale (normalized vs
    pixels) doesn't matter as long as it's consistent across people.
    """
    body = _flat_to_triples(person.get("pose_keypoints_2d"))
    vals = [(p[0] if axis == "x" else p[1]) for p in body if p[2] > 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def order_people(people, order="detection"):
    """Return the people list reordered so person_index is spatially stable.

    order:
        "detection" -> keep the detector's original order (default)
        "left_right"  -> index 0 = leftmost character (smallest mean x)
        "right_left"  -> index 0 = rightmost character
        "top_bottom"  -> index 0 = highest character (smallest mean y)
    """
    people = list(people or [])
    if order == "left_right":
        people.sort(key=lambda p: _person_center(p, "x"))
    elif order == "right_left":
        people.sort(key=lambda p: _person_center(p, "x"), reverse=True)
    elif order == "top_bottom":
        people.sort(key=lambda p: _person_center(p, "y"))
    return people


def get_person(frame, person_index, order="detection"):
    if not frame:
        return None
    people = order_people(frame.get("people"), order)
    if not people:
        return None
    if person_index < 0:
        person_index = 0
    if person_index >= len(people):
        person_index = len(people) - 1
    return people[person_index]


def parse_indices(spec):
    """Parse a keypoint-index spec string into a list of ints.

    Accepts comma/space/semicolon separated numbers and inclusive ranges:
        "4"        -> [4]
        "2,3,4"    -> [2, 3, 4]
        "2-4"      -> [2, 3, 4]
        "0, 14-17" -> [0, 14, 15, 16, 17]
    """
    out = []
    if not spec:
        return out
    text = str(spec).replace(";", ",").replace(" ", ",")
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:  # a range like "2-4" (leading '-' = negative, ignored)
            a, _, b = tok.partition("-")
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                continue
    return out


def keypoint_bbox(parsed, indices, group="body"):
    """Bbox (x1,y1,x2,y2) enclosing the given keypoint indices, or None.

    group is one of "body", "face", "hand_left", "hand_right". Indices that
    are out of range or below the confidence threshold are simply skipped.
    """
    pts = parsed.get(group) or []
    chosen = []
    for i in indices:
        if 0 <= i < len(pts) and pts[i] is not None:
            chosen.append(pts[i])
    return _bbox_of(chosen)


def num_people(frame):
    if not frame:
        return 0
    return len(frame.get("people") or [])


def _detect_scale(triples_groups, canvas_w, canvas_h):
    """
    Decide how to map stored coordinates to 0..1.

    Returns a function (x, y) -> (nx, ny) normalized to 0..1.
    Most producers store normalized coords; some store pixels.
    """
    max_val = 0.0
    for group in triples_groups:
        for (x, y, c) in group:
            if c <= 0:
                continue
            max_val = max(max_val, abs(x), abs(y))
    if max_val <= 1.5:
        # Already normalized.
        return lambda x, y: (x, y)
    # Pixel coordinates -> divide by canvas size (fall back to max observed).
    cw = canvas_w if canvas_w and canvas_w > 1 else max_val
    ch = canvas_h if canvas_h and canvas_h > 1 else max_val
    return lambda x, y: (x / cw, y / ch)


def parse_person(frame, person, img_w, img_h, conf_threshold=0.05):
    """
    Return a dict of named keypoint groups, each a list of (px, py, conf) in
    *pixel* coordinates relative to (img_w, img_h). Invalid points are dropped.
    """
    canvas_w = (frame or {}).get("canvas_width", 0)
    canvas_h = (frame or {}).get("canvas_height", 0)

    body = _flat_to_triples(person.get("pose_keypoints_2d"))
    face = _flat_to_triples(person.get("face_keypoints_2d"))
    hand_l = _flat_to_triples(person.get("hand_left_keypoints_2d"))
    hand_r = _flat_to_triples(person.get("hand_right_keypoints_2d"))

    norm = _detect_scale([body, face, hand_l, hand_r], canvas_w, canvas_h)

    def to_px(triples):
        out = []
        for (x, y, c) in triples:
            if c < conf_threshold:
                out.append(None)
                continue
            nx, ny = norm(x, y)
            out.append((nx * img_w, ny * img_h, c))
        return out

    return {
        "body": to_px(body),
        "face": to_px(face),
        "hand_left": to_px(hand_l),
        "hand_right": to_px(hand_r),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _valid(pts, idx):
    if idx < len(pts) and pts[idx] is not None:
        return pts[idx]
    return None


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _bbox_of(points):
    """Tight bbox (x1,y1,x2,y2) of a list of (x,y,...) points, or None."""
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _square_around(cx, cy, half):
    return (cx - half, cy - half, cx + half, cy + half)


# ---------------------------------------------------------------------------
# Per-region bounding boxes (in pixels). Each returns (x1,y1,x2,y2) or None.
# These mirror OpenPose's own hand/face heuristics (see open_pose/util.py)
# so that hands/feet are sized sensibly even when only body joints exist.
# ---------------------------------------------------------------------------
def _hand_box(body, hand_pts, wrist_i, elbow_i, shoulder_i):
    # Prefer detailed hand keypoints if available.
    hb = _bbox_of(hand_pts)
    if hb is not None:
        return hb
    # Fall back to OpenPose handDetect heuristic from wrist/elbow/shoulder.
    wrist = _valid(body, wrist_i)
    elbow = _valid(body, elbow_i)
    if wrist is None:
        return None
    if elbow is None:
        # No forearm direction: small box on the wrist.
        return _square_around(wrist[0], wrist[1], 24)
    ratio = 0.33
    cx = wrist[0] + ratio * (wrist[0] - elbow[0])
    cy = wrist[1] + ratio * (wrist[1] - elbow[1])
    d_we = _dist(wrist, elbow)
    width = 1.5 * d_we
    shoulder = _valid(body, shoulder_i)
    if shoulder is not None:
        d_es = _dist(elbow, shoulder)
        width = 1.5 * max(d_we, 0.9 * d_es)
    half = max(width / 2.0, 16)
    return _square_around(cx, cy, half)


def _foot_box(body, ankle_i, knee_i):
    ankle = _valid(body, ankle_i)
    if ankle is None:
        return None
    knee = _valid(body, knee_i)
    if knee is None:
        return _square_around(ankle[0], ankle[1], 28)
    shin = _dist(ankle, knee)
    half = max(0.45 * shin, 18)
    # Bias the box slightly toward the toes (below the ankle).
    return _square_around(ankle[0], ankle[1] + 0.25 * half, half)


def _face_box(body, face_pts):
    fb = _bbox_of(face_pts)
    if fb is not None:
        return fb
    # OpenPose faceDetect heuristic from nose/eyes/ears.
    head = _valid(body, NOSE)
    eyes_ears = [
        (_valid(body, L_EYE), 3.0),
        (_valid(body, R_EYE), 3.0),
        (_valid(body, L_EAR), 1.5),
        (_valid(body, R_EAR), 1.5),
    ]
    if head is None:
        pts = [p for p, _ in eyes_ears if p is not None]
        return _bbox_of(pts)
    # Estimate the face SIZE (full side length) from how far the eyes/ears sit
    # from the nose. Robustness: ignore an eye/ear that is implausibly far from
    # the nose (a cross-person or limb-line mis-detection) so one bad keypoint
    # can't blow the box up to whole-image size. The plausibility limit scales
    # with the closest reliable head point.
    dists = []
    for p, mult in eyes_ears:
        if p is None:
            continue
        d = max(abs(head[0] - p[0]), abs(head[1] - p[1]))
        dists.append(d * mult)
    size = 0.0
    if dists:
        # Reject outliers > 2.5x the smallest weighted distance.
        lo = min(dists)
        size = max(d for d in dists if d <= max(lo * 2.5, lo + 1.0))
    if size <= 0:
        size = 30.0
    # _square_around expects a HALF-extent (like _hand_box / _foot_box), so
    # pass size / 2 -- otherwise the box comes out at twice the intended size
    # and a "head"/"face" crop ends up larger than the whole body.
    return _square_around(head[0], head[1], size / 2.0)


def _head_box(body, face_pts):
    # Head = face region plus the neck, for a bit more context.
    fb = _face_box(body, face_pts)
    neck = _valid(body, NECK)
    if fb is None:
        return None
    if neck is None:
        return fb
    return (
        min(fb[0], neck[0]),
        min(fb[1], neck[1]),
        max(fb[2], neck[0]),
        max(fb[3], neck[1]),
    )


def region_bbox(parsed, region):
    """
    Compute the pixel bounding box (x1, y1, x2, y2) for the named region,
    or None if the required keypoints are missing.
    """
    body = parsed["body"]
    face = parsed["face"]
    hand_l = parsed["hand_left"]
    hand_r = parsed["hand_right"]

    if region == "head":
        return _head_box(body, face)
    if region == "face":
        return _face_box(body, face)
    if region == "left_hand":
        return _hand_box(body, hand_l, L_WRIST, L_ELBOW, L_SHOULDER)
    if region == "right_hand":
        return _hand_box(body, hand_r, R_WRIST, R_ELBOW, R_SHOULDER)
    if region == "both_hands":
        boxes = [
            _hand_box(body, hand_l, L_WRIST, L_ELBOW, L_SHOULDER),
            _hand_box(body, hand_r, R_WRIST, R_ELBOW, R_SHOULDER),
        ]
        return _union(boxes)
    if region == "left_foot":
        return _foot_box(body, L_ANKLE, L_KNEE)
    if region == "right_foot":
        return _foot_box(body, R_ANKLE, R_KNEE)
    if region == "both_feet":
        return _union([_foot_box(body, L_ANKLE, L_KNEE), _foot_box(body, R_ANKLE, R_KNEE)])
    if region == "left_arm":
        return _bbox_of([_valid(body, L_SHOULDER), _valid(body, L_ELBOW), _valid(body, L_WRIST)])
    if region == "right_arm":
        return _bbox_of([_valid(body, R_SHOULDER), _valid(body, R_ELBOW), _valid(body, R_WRIST)])
    if region == "left_leg":
        return _bbox_of([_valid(body, L_HIP), _valid(body, L_KNEE), _valid(body, L_ANKLE)])
    if region == "right_leg":
        return _bbox_of([_valid(body, R_HIP), _valid(body, R_KNEE), _valid(body, R_ANKLE)])
    if region == "torso":
        return _bbox_of([
            _valid(body, NECK), _valid(body, R_SHOULDER), _valid(body, L_SHOULDER),
            _valid(body, R_HIP), _valid(body, L_HIP),
        ])
    if region == "full_body":
        return _bbox_of(body)
    return None


def _union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


# ---------------------------------------------------------------------------
# Box post-processing: padding, square, min-size, multiple-of, clamp.
# ---------------------------------------------------------------------------
def finalize_box(box, img_w, img_h, padding=1.2, make_square=False,
                 min_size=64, multiple_of=8):
    """
    Expand / shape the raw box and clamp it to the image. Returns integer
    (x, y, w, h) fully inside the image, or None.
    """
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(x2 - x1, 1.0) * padding
    h = max(y2 - y1, 1.0) * padding

    if make_square:
        w = h = max(w, h)

    w = max(w, float(min_size))
    h = max(h, float(min_size))

    # Can't be larger than the image.
    w = min(w, float(img_w))
    h = min(h, float(img_h))

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0

    # Round and snap to multiple_of by growing the box.
    if multiple_of and multiple_of > 1:
        w = _round_up(w, multiple_of)
        h = _round_up(h, multiple_of)
        w = min(w, _round_down(img_w, multiple_of) or img_w)
        h = min(h, _round_down(img_h, multiple_of) or img_h)

    x = int(round(x1))
    y = int(round(y1))
    w = int(round(w))
    h = int(round(h))

    # Shift back inside the image.
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img_w:
        x = img_w - w
    if y + h > img_h:
        y = img_h - h
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, img_w)
    h = min(h, img_h)

    if w < 1 or h < 1:
        return None
    return (x, y, w, h)


def _round_up(v, m):
    return int(math.ceil(v / m) * m)


def _round_down(v, m):
    return int(math.floor(v / m) * m)
