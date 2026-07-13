"""
Face Area Batch Split — ConsistentCharacterCreator
==================================================

Splits an IMAGE batch into two groups based on the relative face area:
  - images_to_detail  : images whose (largest) face fills LESS than `area_threshold`
                        of the image area  -> these should be detailed / upscaled
  - images_passthrough: all remaining images (large face or no face)

A second node (Merger) recombines the — possibly detail-branch-modified — images
back into the original order. A third node (SEGS filter) offers the same
relative-area decision for the SEGS / FaceDetailer path.

These classes were previously a standalone single-file custom node
(`custom_nodes/face_area_batch_split.py`); they now live inside the
ConsistentCharacterCreator pack. Registration (with the CCC_ prefix) happens in
the pack's __init__.py — this module only defines the classes.

Requires `ultralytics` (installed with the Impact subpack) OR an
UltralyticsDetectorProvider node wired into `bbox_detector`. The face model
(e.g. face_yolov8m.pt) is resolved via ComfyUI's folder_paths under
ComfyUI/models/ultralytics/bbox/.
"""

import os
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def _resolve_model(model_path: str) -> str:
    """Accepts an absolute path OR a filename resolved via ComfyUI's folder_paths
    against the ultralytics model folders."""
    if os.path.isfile(model_path):
        return model_path
    try:
        import folder_paths  # only available inside ComfyUI
        for key in ("ultralytics_bbox", "ultralytics", "ultralytics_segm"):
            try:
                p = folder_paths.get_full_path(key, model_path)
                if p and os.path.isfile(p):
                    return p
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: hand the name to YOLO (it can also fetch official model names itself)
    return model_path


def _load_yolo(model_path: str):
    resolved = _resolve_model(model_path)
    if resolved not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "ultralytics is not installed. Either connect an "
                "UltralyticsDetectorProvider node to the 'bbox_detector' input "
                "(then this node does not need ultralytics directly), or run "
                "'pip install ultralytics' in the ComfyUI Python environment "
                "(or install the Impact subpack)."
            ) from e
        _MODEL_CACHE[resolved] = YOLO(resolved)
    return _MODEL_CACHE[resolved]


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """[H, W, C] float 0-1  ->  PIL RGB (Ultralytics treats PIL correctly as RGB)."""
    arr = (img_tensor.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _mask_from_boxes(boxes, h, w, device):
    """Builds an [H, W] mask (float 0/1): white inside each face bbox, black else.
    boxes are (x1, y1, x2, y2) in this image's pixel space."""
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    for (x1, y1, x2, y2) in boxes:
        xi1 = max(0, min(w, int(round(x1))))
        xi2 = max(0, min(w, int(round(x2))))
        yi1 = max(0, min(h, int(round(y1))))
        yi2 = max(0, min(h, int(round(y2))))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = 1.0
    return mask


def _boxes_from_detector(detector, image_single, confidence):
    """Uses Impact's BBOX_DETECTOR (UltralyticsDetectorProvider -> .detect()) and
    returns a list of (x1, y1, x2, y2). image_single has shape [1, H, W, C]."""
    # Impact signature: detect(image, threshold, dilation, crop_factor, drop_size, ...)
    # dilation/crop_factor only affect mask/crop, not the bbox -> irrelevant for area.
    try:
        segs = detector.detect(image_single, confidence, 4, 3.0, 1)
    except TypeError:
        segs = detector.detect(image_single, confidence, 4, 3.0)
    seg_list = segs[1]
    boxes = []
    for seg in seg_list:
        x1, y1, x2, y2 = seg.bbox
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def _boxes_from_yolo(model, image_single, confidence):
    """Fallback without BBOX_DETECTOR: load/use YOLO directly via ultralytics."""
    results = model(_to_pil(image_single[0]), conf=confidence, verbose=False)
    boxes_obj = results[0].boxes
    boxes = []
    if boxes_obj is not None and len(boxes_obj) > 0:
        for xyxy in boxes_obj.xyxy.cpu().numpy():
            boxes.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])))
    return boxes


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class FaceAreaBatchSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "area_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Route to detail branch when the largest face fills LESS "
                               "than this fraction of the image area (0.5 = 50%).",
                }),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "no_face_action": (["passthrough", "detail"], {"default": "passthrough"}),
            },
            "optional": {
                # Recommended: connect an UltralyticsDetectorProvider here. Then this
                # node does NOT load the model itself and needs no direct ultralytics.
                "bbox_detector": ("BBOX_DETECTOR", {
                    "tooltip": "Connect an UltralyticsDetectorProvider node here "
                               "(recommended). That node requires the ComfyUI Impact "
                               "Subpack to be installed. With it connected, this node "
                               "does not load the model itself. If left empty, it falls "
                               "back to the 'model_path' below (needs ultralytics).",
                }),
                # Fallback (only used when NO bbox_detector is connected):
                "model_path": ("STRING", {"default": "face_yolov8m.pt"}),
                # APPENDED (last widget slot) so existing workflows don't get their
                # model_path values shifted into max_faces.
                "max_faces": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Keep only the N largest faces per image (largest first) "
                               "for the mask. 0 = all faces.",
                }),
            },
        }

    # face_masks is APPENDED (last slot) so existing wiring of detail_indices/
    # total_count does not shift. Masks are aligned to images_to_detail: same
    # count and order.
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "MASK")
    RETURN_NAMES = (
        "images_to_detail", "images_passthrough", "detail_indices", "total_count",
        "face_masks",
    )
    FUNCTION = "split"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"

    def split(self, images, area_threshold, confidence, no_face_action,
              bbox_detector=None, model_path="face_yolov8m.pt", max_faces=0):
        use_detector = bbox_detector is not None
        model = None if use_detector else _load_yolo(model_path)

        to_detail, passthrough, detail_idx, detail_masks = [], [], [], []

        for i in range(images.shape[0]):
            img = images[i]                      # [H, W, C]
            h, w = int(img.shape[0]), int(img.shape[1])
            image_area = float(h * w)
            single = images[i:i + 1]             # [1, H, W, C]

            if use_detector:
                boxes = _boxes_from_detector(bbox_detector, single, confidence)
            else:
                boxes = _boxes_from_yolo(model, single, confidence)

            # Sort by area (largest face first) and optionally cap to the N largest.
            boxes.sort(
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
            )
            if max_faces > 0:
                boxes = boxes[:max_faces]

            if boxes:
                # boxes are sorted by area -> boxes[0] is the largest.
                x1, y1, x2, y2 = boxes[0]
                max_ratio = (((x2 - x1) * (y2 - y1)) / image_area) if image_area > 0 else 0.0
                # detail when the largest face is SMALLER than the threshold
                route_detail = max_ratio < area_threshold
            else:
                route_detail = (no_face_action == "detail")

            if route_detail:
                to_detail.append(img)
                detail_idx.append(i)
                # Mask from the (up to max_faces) largest face bboxes.
                detail_masks.append(_mask_from_boxes(boxes, h, w, images.device))
            else:
                passthrough.append(img)

        c = int(images.shape[3])
        h0, w0 = int(images.shape[1]), int(images.shape[2])

        def stack(lst):
            if len(lst) == 0:
                return torch.zeros((0, h0, w0, c), dtype=images.dtype, device=images.device)
            return torch.stack(lst, dim=0)

        def stack_masks(lst):
            if len(lst) == 0:
                return torch.zeros((0, h0, w0), dtype=torch.float32, device=images.device)
            return torch.stack(lst, dim=0)

        indices_str = ",".join(str(x) for x in detail_idx)
        return (
            stack(to_detail), stack(passthrough), indices_str, int(images.shape[0]),
            stack_masks(detail_masks),
        )


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

class FaceAreaBatchMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_to_detail": ("IMAGE",),       # possibly detailed, order = detail_indices
                "images_passthrough": ("IMAGE",),
                "detail_indices": ("STRING", {"default": ""}),
                "total_count": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "merge"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"

    def merge(self, images_to_detail, images_passthrough, detail_indices, total_count):
        detail_set = {int(x) for x in detail_indices.split(",") if x.strip() != ""}

        out = [None] * total_count
        di = pi = 0
        for i in range(total_count):
            if i in detail_set:
                out[i] = images_to_detail[di]
                di += 1
            else:
                out[i] = images_passthrough[pi]
                pi += 1

        if total_count == 0 or any(o is None for o in out):
            # Fallback: nothing / incomplete -> return what is present
            ref = images_to_detail if images_to_detail.shape[0] > 0 else images_passthrough
            return (ref,)

        return (torch.stack(out, dim=0),)


# ---------------------------------------------------------------------------
# Relative-Area SEGS Filter  (for the SEGS-/FaceDetailer path)
# ---------------------------------------------------------------------------

class SEGSRelativeAreaFilter:
    """Filters SEGS by area RELATIVE to the image area.

    Solves the problem that Impact's 'SEGS Filter (range)' only knows ABSOLUTE
    pixel areas: there you must wire width*height*ratio by hand, and if max_value
    stays at the huge default (~67M px^2), even (nearly) frame-filling faces slip
    through. This node computes the ratio itself.

    Usage: place between 'BBOX Detector (SEGS)' and 'DetailerForEach' to replace
    the ImpactSEGSRangeFilter + math nodes entirely.
      keep = "smaller_than_threshold" + area_threshold = 0.5
        -> keeps only faces filling LESS than 50% of the image area
           (exactly the ones you want to detail / upscale).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "area_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold as a fraction of the image area (0.5 = 50%).",
                }),
                "keep": (["smaller_than_threshold", "larger_than_threshold"],
                         {"default": "smaller_than_threshold"}),
            }
        }

    RETURN_TYPES = ("SEGS", "SEGS")
    RETURN_NAMES = ("filtered_SEGS", "removed_SEGS")
    FUNCTION = "filter"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"

    def filter(self, segs, area_threshold, keep):
        # SEGS = (shape, [SEG, ...]); shape is (H, W) or (H, W, C)
        shape, seg_list = segs[0], segs[1]
        H, W = int(shape[0]), int(shape[1])
        image_area = float(H * W) if H > 0 and W > 0 else 0.0

        keep_smaller = (keep == "smaller_than_threshold")
        kept, removed = [], []

        for seg in seg_list:
            x1, y1, x2, y2 = seg.bbox
            face_area = float((x2 - x1) * (y2 - y1))
            ratio = (face_area / image_area) if image_area > 0 else 0.0
            is_smaller = ratio < area_threshold
            if is_smaller == keep_smaller:
                kept.append(seg)
            else:
                removed.append(seg)

        return ((shape, kept), (shape, removed))


NODE_CLASS_MAPPINGS = {
    "CCC_FaceAreaBatchSplitter": FaceAreaBatchSplitter,
    "CCC_FaceAreaBatchMerger": FaceAreaBatchMerger,
    "CCC_SEGSRelativeAreaFilter": SEGSRelativeAreaFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCC_FaceAreaBatchSplitter": "Face Area Batch Splitter",
    "CCC_FaceAreaBatchMerger": "Face Area Batch Merger",
    "CCC_SEGSRelativeAreaFilter": "SEGS Filter by Relative Area",
}
