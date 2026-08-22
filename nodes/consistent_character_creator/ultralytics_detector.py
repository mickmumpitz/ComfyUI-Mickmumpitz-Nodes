"""
Ultralytics Detector Provider — clean-room replacement
======================================================

A self-contained loader for Ultralytics YOLO detection / segmentation models
that emits ``BBOX_DETECTOR`` / ``SEGM_DETECTOR`` objects. These objects are
duck-type compatible with the detector interface consumed by the ComfyUI
Impact-Pack nodes (FaceDetailer, DetailerForEach, SEGS utilities, …) as well as
this pack's own ``CCC_FaceAreaBatchSplitter``.

Why this exists
---------------
The ``UltralyticsDetectorProvider`` node ships in the *Impact Subpack*, whose
license is not permissive enough to copy & patch. Users repeatedly hit
install / import / version problems with that subpack. This module provides an
independent implementation. It was written only from the observable interface
that downstream nodes rely on — the ordered fields of a per-detection record and
the positional arguments they pass when they ask a detector to run. No source
code was copied from the Impact packages and nothing is imported from them at
runtime; all geometry (crop regions, masks, dilation) is derived here from first
principles with standard array operations.

It only needs ``ultralytics`` (which most Impact users already have). It does
NOT need the Impact Subpack installed at all.

Model resolution
----------------
Models live under ``ComfyUI/models/ultralytics/bbox`` and
``ComfyUI/models/ultralytics/segm``. This module registers those folders with
ComfyUI's ``folder_paths`` on import (idempotent), so the dropdown works even
when the subpack is absent. The dropdown values use the same ``bbox/xxx.pt`` /
``segm/xxx.pt`` form the subpack used, so existing workflow widget values load
unchanged.
"""

import os
import logging
import threading
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# folder_paths registration (so the model dropdown works standalone)
# ---------------------------------------------------------------------------

try:
    import folder_paths

    _MODELS_DIR = folder_paths.models_dir
    _ULTRA_ROOT = os.path.join(_MODELS_DIR, "ultralytics")
    _ULTRA_BBOX = os.path.join(_ULTRA_ROOT, "bbox")
    _ULTRA_SEGM = os.path.join(_ULTRA_ROOT, "segm")

    def _register_folder(key, paths):
        # Merge with anything the subpack (or a prior import) already registered.
        existing = folder_paths.folder_names_and_paths.get(key)
        if existing is None:
            folder_paths.folder_names_and_paths[key] = (list(paths), {".pt"})
        else:
            cur_paths, exts = existing
            for p in paths:
                if p not in cur_paths:
                    cur_paths.append(p)
            # keep .pt in the accepted extension set
            try:
                exts.add(".pt")
            except AttributeError:
                pass

    _register_folder("ultralytics", [_ULTRA_ROOT])
    _register_folder("ultralytics_bbox", [_ULTRA_BBOX])
    _register_folder("ultralytics_segm", [_ULTRA_SEGM])
except Exception:  # pragma: no cover - folder_paths only exists inside ComfyUI
    folder_paths = None
    _ULTRA_ROOT = _ULTRA_BBOX = _ULTRA_SEGM = None


# ---------------------------------------------------------------------------
# One-time default-model downloader
# ---------------------------------------------------------------------------
# On first run (empty model folders) we fetch a small set of common starter
# detection models so the node is usable out of the box, mirroring the
# convenience the Impact-Subpack offered. Already-present files are never
# re-downloaded. It runs on a background thread so it never blocks ComfyUI
# startup, and it can be disabled entirely (see _download_disabled).
#
# The model files are public weights hosted on Hugging Face and are fetched
# with the standard ``huggingface_hub`` downloader (the same library ComfyUI
# and most model nodes already use for weight retrieval).

_DEFAULT_MODELS = [
    # (subfolder, local filename, hf repo id, filename within the repo)
    ("bbox", "face_yolov8m.pt", "Bingsu/adetailer", "face_yolov8m.pt"),
    ("bbox", "hand_yolov8s.pt", "Bingsu/adetailer", "hand_yolov8s.pt"),
    ("segm", "person_yolov8m-seg.pt", "Bingsu/adetailer", "person_yolov8m-seg.pt"),
]

_DOWNLOAD_STARTED = False


def _download_disabled():
    """Opt-out: a ``skip_download_model`` marker file placed next to this pack or
    in the custom_nodes directory disables the one-time model download."""
    here = os.path.dirname(os.path.abspath(__file__))
    # .../custom_nodes/ComfyUI-Mickmumpitz-Nodes/nodes/consistent_character_creator
    pack_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    custom_nodes = os.path.abspath(os.path.join(pack_root, ".."))
    for base in (pack_root, custom_nodes):
        if os.path.exists(os.path.join(base, "skip_download_model")):
            return True
    return False


def _ensure_default_models(background=True):
    """Download any missing default models. No-op if folders are unavailable,
    downloads are disabled, or every default already exists."""
    global _DOWNLOAD_STARTED
    if _DOWNLOAD_STARTED or folder_paths is None or _ULTRA_ROOT is None:
        return
    if _download_disabled():
        return

    targets = []
    for sub, name, repo, filename in _DEFAULT_MODELS:
        folder = _ULTRA_BBOX if sub == "bbox" else _ULTRA_SEGM
        dest = os.path.join(folder, name)
        if not os.path.exists(dest):
            targets.append((repo, filename, dest, folder))
    if not targets:
        return

    _DOWNLOAD_STARTED = True

    def _worker():
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            logger.warning("[MMZ] huggingface_hub is unavailable (%s); skipping the "
                           "default detection-model download. Place the models under "
                           "%s manually if you need them.", e, _ULTRA_ROOT)
            return
        for repo, filename, dest, folder in targets:
            try:
                os.makedirs(folder, exist_ok=True)
                logger.info("[MMZ] Downloading detection model %s ...",
                            os.path.basename(dest))
                hf_hub_download(repo_id=repo, repo_type="model",
                                filename=filename, local_dir=folder)
                logger.info("[MMZ] Saved %s", dest)
            except Exception as e:  # network/offline/permission — non-fatal
                logger.warning("[MMZ] Could not download %s (%s). Place the "
                               "model in %s manually if you need it.",
                               os.path.basename(dest), e, folder)
        logger.info("[MMZ] Default detection models ready (restart or refresh "
                    "to see new files in the model list).")

    if background:
        threading.Thread(target=_worker, name="mmz-ultra-model-dl",
                         daemon=True).start()
    else:
        _worker()


# ---------------------------------------------------------------------------
# Segment record (interface contract)
# ---------------------------------------------------------------------------
# Downstream detail nodes consume a per-detection record positionally: they read
# the seven fields below in this order and rebuild their own record type from
# them. A field layout is a functional interface, not something we borrow — this
# is our own definition and nothing is imported from the Impact packages.
SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

_YOLO_CACHE = {}


def _resolve_model_path(model_name: str) -> str:
    """``bbox/face_yolov8m.pt`` -> absolute path via folder_paths, with a couple
    of fallbacks. An absolute/existing path is returned as-is."""
    if os.path.isfile(model_name):
        return model_name
    if folder_paths is not None:
        # get_full_path joins the sub-path ("bbox/xxx.pt") onto the ultralytics root
        for key in ("ultralytics", "ultralytics_bbox", "ultralytics_segm"):
            try:
                p = folder_paths.get_full_path(key, model_name)
                if p and os.path.isfile(p):
                    return p
            except Exception:
                pass
        # Try stripping the bbox/ | segm/ prefix against the matching key
        if "/" in model_name:
            prefix, rest = model_name.split("/", 1)
            key = "ultralytics_bbox" if prefix == "bbox" else "ultralytics_segm"
            try:
                p = folder_paths.get_full_path(key, rest)
                if p and os.path.isfile(p):
                    return p
            except Exception:
                pass
    return model_name  # hand the raw name to YOLO (it can fetch known names)


def _load_yolo(model_name: str):
    resolved = _resolve_model_path(model_name)
    if resolved in _YOLO_CACHE:
        return _YOLO_CACHE[resolved]
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The 'ultralytics' package is required to load YOLO detection models. "
            "Install it in the ComfyUI Python environment with:\n"
            "    pip install ultralytics"
        ) from e

    # PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the
    # pickled ultralytics model classes. Recent ultralytics handles this itself;
    # for older combos we allow-list the needed globals as a best effort.
    try:
        model = YOLO(resolved)
    except Exception:
        try:
            import torch.serialization as _ts
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel  # type: ignore

            _ts.add_safe_globals([DetectionModel, SegmentationModel])
        except Exception:
            pass
        model = YOLO(resolved)  # retry (re-raises the original if still failing)

    _YOLO_CACHE[resolved] = model
    return model


# ---------------------------------------------------------------------------
# Geometry / mask helpers (standard operations, re-derived)
# ---------------------------------------------------------------------------

def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """[B,H,W,C] or [H,W,C] float 0-1 -> PIL RGB (first frame)."""
    if image.dim() == 4:
        image = image[0]
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr[:, :, :3])


def _make_crop_region(w, h, bbox, crop_factor):
    """Enlarge a tight bbox by ``crop_factor`` about its centre, clamped to the
    image. bbox = (x1, y1, x2, y2). Returns integer (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cw = bw * crop_factor
    ch = bh * crop_factor
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    nx1 = int(cx - cw / 2.0)
    ny1 = int(cy - ch / 2.0)
    nx2 = int(cx + cw / 2.0)
    ny2 = int(cy + ch / 2.0)
    nx1 = max(0, min(nx1, w))
    ny1 = max(0, min(ny1, h))
    nx2 = max(0, min(nx2, w))
    ny2 = max(0, min(ny2, h))
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _dilate_mask(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Grow (dilation>0) or shrink (dilation<0) a float 0/1 mask by |dilation|
    pixels. Uses cv2 if present, else scipy, else returns the mask unchanged."""
    if not dilation:
        return mask
    kernel = 2 * abs(int(dilation)) + 1
    try:
        import cv2

        k = np.ones((kernel, kernel), np.uint8)
        m = (mask > 0.5).astype(np.uint8)
        m = cv2.dilate(m, k) if dilation > 0 else cv2.erode(m, k)
        return m.astype(np.float32)
    except Exception:
        pass
    try:
        from scipy import ndimage

        m = mask > 0.5
        it = abs(int(dilation))
        m = ndimage.binary_dilation(m, iterations=it) if dilation > 0 \
            else ndimage.binary_erosion(m, iterations=it)
        return m.astype(np.float32)
    except Exception:
        return mask


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour resize a 2D float mask to (h, w) using PIL (no cv2 dep)."""
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    img = img.resize((w, h), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


# ---------------------------------------------------------------------------
# Detector cores
# ---------------------------------------------------------------------------

class _UltraDetector:
    """Shared inference + SEG assembly for both bbox and segm detectors."""

    def __init__(self, model, use_masks: bool):
        self.model = model
        self.use_masks = use_masks
        self.aux = None  # some callers set a label hint via setAux()

    # Some callers announce an expected label (e.g. a class name) on the
    # detector before running it. We keep it as an optional label hint; it has
    # no other effect.
    def setAux(self, x):
        self.aux = x

    def _infer(self, image, threshold):
        pil = _tensor_to_pil(image)
        results = self.model.predict(
            pil, conf=float(max(0.0, min(1.0, threshold))), verbose=False
        )
        return results[0] if results else None

    def _iter_detections(self, result, w, h):
        """Yield (bbox_xyxy_ints, confidence, label, full_mask[h,w])."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return

        names = getattr(self.model, "names", {}) or {}
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        clss = result.boxes.cls.detach().cpu().numpy().astype(int)

        seg_masks = None
        if self.use_masks and getattr(result, "masks", None) is not None:
            try:
                seg_masks = result.masks.data.detach().cpu().numpy()  # [n, mh, mw]
            except Exception:
                seg_masks = None

        n = len(boxes_xyxy)
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1 = max(0, min(int(round(x1)), w))
            y1 = max(0, min(int(round(y1)), h))
            x2 = max(0, min(int(round(x2)), w))
            y2 = max(0, min(int(round(y2)), h))
            bbox = [x1, y1, x2, y2]
            conf = float(confs[i])
            label = str(names.get(int(clss[i]), self.aux if self.aux else "A"))

            if seg_masks is not None and i < len(seg_masks):
                full = _resize_mask(seg_masks[i].astype(np.float32), w, h)
            else:
                # bbox detector (or masks unavailable): fill the bbox rectangle
                full = np.zeros((h, w), dtype=np.float32)
                if x2 > x1 and y2 > y1:
                    full[y1:y2, x1:x2] = 1.0
            yield bbox, conf, label, full

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1,
               detailer_hook=None):
        """Run detection and return ``(image_shape, [SEG, ...])``.

        ``image`` is a ComfyUI IMAGE tensor [B, H, W, C]. dilation/crop_factor/
        drop_size follow the same meaning downstream nodes expect."""
        drop_size = max(int(drop_size), 1)
        h = int(image.shape[1])
        w = int(image.shape[2])

        items = []
        result = self._infer(image, threshold)
        for bbox, conf, label, full in self._iter_detections(result, w, h):
            x1, y1, x2, y2 = bbox
            if (x2 - x1) <= drop_size or (y2 - y1) <= drop_size:
                continue

            if dilation:
                full = _dilate_mask(full, dilation)

            crop_region = _make_crop_region(w, h, bbox, crop_factor)
            cx1, cy1, cx2, cy2 = crop_region
            cropped_image = image[:, cy1:cy2, cx1:cx2, :]
            cropped_mask = full[cy1:cy2, cx1:cx2].copy()

            seg = SEG(cropped_image, cropped_mask, conf, crop_region, bbox,
                      label, None)
            items.append(seg)

            if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
                try:
                    items = detailer_hook.post_detection(items)
                except Exception:
                    pass

        # segs[0] is the reference (height, width) of the source image — a plain
        # 2-tuple. Downstream rescales masks/crops against it, so it must NOT be
        # the full [B, H, W, C] tensor shape.
        return (h, w), items

    def detect_combined(self, image, threshold, dilation):
        """Return a single [H, W] float mask combining all detections (used by
        the *ForEach*-style combined-mask consumers)."""
        h = int(image.shape[1])
        w = int(image.shape[2])
        combined = np.zeros((h, w), dtype=np.float32)
        result = self._infer(image, threshold)
        for bbox, conf, label, full in self._iter_detections(result, w, h):
            if dilation:
                full = _dilate_mask(full, dilation)
            combined = np.maximum(combined, full)
        return torch.from_numpy(combined)


# Distinct public classes so a caller can tell bbox vs segm apart if it checks.
class BBoxDetector(_UltraDetector):
    def __init__(self, model):
        super().__init__(model, use_masks=False)


class SegmDetector(_UltraDetector):
    def __init__(self, model):
        super().__init__(model, use_masks=True)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _list_models():
    choices = []
    if folder_paths is not None:
        try:
            choices += ["bbox/" + n for n in folder_paths.get_filename_list("ultralytics_bbox")]
        except Exception:
            pass
        try:
            choices += ["segm/" + n for n in folder_paths.get_filename_list("ultralytics_segm")]
        except Exception:
            pass
    return choices or ["bbox/face_yolov8m.pt"]


class UltralyticsDetectorProvider:
    """MickMumpitz clean-room provider for Ultralytics YOLO detectors.

    Outputs a BBOX_DETECTOR and a SEGM_DETECTOR wrapping the chosen model.
    Choose a ``bbox/...`` model for face/hand/object boxes, or a ``segm/...``
    model for instance masks. Downstream (FaceDetailer, SEGS tools, the CCC
    Face-Area splitter) consumes these exactly like the Impact Subpack node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_models(),),
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    RETURN_NAMES = ("bbox_detector", "segm_detector")
    FUNCTION = "load"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"

    def load(self, model_name):
        # If a default model is picked but not on disk yet (e.g. first run before
        # the background download finished), fetch it synchronously now so the
        # run can proceed instead of failing.
        if not os.path.isfile(_resolve_model_path(model_name)):
            _ensure_default_models(background=False)
        model = _load_yolo(model_name)
        bbox = BBoxDetector(model)
        segm = SegmDetector(model)
        return (bbox, segm)


# NOTE: We deliberately do NOT download anything at import time. The one-time
# default-model fetch is triggered lazily from UltralyticsDetectorProvider.load()
# the first time the node actually runs and only if the chosen model is missing.
# This avoids any network activity merely from ComfyUI loading the pack.


NODE_CLASS_MAPPINGS = {
    "CCC_UltralyticsDetectorProvider": UltralyticsDetectorProvider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCC_UltralyticsDetectorProvider": "Ultralytics Detector Provider",
}
