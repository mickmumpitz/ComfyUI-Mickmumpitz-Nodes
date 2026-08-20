"""
ConsistentCharacterCreator (CCC)
================================

LoRA dataset-building nodes: batch loaders, the model-specific captioning Prompt
Studio, the Ideogram 4 tagger, bbox converters, the interactive Dataset Reviewer,
face-area batch routing, and the branch gate.

All nodes live under the `Mickmumpitz/ConsistentCharacterCreator` category and
keep the `CCC_` node-key prefix they were authored with, so workflows built
against the original standalone pack keep loading.
"""

from .dataset_creation import NODE_CLASS_MAPPINGS as DATASET_CREATION_MAPPINGS
from .dataset_creation import NODE_DISPLAY_NAME_MAPPINGS as DATASET_CREATION_DISPLAY_MAPPINGS
from .face_area_batch import NODE_CLASS_MAPPINGS as FACE_AREA_BATCH_MAPPINGS
from .face_area_batch import NODE_DISPLAY_NAME_MAPPINGS as FACE_AREA_BATCH_DISPLAY_MAPPINGS
from .branch_gate import NODE_CLASS_MAPPINGS as BRANCH_GATE_MAPPINGS
from .branch_gate import NODE_DISPLAY_NAME_MAPPINGS as BRANCH_GATE_DISPLAY_MAPPINGS
# Clean-room Ultralytics detector provider (replaces the Impact-Subpack node) —
# lives here so it groups with the CCC face/detail nodes that consume it.
from .ultralytics_detector import NODE_CLASS_MAPPINGS as ULTRA_DETECTOR_MAPPINGS
from .ultralytics_detector import NODE_DISPLAY_NAME_MAPPINGS as ULTRA_DETECTOR_DISPLAY_MAPPINGS
from .ultralytics_detector import UltralyticsDetectorProvider as _ULTRA_CLASS

NODE_CLASS_MAPPINGS = {
    **DATASET_CREATION_MAPPINGS,
    **FACE_AREA_BATCH_MAPPINGS,
    **BRANCH_GATE_MAPPINGS,
    **ULTRA_DETECTOR_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DATASET_CREATION_DISPLAY_MAPPINGS,
    **FACE_AREA_BATCH_DISPLAY_MAPPINGS,
    **BRANCH_GATE_DISPLAY_MAPPINGS,
    **ULTRA_DETECTOR_DISPLAY_MAPPINGS,
}

# Drop-in alias: also register our class under the original subpack name, but ONLY
# if nothing else already claims it. When the Impact Subpack is installed it wins
# (no clash); when it's absent/broken, existing workflows that reference
# "UltralyticsDetectorProvider" transparently load our replacement instead.
try:
    import nodes as _comfy_nodes  # ComfyUI's global node registry

    _global = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {})
    if ("UltralyticsDetectorProvider" not in _global
            and "UltralyticsDetectorProvider" not in NODE_CLASS_MAPPINGS):
        NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"] = _ULTRA_CLASS
except Exception:
    pass
