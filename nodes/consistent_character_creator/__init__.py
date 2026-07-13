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

NODE_CLASS_MAPPINGS = {
    **DATASET_CREATION_MAPPINGS,
    **FACE_AREA_BATCH_MAPPINGS,
    **BRANCH_GATE_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DATASET_CREATION_DISPLAY_MAPPINGS,
    **FACE_AREA_BATCH_DISPLAY_MAPPINGS,
    **BRANCH_GATE_DISPLAY_MAPPINGS,
}
