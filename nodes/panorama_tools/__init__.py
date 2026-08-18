"""
Panorama Tools
==============

Turn one ordinary photo into a partial equirectangular (360) canvas and clean up the
result.

Nodes
-----
* Perspective to Pano Warp   : photo -> ERP canvas + masks + geometry guide
* Estimate FOV               : recover horizontal FOV + pitch (DEPRECATED -- type it by hand)
* Pano Seam Roll             : roll the wrap seam to the centre + seam mask (its own inverse)
* Stage Switch               : lazy on/off bypass between two images
* Harmonize Boundary         : colour-match the generated surroundings to the placed photo
* Unfilled Area Mask         : mask the near-black regions an outpaint left unfilled
* Krea2 Full-Res Reference   : full-resolution Krea-2 reference latents (fixes top-left anchor)
* Pano Roll Horizontal       : roll a pano by any fraction of its width (seam to centre and back)
* Pano Seam Mask             : centred seam-strip mask with in-strip feather, from width/height

All nodes live under the `Mickmumpitz/Panorama` category.

NOTE: the display titles below are intentionally neutral placeholders -- edit them freely.
"""

from .nodes import (
    PerspToErpWarp,
    EstimateFOV,
    SeamRoll,
    StageSwitch,
    HarmonizeBoundary,
    UnfilledMask,
    Krea2FullResReference,
    PanoRollHorizontal,
    PanoSeamMask,
)

NODE_CLASS_MAPPINGS = {
    "MickmumpitzPanoWarp": PerspToErpWarp,
    "MickmumpitzPanoEstimateFOV": EstimateFOV,
    "MickmumpitzPanoSeamRoll": SeamRoll,
    "MickmumpitzPanoStageSwitch": StageSwitch,
    "MickmumpitzPanoHarmonizeBoundary": HarmonizeBoundary,
    "MickmumpitzPanoUnfilledMask": UnfilledMask,
    "MickmumpitzPanoKrea2Reference": Krea2FullResReference,
    "MickmumpitzPanoRollHorizontal": PanoRollHorizontal,
    "MickmumpitzPanoSeamMask": PanoSeamMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzPanoWarp": "Perspective to Pano Warp",
    "MickmumpitzPanoEstimateFOV": "Estimate FOV",
    "MickmumpitzPanoSeamRoll": "Pano Seam Roll",
    "MickmumpitzPanoStageSwitch": "Stage Switch",
    "MickmumpitzPanoHarmonizeBoundary": "Harmonize Boundary",
    "MickmumpitzPanoUnfilledMask": "Unfilled Area Mask",
    "MickmumpitzPanoKrea2Reference": "Krea2 Full-Res Reference",
    "MickmumpitzPanoRollHorizontal": "Pano Roll Horizontal",
    "MickmumpitzPanoSeamMask": "Pano Seam Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
