"""
Panorama Tools
==============

Turn one ordinary photo into a partial equirectangular (360) canvas and clean up the
result.

Nodes
-----
* Perspective to Pano Warp   : photo -> ERP canvas + masks + geometry guide
* Estimate FOV               : recover horizontal FOV + pitch (vanishing points, no EXIF)
* Pano Seam Roll             : roll the wrap seam to the centre + seam mask (its own inverse)
* Stage Switch               : lazy on/off bypass between two images
* Harmonize Boundary         : colour-match the generated surroundings to the placed photo
* Unfilled Area Mask         : mask the near-black regions an outpaint left unfilled

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
)

NODE_CLASS_MAPPINGS = {
    "MickmumpitzPanoWarp": PerspToErpWarp,
    "MickmumpitzPanoEstimateFOV": EstimateFOV,
    "MickmumpitzPanoSeamRoll": SeamRoll,
    "MickmumpitzPanoStageSwitch": StageSwitch,
    "MickmumpitzPanoHarmonizeBoundary": HarmonizeBoundary,
    "MickmumpitzPanoUnfilledMask": UnfilledMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzPanoWarp": "Perspective to Pano Warp",
    "MickmumpitzPanoEstimateFOV": "Estimate FOV",
    "MickmumpitzPanoSeamRoll": "Pano Seam Roll",
    "MickmumpitzPanoStageSwitch": "Stage Switch",
    "MickmumpitzPanoHarmonizeBoundary": "Harmonize Boundary",
    "MickmumpitzPanoUnfilledMask": "Unfilled Area Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
