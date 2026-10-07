"""
Storyboard
==========

Shot list -> one image per shot -> storyboard sheet.

Nodes
-----
* Make Storyboard Data : the shot list (title / frames / prompt per shot)
* Get Storyboard Shot  : pull one shot out of the list by index
* Make Storyboard Grid : lay the shot images out as one sheet with titles + timeline

All nodes live under the `Mickmumpitz/Storyboard` category.
"""

from .data import NODE_CLASS_MAPPINGS as DATA_MAPPINGS
from .data import NODE_DISPLAY_NAME_MAPPINGS as DATA_DISPLAY_MAPPINGS
from .get_shot import NODE_CLASS_MAPPINGS as GET_SHOT_MAPPINGS
from .get_shot import NODE_DISPLAY_NAME_MAPPINGS as GET_SHOT_DISPLAY_MAPPINGS
from .grid import NODE_CLASS_MAPPINGS as GRID_MAPPINGS
from .grid import NODE_DISPLAY_NAME_MAPPINGS as GRID_DISPLAY_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **DATA_MAPPINGS,
    **GET_SHOT_MAPPINGS,
    **GRID_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DATA_DISPLAY_MAPPINGS,
    **GET_SHOT_DISPLAY_MAPPINGS,
    **GRID_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
