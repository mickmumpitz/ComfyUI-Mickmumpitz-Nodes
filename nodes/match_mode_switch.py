"""
Match Mode Switch Node for ComfyUI
==================================
A 3-way IMAGE switch for toggling how an inserted element is processed, without
rewiring the graph. Pick one of three branches with a dropdown and that input is
passed to the output:

  - Compose + Match        -> output of the Compose + Match node
  - Color Match            -> output of a color-match node
  - Disabled (pass-through)-> the input wired to ``passthrough`` (typically the
                              original / unprocessed footage), i.e. bypass matching

The inputs are **lazy**: only the selected branch is evaluated, so the other two
(potentially expensive) graphs don't run. Switching to "Disabled" therefore skips
both matching branches entirely.
"""

import torch


class MatchModeSwitch:
    # dropdown option -> input slot name
    _INPUT_FOR = {
        "Disabled (pass-through)": "passthrough",
        "Color Match": "color_match",
        "Compose + Match": "compose_match",
    }
    MODES = list(_INPUT_FOR.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "match_mode": (
                    cls.MODES,
                    {
                        "default": "Compose + Match",
                        "tooltip": "Selects how the generated VFX is processed before "
                        "output (only the selected branch runs):\n"
                        "- Disabled (pass-through): outputs the generated video unchanged.\n"
                        "- Color Match: only works from the second iteration on; matches "
                        "the video based on the first generation.\n"
                        "- Compose + Match: composites the VFX based on the mask and "
                        "matches its grading to the plate (heavier).",
                    },
                ),
            },
            "optional": {
                "passthrough": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "Disabled branch: wire the generated video here to "
                        "output it unchanged.",
                    },
                ),
                "color_match": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "Color Match branch: the color-matched video "
                        "(matched to the first generation; from the 2nd iteration on).",
                    },
                ),
                "compose_match": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "Compose + Match branch: output of the Compose + "
                        "Match node (composites the VFX and grades it to the plate).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Mickmumpitz/Image"
    DESCRIPTION = (
        "3-way image switch: route Compose + Match, Color Match, or a "
        "pass-through (Disabled) input to the output. Only the selected branch "
        "is evaluated, so unused branches don't waste compute."
    )

    def check_lazy_status(self, match_mode, compose_match=None, color_match=None, passthrough=None):
        # Only ask ComfyUI to evaluate the branch we actually need.
        needed = self._INPUT_FOR[match_mode]
        values = {
            "compose_match": compose_match,
            "color_match": color_match,
            "passthrough": passthrough,
        }
        if values[needed] is None:
            return [needed]
        return []

    def switch(self, match_mode, compose_match=None, color_match=None, passthrough=None):
        values = {
            "compose_match": compose_match,
            "color_match": color_match,
            "passthrough": passthrough,
        }
        out = values[self._INPUT_FOR[match_mode]]

        if out is None:
            # Selected branch isn't connected; fall back to any wired input so the
            # graph still produces a valid IMAGE rather than erroring.
            for v in (passthrough, compose_match, color_match):
                if v is not None:
                    out = v
                    break

        if out is None:
            out = torch.zeros((1, 1, 1, 3))

        return (out,)


NODE_CLASS_MAPPINGS = {
    "MatchModeSwitch": MatchModeSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatchModeSwitch": "Match Mode Switch",
}
