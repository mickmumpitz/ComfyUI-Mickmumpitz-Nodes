"""
WAN Video Resolution Picker Node for ComfyUI
Resolution presets compatible with WAN 2.1 (divisible by 16).
"""

RESOLUTIONS = {
    "512p": (912, 512),
    "576p": (1024, 576),
    "720p": (1280, 720),
    "1024p square": (1024, 1024),
    "Custom": None,
}

RESOLUTION_LIST = list(RESOLUTIONS.keys())
ASPECT_RATIOS = ["16:9", "9:16 (Vertical)"]


class WanResolutionPicker:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (RESOLUTION_LIST, {"default": "720p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "custom_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 16}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "Mickmumpitz/utils"

    def get_resolution(self, resolution: str, aspect_ratio: str, custom_width: int, custom_height: int):
        if resolution == "Custom":
            return (custom_width, custom_height)

        if resolution == "1024p square":
            return (1024, 1024)

        width, height = RESOLUTIONS[resolution]

        if aspect_ratio == "9:16 (Vertical)":
            return (height, width)
        return (width, height)


NODE_CLASS_MAPPINGS = {
    "WanResolutionPicker": WanResolutionPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanResolutionPicker": "WAN Resolution Picker",
}
