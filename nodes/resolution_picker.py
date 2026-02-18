"""
Resolution Picker Node for ComfyUI
Standard resolution presets with aspect ratio options.
"""

# Resolution presets: name -> (width_16:9, height_16:9)
RESOLUTIONS = {
    "512p": (912, 512),
    "576p": (1024, 576),
    "720p": (1280, 720),
    "1024p square": (1024, 1024),
    "1080p": (1920, 1080),
    "1152p": (2048, 1152),
    "2160p": (3840, 2160),
    "Custom": None,
}

RESOLUTION_LIST = list(RESOLUTIONS.keys())
ASPECT_RATIOS = ["16:9", "9:16 (Vertical)", "1:1 (Square)"]


class ResolutionPicker:
    """
    Pick from standard resolutions with aspect ratio options.
    Custom mode allows manual width/height entry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (RESOLUTION_LIST, {"default": "1080p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "custom_width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "Mickmumpitz/utils"

    def get_resolution(self, resolution: str, aspect_ratio: str, custom_width: int, custom_height: int):
        if resolution == "Custom":
            return (custom_width, custom_height)

        base = RESOLUTIONS[resolution]

        # Special case for 1024p square - always square
        if resolution == "1024p square":
            return (1024, 1024)

        width, height = base

        if aspect_ratio == "16:9":
            return (width, height)
        elif aspect_ratio == "9:16 (Vertical)":
            return (height, width)  # Swap for vertical
        else:  # 1:1 (Square)
            size = height  # Use the shorter dimension
            return (size, size)


NODE_CLASS_MAPPINGS = {
    "ResolutionPicker": ResolutionPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionPicker": "Resolution Picker",
}
