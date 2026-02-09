"""
Preprocess Settings Node for ComfyUI
A toggle + dropdown selector that outputs a signal for conditional processing.
"""

PREPROCESS_OPTIONS = ["Depth", "Canny"]


class PreprocessSettings:
    """
    Control preprocessing with a toggle and method selector.
    Outputs a signal string that can be used with PreprocessSwitch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "activate": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "method": (PREPROCESS_OPTIONS, {"default": PREPROCESS_OPTIONS[0]}),
            },
        }

    RETURN_TYPES = ("PREPROCESS_SIGNAL",)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_signal"
    CATEGORY = "Mickmumpitz/utils"

    def get_signal(self, activate: bool, method: str):
        if not activate:
            signal = "none"
        else:
            signal = method.lower()
        return (signal,)


class PreprocessSwitch:
    """
    Routes images based on the preprocess signal.
    Uses lazy evaluation - only the selected input path gets executed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("PREPROCESS_SIGNAL",),
            },
            "optional": {
                "original": ("IMAGE", {"lazy": True}),
                "depth": ("IMAGE", {"lazy": True}),
                "canny": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Mickmumpitz/utils"

    def check_lazy_status(self, signal, original=None, depth=None, canny=None):
        """Tell ComfyUI which inputs we actually need based on the signal."""
        needed = []
        if signal == "depth":
            if depth is None:
                needed.append("depth")
        elif signal == "canny":
            if canny is None:
                needed.append("canny")
        else:  # "none" - use original
            if original is None:
                needed.append("original")
        return needed

    def switch(self, signal: str, original=None, depth=None, canny=None):
        if signal == "depth" and depth is not None:
            return (depth,)
        elif signal == "canny" and canny is not None:
            return (canny,)
        elif original is not None:
            return (original,)
        else:
            raise ValueError("Required image input not connected")


NODE_CLASS_MAPPINGS = {
    "PreprocessSettings": PreprocessSettings,
    "PreprocessSwitch": PreprocessSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreprocessSettings": "Preprocess Settings",
    "PreprocessSwitch": "Preprocess Switch",
}
