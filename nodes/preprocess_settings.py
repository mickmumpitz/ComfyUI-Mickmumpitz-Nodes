"""
Preprocess Settings Node for ComfyUI
A toggle + dropdown selector that outputs a signal for conditional processing.
"""

PREPROCESS_OPTIONS = [
    "depth",
    "canny",
    "pose",
    "depth + canny",
    "depth + pose",
    "canny + pose",
]

PREPROCESS_OPTIONS_SINGLE = [
    "depth",
    "canny",
    "pose",
]

# Map combo signals to the individual inputs they need
SIGNAL_INPUTS = {
    "depth": ["depth"],
    "canny": ["canny"],
    "pose": ["pose"],
    "depth + canny": ["depth", "canny"],
    "depth + pose": ["depth", "pose"],
    "canny + pose": ["canny", "pose"],
}


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
            signal = method
        return (signal,)


class PreprocessSettingsSimple:
    """
    Preprocessing method selector without a toggle.
    Always outputs the selected method as the signal.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (PREPROCESS_OPTIONS, {"default": PREPROCESS_OPTIONS[0]}),
            },
        }

    RETURN_TYPES = ("PREPROCESS_SIGNAL",)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_signal"
    CATEGORY = "Mickmumpitz/utils"

    def get_signal(self, method: str):
        return (method,)


class PreprocessSettingsSingle:
    """
    Preprocessing method selector with only single method options (no combos).
    No activate toggle — always outputs the selected method.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (PREPROCESS_OPTIONS_SINGLE, {"default": PREPROCESS_OPTIONS_SINGLE[0]}),
            },
        }

    RETURN_TYPES = ("PREPROCESS_SIGNAL",)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_signal"
    CATEGORY = "Mickmumpitz/utils"

    def get_signal(self, method: str):
        return (method,)


class PreprocessSwitch:
    """
    Routes images based on the preprocess signal.
    Uses lazy evaluation - only the selected input paths get executed.
    For combo signals (e.g. "depth + canny"), returns a batch of both images.
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
                "pose": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Mickmumpitz/utils"

    def check_lazy_status(self, signal, original=None, depth=None, canny=None, pose=None):
        """Tell ComfyUI which inputs we actually need based on the signal."""
        inputs = {"depth": depth, "canny": canny, "pose": pose}
        needed_keys = SIGNAL_INPUTS.get(signal)

        if needed_keys is None:
            # "none" — use original
            if original is None:
                return ["original"]
            return []

        return [k for k in needed_keys if inputs[k] is None]

    def switch(self, signal: str, original=None, depth=None, canny=None, pose=None):
        import torch

        inputs = {"depth": depth, "canny": canny, "pose": pose}
        needed_keys = SIGNAL_INPUTS.get(signal)

        if needed_keys is None:
            if original is not None:
                return (original,)
            raise ValueError("Required image input not connected")

        images = [inputs[k] for k in needed_keys if inputs[k] is not None]
        if not images:
            raise ValueError("Required image input(s) not connected")

        if len(images) == 1:
            return (images[0],)

        return (torch.cat(images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "PreprocessSettings": PreprocessSettings,
    "PreprocessSettingsSimple": PreprocessSettingsSimple,
    "PreprocessSettingsSingle": PreprocessSettingsSingle,
    "PreprocessSwitch": PreprocessSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreprocessSettings": "Preprocess Settings",
    "PreprocessSettingsSimple": "Preprocess Settings (Simple)",
    "PreprocessSettingsSingle": "Preprocess Settings (Single)",
    "PreprocessSwitch": "Preprocess Switch",
}
