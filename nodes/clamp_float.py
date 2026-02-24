_clamp = lambda v, lo, hi: lo if v < lo else hi if v > hi else v


class ClampFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "min": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0, "min": -1e9, "max": 1e9, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "clamp"
    CATEGORY = "Mickmumpitz/Utils"

    def clamp(self, value, min, max):
        if min > max:
            min, max = max, min
        return (_clamp(value, min, max),)


NODE_CLASS_MAPPINGS = {
    "ClampFloat": ClampFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClampFloat": "Clamp Float",
}
