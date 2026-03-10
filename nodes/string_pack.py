class StringPack:
    """Pack multiple strings into a STRING_PACK for batch prompt workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 25):
            optional[f"string_{i}"] = ("STRING", {"multiline": True, "default": ""})
        return {
            "required": {
                "num_fields": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING_PACK",)
    RETURN_NAMES = ("string_pack",)
    FUNCTION = "pack"
    CATEGORY = "Mickmumpitz/String Pack"

    def pack(self, num_fields, **kwargs):
        strings = []
        for i in range(1, num_fields + 1):
            strings.append(kwargs.get(f"string_{i}", ""))
        return (tuple(strings),)


class PackSelector:
    """Select one STRING_PACK from up to 5 inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 6):
            optional[f"pack_{i}"] = ("STRING_PACK",)
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING_PACK",)
    RETURN_NAMES = ("string_pack",)
    FUNCTION = "select_pack"
    CATEGORY = "Mickmumpitz/String Pack"

    def select_pack(self, select, **kwargs):
        pack = kwargs.get(f"pack_{select}")
        if pack is None:
            return (("",),)
        return (pack,)


class StringSelector:
    """Select a single string from a STRING_PACK by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_pack": ("STRING_PACK",),
                "select": ("INT", {"default": 1, "min": 1, "max": 24, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "select_string"
    CATEGORY = "Mickmumpitz/String Pack"

    def select_string(self, string_pack, select):
        index = max(0, min(select - 1, len(string_pack) - 1))
        return (string_pack[index],)


class PromptStitcher:
    """Stitch a style string onto each prompt in a STRING_PACK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_pack": ("STRING_PACK",),
                "style": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
                "num_outputs": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",) * 24
    RETURN_NAMES = tuple(f"prompt_{i}" for i in range(1, 25))
    FUNCTION = "stitch"
    CATEGORY = "Mickmumpitz/String Pack"

    def stitch(self, string_pack, style, separator, num_outputs):
        results = []
        for i in range(24):
            if i < len(string_pack) and i < num_outputs:
                prompt = string_pack[i]
                if style.strip():
                    prompt = prompt + separator + style
                results.append(prompt)
            else:
                results.append("")
        return tuple(results)


NODE_CLASS_MAPPINGS = {
    "StringPack": StringPack,
    "PackSelector": PackSelector,
    "StringSelector": StringSelector,
    "PromptStitcher": PromptStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringPack": "String Pack",
    "PackSelector": "Pack Selector",
    "StringSelector": "String Selector",
    "PromptStitcher": "Prompt Stitcher",
}
