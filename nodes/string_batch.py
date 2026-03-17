class StringBatch:
    """Batch multiple strings into a STRING_BATCH for multi-prompt workflows."""

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

    RETURN_TYPES = ("STRING_BATCH",)
    RETURN_NAMES = ("string_batch",)
    FUNCTION = "batch"
    CATEGORY = "Mickmumpitz/String Batch"

    def batch(self, num_fields, **kwargs):
        strings = []
        for i in range(1, num_fields + 1):
            strings.append(kwargs.get(f"string_{i}", ""))
        return (tuple(strings),)


class BatchSelector:
    """Select one STRING_BATCH from up to 5 inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 6):
            optional[f"batch_{i}"] = ("STRING_BATCH",)
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING_BATCH",)
    RETURN_NAMES = ("string_batch",)
    FUNCTION = "select_batch"
    CATEGORY = "Mickmumpitz/String Batch"

    def select_batch(self, select, **kwargs):
        batch = kwargs.get(f"batch_{select}")
        if batch is None:
            return (("",),)
        return (batch,)


class StringSelector:
    """Select a single string from a STRING_BATCH by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_batch": ("STRING_BATCH",),
                "select": ("INT", {"default": 1, "min": 1, "max": 24, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "select_string"
    CATEGORY = "Mickmumpitz/String Batch"

    def select_string(self, string_batch, select):
        index = max(0, min(select - 1, len(string_batch) - 1))
        return (string_batch[index],)


class PromptStitcher:
    """Stitch a style string onto each prompt in a STRING_BATCH."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_batch": ("STRING_BATCH",),
                "style": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
                "num_outputs": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",) * 24
    RETURN_NAMES = tuple(f"prompt_{i}" for i in range(1, 25))
    FUNCTION = "stitch"
    CATEGORY = "Mickmumpitz/String Batch"

    def stitch(self, string_batch, style, separator, num_outputs):
        results = []
        for i in range(24):
            if i < len(string_batch) and i < num_outputs:
                prompt = string_batch[i]
                if style.strip():
                    prompt = prompt + separator + style
                results.append(prompt)
            else:
                results.append("")
        return tuple(results)


NODE_CLASS_MAPPINGS = {
    "StringBatch": StringBatch,
    "BatchSelector": BatchSelector,
    "StringSelector": StringSelector,
    "PromptStitcher": PromptStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringBatch": "String Batch",
    "BatchSelector": "Batch Selector",
    "StringSelector": "String Selector",
    "PromptStitcher": "Prompt Stitcher",
}
