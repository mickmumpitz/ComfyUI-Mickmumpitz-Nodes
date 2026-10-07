"""
Storyboard Get Shot: pulls one shot (title / prompt / frames) out of the
Storyboard Data bundle by its 1-based shot_index.
"""

class MickmumpitzStoryboardGetShot:
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("title", "prompt", "frames")
    FUNCTION = "run"
    CATEGORY = "Mickmumpitz/Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": ("STORYBOARD_DATA",),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
            }
        }

    def run(self, storyboard_data, shot_index):
        for shot in storyboard_data:
            if shot["index"] == shot_index:
                return (shot["title"], shot["prompt"], shot["frames"])
        # shot_index not found (out of range for current shot_count) -> safe fallback
        return ("", "", 0)


NODE_CLASS_MAPPINGS = {
    "MickmumpitzStoryboardGetShot": MickmumpitzStoryboardGetShot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzStoryboardGetShot": "Get Storyboard Shot",
}
