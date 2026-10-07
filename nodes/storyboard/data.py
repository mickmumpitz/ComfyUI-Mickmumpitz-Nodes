"""
Storyboard Data: the shot list.

One row per shot (title -> frames -> prompt). Outputs a single STORYBOARD_DATA
bundle plus the comma-joined titles and frame lengths, which plug straight into
Storyboard Grid. Use one Storyboard Get Shot node per generation group.
"""

# Cap for the data-entry widgets on THIS node (how many shot rows you can
# fill in). Raise if you need more than 20.
MAX_SHOTS = 20


class MickmumpitzStoryboardData:
    RETURN_TYPES = ("STORYBOARD_DATA", "STRING", "STRING")
    RETURN_NAMES = ("storyboard_data", "shot_names_joined", "frame_lengths_joined")
    FUNCTION = "run"
    CATEGORY = "Mickmumpitz/Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "shot_count": ("INT", {"default": 8, "min": 1, "max": MAX_SHOTS, "step": 1}),
        }
        optional = {}
        for i in range(1, MAX_SHOTS + 1):
            optional[f"shot{i}_title"] = ("STRING", {"default": f"Shot {i}", "multiline": False})
            optional[f"shot{i}_frames"] = ("INT", {"default": 24, "min": 1, "max": 100000, "step": 1})
            optional[f"shot{i}_prompt"] = ("STRING", {"default": "", "multiline": True})
        return {"required": required, "optional": optional}

    def run(self, shot_count, **kwargs):
        shots, names, lengths = [], [], []
        for i in range(1, shot_count + 1):
            title = kwargs.get(f"shot{i}_title", f"Shot {i}")
            frames = int(kwargs.get(f"shot{i}_frames", 24))
            prompt = kwargs.get(f"shot{i}_prompt", "")
            shots.append({
                "index": i,
                "title": title,
                "frames": frames,
                "prompt": prompt,
            })
            names.append(title)
            lengths.append(str(frames))
        return (shots, ",".join(names), ",".join(lengths))


NODE_CLASS_MAPPINGS = {
    "MickmumpitzStoryboardData": MickmumpitzStoryboardData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzStoryboardData": "Make Storyboard Data",
}
