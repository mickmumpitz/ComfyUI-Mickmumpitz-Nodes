STORY_OPTIONS = ["Motorcycle vs. Insect", "A Knights Tale", "Cyberpunk"]
STYLE_OPTIONS = ["Realistic", "Anime", "Stop Motion"]


class StoryStyleSelector:
    """Select a story and style by name, outputs their 1-based indices."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story": (STORY_OPTIONS, {"default": STORY_OPTIONS[0]}),
                "style": (STYLE_OPTIONS, {"default": STYLE_OPTIONS[0]}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("story_index", "style_index")
    FUNCTION = "select"
    CATEGORY = "Mickmumpitz/String Batch"

    def select(self, story, style):
        story_index = STORY_OPTIONS.index(story) + 1
        style_index = STYLE_OPTIONS.index(style) + 1
        return (story_index, style_index)


NODE_CLASS_MAPPINGS = {
    "StoryStyleSelector": StoryStyleSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryStyleSelector": "Story & Style Selector",
}
