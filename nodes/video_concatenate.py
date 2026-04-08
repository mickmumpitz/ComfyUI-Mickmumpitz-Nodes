from ._video_utils import concatenate_videos


class VideoConcatenate:
    """Concatenate multiple VIDEO inputs end-to-end, preserving audio."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"video_{i}"] = ("VIDEO",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "concatenate"
    CATEGORY = "Mickmumpitz/Video"

    def concatenate(self, **kwargs):
        videos = []
        for i in range(1, 11):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                videos.append(v)

        if len(videos) == 0:
            raise ValueError("VideoConcatenate: at least one video input must be connected.")

        if len(videos) == 1:
            return (videos[0],)

        return (concatenate_videos(videos),)


NODE_CLASS_MAPPINGS = {
    "VideoConcatenate": VideoConcatenate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatenate": "Video Concatenate",
}
