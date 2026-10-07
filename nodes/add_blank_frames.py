import torch


class MickmumpitzAddBlankFrames:
    """Adds black frames at the start or end of a ComfyUI IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blank_frames_count": (
                    "INT",
                    {"default": 1, "min": 0, "max": 1000, "step": 1},
                ),
                "insert_position": (["end", "start"], {"default": "end"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("new_batch",)
    FUNCTION = "add_blank_frames"
    CATEGORY = "Mickmumpitz/Video"
    DESCRIPTION = (
        "Adds black frames in the same resolution, dtype, and device as the input "
        "IMAGE batch. Insert frames at the start or end."
    )

    def add_blank_frames(self, images, blank_frames_count, insert_position):
        if images is None:
            raise ValueError("The 'images' input is required.")

        # Standard ComfyUI IMAGE tensor format: [batch, height, width, channels].
        if not isinstance(images, torch.Tensor):
            raise TypeError("Expected a ComfyUI IMAGE tensor as input.")
        if images.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor shape [B, H, W, C], got {tuple(images.shape)}."
            )

        count = max(0, int(blank_frames_count))
        if count == 0:
            return (images,)

        blank_frames = torch.zeros(
            (count, images.shape[1], images.shape[2], images.shape[3]),
            dtype=images.dtype,
            device=images.device,
        )

        if insert_position == "start":
            new_batch = torch.cat((blank_frames, images), dim=0)
        else:
            new_batch = torch.cat((images, blank_frames), dim=0)

        return (new_batch,)


NODE_CLASS_MAPPINGS = {
    "MickmumpitzAddBlankFrames": MickmumpitzAddBlankFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzAddBlankFrames": "Add Blank Frames",
}
