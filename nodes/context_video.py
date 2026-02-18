"""
Context Video Nodes

Pre-generate context frames for iterative video workflows.
Generate a full context video with WAN VIDEO, then extract the boundary
frames (start of iter 0 + last num_start_frames of each iteration).
"""

import torch


class ContextImageExtractor:
    """Extracts the boundary frames from a pre-generated context video.

    Given a context video of total_iterations * frames_per_iteration frames,
    extracts:
      - The first frame (start of iteration 0)
      - The last num_start_frames of each iteration (start frames for the next)

    Example: num_start_frames=4, total_iterations=3, frames_per_iteration=81
    Output frames (1-indexed): 1, 78,79,80,81, 159,160,161,162
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_images": ("IMAGE",),
                "num_start_frames": ("INT", {"default": 4, "min": 1, "max": 99}),
                "total_iterations": ("INT", {"default": 5, "min": 1, "max": 9999}),
                "frames_per_iteration": ("INT", {"default": 81, "min": 1, "max": 9999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("boundary_frames",)
    FUNCTION = "extract"
    CATEGORY = "Mickmumpitz/video/context"

    def extract(self, context_images, num_start_frames,
                total_iterations, frames_per_iteration):
        max_idx = context_images.shape[0] - 1

        # First frame: start of iteration 0
        indices = [0]

        # Last num_start_frames of each iteration (except the final one)
        for i in range(1, total_iterations):
            end_of_prev = i * frames_per_iteration
            for j in range(num_start_frames, 0, -1):
                indices.append(min(end_of_prev - j, max_idx))

        return (context_images[indices],)


class ControlEndFrameExtractor:
    """Extracts start frame + end frames from control images for iterative video.

    Given control images and iteration parameters, extracts:
      - Frame 0 (start frame)
      - Frame (frames_per_iteration - 1)  (end of iteration 0)
      - Frame (frames_per_iteration - 1) + (frames_per_iteration - num_start_frames)  (end of iteration 1)
      - ... and so on until the input is exhausted.

    Example: num_start_frames=4, frames_per_iteration=81
    Extracted indices (0-based): 0, 80, 157, 234, ...
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_start_frames": ("INT", {"default": 4, "min": 1, "max": 99}),
                "frames_per_iteration": ("INT", {"default": 81, "min": 2, "max": 9999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("end_frames",)
    FUNCTION = "extract"
    CATEGORY = "Mickmumpitz/video/context"

    def extract(self, images, num_start_frames, frames_per_iteration):
        total = images.shape[0]

        # Start frame
        indices = [0]

        # End of iteration 0
        end_idx = frames_per_iteration - 1
        step = frames_per_iteration - num_start_frames

        while end_idx < total:
            indices.append(end_idx)
            end_idx += step

        return (images[indices],)


NODE_CLASS_MAPPINGS = {
    "ContextImageExtractor": ContextImageExtractor,
    "ControlEndFrameExtractor": ControlEndFrameExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContextImageExtractor": "Context Image Extractor",
    "ControlEndFrameExtractor": "Control End Frame Extractor",
}
