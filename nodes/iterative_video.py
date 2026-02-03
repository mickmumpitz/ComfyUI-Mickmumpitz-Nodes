"""
Iterative Video Generation Nodes

Queue-based loop nodes for iterative video generation in ComfyUI.
Uses server events + client-side JS to re-queue the workflow each iteration.
"""

import os
import uuid
import torch
import numpy as np
from PIL import Image

import folder_paths
from server import PromptServer

# Global frame buffer: session_id -> tensor (B, H, W, C) on CPU
FRAME_BUFFERS = {}


def save_last_frame_to_temp(last_frame: torch.Tensor, session_id: int) -> str:
    """Save the last frame tensor to ComfyUI temp dir, return the path."""
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"mmz_iter_lastframe_{session_id}.png"
    filepath = os.path.join(temp_dir, filename)

    # last_frame shape: (1, H, W, C), values 0-1
    img_np = (last_frame[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(filepath)
    return filepath


def load_image_as_tensor(filepath: str) -> torch.Tensor:
    """Load an image file and return as (1, H, W, C) tensor, values 0-1."""
    img = Image.open(filepath).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def save_iteration_frames(frames: torch.Tensor, session_id: int, iteration: int):
    """Save individual frames from an iteration to the temp directory."""
    temp_dir = folder_paths.get_temp_directory()
    iter_dir = os.path.join(temp_dir, f"mmz_iter_{session_id}", f"iter_{iteration:04d}")
    os.makedirs(iter_dir, exist_ok=True)

    for i in range(frames.shape[0]):
        img_np = (frames[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(iter_dir, f"frame_{i:06d}.png"))


class IterVideoRouter:
    """Routes between initial start image (iteration 0) and saved last frame (iteration > 0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),
                "iteration": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "previous_frame_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("current_start",)
    FUNCTION = "route"
    CATEGORY = "Mickmumpitz/video/iteration"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def route(self, start_image, iteration, previous_frame_path=""):
        if iteration == 0 or not previous_frame_path or not os.path.isfile(previous_frame_path):
            return (start_image,)

        loaded = load_image_as_tensor(previous_frame_path)
        # Match device of start_image
        loaded = loaded.to(start_image.device)
        return (loaded,)


class ControlImageSlicer:
    """Slices the full control image batch for the current iteration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_images": ("IMAGE",),
                "frames_per_iteration": ("INT", {"default": 81, "min": 1, "max": 9999}),
                "iteration": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 999}),
                "extend_mode": (["none", "repeat_last", "loop"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("control_slice", "has_frames")
    FUNCTION = "slice"
    CATEGORY = "Mickmumpitz/video/iteration"

    def slice(self, control_images, frames_per_iteration, iteration,
              overlap_frames=0, extend_mode="none"):
        total = control_images.shape[0]

        if overlap_frames > 0:
            start = iteration * (frames_per_iteration - overlap_frames)
        else:
            start = iteration * frames_per_iteration
        end = start + frames_per_iteration

        # Check if we have enough frames
        has_frames = start < total

        if start >= total:
            # No frames available at all for this iteration
            if extend_mode == "loop":
                # Wrap around
                indices = [i % total for i in range(start, end)]
                sliced = control_images[indices]
                return (sliced, True)
            elif extend_mode == "repeat_last":
                # Return frames_per_iteration copies of the last frame
                last = control_images[-1:]
                sliced = last.repeat(frames_per_iteration, 1, 1, 1)
                return (sliced, False)
            else:
                # "none" - return just the last frame as a minimal batch
                return (control_images[-1:], False)

        if end <= total:
            # Full slice available
            sliced = control_images[start:end]
            return (sliced, True)

        # Partial frames available
        available = control_images[start:total]

        if extend_mode == "repeat_last":
            pad_count = end - total
            last = control_images[-1:]
            padding = last.repeat(pad_count, 1, 1, 1)
            sliced = torch.cat([available, padding], dim=0)
            return (sliced, False)
        elif extend_mode == "loop":
            needed = end - total
            indices = [i % total for i in range(needed)]
            padding = control_images[indices]
            sliced = torch.cat([available, padding], dim=0)
            return (sliced, False)
        else:
            # "none" - return short batch
            return (available, False)


class FrameAccumulator:
    """Accumulates frames across iterations and controls the re-queue loop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_frames": ("IMAGE",),
                "iteration": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "total_iterations": ("INT", {"default": 5, "min": 1, "max": 9999}),
                "session_id": ("INT", {"default": 1, "min": 1, "max": 99999}),
            },
            "optional": {
                "trim_first_n": ("INT", {"default": 0, "min": 0, "max": 999}),
                "save_intermediate": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("all_frames", "frame_count", "last_frame")
    FUNCTION = "accumulate"
    CATEGORY = "Mickmumpitz/video/iteration"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def accumulate(self, new_frames, iteration, total_iterations, session_id,
                   trim_first_n=0, save_intermediate=False,
                   unique_id=None, prompt=None, extra_pnginfo=None):
        # Trim overlap frames if specified
        if trim_first_n > 0 and iteration > 0:
            new_frames = new_frames[trim_first_n:]

        # Accumulate in global buffer
        if iteration == 0:
            FRAME_BUFFERS[session_id] = new_frames.cpu()
        else:
            FRAME_BUFFERS[session_id] = torch.cat(
                [FRAME_BUFFERS[session_id], new_frames.cpu()], dim=0
            )

        all_frames = FRAME_BUFFERS[session_id]
        last_frame = new_frames[-1:]

        # Save last frame to temp dir for next iteration's IterVideoRouter
        last_frame_path = save_last_frame_to_temp(last_frame, session_id)

        # Optionally save intermediate frames to disk
        if save_intermediate:
            save_iteration_frames(new_frames, session_id, iteration)

        # Loop control via server events
        if iteration < total_iterations - 1:
            PromptServer.instance.send_sync("mmz-iter-update", {
                "session_id": session_id,
                "iteration": iteration + 1,
                "last_frame_path": last_frame_path,
            })
            PromptServer.instance.send_sync("mmz-add-queue", {})
        else:
            # Final iteration - clean up buffer reference (frames already returned)
            if session_id in FRAME_BUFFERS:
                del FRAME_BUFFERS[session_id]

        return (all_frames, all_frames.shape[0], last_frame)


NODE_CLASS_MAPPINGS = {
    "IterVideoRouter": IterVideoRouter,
    "ControlImageSlicer": ControlImageSlicer,
    "FrameAccumulator": FrameAccumulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IterVideoRouter": "Iter Video Router",
    "ControlImageSlicer": "Control Image Slicer",
    "FrameAccumulator": "Frame Accumulator",
}
