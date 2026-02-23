"""
Iterative Video Generation Nodes

Queue-based loop nodes for iterative video generation in ComfyUI.
Uses server events + client-side JS to re-queue the workflow each iteration.
"""

import os
import torch
import numpy as np
from PIL import Image

import folder_paths
from server import PromptServer

# Global state
FRAME_BUFFERS = {}   # buffer_key -> tensor (B, H, W, C) on CPU
FRAME_SIZES = {}     # buffer_key -> list of cumulative frame counts after each iteration
_ACTIVE_SESSION = None  # buffer_key of the last active FrameAccumulator
_ITERATION_STATE = {}   # {"iteration": int, "last_frame_path": str}


# Iter node class names whose "iteration" input should be overridden
_ITER_NODE_TYPES = {"IterVideoRouter", "IterationSwitch", "ControlImageSlicer", "FrameAccumulator", "EndFrameInjector", "BoundaryFrameExtractor", "BoundaryFrameSplicer"}


def _on_prompt_handler(json_data):
    """Intercept every POST /prompt to inject the correct iteration value.

    This is necessary because inside ComfyUI group nodes (subgraphs), inner
    nodes are NOT in app.graph._nodes, so the JS widget-update approach fails.
    The prompt dict, however, always contains the expanded inner nodes.

    Auto-requeue is detected via a per-prompt marker in extra_data (not a
    global flag), so concurrent/queued prompts can't corrupt each other.
    """
    prompt = json_data.get("prompt", {})
    extra_data = json_data.get("extra_data", {})
    is_auto = extra_data.get("mmz_auto_requeue", False)

    if is_auto and _ITERATION_STATE:
        # Auto-requeue: inject iteration into all iter nodes from Python-side state.
        # Always set (not just override) since iteration is now a hidden input.
        iteration = _ITERATION_STATE["iteration"]
        last_frame_path = _ITERATION_STATE.get("last_frame_path", "")
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            if class_type not in _ITER_NODE_TYPES:
                continue
            inputs = node_data.setdefault("inputs", {})
            inputs["iteration"] = iteration
            if class_type == "IterVideoRouter":
                inputs["previous_frame_path"] = last_frame_path
        return json_data

    # Not an auto-requeue — check for resume
    if not _ITERATION_STATE:
        global _ACTIVE_SESSION
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") != "FrameAccumulator":
                continue
            inputs = node_data.get("inputs", {})
            resume_val = inputs.get("resume_from_iteration")
            # Only detect resume if it's a literal int > 0 (not a link)
            if isinstance(resume_val, (int, float)) and int(resume_val) > 0:
                buffer_key = str(node_id)
                if buffer_key in FRAME_BUFFERS:
                    resume_iter = int(resume_val)
                    # Restore active session so IterVideoRouter can find the buffer
                    _ACTIVE_SESSION = buffer_key
                    # Inject iteration into all iter nodes
                    for nid, nd in prompt.items():
                        ct = nd.get("class_type", "")
                        if ct in _ITER_NODE_TYPES:
                            nd.setdefault("inputs", {})["iteration"] = resume_iter
                    break
    return json_data


PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)


def _safe_filename(buffer_key: str) -> str:
    """Sanitize buffer_key for use in file/directory names (colon is illegal on Windows)."""
    return buffer_key.replace(":", "_")


def save_last_frame_to_temp(last_frame: torch.Tensor, buffer_key: str) -> str:
    """Save the last frame tensor to ComfyUI temp dir, return the path."""
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"mmz_iter_lastframe_{_safe_filename(buffer_key)}.png"
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


def save_iteration_frames(frames: torch.Tensor, buffer_key: str, iteration: int):
    """Save individual frames from an iteration to the temp directory."""
    temp_dir = folder_paths.get_temp_directory()
    iter_dir = os.path.join(temp_dir, f"mmz_iter_{_safe_filename(buffer_key)}", f"iter_{iteration:04d}")
    os.makedirs(iter_dir, exist_ok=True)

    for i in range(frames.shape[0]):
        img_np = (frames[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(iter_dir, f"frame_{i:06d}.png"))


class IterVideoRouter:
    """Routes between initial start image (iteration 0) and last N frames from the buffer (iteration > 0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),
            },
            "optional": {
                "num_start_frames": ("INT", {"default": 1, "min": 1, "max": 99}),
            },
            "hidden": {
                "iteration": "INT",
                "previous_frame_path": "STRING",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("current_start", "num_start_frames")
    FUNCTION = "route"
    CATEGORY = "Mickmumpitz/video/iteration"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def route(self, start_image, iteration=0, num_start_frames=1, previous_frame_path=""):
        if iteration == 0:
            return (start_image, num_start_frames)

        # Pull last N frames from the in-memory buffer
        if _ACTIVE_SESSION and _ACTIVE_SESSION in FRAME_BUFFERS:
            buffer = FRAME_BUFFERS[_ACTIVE_SESSION]

            # Use FRAME_SIZES to find the correct read position.
            # Critical during resume: the buffer hasn't been truncated yet,
            # so reading from the end would grab frames from the wrong iteration.
            if (_ACTIVE_SESSION in FRAME_SIZES
                    and len(FRAME_SIZES[_ACTIVE_SESSION]) >= iteration):
                end_pos = FRAME_SIZES[_ACTIVE_SESSION][iteration - 1]
                start_pos = max(0, end_pos - num_start_frames)
                start_frames = buffer[start_pos:end_pos]
            else:
                start_frames = buffer[-num_start_frames:]

            return (start_frames.to(start_image.device), num_start_frames)

        # Fallback: load single frame from disk
        if previous_frame_path and os.path.isfile(previous_frame_path):
            loaded = load_image_as_tensor(previous_frame_path)
            return (loaded.to(start_image.device), num_start_frames)

        return (start_image, num_start_frames)


class IterationSwitch:
    """Passes original on iteration 0, processed on iteration 1+.
    Use this to conditionally apply processing (e.g. color correction) only after the first iteration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "processed": ("IMAGE",),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Mickmumpitz/video/iteration"

    def switch(self, original, processed, iteration=0):
        if iteration == 0:
            return (original,)
        return (processed,)


class ControlImageSlicer:
    """Slices the full control image batch for the current iteration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_images": ("IMAGE",),
                "frames_per_iteration": ("INT", {"default": 81, "min": 1, "max": 9999}),
            },
            "optional": {
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 999}),
                "extend_mode": (["none", "repeat_last", "loop"],),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("control_slice", "has_frames")
    FUNCTION = "slice"
    CATEGORY = "Mickmumpitz/video/iteration"

    def slice(self, control_images, frames_per_iteration, iteration=0,
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
                "total_iterations": ("INT", {"default": 5, "min": 1, "max": 9999}),
            },
            "optional": {
                "num_start_frames": ("INT", {"default": 0, "min": 0, "max": 99}),
                "blend_overlap": ("BOOLEAN", {"default": False}),
                "resume_from_iteration": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "save_intermediate": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "iteration": "INT",
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

    def accumulate(self, new_frames, total_iterations,
                   iteration=0, num_start_frames=0, blend_overlap=False,
                   resume_from_iteration=0, save_intermediate=False,
                   unique_id=None, prompt=None, extra_pnginfo=None):
        global _ACTIVE_SESSION
        buffer_key = str(unique_id)
        _ACTIVE_SESSION = buffer_key

        # Safety: clear iteration state on fresh start so stale state can't persist
        if iteration == 0:
            _ITERATION_STATE.clear()

        # Handle resume: truncate buffer to before this iteration
        if (resume_from_iteration > 0
                and iteration == resume_from_iteration
                and buffer_key in FRAME_BUFFERS
                and buffer_key in FRAME_SIZES):
            sizes = FRAME_SIZES[buffer_key]
            if len(sizes) >= resume_from_iteration:
                target_size = sizes[resume_from_iteration - 1]
                FRAME_BUFFERS[buffer_key] = FRAME_BUFFERS[buffer_key][:target_size]
                FRAME_SIZES[buffer_key] = sizes[:resume_from_iteration]

        # Trim/blend start frames that overlap with the previous iteration
        if num_start_frames > 0 and iteration > 0:
            if blend_overlap and buffer_key in FRAME_BUFFERS:
                # Cross-fade the overlapping region instead of hard-cutting
                n = num_start_frames
                buffer_tail = FRAME_BUFFERS[buffer_key][-n:]
                new_head = new_frames[:n].cpu()
                # Alpha ramp: 1.0 (keep buffer) -> 0.0 (keep new)
                alpha = torch.linspace(1.0, 0.0, n).view(n, 1, 1, 1)
                blended = buffer_tail * alpha + new_head * (1.0 - alpha)
                FRAME_BUFFERS[buffer_key][-n:] = blended
                new_frames = new_frames[n:]
            else:
                # Hard trim: discard duplicate start frames
                new_frames = new_frames[num_start_frames:]

        # Accumulate in global buffer
        if iteration == 0:
            FRAME_BUFFERS[buffer_key] = new_frames.cpu()
            FRAME_SIZES[buffer_key] = []
        else:
            if buffer_key not in FRAME_BUFFERS:
                # No buffer exists (e.g., after restart), start fresh
                FRAME_BUFFERS[buffer_key] = new_frames.cpu()
                FRAME_SIZES[buffer_key] = []
            else:
                FRAME_BUFFERS[buffer_key] = torch.cat(
                    [FRAME_BUFFERS[buffer_key], new_frames.cpu()], dim=0
                )

        # Track buffer size after this iteration
        if buffer_key not in FRAME_SIZES:
            FRAME_SIZES[buffer_key] = []
        FRAME_SIZES[buffer_key].append(FRAME_BUFFERS[buffer_key].shape[0])

        all_frames = FRAME_BUFFERS[buffer_key]
        last_frame = new_frames[-1:]

        # Save last frame to temp dir for next iteration's IterVideoRouter
        last_frame_path = save_last_frame_to_temp(last_frame, buffer_key)

        # Optionally save intermediate frames to disk
        if save_intermediate:
            save_iteration_frames(new_frames, buffer_key, iteration)

        # Loop control via server events
        if iteration < total_iterations - 1:
            # Set Python-side state BEFORE triggering re-queue
            _ITERATION_STATE["iteration"] = iteration + 1
            _ITERATION_STATE["last_frame_path"] = last_frame_path

            PromptServer.instance.send_sync("mmz-add-queue", {}, PromptServer.instance.client_id)
        else:
            # Final iteration — clear loop state (buffer kept for potential resume)
            _ITERATION_STATE.clear()
            PromptServer.instance.send_sync("mmz-iter-reset", {}, PromptServer.instance.client_id)

        return (all_frames, all_frames.shape[0], last_frame)


class BoundaryFrameExtractor:
    """Extracts the boundary frames around the broken region for external interpolation.

    On iteration 0: outputs the original images as-is for both boundary frames
    (passthrough, no fixing needed).
    On iteration 1+: outputs the left and right boundary frames so you can feed
    them into any interpolation node (RIFE, FILM, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "num_start_frames": ("INT", {"default": 1, "min": 1, "max": 99,
                                              "tooltip": "Number of overlap/start frames (connect from IterVideoRouter)"}),
                "frame_offset": ("INT", {"default": 1, "min": 0, "max": 99,
                                          "tooltip": "Position of first broken frame relative to end of start frames (1 = first frame after overlap)"}),
                "num_replace": ("INT", {"default": 1, "min": 1, "max": 99,
                                         "tooltip": "Number of consecutive frames to replace"}),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BOOLEAN", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_before", "frame_after", "enabled", "num_start_frames", "frame_offset", "num_replace")
    FUNCTION = "extract"
    CATEGORY = "Mickmumpitz/video/iteration"

    def extract(self, images, enabled=True, num_start_frames=1, frame_offset=1, num_replace=1, iteration=0):
        if not enabled or iteration == 0:
            return (images, images[:1], images[:1], enabled, num_start_frames, frame_offset, num_replace)

        total = images.shape[0]
        target_start = num_start_frames + frame_offset - 1
        target_end = target_start + num_replace  # exclusive

        if target_start < 1 or target_end >= total:
            return (images, images[:1], images[-1:], enabled, num_start_frames, frame_offset, num_replace)

        frame_before = images[target_start - 1:target_start]  # (1, H, W, C)
        frame_after = images[target_end:target_end + 1]        # (1, H, W, C)

        return (images, frame_before, frame_after, enabled, num_start_frames, frame_offset, num_replace)


class BoundaryFrameSplicer:
    """Splices interpolated frames back into the original batch, replacing the broken region.

    On iteration 0: passes images through unchanged.
    On iteration 1+: replaces the broken frames with the interpolated ones.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "interpolated_frames": ("IMAGE",),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
                "num_start_frames": ("INT", {"default": 1, "min": 1, "max": 99,
                                              "tooltip": "Must match BoundaryFrameExtractor"}),
                "frame_offset": ("INT", {"default": 1, "min": 0, "max": 99,
                                          "tooltip": "Must match BoundaryFrameExtractor"}),
                "num_replace": ("INT", {"default": 1, "min": 1, "max": 99,
                                         "tooltip": "Must match BoundaryFrameExtractor"}),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "splice"
    CATEGORY = "Mickmumpitz/video/iteration"

    def splice(self, images, interpolated_frames, enabled=True, num_start_frames=1,
               frame_offset=1, num_replace=1, iteration=0):
        if not enabled or iteration == 0:
            return (images,)

        total = images.shape[0]
        target_start = num_start_frames + frame_offset - 1
        target_end = target_start + num_replace

        if target_start < 1 or target_end > total:
            return (images,)

        # Interpolation nodes typically include the boundary pair in the output:
        # [frame_before, interp1, ..., interpN, frame_after]
        # Strip the first and last frames to get only the interpolated middles.
        n_interp = interpolated_frames.shape[0]
        if n_interp > 2:
            inner = interpolated_frames[1:-1]
        elif n_interp == 2:
            # Only the two boundary frames came back, no actual interpolation —
            # fall back to averaging them as a single replacement frame
            inner = (interpolated_frames[:1] + interpolated_frames[1:]) / 2.0
        else:
            # Single frame returned — use it directly
            inner = interpolated_frames

        result = images.clone()
        n_to_splice = min(inner.shape[0], num_replace)
        result[target_start:target_start + n_to_splice] = inner[:n_to_splice]

        return (result,)


NODE_CLASS_MAPPINGS = {
    "IterVideoRouter": IterVideoRouter,
    "IterationSwitch": IterationSwitch,
    "ControlImageSlicer": ControlImageSlicer,
    "FrameAccumulator": FrameAccumulator,
    "BoundaryFrameExtractor": BoundaryFrameExtractor,
    "BoundaryFrameSplicer": BoundaryFrameSplicer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IterVideoRouter": "Iter Video Router",
    "IterationSwitch": "Iteration Switch",
    "ControlImageSlicer": "Control Image Slicer",
    "FrameAccumulator": "Frame Accumulator",
    "BoundaryFrameExtractor": "Boundary Frame Extractor",
    "BoundaryFrameSplicer": "Boundary Frame Splicer",
}
