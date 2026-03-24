"""
Iterative Video Generation Nodes

Queue-based loop nodes for iterative video generation in ComfyUI.
Re-queues directly into ComfyUI's PromptQueue from Python — no JS roundtrip.
"""

import copy
import logging
import os
import time
import uuid

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
_QUEUE_SNAPSHOT = None   # {"prompt": dict, "extra_data": dict, "outputs_to_execute": list}


# Iter node class names whose "iteration" input should be overridden
_ITER_NODE_TYPES = {"IterVideoRouter", "IterationSwitch", "ControlImageSlicer", "MultiChannelSlicer", "FrameAccumulator", "EndFrameInjector", "BoundaryFrameExtractor", "BoundaryFrameSplicer", "IterStringSelector", "ControlCrossfadeIterationFix"}


def _compute_outputs_to_execute(prompt):
    """Derive OUTPUT_NODE IDs from the prompt dict."""
    import nodes as comfy_nodes
    return [nid for nid, nd in prompt.items()
            if getattr(comfy_nodes.NODE_CLASS_MAPPINGS.get(nd.get("class_type", ""), object),
                       "OUTPUT_NODE", False)]


def _enqueue_next_iteration(iteration, last_frame_path):
    """Deep-copy the snapshot prompt, inject iteration values, and queue directly."""
    global _QUEUE_SNAPSHOT
    if _QUEUE_SNAPSHOT is None:
        logging.warning("[MMZ Iter] No queue snapshot available, cannot enqueue next iteration")
        return

    prompt = copy.deepcopy(_QUEUE_SNAPSHOT["prompt"])

    # Inject iteration + last_frame_path into all iter nodes
    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type", "")
        if class_type not in _ITER_NODE_TYPES:
            continue
        inputs = node_data.setdefault("inputs", {})
        inputs["iteration"] = iteration
        if class_type == "IterVideoRouter":
            inputs["previous_frame_path"] = last_frame_path

    extra_data = copy.deepcopy(_QUEUE_SNAPSHOT["extra_data"])
    extra_data["create_time"] = int(time.time() * 1000)

    prompt_id = str(uuid.uuid4())
    outputs_to_execute = list(_QUEUE_SNAPSHOT["outputs_to_execute"])

    # Priority -inf → front of queue (before any user-queued jobs)
    queue_tuple = (-float('inf'), prompt_id, prompt, extra_data, outputs_to_execute, {})
    PromptServer.instance.prompt_queue.put(queue_tuple)
    logging.info("[MMZ Iter] Enqueued iteration %d (prompt_id=%s)", iteration, prompt_id)


def _on_prompt_handler(json_data):
    """Intercept POST /prompt to handle resume detection.

    Iteration re-queuing is done directly via PromptQueue.put() from
    FrameAccumulator — this handler only needs to handle resume and
    capture client_id for the first queue.
    """
    prompt = json_data.get("prompt", {})

    if not _ITERATION_STATE:
        # Check for resume
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

            device = start_image.device if start_image is not None else start_frames.device
            return (start_frames.to(device), num_start_frames)

        # Fallback: load single frame from disk
        if previous_frame_path and os.path.isfile(previous_frame_path):
            loaded = load_image_as_tensor(previous_frame_path)
            device = start_image.device if start_image is not None else loaded.device
            return (loaded.to(device), num_start_frames)

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


class IterStringSelector:
    """Selects a string from a STRING_BATCH based on the current iteration.

    Falls back to the last non-empty string if the current iteration's
    string is empty or out of range.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_batch": ("STRING_BATCH",),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "select"
    CATEGORY = "Mickmumpitz/video/iteration"

    def select(self, string_batch, iteration=0):
        return (_select_iter_string(string_batch, iteration),)


class IterSeedBatch:
    """Batches multiple seeds into a SEED_BATCH for per-iteration seed control.

    Seeds set to -1 are treated as unset — the slicers will fall back to the
    last valid (>= 0) seed from a previous iteration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 25):
            optional[f"seed_{i}"] = ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF})
        return {
            "required": {
                "num_fields": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("SEED_BATCH",)
    RETURN_NAMES = ("seed_batch",)
    FUNCTION = "batch"
    CATEGORY = "Mickmumpitz/video/iteration"

    def batch(self, num_fields, **kwargs):
        seeds = []
        for i in range(1, num_fields + 1):
            seeds.append(kwargs.get(f"seed_{i}", -1))
        return (tuple(seeds),)


class IterPromptBuilder:
    """Batches per-iteration prompts and extracts preview frames from the input video.

    Outputs a STRING_BATCH plus an IMAGE batch of the frames where each
    iteration's new content starts (one frame per field).
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 25):
            optional[f"string_{i}"] = ("STRING", {"multiline": True, "default": ""})
        return {
            "required": {
                "input_video": ("IMAGE",),
                "frames_per_iteration": ("INT", {"default": 81, "min": 1, "max": 9999}),
                "num_start_frames": ("INT", {"default": 1, "min": 0, "max": 999}),
                "num_fields": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING_BATCH", "IMAGE")
    RETURN_NAMES = ("string_batch", "iteration_start_frames")
    FUNCTION = "build"
    CATEGORY = "Mickmumpitz/video/iteration"

    def build(self, input_video, frames_per_iteration, num_start_frames,
              num_fields, **kwargs):
        strings = []
        for i in range(1, num_fields + 1):
            strings.append(kwargs.get(f"string_{i}", ""))

        # Extract the frame where each iteration's new content begins
        step = max(1, frames_per_iteration - num_start_frames)
        total_frames = input_video.shape[0]

        indices = []
        for i in range(num_fields):
            indices.append(min(i * step, total_frames - 1))

        preview_frames = input_video[indices]

        return (tuple(strings), preview_frames)


def _select_iter_string(string_batch, iteration):
    """Select a string from a batch by iteration, falling back to last non-empty."""
    if not string_batch:
        return ""
    last_nonempty = ""
    for i in range(len(string_batch)):
        if string_batch[i].strip():
            last_nonempty = string_batch[i]
        if i == iteration:
            break
    return last_nonempty


def _select_iter_seed(seed_batch, iteration):
    """Select a seed from a batch by iteration, falling back to last valid (>= 0)."""
    if not seed_batch:
        return 0
    last_valid = 0
    for i in range(len(seed_batch)):
        if seed_batch[i] >= 0:
            last_valid = seed_batch[i]
        if i == iteration:
            break
    return last_valid


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
                "string_batch": ("STRING_BATCH",),
                "seed_batch": ("SEED_BATCH",),
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 999}),
                "extend_mode": (["none", "repeat_last", "loop"],),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("control_slice", "has_frames", "string", "seed")
    FUNCTION = "slice"
    CATEGORY = "Mickmumpitz/video/iteration"

    def slice(self, control_images, frames_per_iteration, iteration=0,
              string_batch=None, seed_batch=None, overlap_frames=0, extend_mode="none"):
        total = control_images.shape[0]

        # Select string/seed for this iteration (fallback to last valid)
        selected_string = _select_iter_string(string_batch, iteration)
        selected_seed = _select_iter_seed(seed_batch, iteration)

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
                return (sliced, True, selected_string, selected_seed)
            elif extend_mode == "repeat_last":
                # Return frames_per_iteration copies of the last frame
                last = control_images[-1:]
                sliced = last.repeat(frames_per_iteration, 1, 1, 1)
                return (sliced, False, selected_string, selected_seed)
            else:
                # "none" - return just the last frame as a minimal batch
                return (control_images[-1:], False, selected_string, selected_seed)

        if end <= total:
            # Full slice available
            sliced = control_images[start:end]
            return (sliced, True, selected_string, selected_seed)

        # Partial frames available
        available = control_images[start:total]

        if extend_mode == "repeat_last":
            pad_count = end - total
            last = control_images[-1:]
            padding = last.repeat(pad_count, 1, 1, 1)
            sliced = torch.cat([available, padding], dim=0)
            return (sliced, False, selected_string, selected_seed)
        elif extend_mode == "loop":
            needed = end - total
            indices = [i % total for i in range(needed)]
            padding = control_images[indices]
            sliced = torch.cat([available, padding], dim=0)
            return (sliced, False, selected_string, selected_seed)
        else:
            # "none" - return short batch
            return (available, False, selected_string, selected_seed)


class MultiChannelSlicer:
    """Slices control images, plate, and mask batches for the current iteration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_images": ("IMAGE",),
                "frames_per_iteration": ("INT", {"default": 81, "min": 1, "max": 9999}),
            },
            "optional": {
                "string_batch": ("STRING_BATCH",),
                "seed_batch": ("SEED_BATCH",),
                "plate": ("IMAGE",),
                "mask": ("MASK",),
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 999}),
                "extend_mode": (["none", "repeat_last", "loop"],),
            },
            "hidden": {
                "iteration": "INT",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("control_slice", "plate_slice", "mask_slice", "has_frames", "string", "seed")
    FUNCTION = "slice"
    CATEGORY = "Mickmumpitz/video/iteration"

    def _slice_batch(self, batch, frames_per_iteration, start, end, total, extend_mode):
        """Slice a single batch tensor using the computed start/end indices."""
        if start >= total:
            if extend_mode == "loop":
                indices = [i % total for i in range(start, end)]
                return batch[indices]
            elif extend_mode == "repeat_last":
                last = batch[-1:]
                return last.repeat(frames_per_iteration, *([1] * (batch.dim() - 1)))
            else:
                return batch[-1:]

        if end <= total:
            return batch[start:end]

        available = batch[start:total]
        if extend_mode == "repeat_last":
            pad_count = end - total
            last = batch[-1:]
            padding = last.repeat(pad_count, *([1] * (batch.dim() - 1)))
            return torch.cat([available, padding], dim=0)
        elif extend_mode == "loop":
            needed = end - total
            indices = [i % total for i in range(needed)]
            padding = batch[indices]
            return torch.cat([available, padding], dim=0)
        else:
            return available

    def slice(self, control_images, frames_per_iteration, iteration=0,
              string_batch=None, seed_batch=None, plate=None, mask=None,
              overlap_frames=0, extend_mode="none"):
        total = control_images.shape[0]

        # Select string/seed for this iteration (fallback to last valid)
        selected_string = _select_iter_string(string_batch, iteration)
        selected_seed = _select_iter_seed(seed_batch, iteration)

        if overlap_frames > 0:
            start = iteration * (frames_per_iteration - overlap_frames)
        else:
            start = iteration * frames_per_iteration
        end = start + frames_per_iteration

        has_frames = start < total

        # Slice control images
        control_slice = self._slice_batch(control_images, frames_per_iteration, start, end, total, extend_mode)

        # Slice plate
        if plate is not None:
            plate_total = plate.shape[0]
            plate_slice = self._slice_batch(plate, frames_per_iteration, start, end, plate_total, extend_mode)
        else:
            plate_slice = control_slice

        # Slice mask
        if mask is not None:
            mask_total = mask.shape[0]
            mask_slice = self._slice_batch(mask, frames_per_iteration, start, end, mask_total, extend_mode)
        else:
            # Return an empty mask matching the control slice spatial dims
            if control_slice.dim() == 4:
                mask_slice = torch.zeros(control_slice.shape[0], control_slice.shape[1], control_slice.shape[2])
            else:
                mask_slice = torch.zeros(1, 1, 1)

        # Adjust has_frames for extend modes
        if not has_frames and extend_mode in ("loop", "repeat_last"):
            if extend_mode == "loop":
                has_frames = True

        return (control_slice, plate_slice, mask_slice, has_frames, selected_string, selected_seed)


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
        global _ACTIVE_SESSION, _QUEUE_SNAPSHOT
        buffer_key = str(unique_id)
        _ACTIVE_SESSION = buffer_key

        # Snapshot the post-replacement prompt on first iteration (or whenever
        # snapshot is missing). This prompt comes from the hidden PROMPT input,
        # which is the fully-processed, post-replacement version — safe to
        # queue directly without going through apply_replacements again.
        if _QUEUE_SNAPSHOT is None and prompt is not None:
            client_id = PromptServer.instance.client_id
            extra_data = {"client_id": client_id} if client_id else {}
            _QUEUE_SNAPSHOT = {
                "prompt": copy.deepcopy(prompt),
                "extra_data": extra_data,
                "outputs_to_execute": _compute_outputs_to_execute(prompt),
            }

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

        # Loop control: queue next iteration directly into PromptQueue
        if iteration < total_iterations - 1:
            _ITERATION_STATE["iteration"] = iteration + 1
            _ITERATION_STATE["last_frame_path"] = last_frame_path

            _enqueue_next_iteration(iteration + 1, last_frame_path)
        else:
            # Final iteration — clear loop state (buffer kept for potential resume)
            _ITERATION_STATE.clear()
            _QUEUE_SNAPSHOT = None
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
    DESCRIPTION = "Extracts the two boundary frames surrounding a broken region in a video batch for external interpolation (RIFE, FILM, etc.). On iteration 0 the images pass through unchanged. On iteration 1+ the node outputs the frame immediately before and the frame immediately after the broken region so an interpolation node can generate replacements. Connect the outputs to BoundaryFrameSplicer to splice the interpolated frames back in."

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
    DESCRIPTION = "Splices interpolated frames back into the original video batch, replacing the broken region. Pair with BoundaryFrameExtractor. Automatically strips the boundary pair that most interpolation nodes include in their output, keeping only the interpolated middle frames. On iteration 0 images pass through unchanged. The num_start_frames, frame_offset, and num_replace values must match the connected BoundaryFrameExtractor."

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
    "MultiChannelSlicer": MultiChannelSlicer,
    "FrameAccumulator": FrameAccumulator,
    "BoundaryFrameExtractor": BoundaryFrameExtractor,
    "BoundaryFrameSplicer": BoundaryFrameSplicer,
    "IterStringSelector": IterStringSelector,
    "IterPromptBuilder": IterPromptBuilder,
    "IterSeedBatch": IterSeedBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IterVideoRouter": "Iter Video Router",
    "IterationSwitch": "Iteration Switch",
    "ControlImageSlicer": "Control Image Slicer",
    "MultiChannelSlicer": "Multi Channel Slicer",
    "FrameAccumulator": "Frame Accumulator",
    "BoundaryFrameExtractor": "Boundary Frame Extractor",
    "BoundaryFrameSplicer": "Boundary Frame Splicer",
    "IterStringSelector": "Iter String Selector",
    "IterPromptBuilder": "Iter Prompt Builder",
    "IterSeedBatch": "Iter Seed Batch",
}
