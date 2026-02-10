"""
ComfyUI Custom Node: Frame Context Fit
Fits a long image sequence into a smaller context window by:
  1. Repeating key frames (every N input frames) a configurable number of times
  2. Selecting intermediate frames with configurable easing
"""

import json
import torch
import math
import numpy as np


def ease_linear(t):
    return t

def ease_in_quad(t):
    return t * t

def ease_out_quad(t):
    return 1.0 - (1.0 - t) * (1.0 - t)

def ease_in_out_quad(t):
    return 2.0 * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0

def ease_in_cubic(t):
    return t * t * t

def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3

def ease_in_out_cubic(t):
    return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

def ease_in_sine(t):
    return 1.0 - math.cos(t * math.pi / 2.0)

def ease_out_sine(t):
    return math.sin(t * math.pi / 2.0)

def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1.0) / 2.0

def ease_in_expo(t):
    return 0.0 if t == 0.0 else 2.0 ** (10.0 * t - 10.0)

def ease_out_expo(t):
    return 1.0 if t == 1.0 else 1.0 - 2.0 ** (-10.0 * t)

def ease_in_out_expo(t):
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    if t < 0.5:
        return 2.0 ** (20.0 * t - 10.0) / 2.0
    return (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0

def ease_in_back(t):
    c1 = 1.70158
    c3 = c1 + 1.0
    return c3 * t * t * t - c1 * t * t

def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((t - 1.0) ** 3) + c1 * ((t - 1.0) ** 2)

def ease_in_out_back(t):
    c1 = 1.70158
    c2 = c1 * 1.525
    if t < 0.5:
        return ((2.0 * t) ** 2 * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
    return ((2.0 * t - 2.0) ** 2 * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0


EASING_FUNCTIONS = {
    "linear": ease_linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_sine": ease_in_sine,
    "ease_out_sine": ease_out_sine,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_in_expo": ease_in_expo,
    "ease_out_expo": ease_out_expo,
    "ease_in_out_expo": ease_in_out_expo,
    "ease_in_back": ease_in_back,
    "ease_out_back": ease_out_back,
    "ease_in_out_back": ease_in_out_back,
}


class FrameContextFit:
    """
    Fits an image sequence into a context window with key-frame repetition and easing.

    Given an input batch of images (e.g. 240 frames), this node produces an output
    batch of `context_window` frames structured as repeating segments:

        [key_frame × repeats] [eased intermediate frames] [key_frame × repeats] ...

    Key frames are sampled every `key_frame_interval` input frames (0, 80, 160, ...).
    Between key frames, intermediate frames are selected using the chosen easing curve
    to control temporal density (e.g. ease_out = more frames near the start of each segment).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_window": ("INT", {
                    "default": 80,
                    "min": 4,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Total number of output frames (the context window size).",
                }),
                "key_frame_interval": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Pick a key frame every N input frames (e.g. 80 → frames 0, 80, 160 ...). When using iterative video generation with num_start_frames > 0, the first interval is this value and subsequent intervals are (key_frame_interval - num_start_frames).",
                }),
                "num_start_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of overlapping start frames in iterative video generation. When > 0, key frames are placed at iteration boundaries: first at key_frame_interval, then every (key_frame_interval - num_start_frames) frames. Set to 0 for uniform spacing.",
                }),
                "key_frame_repeats": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many times each key frame is repeated in the output.",
                }),
                "easing_mode": (list(EASING_FUNCTIONS.keys()), {
                    "default": "linear",
                    "tooltip": "Easing curve applied when selecting intermediate frames between key frames.",
                }),
                "include_last_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, the very last input frame is always included as a final key frame (with repeats), even if it doesn't land exactly on an interval boundary.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("images", "key_frame_data", "frame_indices_report",)
    FUNCTION = "execute"
    CATEGORY = "image/sequence"
    DESCRIPTION = (
        "Compresses a long image sequence into a fixed-size context window. "
        "Key frames are repeated, and intermediate frames are selected with easing."
    )

    def execute(self, images: torch.Tensor, context_window: int, key_frame_interval: int,
                num_start_frames: int, key_frame_repeats: int, easing_mode: str,
                include_last_frame: bool):

        total_input = images.shape[0]
        ease_fn = EASING_FUNCTIONS[easing_mode]

        # ── Determine key frame positions (input indices) ──
        if num_start_frames > 0 and num_start_frames < key_frame_interval:
            # Iterative video generation mode:
            #   Iteration 0 ends at key_frame_interval
            #   Iteration N ends at previous + (key_frame_interval - num_start_frames)
            key_positions = [0]
            pos = key_frame_interval
            step = key_frame_interval - num_start_frames
            while pos < total_input:
                key_positions.append(pos)
                pos += step
        else:
            key_positions = list(range(0, total_input, key_frame_interval))
        if include_last_frame and key_positions[-1] != total_input - 1:
            key_positions.append(total_input - 1)

        num_keys = len(key_positions)
        total_key_slots = num_keys * key_frame_repeats

        if total_key_slots >= context_window:
            # Edge case: key frames alone fill or exceed the context window.
            # Just output repeated key frames, trimmed to context_window.
            indices = []
            kf_output_indices = []
            for kp in key_positions:
                kf_output_indices.append(len(indices))
                indices.extend([kp] * key_frame_repeats)
            indices = indices[:context_window]
            kf_output_indices = [i for i in kf_output_indices if i < context_window]
            key_frame_data = self._build_key_frame_data(
                key_positions, kf_output_indices, key_frame_repeats, total_input, len(indices))
            report = self._build_report(indices, key_positions, key_frame_repeats,
                                        easing_mode, total_input, context_window)
            return (images[indices], key_frame_data, report)

        # ── Determine how many segments need intermediate frames ──
        # Segments are the gaps *between* consecutive key frames.
        num_segments = num_keys - 1
        remaining_slots = context_window - total_key_slots

        if num_segments == 0 or remaining_slots <= 0:
            # Only one key frame (or no room for intermediates)
            kf_output_indices = [0]
            indices = [key_positions[0]] * min(key_frame_repeats, context_window)
            indices = indices[:context_window]
            key_frame_data = self._build_key_frame_data(
                key_positions[:1], kf_output_indices, key_frame_repeats, total_input, len(indices))
            report = self._build_report(indices, key_positions, key_frame_repeats,
                                        easing_mode, total_input, context_window)
            return (images[indices], key_frame_data, report)

        # Distribute intermediate slots across segments proportionally to their span
        segment_spans = []
        for i in range(num_segments):
            span = key_positions[i + 1] - key_positions[i] - 1  # excludable frames between keys
            segment_spans.append(max(span, 0))

        total_span = sum(segment_spans)
        if total_span == 0:
            # All key frames are adjacent; no intermediates possible
            slots_per_segment = [0] * num_segments
        else:
            # Proportional allocation with rounding
            slots_float = [(s / total_span) * remaining_slots for s in segment_spans]
            slots_per_segment = [int(math.floor(sf)) for sf in slots_float]
            # Distribute leftover slots by largest remainder
            leftover = remaining_slots - sum(slots_per_segment)
            remainders = [(slots_float[i] - slots_per_segment[i], i) for i in range(num_segments)]
            remainders.sort(reverse=True, key=lambda x: x[0])
            for j in range(leftover):
                slots_per_segment[remainders[j][1]] += 1

        # ── Build the full output index list ──
        indices = []
        kf_output_indices = []
        for seg_idx in range(num_segments):
            kp_start = key_positions[seg_idx]
            kp_end = key_positions[seg_idx + 1]

            # Key frame repeats
            kf_output_indices.append(len(indices))
            indices.extend([kp_start] * key_frame_repeats)

            # Intermediate frames with easing
            n_inter = slots_per_segment[seg_idx]
            if n_inter > 0 and (kp_end - kp_start) > 1:
                for i in range(n_inter):
                    # t goes from 0→1 across the intermediate slots
                    t = (i + 1) / (n_inter + 1)
                    eased_t = ease_fn(t)
                    # Map eased_t to an input frame index between kp_start+1 and kp_end-1
                    frame_idx = kp_start + eased_t * (kp_end - kp_start)
                    frame_idx = int(round(frame_idx))
                    frame_idx = max(kp_start + 1, min(kp_end - 1, frame_idx))
                    indices.append(frame_idx)
            elif n_inter > 0:
                # Adjacent key frames with slots to fill → just repeat start
                indices.extend([kp_start] * n_inter)

        # Final key frame repeats
        kf_output_indices.append(len(indices))
        indices.extend([key_positions[-1]] * key_frame_repeats)

        # Trim or pad to exact context_window size
        if len(indices) > context_window:
            indices = indices[:context_window]
            kf_output_indices = [i for i in kf_output_indices if i < context_window]
        elif len(indices) < context_window:
            # Pad with last frame
            indices.extend([indices[-1]] * (context_window - len(indices)))

        key_frame_data = self._build_key_frame_data(
            key_positions, kf_output_indices, key_frame_repeats, total_input, len(indices))
        report = self._build_report(indices, key_positions, key_frame_repeats,
                                    easing_mode, total_input, context_window)
        return (images[indices], key_frame_data, report)

    @staticmethod
    def _build_key_frame_data(key_positions, kf_output_indices, key_frame_repeats,
                              total_input, total_output):
        return json.dumps({
            "key_frame_positions": key_positions,
            "key_frame_output_indices": kf_output_indices,
            "key_frame_repeats": key_frame_repeats,
            "total_input_frames": total_input,
            "total_output_frames": total_output,
        })

    @staticmethod
    def _build_report(indices, key_positions, key_frame_repeats, easing_mode,
                      total_input, context_window):
        unique_frames = sorted(set(indices))
        lines = [
            f"=== Frame Context Fit Report ===",
            f"Input frames:      {total_input}",
            f"Output frames:     {len(indices)}",
            f"Context window:    {context_window}",
            f"Key frame positions (input): {key_positions}",
            f"Key frame repeats: {key_frame_repeats}",
            f"Easing mode:       {easing_mode}",
            f"Unique frames used: {len(unique_frames)}",
            f"",
            f"Output index sequence:",
            f"  {indices}",
        ]
        return "\n".join(lines)


class AnchorFrameExtractor:
    """
    Extracts one representative frame per key-frame group from Frame Context Fit's
    compressed output.

    Connect the compressed images and key_frame_data from Frame Context Fit.
    For each key-frame repeat group (e.g. [81,81,81,81,81] at output positions 22-26),
    this node picks one frame based on the chosen pick_position strategy.
    """

    PICK_POSITIONS = ["second_to_last", "first", "middle", "last"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "key_frame_data": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON key frame data from Frame Context Fit's key_frame_data output.",
                }),
                "pick_position": (cls.PICK_POSITIONS, {
                    "default": "second_to_last",
                    "tooltip": "Which frame to pick from each key-frame repeat group. 'second_to_last' picks the one before the last repeat.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("anchor_frames", "anchor_data", "report",)
    FUNCTION = "execute"
    CATEGORY = "image/sequence"
    DESCRIPTION = (
        "Extracts one anchor frame per key-frame group from Frame Context Fit's "
        "compressed output. Connect the compressed images and key_frame_data."
    )

    def execute(self, images: torch.Tensor, key_frame_data: str, pick_position: str):

        total_output = images.shape[0]
        data = json.loads(key_frame_data)
        kf_output_indices = data["key_frame_output_indices"]
        repeats = data["key_frame_repeats"]
        key_positions = data["key_frame_positions"]
        num_groups = len(kf_output_indices)

        pick_indices = []
        for g in range(num_groups):
            start = kf_output_indices[g]
            # Effective group size: capped by next group start or end of output
            if g + 1 < num_groups:
                group_size = min(repeats, kf_output_indices[g + 1] - start)
            else:
                group_size = min(repeats, total_output - start)
            group_size = max(group_size, 1)

            if pick_position == "first":
                pick = start
            elif pick_position == "last":
                pick = start + group_size - 1
            elif pick_position == "middle":
                pick = start + group_size // 2
            else:  # second_to_last
                pick = start + max(0, group_size - 2)

            pick_indices.append(pick)

        # Build JSON metadata for injector
        anchor_data = json.dumps({
            "key_frame_positions": key_positions[:num_groups],
            "pick_output_indices": pick_indices,
            "total_output_frames": total_output,
        })

        # Build human-readable report
        report_lines = [
            "=== Anchor Frame Extractor Report ===",
            f"Compressed frames:     {total_output}",
            f"Key frame groups:      {num_groups}",
            f"Key frame repeats:     {repeats}",
            f"Pick position:         {pick_position}",
            f"Group start indices:   {kf_output_indices}",
            f"Picked output indices: {pick_indices}",
            f"Original input frames: {key_positions[:num_groups]}",
        ]

        return (images[pick_indices], anchor_data, "\n".join(report_lines))


# ── ComfyUI Registration ──
NODE_CLASS_MAPPINGS = {
    "FrameContextFit": FrameContextFit,
    "AnchorFrameExtractor": AnchorFrameExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameContextFit": "Frame Context Fit 🎞️",
    "AnchorFrameExtractor": "Anchor Frame Extractor 🎯",
}
