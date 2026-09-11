"""Canny edges for video batches, a few frames at a time.

The core `Canny` node hands the whole batch to kornia in one call. That is
fine for a picture and fatal for a clip: measured at 1344x768 it peaks at
about 170 MB of VRAM per frame, so a 158-frame pass wants 27 GB and a
362-frame pass 57 GB, before any model is loaded. The edges themselves do
not depend on the neighbouring frames - hysteresis runs per image - so the
batch can be cut into slices and the result is the same to the bit.

Same thresholds, same output as the core node. `frames_per_pass` is the
only knob: 8 frames need about 1.5 GB at 1344x768 and the whole clip takes
a few seconds either way.
"""

import torch
from kornia.filters import canny

import comfy.model_management


class MickmumpitzCanny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Weak edges above this join a strong edge. Lower for more lines."}),
                "high_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Edges above this always count. Raise to keep only silhouettes."}),
                "frames_per_pass": ("INT", {
                    "default": 8, "min": 1, "max": 512,
                    "tooltip": "Frames sent to the GPU at once. 8 needs about 1.5 GB at 1344x768; "
                               "the result does not depend on this number."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "Mickmumpitz/utils"
    DESCRIPTION = ("Canny edge detection for image batches and video frames. Identical to the core "
                   "Detect Edges (Canny) node, but processes the clip a few frames at a time so a "
                   "long video does not run out of VRAM.")

    def detect(self, image, low_threshold, high_threshold, frames_per_pass):
        device = comfy.model_management.get_torch_device()
        out_device = comfy.model_management.intermediate_device()
        out = []
        for start in range(0, image.shape[0], frames_per_pass):
            chunk = image[start:start + frames_per_pass]
            chunk = chunk.to(device=device, dtype=torch.float32).movedim(-1, 1)
            edges = canny(chunk, low_threshold, high_threshold)[1]
            out.append(edges.to(device=out_device, dtype=image.dtype)
                       .repeat(1, 3, 1, 1).movedim(1, -1))
            del chunk, edges
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "MickmumpitzCanny": MickmumpitzCanny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzCanny": "Canny Edges (batched)",
}
