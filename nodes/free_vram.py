"""
Conditional VRAM cleanup nodes.

Two complementary helpers for swapping between video and image models
without forcing a reload every run:

- FreeVRAMIfLoaded: unload everything when a target pattern *is* present
  in VRAM (e.g. ``ltx`` — purge after the video model has been used so
  the next image model has room).
- FreeVRAMUnlessLoaded: unload everything when a target pattern is *not*
  present in VRAM (e.g. ``flux`` — only purge if the image model isn't
  already there, so we don't churn it on every image run).

Cleanup runs at *queue time* via an on_prompt_handler, not during node
execution. This keeps the nodes out of the cache key signature of any
downstream node — your cached image/video generation outputs are not
invalidated when the cleanup state flips.
"""

import gc
import logging

import torch
import comfy.model_management as mm
from server import PromptServer


# --- helpers -----------------------------------------------------------------


def _model_identifier(loaded_model):
    """Best-effort class-name string for a LoadedModel.

    ``LoadedModel.model`` is typically a ``ModelPatcher``; ``.model.model``
    is the underlying ``nn.Module``. ComfyUI itself uses
    ``.model.model.__class__.__name__`` for unload logs, so we treat that
    as the canonical identity, plus the patcher class name for matching
    against custom wrappers.
    """
    parts = []
    patcher = getattr(loaded_model, "model", None)
    if patcher is not None:
        parts.append(type(patcher).__name__)
        inner = getattr(patcher, "model", None)
        if inner is not None:
            parts.append(type(inner).__name__)
    return " ".join(parts)


def _is_pattern_loaded(pattern):
    if not pattern:
        return False
    needle = pattern.lower()
    for lm in list(mm.current_loaded_models):
        if needle in _model_identifier(lm).lower():
            return True
    return False


def _free_all():
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _literal_or(val, default):
    """Return val if it's a literal primitive, else default.

    Inside subgraphs, widget inputs can appear as link lists like
    ``["upstream_id", 0]`` in the expanded prompt. We can't resolve those
    at handler time, so we fall back to a default.
    """
    if isinstance(val, (int, float, bool, str)):
        return val
    return default


# --- queue-time handler ------------------------------------------------------


def _on_prompt_handler(json_data):
    """Run conditional VRAM cleanup before the prompt executes.

    Walks the submitted prompt for FreeVRAM* nodes and applies their
    rules using the *current* VRAM state. This runs even when downstream
    nodes will all cache-hit, so cleanup happens reliably without
    poisoning anyone's cache key.
    """
    prompt = json_data.get("prompt", {})

    if not prompt:
        return json_data

    should_free = False
    reasons = []

    for _node_id, node_data in prompt.items():
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        if class_type == "FreeVRAMIfLoaded":
            pattern = _literal_or(inputs.get("if_loaded", ""), "")
            if pattern and _is_pattern_loaded(pattern):
                should_free = True
                reasons.append(f"'{pattern}' is loaded")
        elif class_type == "FreeVRAMUnlessLoaded":
            pattern = _literal_or(inputs.get("unless_loaded", ""), "")
            if pattern and not _is_pattern_loaded(pattern):
                should_free = True
                reasons.append(f"'{pattern}' is not loaded")

    if should_free:
        logging.info(f"[FreeVRAM] unloading all models — {'; '.join(reasons)}")
        _free_all()

    return json_data


PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)


# --- nodes (config-only, no dataflow) ---------------------------------------


class FreeVRAMIfLoaded:
    """Unload all models when ``if_loaded`` is currently in VRAM.

    Drop this node onto the graph and set ``if_loaded`` to a substring
    of the video model's class name (e.g. ``ltx`` or ``wan``). When you
    queue a prompt and that model is detected in VRAM, everything is
    unloaded *before* execution starts. No wiring needed — this node is
    a config marker, not part of the dataflow, so it never invalidates
    downstream caches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "if_loaded": ("STRING", {"default": "ltx"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Mickmumpitz/Utils"
    OUTPUT_NODE = True

    def noop(self, if_loaded):
        return ()


class FreeVRAMUnlessLoaded:
    """Unload all models *unless* ``unless_loaded`` is already in VRAM.

    Drop this node onto the graph and set ``unless_loaded`` to a
    substring of the image model's class name (e.g. ``flux``). When you
    queue a prompt and that model isn't detected, everything else is
    unloaded *before* execution starts so the image loader has room.
    Once the image model is loaded on a subsequent queue, this no-ops.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unless_loaded": ("STRING", {"default": "flux"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Mickmumpitz/Utils"
    OUTPUT_NODE = True

    def noop(self, unless_loaded):
        return ()


NODE_CLASS_MAPPINGS = {
    "FreeVRAMIfLoaded": FreeVRAMIfLoaded,
    "FreeVRAMUnlessLoaded": FreeVRAMUnlessLoaded,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVRAMIfLoaded": "Free VRAM If Loaded",
    "FreeVRAMUnlessLoaded": "Free VRAM Unless Loaded",
}
