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
"""

import gc
import logging

import torch
import comfy.model_management as mm


def _model_identifier(loaded_model):
    """Best-effort class-name string for a LoadedModel.

    ``LoadedModel.model`` is typically a ``ModelPatcher``; ``.model.model``
    is the underlying ``nn.Module``. ComfyUI itself uses
    ``.model.model.__class__.__name__`` for unload logs, so we treat that
    as the canonical identity. We also tack on the patcher's class name
    in case the user wants to match against e.g. a custom wrapper.
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


class FreeVRAMIfLoaded:
    """Unload all models when ``if_loaded`` is currently in VRAM.

    Wire this after video gen: set ``if_loaded`` to a substring of the
    video model's class name (e.g. ``ltx`` or ``wan``). When the video
    model is detected, everything is unloaded. When it's already gone,
    this is a no-op passthrough.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "if_loaded": ("STRING", {"default": "ltx"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "maybe_free"
    CATEGORY = "Mickmumpitz/Utils"

    @classmethod
    def IS_CHANGED(cls, value, if_loaded):
        # State-aware: invalidate only when the "is the pattern loaded"
        # state flips. Avoids forcing a re-run (and downstream cache miss)
        # on every queue while still triggering cleanup the moment the
        # video model appears in VRAM.
        return f"loaded={_is_pattern_loaded(if_loaded)}"

    def maybe_free(self, value, if_loaded):
        if _is_pattern_loaded(if_loaded):
            logging.info(f"[FreeVRAMIfLoaded] '{if_loaded}' detected in VRAM — unloading all models")
            _free_all()
        return (value,)


class FreeVRAMUnlessLoaded:
    """Unload all models *unless* ``unless_loaded`` is already in VRAM.

    Wire this in front of the image checkpoint loader: set
    ``unless_loaded`` to a substring of the image model's class name
    (e.g. ``flux``). On the first run the image model isn't there,
    so we purge to make room. On subsequent runs the image model is
    already loaded — VRAM is left alone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "unless_loaded": ("STRING", {"default": "flux"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "maybe_free"
    CATEGORY = "Mickmumpitz/Utils"

    @classmethod
    def IS_CHANGED(cls, value, unless_loaded):
        return f"loaded={_is_pattern_loaded(unless_loaded)}"

    def maybe_free(self, value, unless_loaded):
        if not _is_pattern_loaded(unless_loaded):
            logging.info(f"[FreeVRAMUnlessLoaded] '{unless_loaded}' not in VRAM — unloading all models")
            _free_all()
        return (value,)


NODE_CLASS_MAPPINGS = {
    "FreeVRAMIfLoaded": FreeVRAMIfLoaded,
    "FreeVRAMUnlessLoaded": FreeVRAMUnlessLoaded,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVRAMIfLoaded": "Free VRAM If Loaded",
    "FreeVRAMUnlessLoaded": "Free VRAM Unless Loaded",
}
