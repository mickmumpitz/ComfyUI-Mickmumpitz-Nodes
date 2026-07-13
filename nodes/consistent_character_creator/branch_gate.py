"""
Branch Gate — ConsistentCharacterCreator
========================================

A tiny switch that hard-skips an expensive branch (e.g. an upscaler) when a
boolean is off.

Why this exists
---------------
A normal value-switch ("If/Else Switch") on the OUTPUT of an expensive node only
chooses which result flows downstream — the expensive node can still be forced to
run if anything else references its output (a preview, an Image Comparer, a
merge, or a lazy-evaluation quirk introduced by Get/Set virtual nodes). The
result: the upscaler runs even when "Use Upscaler" is off.

Placing this gate on the INPUT of the expensive branch fixes that at the source:
when `enabled` is False it returns an ExecutionBlocker, so ComfyUI skips the
upscaler (and its model loaders) entirely — nothing downstream can force it. Wire
`enabled` to the SAME boolean that drives your output switch, so one toggle
controls both.
"""

from comfy_execution.graph_utils import ExecutionBlocker


class BranchGate:
    """Passes `images` through when `enabled` is True; returns an ExecutionBlocker
    when False so the downstream branch (upscaler etc.) is skipped, not run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # forceInput: drive this from the same boolean as the output switch
                # so a single toggle controls the whole branch.
                "enabled": ("BOOLEAN", {
                    "default": True, "forceInput": True,
                    "tooltip": "When False, blocks this branch so the downstream "
                               "upscaler is skipped entirely (not run-and-discarded).",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "gate"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Gate for an expensive branch. Passes images when enabled; when disabled "
        "returns an ExecutionBlocker so the downstream upscaler is skipped entirely. "
        "Put it on the upscaler's input and drive 'enabled' from the same boolean as "
        "your output switch."
    )

    def gate(self, images, enabled):
        if enabled:
            return (images,)
        return (ExecutionBlocker(None),)


NODE_CLASS_MAPPINGS = {
    "CCC_BranchGate": BranchGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCC_BranchGate": "Branch Gate (skip upscaler when off)",
}
