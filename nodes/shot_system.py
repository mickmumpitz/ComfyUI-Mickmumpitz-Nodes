from server import PromptServer

from ._video_utils import concatenate_videos

# Maximum hidden dependency slots on ShotAssembler
_MAX_DEPS = 20


def _literal_or(val, default):
    """Return val if it's a literal primitive, else default.

    Inside subgraphs, widget inputs can appear as link lists like
    ``["upstream_id", 0]`` in the expanded prompt. For those we can't
    resolve the value at handler time, so we fall back to a default.
    """
    if isinstance(val, (int, float, bool, str)):
        return val
    return default


def _on_prompt_handler(json_data):
    """Auto-wire ShotVideoOutput -> ShotAssembler dependencies in shot order.

    Shots are wired to the assembler's hidden ``_dep_N`` slots sorted by
    ``shot_number``. Because ComfyUI passes cached outputs through
    dependencies, the assembler receives videos for ALL shots on every run
    — including shots whose ShotVideoOutput was cached and therefore never
    re-executed. This is what lets a single-shot regeneration still produce
    a complete assembled video.
    """
    prompt = json_data.get("prompt", {})
    shot_entries = []  # list of (shot_number, node_id)
    assembler_ids = []

    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type")
        if class_type == "ShotVideoOutput":
            inputs = node_data.get("inputs", {})
            enabled = _literal_or(inputs.get("enabled", True), True)
            if not enabled:
                continue
            # Fallback to 10_000 pushes link-bound shot_numbers after literal ones
            shot_number = _literal_or(inputs.get("shot_number", 1), 10_000)
            shot_entries.append((shot_number, node_id))
        elif class_type == "ShotAssembler":
            assembler_ids.append(node_id)

    # Sort by shot_number so dep slots are populated in playback order.
    # node_id tiebreaker keeps ordering deterministic if numbers collide.
    shot_entries.sort(key=lambda x: (x[0], x[1]))

    for assembler_id in assembler_ids:
        inputs = prompt[assembler_id].setdefault("inputs", {})
        # Clear any prior dep slots so stale wirings from a previous prompt
        # can't linger between runs.
        for i in range(1, _MAX_DEPS + 1):
            inputs.pop(f"_dep_{i}", None)
        for i, (_, shot_id) in enumerate(shot_entries):
            if i >= _MAX_DEPS:
                break
            inputs[f"_dep_{i + 1}"] = [shot_id, 0]  # output slot 0 (video)

    return json_data


PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)


# ---------------------------------------------------------------------------
# ShotVideoOutput
# ---------------------------------------------------------------------------
class ShotVideoOutput:
    """Tag a video with a shot number for automatic assembly.

    This is a passthrough: it simply exposes the video to the ShotAssembler
    via the auto-wired dependency slots. The ``shot_number`` and ``enabled``
    widgets are consumed by the prompt handler, not this method.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "shot_number": ("INT", {"default": 1, "min": 1, "max": 99}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "passthrough"
    CATEGORY = "Mickmumpitz/Shot"
    OUTPUT_NODE = True

    def passthrough(self, video, shot_number, enabled=True):
        return (video,)


# ---------------------------------------------------------------------------
# ShotAssembler
# ---------------------------------------------------------------------------
class ShotAssembler:
    """Collect all shot videos via hidden dep slots and concatenate in order.

    Dep slots are injected by the prompt handler in ``shot_number`` order,
    so iterating ``_dep_1 .. _dep_N`` naturally walks the shots in playback
    order. Cached ShotVideoOutputs still feed their videos through these
    dep slots, which is why the assembler always sees the full set — not
    just the shots that were re-executed this run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_DEPS + 1):
            optional[f"_dep_{i}"] = ("*",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("assembled_video",)
    FUNCTION = "assemble"
    CATEGORY = "Mickmumpitz/Shot"
    OUTPUT_NODE = True

    def assemble(self, **kwargs):
        videos = []
        for i in range(1, _MAX_DEPS + 1):
            v = kwargs.get(f"_dep_{i}")
            if v is not None and hasattr(v, "get_components"):
                videos.append(v)

        if not videos:
            raise ValueError(
                "ShotAssembler: no videos found. Ensure ShotVideoOutput nodes "
                "are in the workflow."
            )

        if len(videos) == 1:
            return (videos[0],)

        return (concatenate_videos(videos),)


NODE_CLASS_MAPPINGS = {
    "ShotVideoOutput": ShotVideoOutput,
    "ShotAssembler": ShotAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotVideoOutput": "Shot Video Output",
    "ShotAssembler": "Shot Assembler",
}
