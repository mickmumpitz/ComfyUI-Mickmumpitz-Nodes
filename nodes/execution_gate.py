from server import PromptServer

# Max hidden check slots injected onto each ExecutionGate by the handler
_MAX_CHECKS = 10


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
    """Auto-wire CheckOutput -> ExecutionGate dependencies by channel.

    For every ExecutionGate, inject hidden ``_check_N`` dependency slots
    pointing at all CheckOutput nodes sharing the same ``channel``. This
    forces the gate to wait for those nodes before releasing its value.

    An ExecutionGateControl on the same channel with ``enabled=False``
    suppresses wiring entirely — the gate passes through immediately.
    """
    prompt = json_data.get("prompt", {})

    control_enabled = {}          # channel -> bool
    checks_by_channel = {}        # channel -> list[node_id]
    gate_ids = []

    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        if class_type == "CheckOutput":
            channel = _literal_or(inputs.get("channel", "default"), "default")
            checks_by_channel.setdefault(channel, []).append(node_id)
        elif class_type == "ExecutionGate":
            gate_ids.append(node_id)
        elif class_type == "ExecutionGateControl":
            channel = _literal_or(inputs.get("channel", "default"), "default")
            enabled = _literal_or(inputs.get("enabled", True), True)
            control_enabled[channel] = bool(enabled)

    for gate_id in gate_ids:
        inputs = prompt[gate_id].setdefault("inputs", {})
        # Clear stale _check_N slots from prior runs
        for i in range(1, _MAX_CHECKS + 1):
            inputs.pop(f"_check_{i}", None)

        channel = _literal_or(inputs.get("channel", "default"), "default")

        if control_enabled.get(channel, True) is False:
            continue

        matching = checks_by_channel.get(channel, [])
        for i, check_id in enumerate(matching[:_MAX_CHECKS]):
            inputs[f"_check_{i + 1}"] = [check_id, 0]

    return json_data


PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)


class CheckOutput:
    """Signals graph completion up to this node via an auto-wired dependency.

    Connect any upstream value. Because this is an ``OUTPUT_NODE``, it
    always executes when its subgraph is reached. The prompt handler
    wires this node into every ExecutionGate sharing the same
    ``channel``, so the gate waits for it before releasing its value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "channel": ("STRING", {"default": "default"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "passthrough"
    CATEGORY = "Mickmumpitz/Utils"
    OUTPUT_NODE = True

    def passthrough(self, value, channel):
        return (value,)


class ExecutionGate:
    """Pass-through whose execution waits for CheckOutput nodes on its channel.

    Hidden ``_check_N`` dependencies are injected at submit time by the
    prompt handler from every CheckOutput with a matching ``channel``.
    An ExecutionGateControl on the same channel with ``enabled=False``
    bypasses the gate — it passes through without waiting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_CHECKS + 1):
            optional[f"_check_{i}"] = ("*",)
        return {
            "required": {
                "value": ("*",),
                "channel": ("STRING", {"default": "default"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "Mickmumpitz/Utils"

    def gate(self, value, channel="default", **kwargs):
        return (value,)


class ExecutionGateControl:
    """Centralized on/off switch for ExecutionGates on a channel.

    When ``enabled=False``, the prompt handler skips wiring CheckOutput
    dependencies into gates on the matching channel, so they pass
    through immediately. When ``enabled=True`` (or no control exists),
    gates wait for all CheckOutputs on their channel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channel": ("STRING", {"default": "default"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Mickmumpitz/Utils"
    OUTPUT_NODE = True

    def noop(self, channel, enabled):
        return ()


NODE_CLASS_MAPPINGS = {
    "ExecutionGate": ExecutionGate,
    "CheckOutput": CheckOutput,
    "ExecutionGateControl": ExecutionGateControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExecutionGate": "Execution Gate",
    "CheckOutput": "Check Output",
    "ExecutionGateControl": "Execution Gate Control",
}
