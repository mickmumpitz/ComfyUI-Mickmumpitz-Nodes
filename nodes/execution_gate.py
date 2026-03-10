class ExecutionGate:
    """Pass-through node that creates an artificial execution dependency.

    Connect outputs from nodes that must finish first to the wait_for slots.
    The value input is passed through unchanged once all connected inputs are ready.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "wait_for_1": ("*",),
                "wait_for_2": ("*",),
                "wait_for_3": ("*",),
                "wait_for_4": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "Mickmumpitz/Utils"

    def gate(self, value, **kwargs):
        return (value,)


NODE_CLASS_MAPPINGS = {
    "ExecutionGate": ExecutionGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExecutionGate": "Execution Gate",
}
