import json


MAX_SLOTS = 32


class Bundle:
    """Collapse many typed wires into a single BUNDLE cable.

    Slots are typed as wildcard ``*``; the companion JS extension retypes each
    slot to match the source output on connect and keeps one trailing empty
    slot visible. ``bundle_meta`` is a hidden JSON widget populated by JS with
    the ordered ``{names, types}`` for each connected slot, so the payload can
    be split downstream even across copy-paste / Reroute boundaries. The
    user-editable ``name`` widget lets ``UnbundleByName`` find this Bundle by
    name (even across subgraph boundaries).
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"input_{i}": ("*",) for i in range(1, MAX_SLOTS + 1)}
        optional["name"] = ("STRING", {"default": "", "multiline": False})
        optional["bundle_meta"] = ("STRING", {"default": "{}", "multiline": False})
        optional["slot_names"] = ("STRING", {"default": "{}", "multiline": False})
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("BUNDLE",)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "bundle"
    CATEGORY = "Mickmumpitz/Utils"

    def bundle(self, name="", bundle_meta="{}", slot_names="{}", **kwargs):
        try:
            meta = json.loads(bundle_meta or "{}")
        except (ValueError, TypeError):
            meta = {}

        names = list(meta.get("names") or [])
        types = list(meta.get("types") or [])
        sources = list(meta.get("sources") or [])

        if sources:
            # JS records the actual input slot name for each connected slot in
            # `sources`. Slot names are not contiguous (disconnecting a middle
            # slot can leave e.g. input_2, input_3, input_1-empty), so we MUST
            # key kwargs by the recorded source name rather than by position.
            values = [kwargs.get(s) for s in sources]
            while len(names) < len(values):
                names.append(f"output_{len(names) + 1}")
            while len(types) < len(values):
                types.append("*")
            names = names[:len(values)]
            types = types[:len(values)]
        elif names:
            values = [kwargs.get(f"input_{i + 1}") for i in range(len(names))]
        else:
            values = []
            for i in range(1, MAX_SLOTS + 1):
                v = kwargs.get(f"input_{i}")
                if v is not None:
                    values.append(v)
                    names.append(f"output_{len(values)}")
                    types.append("*")
                    sources.append(f"input_{i}")

        payload = {
            "name": name,
            "names": names,
            "types": types,
            "sources": sources,
            "values": values,
        }
        return (payload,)


class Unbundle:
    """Split a BUNDLE cable back into its individual wires.

    Outputs are pre-declared as 32 wildcard slots; the companion JS extension
    retypes and renames them by inspecting the source Bundle node (walking
    through Reroute nodes) at graph-design time. Python unpacks ``values`` in
    order, padding with ``None`` for unused outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle": ("BUNDLE",),
            },
            "optional": {
                "saved_meta": ("STRING", {"default": "{}", "multiline": False}),
            },
        }

    RETURN_TYPES = ("*",) * MAX_SLOTS
    RETURN_NAMES = tuple(f"output_{i}" for i in range(1, MAX_SLOTS + 1))
    FUNCTION = "unbundle"
    CATEGORY = "Mickmumpitz/Utils"

    def unbundle(self, bundle, saved_meta="{}"):
        if isinstance(bundle, dict):
            values = list(bundle.get("values") or [])
        else:
            values = []
        values = values + [None] * MAX_SLOTS
        return tuple(values[:MAX_SLOTS])


class UnbundleByName:
    """Split a BUNDLE cable by looking up its source Bundle node by name.

    Has no visible ``bundle`` input — the companion prompt handler scans the
    submitted prompt, finds the Bundle whose ``name`` widget matches
    ``source_bundle``, and injects the link. This works across subgraph
    boundaries because the frontend flattens subgraphs into the prompt dict
    before submission.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_bundle": ("STRING", {"default": "", "multiline": False}),
                "bundle": ("BUNDLE",),
            },
            "optional": {
                "saved_meta": ("STRING", {"default": "{}", "multiline": False}),
            },
        }

    RETURN_TYPES = ("*",) * MAX_SLOTS
    RETURN_NAMES = tuple(f"output_{i}" for i in range(1, MAX_SLOTS + 1))
    FUNCTION = "unbundle"
    CATEGORY = "Mickmumpitz/Utils"

    def unbundle(self, source_bundle, bundle, saved_meta="{}"):
        if isinstance(bundle, dict):
            values = list(bundle.get("values") or [])
        else:
            values = []
        values = values + [None] * MAX_SLOTS
        return tuple(values[:MAX_SLOTS])


def _on_prompt_handler(json_data):
    """Inject the BUNDLE link for each UnbundleByName based on its selected name.

    The frontend flattens subgraphs into ``prompt``, so a name-based lookup
    reaches across subgraph boundaries.
    """
    try:
        prompt = json_data.get("prompt", {}) if isinstance(json_data, dict) else {}
        if not isinstance(prompt, dict):
            return json_data

        bundles_by_name = {}
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "Bundle":
                continue
            inputs = node.get("inputs") or {}
            raw_name = inputs.get("name", "")
            if isinstance(raw_name, list):
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            bundles_by_name[name] = node_id

        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "UnbundleByName":
                continue
            inputs = node.setdefault("inputs", {})
            raw_source = inputs.get("source_bundle", "")
            if isinstance(raw_source, list):
                continue
            source = str(raw_source).strip()
            if source and source in bundles_by_name:
                inputs["bundle"] = [bundles_by_name[source], 0]
    except Exception:
        pass
    return json_data


try:
    from server import PromptServer
    PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)
except Exception:
    pass


NODE_CLASS_MAPPINGS = {
    "Bundle": Bundle,
    "Unbundle": Unbundle,
    "UnbundleByName": UnbundleByName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Bundle": "Bundle",
    "Unbundle": "Unbundle",
    "UnbundleByName": "Unbundle by Name",
}
