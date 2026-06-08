import torch

try:
    from comfy.utils import common_upscale
except Exception:  # pragma: no cover - fallback if comfy isn't importable at import time
    common_upscale = None


# Maximum number of Collect nodes that can share a single name. The Unpack node
# declares this many hidden optional IMAGE inputs so the prompt handler has room
# to inject one link per matching Collect node.
MAX_SLOTS = 128


TARGET_SIZES = ["first", "largest", "smallest"]
FIT_MODES = ["stretch", "pad", "crop"]


def _resize_to(img, h, w):
    """Stretch ``img`` ([B,H,W,C]) to exactly (h, w), ignoring aspect ratio."""
    if common_upscale is not None:
        return common_upscale(img.movedim(-1, 1), w, h, "bilinear", "disabled").movedim(1, -1)
    return torch.nn.functional.interpolate(
        img.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
    ).movedim(1, -1)


def _fit_image(img, target_h, target_w, fit):
    """Fit ``img`` to (target_h, target_w) using the chosen ``fit`` mode.

    - ``stretch``: resize ignoring aspect ratio.
    - ``pad``: scale to fit inside the target (keep aspect), pad the rest black.
    - ``crop``: scale to cover the target (keep aspect), center-crop the overflow.
    """
    h, w = img.shape[1], img.shape[2]
    if (h, w) == (target_h, target_w):
        return img

    if fit == "stretch":
        return _resize_to(img, target_h, target_w)

    if fit == "pad":
        scale = min(target_w / w, target_h / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        scaled = _resize_to(img, new_h, new_w)
        pad_l = (target_w - new_w) // 2
        pad_r = target_w - new_w - pad_l
        pad_t = (target_h - new_h) // 2
        pad_b = target_h - new_h - pad_t
        # F.pad on [B,H,W,C] pads the last dims; pad W then H.
        return torch.nn.functional.pad(
            scaled, (0, 0, pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0
        )

    # crop: cover the target, then center-crop. common_upscale "center" does
    # exactly cover+center-crop in one step.
    if common_upscale is not None:
        return common_upscale(
            img.movedim(-1, 1), target_w, target_h, "bilinear", "center"
        ).movedim(1, -1)
    # Fallback without comfy: scale to cover, then center-crop.
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, round(w * scale))
    new_h = max(target_h, round(h * scale))
    scaled = _resize_to(img, new_h, new_w)
    y = (new_h - target_h) // 2
    x = (new_w - target_w) // 2
    return scaled[:, y:y + target_h, x:x + target_w, :]


def _batch_images(images, target_size="first", fit="stretch"):
    """Concatenate a list of IMAGE tensors into a single batch.

    Channels are padded (e.g. RGB -> RGBA) so all frames stack. Spatial size is
    unified by ``target_size`` (which image's dimensions to target) and ``fit``
    (how each image is resized to those dimensions).
    """
    if not images:
        return torch.zeros((1, 1, 1, 3))
    if len(images) == 1:
        return images[0]

    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]
    if target_size == "largest":
        h, w = max(heights), max(widths)
    elif target_size == "smallest":
        h, w = min(heights), min(widths)
    else:  # "first"
        h, w = images[0].shape[1], images[0].shape[2]

    max_ch = max(img.shape[-1] for img in images)
    total_frames = sum(img.shape[0] for img in images)

    first = images[0]
    out = torch.empty((total_frames, h, w, max_ch), dtype=first.dtype)
    offset = 0
    for img in images:
        img = _fit_image(img, h, w, fit)

        if img.shape[-1] < max_ch:
            img = torch.nn.functional.pad(
                img, (0, max_ch - img.shape[-1]), mode="constant", value=1.0
            )

        n = img.shape[0]
        out[offset:offset + n].copy_(img)
        offset += n

    return out.cpu()


class ImageCollect:
    """Tag an image output so it can be gathered elsewhere by name.

    Drop this on any number of image generation outputs and give them the same
    ``name``. A companion ``ImageCollectUnpack`` node selects that name and
    receives every tagged image as one batch. The node is a pure pass-through:
    the ``image`` output is the input unchanged, so it can also sit inline.

    Collection works across subgraph boundaries because the wiring is done by a
    prompt handler (``_on_prompt_handler``) after the frontend flattens
    subgraphs into the prompt, not by a physical cable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "order": ("INT", {"default": 0, "min": -100000, "max": 100000}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "collect"
    CATEGORY = "Mickmumpitz/Utils"

    def collect(self, image, name="", order=0):
        return (image,)


class ImageCollectUnpack:
    """Gather every ``ImageCollect`` sharing ``source`` into one image batch.

    Has no visible image inputs — the companion prompt handler scans the
    submitted prompt, finds all Collect nodes whose ``name`` matches ``source``,
    sorts them by their ``order`` widget (then node id), and injects one link
    per Collect into the hidden ``input_*`` slots. Batch order follows that
    sort.

    Mixed-resolution collections are unified by ``target_size`` (match the
    first / largest / smallest image's dimensions) and ``fit`` (``stretch`` =
    ignore aspect, ``pad`` = keep aspect + black bars, ``crop`` = keep aspect +
    center-crop). Channels are padded (e.g. RGB -> RGBA) so all frames stack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"input_{i}": ("IMAGE",) for i in range(1, MAX_SLOTS + 1)}
        return {
            "required": {
                "source": ("STRING", {"default": "", "multiline": False}),
                "target_size": (TARGET_SIZES, {"default": "first"}),
                "fit": (FIT_MODES, {"default": "stretch"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "unpack"
    CATEGORY = "Mickmumpitz/Utils"

    def unpack(self, source="", target_size="first", fit="stretch", **kwargs):
        images = []
        for i in range(1, MAX_SLOTS + 1):
            img = kwargs.get(f"input_{i}")
            if img is not None:
                images.append(img)
        return (_batch_images(images, target_size=target_size, fit=fit),)


def _node_id_sort_key(node_id):
    try:
        return (0, int(node_id), "")
    except (ValueError, TypeError):
        return (1, 0, str(node_id))


def _on_prompt_handler(json_data):
    """Wire every ImageCollect into the ImageCollectUnpack that shares its name.

    The frontend flattens subgraphs into ``prompt``, so a name-based lookup
    reaches across subgraph boundaries.
    """
    try:
        prompt = json_data.get("prompt", {}) if isinstance(json_data, dict) else {}
        if not isinstance(prompt, dict):
            return json_data

        # name -> list of (order, node_id)
        collects_by_name = {}
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "ImageCollect":
                continue
            inputs = node.get("inputs") or {}
            raw_name = inputs.get("name", "")
            if isinstance(raw_name, list):
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            raw_order = inputs.get("order", 0)
            try:
                order = int(raw_order) if not isinstance(raw_order, list) else 0
            except (ValueError, TypeError):
                order = 0
            collects_by_name.setdefault(name, []).append((order, node_id))

        for entries in collects_by_name.values():
            entries.sort(key=lambda e: (e[0], _node_id_sort_key(e[1])))

        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "ImageCollectUnpack":
                continue
            inputs = node.setdefault("inputs", {})
            raw_source = inputs.get("source", "")
            if isinstance(raw_source, list):
                continue
            source = str(raw_source).strip()

            # Clear any stale injected slots, then inject fresh links.
            for i in range(1, MAX_SLOTS + 1):
                inputs.pop(f"input_{i}", None)

            entries = collects_by_name.get(source, [])
            for i, (_order, cid) in enumerate(entries[:MAX_SLOTS], start=1):
                inputs[f"input_{i}"] = [cid, 0]
    except Exception:
        pass
    return json_data


try:
    from server import PromptServer
    PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)
except Exception:
    pass


NODE_CLASS_MAPPINGS = {
    "ImageCollect": ImageCollect,
    "ImageCollectUnpack": ImageCollectUnpack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCollect": "Collect Image",
    "ImageCollectUnpack": "Unpack Image Collection",
}
