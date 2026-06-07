import torch

try:
    from comfy.utils import common_upscale
except Exception:  # pragma: no cover - fallback if comfy isn't importable at import time
    common_upscale = None


class ImageBatchMultiSkipEmpty:
    """Batch multiple images together, like Kijai's Image Batch Multi, but
    unconnected inputs are skipped entirely instead of being filled with a
    black image. Set the number of inputs with **inputcount** and click
    **Update inputs**.

    If no input is connected at all, a single 1x1 black image is returned so
    the graph still has something valid to pass downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "Mickmumpitz/Image"

    def combine(self, inputcount, **kwargs):
        # Collect only the connected inputs, in slot order, skipping empties.
        images = []
        for c in range(1, inputcount + 1):
            img = kwargs.get(f"image_{c}")
            if img is not None:
                images.append(img)

        if not images:
            # Nothing connected -> return a single black pixel so downstream
            # nodes still receive a valid IMAGE tensor.
            return (torch.zeros((1, 1, 1, 3)),)

        if len(images) == 1:
            return (images[0],)

        first = images[0]
        h, w = first.shape[1], first.shape[2]
        max_ch = max(img.shape[-1] for img in images)
        total_frames = sum(img.shape[0] for img in images)

        out = torch.empty((total_frames, h, w, max_ch), dtype=first.dtype)
        offset = 0
        for img in images:
            # Match resolution of the first image.
            if img.shape[1:3] != (h, w):
                if common_upscale is not None:
                    img = common_upscale(
                        img.movedim(-1, 1), w, h, "bilinear", "center"
                    ).movedim(1, -1)
                else:
                    img = torch.nn.functional.interpolate(
                        img.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
                    ).movedim(1, -1)

            # Pad channels (e.g. RGB -> RGBA) with 1.0 so alpha stays opaque.
            if img.shape[-1] < max_ch:
                img = torch.nn.functional.pad(
                    img, (0, max_ch - img.shape[-1]), mode="constant", value=1.0
                )

            n = img.shape[0]
            out[offset:offset + n].copy_(img)
            offset += n

        return (out.cpu(),)


NODE_CLASS_MAPPINGS = {
    "ImageBatchMultiSkipEmpty": ImageBatchMultiSkipEmpty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchMultiSkipEmpty": "Image Batch Multi (Skip Empty)",
}
