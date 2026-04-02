import torch


class ImageFallback:
    """Pass through the input image, or output a black image at the given resolution if none is connected."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fallback"
    CATEGORY = "Mickmumpitz/Utils"

    def fallback(self, width, height, image=None):
        if image is not None:
            return (image,)
        return (torch.zeros(1, height, width, 3),)


NODE_CLASS_MAPPINGS = {
    "ImageFallback": ImageFallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFallback": "Image Fallback",
}
