import torch


class ImageExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN",)
    RETURN_NAMES = ("image", "exists",)
    FUNCTION = "check"
    CATEGORY = "Mickmumpitz/Utils"

    def check(self, image=None):
        if image is not None:
            return (image, True)
        # 1x1 black BHWC placeholder so downstream nodes don't choke
        placeholder = torch.zeros(1, 1, 1, 3)
        return (placeholder, False)


NODE_CLASS_MAPPINGS = {
    "ImageExists": ImageExists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageExists": "Image Exists",
}
