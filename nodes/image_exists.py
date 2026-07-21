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
        # 64x64 black BHWC placeholder: large enough to survive VAE encoding
        # (vae_encode_crop_pixels rounds smaller images down to 0x0 and crashes)
        # on setups where a non-lazy switch evaluates the unused branch
        placeholder = torch.zeros(1, 64, 64, 3)
        return (placeholder, False)


NODE_CLASS_MAPPINGS = {
    "ImageExists": ImageExists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageExists": "Image Exists",
}
