import torch


class AudioExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "BOOLEAN",)
    RETURN_NAMES = ("audio", "exists",)
    FUNCTION = "check"
    CATEGORY = "Mickmumpitz/Utils"

    def check(self, audio=None):
        if audio is not None:
            return (audio, True)
        # 1 second of silent stereo at 44.1kHz
        placeholder = {
            "waveform": torch.zeros(1, 2, 44100),
            "sample_rate": 44100,
        }
        return (placeholder, False)


class MaskExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "BOOLEAN",)
    RETURN_NAMES = ("mask", "exists",)
    FUNCTION = "check"
    CATEGORY = "Mickmumpitz/Utils"

    def check(self, mask=None):
        if mask is not None:
            return (mask, True)
        placeholder = torch.zeros(1, 1, 1)
        return (placeholder, False)


class LatentExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN",)
    RETURN_NAMES = ("latent", "exists",)
    FUNCTION = "check"
    CATEGORY = "Mickmumpitz/Utils"

    def check(self, latent=None):
        if latent is not None:
            return (latent, True)
        placeholder = {"samples": torch.zeros(1, 4, 8, 8)}
        return (placeholder, False)


NODE_CLASS_MAPPINGS = {
    "AudioExists": AudioExists,
    "MaskExists": MaskExists,
    "LatentExists": LatentExists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioExists": "Audio Exists",
    "MaskExists": "Mask Exists",
    "LatentExists": "Latent Exists",
}
