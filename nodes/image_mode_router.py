import logging

logger = logging.getLogger(__name__)


class ImageModeRouter:
    """Routes start and reference images based on a selectable mode.

    Modes:
    - Start + Reference: passes both inputs through.
    - Reference Only: disables start output, passes reference through.
    - Auto-Detect: if reference image count > 1, behaves like Reference Only;
      otherwise behaves like Start + Reference.
    """

    MODES = ["Start + Reference", "Reference Only", "Auto-Detect"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Auto-Detect"}),
            },
            "optional": {
                "start_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("start_images", "reference_frames")
    FUNCTION = "route"
    CATEGORY = "Mickmumpitz/Utils"

    def route(self, reference_images, mode, start_image=None):
        ref_count = reference_images.shape[0]
        logger.info("[MMZ ImageModeRouter] mode=%s, ref_count=%d, start_image=%s",
                    mode, ref_count, "provided" if start_image is not None else "None")

        if mode == "Start + Reference":
            return (start_image, reference_images)
        elif mode == "Reference Only":
            return (None, reference_images)
        else:  # Auto-Detect
            if ref_count > 1:
                logger.info("[MMZ ImageModeRouter] Auto-Detect: %d refs → Reference Only", ref_count)
                return (None, reference_images)
            else:
                logger.info("[MMZ ImageModeRouter] Auto-Detect: %d ref → Start + Reference", ref_count)
                return (start_image, reference_images)


NODE_CLASS_MAPPINGS = {
    "ImageModeRouter": ImageModeRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageModeRouter": "Image Mode Router",
}
