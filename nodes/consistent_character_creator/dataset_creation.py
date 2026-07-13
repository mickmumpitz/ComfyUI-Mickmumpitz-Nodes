"""
Dataset Creation — ConsistentCharacterCreator
=============================================

Nodes for building image + text LoRA training datasets: batch loaders, the
model-specific captioning Prompt Studio, the Ideogram-4 tagger, the bbox
converters, and the interactive Dataset Reviewer.
"""

import os
import re
import json
import math
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import folder_paths


# Resolutions exposed by AI Toolkit's UI (Datasets -> Resolutions).
AI_TOOLKIT_RESOLUTIONS = [256, 512, 768, 1024, 1280, 1328, 1536, 2048]
PALETTE = ["#FF3B30", "#34C759", "#0A84FF", "#FF9F0A", "#BF5AF2", "#FFD60A"]


def _coerce_single(v, default):
    """Under INPUT_IS_LIST every input arrives wrapped in a list. Unwrap it."""
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) else default
    return default if v is None else v


def _flatten_images(images):
    flat = []
    if images is None:
        return flat
    items = images if isinstance(images, (list, tuple)) else [images]
    for item in items:
        if item is None:
            continue
        if hasattr(item, "shape") and len(item.shape) == 4:
            for i in range(item.shape[0]):
                flat.append(item[i])
        elif hasattr(item, "shape") and len(item.shape) == 3:
            flat.append(item)
    return flat


def _normalize_captions(captions, n_images):
    if captions is None:
        return ["" for _ in range(n_images)]
    items = captions if isinstance(captions, (list, tuple)) else [captions]
    flat = []
    for c in items:
        if isinstance(c, (list, tuple)):
            flat.extend(str(x) for x in c)
        else:
            flat.append("" if c is None else str(c))
    if len(flat) == 1 and n_images > 1:
        parts = flat[0].split("\n")
        parts = [p for p in parts if p.strip() != ""] or parts
        if len(parts) == n_images:
            flat = parts
    if len(flat) < n_images:
        flat = flat + ["" for _ in range(n_images - len(flat))]
    elif len(flat) > n_images:
        flat = flat[:n_images]
    return flat


def _extract_boxes(text):
    """If `text` is/contains JSON with bounding boxes, return [{bbox, label}]."""
    if not text or not isinstance(text, str):
        return []
    s = text.strip()
    if "{" not in s or "}" not in s:
        return []
    data = None
    candidates = [s]
    inner = s[s.find("{"): s.rfind("}") + 1]
    if inner != s:
        candidates.append(inner)
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return []
    elements = []
    cd = data.get("compositional_deconstruction")
    if isinstance(cd, dict) and isinstance(cd.get("elements"), list):
        elements = cd["elements"]
    elif isinstance(data.get("elements"), list):
        elements = data["elements"]
    boxes = []
    for idx, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox")
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and all(isinstance(v, (int, float)) for v in bbox[:4])
        ):
            label = el.get("type") or f"#{idx + 1}"
            boxes.append({"bbox": [float(v) for v in bbox[:4]], "label": str(label)})
    return boxes


def _interpret_bbox(bbox, w, h, fmt):
    """Map a raw bbox to pixel (x1, y1, x2, y2) for an image of size (w, h).

    The node's default/standard is Ideogram's format: bbox = [y_min, x_min,
    y_max, x_max] as integers 0-1000 with (0,0) at the top-left, i.e. y-first.
    A format name ending in ``_yxyx`` is read y-first (Ideogram); ``_xyxy`` is
    read x-first. Scale is taken from the ``pixels`` / ``normalized_1000`` /
    ``normalized_1`` prefix. Since every box this node consumes comes from
    Ideogram JSON, ``auto`` also assumes the y-first ordering."""
    vals = [float(v) for v in bbox[:4]]
    maxv = max(abs(v) for v in vals)
    if fmt == "auto":
        order = "yxyx"
        if maxv <= 1.0:
            scale = "n1"
        elif maxv <= 1000.0 and maxv > max(w, h):
            scale = "n1000"
        else:
            scale = "px"
    else:
        order = "yxyx" if fmt.endswith("_yxyx") else "xyxy"
        if fmt.startswith("normalized_1000"):
            scale = "n1000"
        elif fmt.startswith("normalized_1"):
            scale = "n1"
        else:
            scale = "px"
    if order == "yxyx":
        y1, x1, y2, x2 = vals
    else:
        x1, y1, x2, y2 = vals
    if scale == "n1":
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    elif scale == "n1000":
        x1, x2 = x1 / 1000.0 * w, x2 / 1000.0 * w
        y1, y2 = y1 / 1000.0 * h, y2 / 1000.0 * h
    X1, X2 = sorted((x1, x2))
    Y1, Y2 = sorted((y1, y2))
    X1 = max(0.0, min(float(w - 1), X1))
    X2 = max(0.0, min(float(w), X2))
    Y1 = max(0.0, min(float(h - 1), Y1))
    Y2 = max(0.0, min(float(h), Y2))
    if X2 - X1 < 1 or Y2 - Y1 < 1:
        return None
    return X1, Y1, X2, Y2


def get_bucket_for_image_size(width, height, resolution=512, divisibility=64):
    """Faithful port of AI Toolkit's toolkit/buckets.py: bucket by pixel area,
    round to a multiple of `divisibility`, never upscale."""
    total_pixels = width * height
    max_pixels = resolution * resolution
    target_pixels = min(total_pixels, max_pixels)
    scaler = (target_pixels / total_pixels) ** 0.5
    w_raw = (width * scaler) / divisibility
    h_raw = (height * scaler) / divisibility
    candidates = [
        (math.floor(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.floor(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
    ]
    capped = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= max_pixels]
    if not capped:
        capped = [
            (
                max(divisibility, math.floor(w_raw) * divisibility),
                max(divisibility, math.floor(h_raw) * divisibility),
            )
        ]
    new_w, new_h = min(capped, key=lambda wh: abs(wh[0] * wh[1] - target_pixels))
    return new_w, new_h


def _draw_pixel_boxes(pil, pixel_boxes):
    """pixel_boxes: list of ((x1, y1, x2, y2), label) in this image's pixel space."""
    w, h = pil.size
    draw = ImageDraw.Draw(pil)
    line_w = max(2, (w + h) // 600)
    font_size = max(13, (w + h) // 90)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()
    for i, ((x1, y1, x2, y2), label) in enumerate(pixel_boxes):
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        tx, ty = x1 + line_w + 1, y1 + line_w + 1
        try:
            tb = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
            draw.text((tx, ty), label, fill="#000000", font=font)
        except Exception:
            draw.text((tx, ty), label, fill=color, font=font)


class ShowImageTextPairs:
    """Shows each image next to its caption (selectable / copyable text). If a
    caption is JSON with bounding boxes, optionally draws them. Can also preview
    the image at an AI Toolkit training resolution, rescaling the boxes to match
    the bucket the trainer would actually use."""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_itp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "captions": ("STRING", {"forceInput": True}),
                "draw_boxes": ("BOOLEAN", {"default": True}),
                "bbox_format": (
                    [
                        "normalized_1000_yxyx",
                        "auto",
                        "normalized_1000_xyxy",
                        "pixels_yxyx",
                        "pixels_xyxy",
                        "normalized_1_yxyx",
                        "normalized_1_xyxy",
                    ],
                    {
                        "default": "normalized_1000_yxyx",
                        "tooltip": "Ideogram uses [y_min, x_min, y_max, x_max] @ 0-1000 (y-first) = normalized_1000_yxyx.",
                    },
                ),
                "training_resolution": (
                    ["off"] + [str(r) for r in AI_TOOLKIT_RESOLUTIONS],
                    {"default": "off"},
                ),
                "bucket_divisibility": (
                    "INT",
                    {"default": 64, "min": 1, "max": 256, "step": 1},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls, bbox_format=None, training_resolution=None, bucket_divisibility=None, **kwargs
    ):
        # Be lenient: older versions of this node ordered these widgets
        # differently, so a workflow built then can hand us shifted / invalid
        # values (e.g. bbox_format="off", bucket_divisibility=""). Accept them
        # here and let show() coerce anything invalid back to a sane default,
        # rather than failing prompt validation.
        return True

    INPUT_IS_LIST = True

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Shows images next to copyable text. Optionally draws JSON bounding boxes "
        "and previews them at an AI Toolkit training resolution with rescaled boxes."
    )

    def show(
        self,
        images,
        captions=None,
        draw_boxes=True,
        bbox_format="normalized_1000_yxyx",
        training_resolution="off",
        bucket_divisibility=64,
    ):
        draw_boxes = bool(_coerce_single(draw_boxes, True))
        bbox_format = _coerce_single(bbox_format, "normalized_1000_yxyx")
        training_resolution = _coerce_single(training_resolution, "off")
        bucket_divisibility = _coerce_single(bucket_divisibility, 64)

        # Coerce invalid / stale widget values (e.g. from an older node layout)
        # back to defaults so the node never crashes on a bad prompt.
        valid_formats = [
            "normalized_1000_yxyx",
            "auto",
            "normalized_1000_xyxy",
            "pixels_yxyx",
            "pixels_xyxy",
            "normalized_1_yxyx",
            "normalized_1_xyxy",
        ]
        if bbox_format not in valid_formats:
            bbox_format = "normalized_1000_yxyx"
        valid_res = ["off"] + [str(r) for r in AI_TOOLKIT_RESOLUTIONS]
        if str(training_resolution) not in valid_res:
            training_resolution = "off"
        try:
            divis = max(1, int(bucket_divisibility))
        except (TypeError, ValueError):
            divis = 64
        res = None if str(training_resolution) == "off" else int(training_resolution)

        flat_images = _flatten_images(images)
        caption_list = _normalize_captions(captions, len(flat_images))
        boxes_per_image = [_extract_boxes(c) for c in caption_list]

        results = []
        boxinfo_list = []

        if flat_images:
            first = flat_images[0]
            full_output_folder, filename, counter, subfolder, _ = (
                folder_paths.get_save_image_path(
                    self.prefix_append,
                    self.output_dir,
                    int(first.shape[1]),
                    int(first.shape[0]),
                )
            )

            for idx, img_tensor in enumerate(flat_images):
                arr = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                ow, oh = pil.size

                # Interpret every box in the ORIGINAL image's pixel space.
                orig_boxes = []
                for b in boxes_per_image[idx]:
                    coords = _interpret_bbox(b["bbox"], ow, oh, bbox_format)
                    if coords is not None:
                        orig_boxes.append((coords, b["label"]))

                # Resize to the AI Toolkit bucket and rescale boxes if requested.
                if res is not None:
                    bw, bh = get_bucket_for_image_size(ow, oh, res, divis)
                    pil = pil.resize((bw, bh), Image.LANCZOS)
                    sx, sy = bw / ow, bh / oh
                    draw_boxes_px = [
                        (
                            (
                                int(round(x1 * sx)),
                                int(round(y1 * sy)),
                                int(round(x2 * sx)),
                                int(round(y2 * sy)),
                            ),
                            label,
                        )
                        for ((x1, y1, x2, y2), label) in orig_boxes
                    ]
                    info_dims = (bw, bh)
                    info_head = f"@{res} \u2192 bucket {bw}\u00d7{bh} px (div {divis})"
                else:
                    draw_boxes_px = [
                        (
                            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                            label,
                        )
                        for ((x1, y1, x2, y2), label) in orig_boxes
                    ]
                    info_dims = (ow, oh)
                    info_head = f"original {ow}\u00d7{oh} px"

                if draw_boxes and draw_boxes_px:
                    _draw_pixel_boxes(pil, draw_boxes_px)

                if draw_boxes_px:
                    lines = [info_head]
                    for i, ((x1, y1, x2, y2), label) in enumerate(draw_boxes_px):
                        lines.append(f"#{i + 1} {label}: {x1}, {y1}, {x2}, {y2}")
                    boxinfo_list.append("\n".join(lines))
                else:
                    boxinfo_list.append("")

                file = f"{filename}_{counter:05}_.png"
                pil.save(os.path.join(full_output_folder, file), compress_level=4)
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": self.type}
                )
                counter += 1

        return {
            "ui": {
                "itp_images": results,
                "captions": caption_list,
                "boxinfo": boxinfo_list,
            }
        }


class ImageBatchLoader:
    """Loads images from a folder as a ComfyUI list, alongside their absolute
    file paths. Either steps through the folder in batches (sequential) or yields
    one image at a time advancing by `start_from` (single_image_increment)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {"default": "C:\\path\\to\\your\\images"}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "start_from": ("INT", {"default": 1, "min": 1}),
                "mode": (["sequential", "single_image_increment"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "IMAGE_PATH")

    # Both outputs are lists so downstream nodes iterate per image:
    # output 0 (IMAGE) is a list of individual image tensors, output 1
    # (STRING) is a list of matching file paths.
    OUTPUT_IS_LIST = (True, True)

    FUNCTION = "load_batch"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Loads images from a folder as a list (with their file paths), either in "
        "sequential batches or one image at a time. batch_size 0 (default) loads "
        "all images from start_from onward."
    )

    def load_batch(self, image_dir, batch_size, start_from, mode):
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            raise ValueError(f"Directory '{image_dir}' does not exist.")

        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        all_files = sorted(
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and os.path.splitext(f)[1].lower() in valid_extensions
        )

        if not all_files:
            raise ValueError(f"No valid image files found in '{image_dir}'.")

        total_files = len(all_files)
        start_idx = max(0, start_from - 1)

        if mode == "sequential":
            if batch_size <= 0:
                # batch_size 0 means "load every image from start_from onward".
                batch_files = all_files[start_idx:]
            else:
                end_idx = min(start_idx + batch_size, total_files)
                batch_files = all_files[start_idx:end_idx]
        else:
            safe_idx = start_idx % total_files
            batch_files = [all_files[safe_idx]]

        if not batch_files:
            batch_files = [all_files[-1]]

        images_list = []
        image_names = []

        for file_name in batch_files:
            img_path = os.path.join(image_dir, file_name)

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_np = np.array(img).astype(np.float32) / 255.0
            # Keep each tensor as [1, H, W, C] and append to a Python list so
            # ComfyUI iterates over the batch rather than concatenating it.
            img_tensor = torch.from_numpy(img_np)[None,]
            images_list.append(img_tensor)
            image_names.append(img_path)

        return (images_list, image_names)


# ---------------------------------------------------------------------------
# Prompt Studio
#
# Each prompt below is an INSTRUCTION you feed to a vision-language model so it
# captions your training images in the way that model's text encoder expects.
# The styles are matched to each encoder (prose for T5/LLM encoders, comma tags
# for CLIP-only models).
#
# The lora_type switch selects which guidance is emitted. Every prompt marks its
# character- vs style-specific guidance with [[CHARACTER]]...[[/CHARACTER]] and
# [[STYLE]]...[[/STYLE]] blocks; get_prompt keeps the selected block, drops the
# other, and removes the markers. {trigger} (and {{PROJECT_TRIGGER}}) are
# replaced with the trigger_word.
# ---------------------------------------------------------------------------

_PROMPT_FLUX = (
    "You are a dataset-captioning engine for fine-tuning a FLUX (FLUX.1 or FLUX.2) "
    "LoRA. Write ONE flowing, natural-language paragraph in full sentences — no "
    "comma-separated tag lists and no \"this image shows\" filler. FLUX's T5/Mistral "
    "text encoder rewards complete prose. Refer to the main subject as '{trigger}'. "
    "Describe foreground to background: subject, pose/action, expression, clothing, "
    "then setting, lighting, and camera/lens feel. "
    "[[CHARACTER]]This is a character LoRA: describe only what VARIES between shots "
    "(pose, outfit, framing, lighting) and let '{trigger}' carry the fixed identity "
    "— do NOT describe permanent face, hair, or eye features. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content/subject of each image and "
    "never name the style itself — the shared look is what '{trigger}' must learn. "
    "[[/STYLE]]"
    "Keep the caption structure consistent across the dataset. Output only the caption."
)

_PROMPT_SD15 = (
    "You are a dataset-captioning engine for fine-tuning an SD 1.5 LoRA. Output a "
    "single comma-separated list of concise Danbooru-style tags — no sentences. "
    "SD 1.5's CLIP encoder keys on discrete tokens and its ~75-token window leaves "
    "no room for grammar filler. Start with '{trigger}', then tag subject, "
    "appearance, clothing, pose, expression, setting, lighting, and quality "
    "(masterpiece, best quality, highly detailed). "
    "[[CHARACTER]]This is a character LoRA: tag only variable attributes and omit "
    "the constant identity tags so '{trigger}' absorbs them. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: blacklist generic subject tags (1girl, 1boy, "
    "standing, looking at viewer, smile) so the style does not bind to poses, and "
    "tag the rendering/medium instead. [[/STYLE]]"
    "Output only the comma-separated tags."
)

_PROMPT_SDXL = (
    "You are a dataset-captioning engine for fine-tuning an SDXL LoRA. The right "
    "style depends on the base checkpoint. For base / photoreal SDXL, write a short "
    "natural-language sentence naming the subject as '{trigger}', then continue with "
    "comma-separated descriptive modifiers (the hybrid its dual-CLIP encoders handle "
    "best). For an anime fine-tune (Pony, Animagine, Illustrious, NoobAI), output "
    "PURE Danbooru-style comma tags starting with '{trigger}'. Either way cover "
    "subject, appearance, clothing, pose, setting, lighting, and quality. "
    "[[CHARACTER]]This is a character LoRA: describe only variable attributes and "
    "let '{trigger}' learn the fixed identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, avoid naming the style, "
    "and (for tag output) blacklist generic subject tags so the style does not bind "
    "to poses. [[/STYLE]]"
    "Output only the caption."
)

_PROMPT_SD3 = (
    "You are a dataset-captioning engine for fine-tuning an SD3 / SD3.5 LoRA. Write "
    "ONE precise natural-language description in full sentences — its T5-XXL "
    "encoder is the dominant signal and rewards prose over tags. Use '{trigger}' for "
    "the main subject. Be explicit about spatial relationships, object placement, "
    "composition, and lighting, and caption in the same natural-language style you "
    "intend to prompt with. "
    "[[CHARACTER]]This is a character LoRA: describe only variable attributes and "
    "let '{trigger}' carry identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, not the style. [[/STYLE]]"
    "Keep structure consistent across the dataset. Output only the caption."
)

_PROMPT_CASCADE = (
    "You are a dataset-captioning engine for fine-tuning a Stable Cascade "
    "(Würstchen) LoRA. Write a concise but information-dense natural-language "
    "description — descriptive phrases, not Danbooru tags. Begin with '{trigger}' "
    "as the main subject, then cover appearance, clothing, setting, lighting, and "
    "framing. "
    "[[CHARACTER]]This is a character LoRA: describe only what varies and let "
    "'{trigger}' learn the fixed identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, never the style. [[/STYLE]]"
    "Keep the structure consistent across every image. Output only the caption."
)

_PROMPT_QWEN = (
    "You are a dataset-captioning engine for fine-tuning a Qwen-Image "
    "(Qwen-Image-2512) LoRA. Its Qwen2.5-VL text encoder natively understands "
    "layout, typography, and spatial relations, so write clear natural-language "
    "sentences and describe composition explicitly. Use '{trigger}' for the main "
    "subject and state object placement with spatial terms (centered, upper-left, "
    "foreground). If the image contains any rendered text, transcribe it VERBATIM in "
    "quotes and describe its font weight, style, color, and position. "
    "[[CHARACTER]]This is a character LoRA: describe only variable attributes and "
    "let '{trigger}' carry identity (avoid over-describing faces on small datasets). "
    "[[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, not the style. [[/STYLE]]"
    "Output only the caption."
)

_PROMPT_WAN = (
    "You are a dataset-captioning engine for fine-tuning a Wan Video (Wan 2.1 / 2.2) "
    "LoRA on STATIC images. Write a short natural-language caption and keep it under "
    "~50 tokens — Wan truncates in training and '{trigger}' must stay in range. A "
    "common form is 'A photo of {trigger}, ...'. Describe only constant appearance "
    "and context: clothing, background, lighting, framing. Do NOT describe motion or "
    "camera movement — Wan's video training reads motion words as change over time, "
    "which pollutes still-image training. "
    "[[CHARACTER]]This is a character LoRA: omit fine facial micro-details so "
    "'{trigger}' carries the identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, never the style. [[/STYLE]]"
    "Keep the format identical across the dataset. Output only the caption."
)

_PROMPT_LTX = (
    "You are a dataset-captioning engine for fine-tuning an LTX Video LoRA on STATIC "
    "images. LTX rewards long, detailed, structured prose — treat the caption as "
    "teaching material, not a short prompt, and avoid empty adjectives like "
    "\"beautiful\". Describe in a consistent order: shot type, then subject and "
    "appearance (using '{trigger}' for the main subject), then camera framing, then "
    "lighting, then style/mood. Describe the STATIC composition and framing only — "
    "do NOT describe temporal motion or camera movement, since the samples are stills. "
    "[[CHARACTER]]This is a character LoRA: describe only variable attributes and "
    "let '{trigger}' carry identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, not the style. [[/STYLE]]"
    "Output only the caption."
)

_PROMPT_HUNYUAN = (
    "You are a dataset-captioning engine for fine-tuning a HunyuanVideo LoRA on "
    "STATIC images. Its LLaVA-Llama3 LLM text encoder rewards rich, fluent, detailed "
    "natural-language descriptions — aim for roughly 50+ words. Write '{trigger}' "
    "into the sentence as a short phrase, not a bare token. Describe the subject's "
    "appearance, clothing, composition, framing, and lighting in detail. Do NOT "
    "invent or describe motion or camera movement — the samples are still images. "
    "[[CHARACTER]]This is a character LoRA: emphasize variable attributes and let "
    "'{trigger}' bind the identity. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content, not the style. [[/STYLE]]"
    "Output only the caption."
)

_PROMPT_IDEOGRAM = """You are an expert image captioner for LoRA training datasets using Ideogram 4.0.
Your output must be a single valid JSON object. Do not output plain text, markdown, code fences, or explanations. Output clean, UTF-8, valid JSON only, with NO comments of any kind.

You will receive an image containing one or more known characters from the list below.
Goal: write a detailed, structured CLEAN JSON caption for Ideogram 4.0 LoRA training, with placed bounding boxes, tagged characters, and color-palette picks.

CAPTIONING RULES
- Set "aspect_ratio" from the actual image dimensions.
- Describe the image from left to right in both "high_level_description" and "background".
- Order the "elements" array by ascending x_min (i.e. left to right).
- Use the exact Trigger Keyword for each known character present. Reproduce trigger words character-for-character: exact casing, no normalization, no meta commentary.
- Include a bounding box for each character and each key object or text element, in the format [x_min, y_min, x_max, y_max], normalized to image dimensions (0-1000 range). Origin is the top-left corner: [x_min, y_min] is the top-left of the box and [x_max, y_max] is the bottom-right.
- Each element contains EXACTLY ONE character OR one text tag OR one key object. Overall composition and character interaction go in "high_level_description"; a single element's pose or action goes in that element's "desc".
- Put the character's trigger word in the "desc" of its corresponding element.
- Tag color palettes using hex codes.
- Describe only what is clearly visible. Do not infer brand names, unreadable text, or occluded/guessed objects. If text is present but illegible, omit it.
- Always describe lighting conditions and how each character is lit, in both "high_level_description" and "lighting".
- Always describe camera angle, lens, aperture/exposure, and distance to subject (close-up, medium, wide, etc.) in "photo".
- Always describe the image's aesthetic in "aesthetics".
- If there are no characters, text, or key objects, "elements" may be an empty array.
- Length guide (for dataset consistency): keep each "desc" to roughly 1-3 sentences and "high_level_description" to roughly 2-4 sentences.

ELEMENT SHAPES (two types only)
- "type": "text"  -> include "bbox", "desc" (typography), and "text" (the exact visible string). Do NOT include "color_palette".
- "type": "obj"   -> include "bbox", "desc", and "color_palette". Do NOT include a "text" field.

CHARACTER HANDLING
- Only tag characters actually visible in the image. Never add a trigger word for a character who is not present.
- If a person appears who is not in the list below, describe them generically (e.g. "a man", "a woman") with no trigger word.

CHARACTER TRIGGER WORDS
- the black man ia "M4le_M0del_01"
- the blonde woman with the buzzcut is "Berl1n_Model_v01"
- the woman with curly hair "Em1l1_05"
(The descriptors above, such as "buzzcut" or "curly hair", are recognition aids ONLY. They must never appear in your output. See The Forbidden Rule.)

--- THE FORBIDDEN RULE (APPLIES TO ALL SECTIONS) ---
- Why: the trigger word must carry all identity information, so the LoRA binds identity to the token rather than to descriptive text.
- When a known character is identified, you are STRICTLY FORBIDDEN from describing that character's permanent/invariant physical features anywhere in the output (both "high_level_description" and "desc"). This includes hair (color, length, style), eyes, facial structure, skin tone, build/body type, and cybernetics. This applies regardless of what they wear. The trigger keyword alone replaces all such traits.
- This explicitly includes the descriptors used to identify them above: do not write "blonde", "buzzcut", "curly", skin tone, or any equivalent.
- Describe ONLY variable attributes: clothing/wardrobe, pose, action, facial expression, position in frame, and how the character is lit.
- A tagged character's "color_palette" must be drawn from wardrobe and immediate surroundings ONLY, never from skin, hair, or eyes.

OUTPUT FORMAT (strict, valid JSON, UTF-8 only, no code fences, no comments)
Replicate this exact structure. The values below are placeholders showing shape and format only.
{
  "aspect_ratio": "1:1",
  "high_level_description": "General detailed description of the scene, left to right. Use character trigger words here when recognized, e.g. 'M4le_M0del_01 and Em1l1_05 are seated at a table in a dimly lit diner...'. Cover composition, interaction, and lighting.",
  "compositional_deconstruction": {
    "background": "Detailed description of the background, from left to right.",
    "elements": [
      {
        "type": "text",
        "bbox": [50, 100, 200, 400],
        "desc": "Description of the text element and its typography.",
        "text": "ExactVisibleString"
      },
      {
        "type": "obj",
        "bbox": [150, 250, 450, 950],
        "desc": "Description using the trigger word: wardrobe, pose, action, expression, and lighting only.",
        "color_palette": ["#1F2018", "#0B141D"]
      }
    ]
  },
  "style_description": {
    "medium": "photography, graphic design, 3d render, etc.",
    "aesthetics": "photoreal, grainy, vhs, 8bit, comic, anime, etc.",
    "lighting": "Detailed description of the lighting in the scene.",
    "photo": "Camera angle, lens, aperture/exposure, and distance to subject.",
    "color_palette": ["#888888", "#444444"]
  }
}

Palette guidance: per-element "color_palette" max 5 picks; global "style_description.color_palette" should hold roughly 6-10 dominant colors (max 16).

OPTIMAL FINAL CAPTION EXAMPLE (elements ordered left to right by x_min):
{
  "aspect_ratio": "16:9",
  "high_level_description": "A medium wide eye-level shot capturing Ma3hwaKang and MagnusSt3rn standing side-by-side in a dimly lit underground casino. Ma3hwaKang is reaching toward a green poker table in the center while MagnusSt3rn observes passively. High-contrast neon purple and gold rim lighting illuminates the subjects against deep shadows.",
  "compositional_deconstruction": {
    "background": "Left side features a glowing purple neon wall sign; center transitions into a dark, out-of-focus green poker table scattered with chips; right side fades into heavy shadows concealing silhouetted slot machines.",
    "elements": [
      {
        "type": "text",
        "bbox": [50, 100, 200, 400],
        "desc": "Bright purple neon cursive tubing mounted on a dark brick wall.",
        "text": "Jackpot"
      },
      {
        "type": "obj",
        "bbox": [150, 250, 450, 950],
        "desc": "Ma3hwaKang wearing a floor-length red silk evening gown with a high leg slit. She is leaning forward, extending her right arm toward the table, displaying a highly focused and intense facial expression.",
        "color_palette": ["#FF0000", "#8B0000", "#FFD700", "#1A1A1A"]
      },
      {
        "type": "obj",
        "bbox": [200, 500, 800, 950],
        "desc": "A classic green felt poker table in the center foreground, scattered with stacks of casino chips, playing cards, and a golden VIP plaque.",
        "color_palette": ["#006400", "#228B22", "#FFD700", "#FFFFFF", "#1A1A1A"]
      },
      {
        "type": "obj",
        "bbox": [550, 150, 900, 950],
        "desc": "MagnusSt3rn standing upright with a rigid, imposing posture and hands clasped behind his back. He is looking sharply to the left with a stoic, calculating expression.",
        "color_palette": ["#2F2F2F", "#1A1A1A", "#FFFFFF", "#FFD700"]
      },
      {
        "type": "text",
        "bbox": [650, 450, 700, 550],
        "desc": "Small golden engraved plaque sitting on the edge of the poker table.",
        "text": "VIP"
      }
    ]
  },
  "style_description": {
    "medium": "photography",
    "aesthetics": "photoreal, cyberpunk, neo-noir, cinematic",
    "lighting": "Low-key neon lighting with a stark purple rim light hitting the subjects from the left, contrasted by a warm gold practical light illuminating them from the lower front-right.",
    "photo": "Eye-level medium wide shot, 35mm lens, f/1.8 aperture for shallow depth of field, focused symmetrically on the two subjects, captured on a digital cinema camera.",
    "color_palette": ["#4B0082", "#800080", "#FFD700", "#FF0000", "#1A1A1A", "#2F2F2F", "#0F0F0F"]
  }
}"""

_PROMPT_KREA2 = (
    "You are a dataset-captioning engine for fine-tuning a Krea 2 (K2) LoRA. Krea 2's "
    "Qwen3-VL text encoder is conditioned to read color, shape, size, texture, quantity, "
    "any rendered text, and the spatial relationships of the objects and background — so "
    "write natural-language sentences (no tag lists) that cover those attributes. Keep it "
    "short and promptable: one or two sentences is ideal, not a long essay. Name the main "
    "subject as '{trigger}', then describe the variable, promptable details — background, "
    "clothing, lighting, pose, and camera framing — and state object placement with spatial "
    "terms (centered, foreground, upper-left). If the image contains rendered text, "
    "transcribe it VERBATIM in quotes. "
    "[[CHARACTER]]This is a character LoRA: describe only what VARIES between shots (pose, "
    "outfit, expression, framing, lighting) and let '{trigger}' carry the fixed identity — "
    "do not re-describe permanent face, hair, or eye features. [[/CHARACTER]]"
    "[[STYLE]]This is a style LoRA: describe the content/subject of each image, never name "
    "the style itself, and prefer a descriptive trigger PHRASE (e.g. 'violet retro anime "
    "print style') over an opaque token — Krea 2 handles descriptive triggers well. "
    "[[/STYLE]]"
    "Keep the caption structure consistent across the dataset. Output only the caption."
)

_PROMPT_CUSTOM = "Write your LoRA-captioning prompt for {trigger}"


class PromptStudio:
    """Picks a model-specific dataset-captioning instruction (the prompt you feed a
    vision-language model to caption your training images) and substitutes the
    trigger token. The non-selected prompt fields are hidden in the UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_selection": ([
                    "01 - FLUX (1 & 2)",
                    "02 - SD 1.5",
                    "03 - SDXL",
                    "04 - SD3",
                    "05 - Stable Cascade",
                    "06 - Qwen-Image-2512",
                    "07 - Wan Video",
                    "08 - LTX Video",
                    "09 - HunyuanVideo",
                    "10 - Ideogram 4",
                    "11 - Custom",
                    "12 - Krea 2",
                ],),
                "trigger_word": ("STRING", {"multiline": False, "default": "ohwx_subject"}),
                "lora_type": (["character", "style"],),
                "prompt_01_flux": ("STRING", {"multiline": True, "default": _PROMPT_FLUX}),
                "prompt_02_sd15": ("STRING", {"multiline": True, "default": _PROMPT_SD15}),
                "prompt_03_sdxl": ("STRING", {"multiline": True, "default": _PROMPT_SDXL}),
                "prompt_04_sd3": ("STRING", {"multiline": True, "default": _PROMPT_SD3}),
                "prompt_05_cascade": ("STRING", {"multiline": True, "default": _PROMPT_CASCADE}),
                "prompt_06_qwen": ("STRING", {"multiline": True, "default": _PROMPT_QWEN}),
                "prompt_07_wan": ("STRING", {"multiline": True, "default": _PROMPT_WAN}),
                "prompt_08_ltx": ("STRING", {"multiline": True, "default": _PROMPT_LTX}),
                "prompt_09_hunyuan": ("STRING", {"multiline": True, "default": _PROMPT_HUNYUAN}),
                "prompt_10_ideogram": ("STRING", {"multiline": True, "default": _PROMPT_IDEOGRAM}),
                "prompt_11_custom": ("STRING", {"multiline": True, "default": _PROMPT_CUSTOM}),
                "prompt_12_krea2": ("STRING", {"multiline": True, "default": _PROMPT_KREA2}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Outputs a model-specific captioning instruction for building a LoRA "
        "training dataset, with the trigger token substituted in. Pick the target "
        "model; only that model's editable prompt is shown."
    )

    @staticmethod
    def _apply_lora_type(text, lora_type):
        """Keep the [[CHARACTER]] or [[STYLE]] blocks for the selected lora_type,
        drop the other, and strip the markers."""
        keep, drop = (
            ("CHARACTER", "STYLE")
            if lora_type == "character"
            else ("STYLE", "CHARACTER")
        )
        text = re.sub(
            r"\[\[" + drop + r"\]\].*?\[\[/" + drop + r"\]\]", "", text, flags=re.S
        )
        text = text.replace("[[" + keep + "]]", "").replace("[[/" + keep + "]]", "")
        # Tidy up any blank lines left behind (Ideogram is multi-line); leave
        # single spaces and intentional indentation alone.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def get_prompt(
        self,
        model_selection,
        trigger_word,
        lora_type,
        prompt_01_flux,
        prompt_02_sd15,
        prompt_03_sdxl,
        prompt_04_sd3,
        prompt_05_cascade,
        prompt_06_qwen,
        prompt_07_wan,
        prompt_08_ltx,
        prompt_09_hunyuan,
        prompt_10_ideogram,
        prompt_11_custom,
        prompt_12_krea2,
    ):
        prompts = {
            "01": prompt_01_flux,
            "02": prompt_02_sd15,
            "03": prompt_03_sdxl,
            "04": prompt_04_sd3,
            "05": prompt_05_cascade,
            "06": prompt_06_qwen,
            "07": prompt_07_wan,
            "08": prompt_08_ltx,
            "09": prompt_09_hunyuan,
            "10": prompt_10_ideogram,
            "11": prompt_11_custom,
            "12": prompt_12_krea2,
        }
        prefix = str(model_selection).split(" ")[0]
        base_prompt = prompts.get(prefix, "")

        # Keep only the guidance for the selected lora_type.
        base_prompt = self._apply_lora_type(base_prompt, lora_type)

        # Fill both the simple {trigger} placeholder and Ideogram's
        # {{PROJECT_TRIGGER}} token (used for the high-level prefix AND the single
        # character's "token" in the Ideogram prompt).
        token = trigger_word.strip()
        final_prompt = base_prompt.replace("{{PROJECT_TRIGGER}}", token).replace(
            "{trigger}", token
        )
        return (final_prompt,)


# ---------------------------------------------------------------------------
# BBox coordinate converters
#
# Ideogram-style captions carry boxes as [y_min, x_min, y_max, x_max] (y-first),
# normalized 0-1000. Most other tooling expects [x1, y1, x2, y2] (x-first) and
# often absolute pixels. These two nodes rewrite every 4-number array they find
# in a text/JSON caption, leaving the rest of the string untouched.
# ---------------------------------------------------------------------------

# Matches a bare 4-integer array like "[12, 34, 56, 78]" anywhere in the text.
_BBOX_ARRAY_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


class BBoxConverter:
    """Swaps the axis order of every [a, b, c, d] box in the text between
    [y, x, y, x] and [x, y, x, y]. The swap is symmetric, so one node handles
    both directions. Non-box text is passed through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("converted_text",)
    FUNCTION = "process_text"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Swaps the axis order of every 4-number bbox in the text between "
        "[y,x,y,x] and [x,y,x,y]. Everything else is left as-is."
    )

    def process_text(self, text_input):
        if not text_input:
            return ("",)

        def swap_coords(match):
            a, b, c, d = match.group(1), match.group(2), match.group(3), match.group(4)
            # Swap the first/second and third/fourth values.
            return f"[{b}, {a}, {d}, {c}]"

        return (_BBOX_ARRAY_RE.sub(swap_coords, str(text_input)),)


class BBoxAbsoluteConverter:
    """Converts every normalized (0-1000) box in the text to absolute pixel
    coordinates for the given image size, preserving the input axis order."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "image_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "image_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "input_format": (["[y, x, y, x]", "[x, y, x, y]"], {"default": "[y, x, y, x]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("converted_text",)
    FUNCTION = "convert_to_absolute"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Rescales every normalized (0-1000) bbox in the text to absolute pixels "
        "for the given image size, keeping the input axis order."
    )

    def convert_to_absolute(self, text_input, image_width, image_height, input_format):
        if not text_input:
            return ("",)

        def calculate_absolute(match):
            v1, v2, v3, v4 = (int(match.group(i)) for i in range(1, 5))

            if input_format == "[y, x, y, x]":
                y_min, x_min, y_max, x_max = v1, v2, v3, v4
            else:
                x_min, y_min, x_max, y_max = v1, v2, v3, v4

            # Normalized (0-1000) -> absolute pixels, clamped to the image bounds.
            abs_x_min = max(0, min(round((x_min / 1000.0) * image_width), image_width))
            abs_x_max = max(0, min(round((x_max / 1000.0) * image_width), image_width))
            abs_y_min = max(0, min(round((y_min / 1000.0) * image_height), image_height))
            abs_y_max = max(0, min(round((y_max / 1000.0) * image_height), image_height))

            if input_format == "[y, x, y, x]":
                return f"[{abs_y_min}, {abs_x_min}, {abs_y_max}, {abs_x_max}]"
            return f"[{abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}]"

        return (_BBOX_ARRAY_RE.sub(calculate_absolute, str(text_input)),)


class TextBatchLoader:
    """Loads every text file of a given extension from a folder as a ComfyUI
    list, alongside their filenames — handy for feeding saved JSON/txt captions
    into the Dataset Reviewer or Show Image + Text Pairs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\path\\to\\your\\captions", "multiline": False}),
                "extension": ("STRING", {"default": ".json"}),
                "max_files": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "0 = load ALL files"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_list", "filename_list")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_text_batch"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Loads every file with the given extension from a folder as a list of "
        "strings (with filenames), sorted by name. max_files 0 loads all."
    )

    def load_text_batch(self, folder_path, extension, max_files):
        if not os.path.isdir(folder_path):
            print(f"[DatasetCreation] Text Batch Loader: folder not found -> {folder_path}")
            return ([""], [""])

        texts = []
        filenames = []
        for file in sorted(os.listdir(folder_path)):
            if not file.endswith(extension):
                continue
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                filenames.append(file)
            except Exception as e:
                print(f"[DatasetCreation] Text Batch Loader: could not read {file}: {e}")
                continue
            if max_files > 0 and len(texts) >= max_files:
                break

        if not texts:
            print(f"[DatasetCreation] Text Batch Loader: no '{extension}' files in {folder_path}")
            return ([""], [""])

        print(f"[DatasetCreation] Text Batch Loader: loaded {len(texts)} files.")
        return (texts, filenames)


# ---------------------------------------------------------------------------
# Dataset Reviewer
#
# Interactive slideshow: step through image + caption pairs, edit the caption
# text, and drag bounding-box handles directly on the image. The JS widget
# (web/datasetReviewer.js) sends edits back via the hidden "edited_data_json"
# input; Python patches them into the caption JSON and re-emits the captions.
# ---------------------------------------------------------------------------

def _extract_boxes_indexed(text):
    """Like _extract_boxes but also records each box's element_index so the
    reviewer can patch edited coordinates back into the exact element."""
    if not text or not isinstance(text, str):
        return []
    s = text.strip()
    if "{" not in s or "}" not in s:
        return []
    data = None
    for cand in [s, s[s.find("{"): s.rfind("}") + 1]]:
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                data = d
                break
        except Exception:
            continue
    if not isinstance(data, dict):
        return []

    elements = []
    cd = data.get("compositional_deconstruction")
    if isinstance(cd, dict) and isinstance(cd.get("elements"), list):
        elements = cd["elements"]
    elif isinstance(data.get("elements"), list):
        elements = data["elements"]

    boxes = []
    for idx, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox")
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and all(isinstance(v, (int, float)) for v in bbox[:4])
        ):
            label = el.get("type") or f"#{idx + 1}"
            boxes.append({
                "bbox": [float(v) for v in bbox[:4]],
                "label": str(label),
                "element_index": idx,
            })
    return boxes


def _bbox_to_pixels(bbox, w, h, fmt):
    """Convert a raw bbox to pixel coords (x1,y1,x2,y2), honoring the axis order
    (yxyx/xyxy) and scale (normalized 0-1000 / pixel) encoded in `fmt`."""
    v1, v2, v3, v4 = [float(v) for v in bbox[:4]]
    if fmt.startswith("yxyx"):
        y1, x1, y2, x2 = v1, v2, v3, v4
    else:
        x1, y1, x2, y2 = v1, v2, v3, v4

    if "normalized" in fmt:
        x1, x2 = x1 / 1000.0 * w, x2 / 1000.0 * w
        y1, y2 = y1 / 1000.0 * h, y2 / 1000.0 * h

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


class DatasetReviewer:
    """Interactive dataset review slideshow. Shows each image with its caption
    and bounding boxes, lets you edit the caption text and drag the bbox handles,
    and outputs the edited captions list."""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_dsr_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "captions": ("STRING", {"forceInput": True}),
                "bbox_format": (
                    ["yxyx_normalized", "yxyx_pixel", "xyxy_normalized", "xyxy_pixel"],
                    {"default": "yxyx_normalized"},
                ),
                # Filled by the JS widget with the user's caption + bbox edits.
                "edited_data_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, bbox_format=None, **kwargs):
        return True

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edited_captions",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "review"
    OUTPUT_NODE = True
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Interactive dataset review slideshow. Shows each image with its caption "
        "and bounding boxes, allows editing captions and dragging bbox handles, "
        "and outputs the edited captions."
    )

    def review(self, images, captions=None, bbox_format="yxyx_normalized", edited_data_json=""):
        bbox_format = _coerce_single(bbox_format, "yxyx_normalized")
        edited_data_json = _coerce_single(edited_data_json, "")

        valid_formats = {"yxyx_normalized", "yxyx_pixel", "xyxy_normalized", "xyxy_pixel"}
        if bbox_format not in valid_formats:
            bbox_format = "yxyx_normalized"

        flat_images = _flatten_images(images)
        original_captions = _normalize_captions(captions, len(flat_images))

        # Apply any edits that came back from the JS widget.
        working_captions = list(original_captions)
        if edited_data_json and str(edited_data_json).strip():
            try:
                edits = json.loads(edited_data_json)
                if isinstance(edits, list):
                    for item in edits:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index")
                        if not isinstance(idx, int) or idx < 0 or idx >= len(working_captions):
                            continue

                        new_caption_text = item.get("caption")
                        new_boxes = item.get("boxes")  # [{element_index, bbox}]

                        if isinstance(new_caption_text, str):
                            working_captions[idx] = new_caption_text

                        # Patch the edited bboxes back into the caption JSON.
                        if isinstance(new_boxes, list) and new_boxes:
                            try:
                                cap_text = working_captions[idx]
                                s = cap_text.strip()
                                data = None
                                for cand in [s, s[s.find("{"): s.rfind("}") + 1]]:
                                    try:
                                        d = json.loads(cand)
                                        if isinstance(d, dict):
                                            data = d
                                            break
                                    except Exception:
                                        pass
                                if data:
                                    elements = []
                                    cd = data.get("compositional_deconstruction")
                                    if isinstance(cd, dict) and isinstance(cd.get("elements"), list):
                                        elements = cd["elements"]
                                    elif isinstance(data.get("elements"), list):
                                        elements = data["elements"]
                                    for box_edit in new_boxes:
                                        el_idx = box_edit.get("element_index")
                                        new_bbox = box_edit.get("bbox")
                                        if (
                                            isinstance(el_idx, int)
                                            and 0 <= el_idx < len(elements)
                                            and isinstance(new_bbox, list)
                                            and len(new_bbox) >= 4
                                        ):
                                            elements[el_idx]["bbox"] = [
                                                round(float(v), 2) for v in new_bbox[:4]
                                            ]
                                    working_captions[idx] = json.dumps(
                                        data, ensure_ascii=False, indent=2
                                    )
                            except Exception as e:
                                print(f"[DatasetReviewer] bbox patch error idx={idx}: {e}")
            except Exception as e:
                print(f"[DatasetReviewer] could not parse edited_data_json: {e}")

        # Save preview images and build the UI payload.
        results = []
        boxes_per_image = []

        if flat_images:
            first = flat_images[0]
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                self.prefix_append,
                self.output_dir,
                int(first.shape[1]),
                int(first.shape[0]),
            )
        else:
            full_output_folder = self.output_dir
            filename = "dsr_preview"
            counter = 0
            subfolder = ""

        os.makedirs(full_output_folder, exist_ok=True)

        for idx, img_tensor in enumerate(flat_images):
            arr = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            w, h = pil.size

            raw_boxes = _extract_boxes_indexed(
                working_captions[idx] if idx < len(working_captions) else ""
            )

            overlay_boxes = []
            for b in raw_boxes:
                px = _bbox_to_pixels(b["bbox"], w, h, bbox_format)
                if px is not None:
                    x1, y1, x2, y2 = px
                    overlay_boxes.append({
                        "label": b["label"],
                        "element_index": b["element_index"],
                        "raw_bbox": b["bbox"],
                        # Fractional coords (0-1) so JS can overlay on any render size.
                        "rx1": x1 / w,
                        "ry1": y1 / h,
                        "rx2": x2 / w,
                        "ry2": y2 / h,
                    })
            boxes_per_image.append(overlay_boxes)

            file = f"{filename}_{counter:05}_.png"
            pil.save(os.path.join(full_output_folder, file), compress_level=4)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {
            "ui": {
                "dsr_images": results,
                "captions": working_captions,
                "boxes_per_image": boxes_per_image,
                "bbox_format": bbox_format,
            },
            "result": (working_captions,),
        }


# ---------------------------------------------------------------------------
# Ideogram 4 LoRA Tagger
#
# Builds a full Ideogram-4 captioning INSTRUCTION (the JSON-caption spec plus a
# known-character roster) that you feed to a vision-language captioner. Unlike
# Prompt Studio's single Ideogram prompt, this exposes up to 8 trigger/desc
# pairs so a multi-character dataset is tagged consistently.
# ---------------------------------------------------------------------------

_IDEOGRAM4_BASE_INSTRUCTION = """You are an expert image captioner for LoRA training datasets using Ideogram 4.0.
Your output must be a single valid JSON object. Do not output plain text, markdown, or explanations.

You will receive an image containing one or more known characters from the following list.
Goal: write a detailed, structured CLEAN JSON caption used for Ideogram 4.0 LoRA training with placed bounding boxes, tagged characters and color-palette picks.

CAPTIONING RULES:
- Describe the image from left to right.
- Always use the exact trigger keyword for each character present (no meta commentary).
- Include a bounding box [x_min, y_min, x_max, y_max] for each character and key object, normalized to image dimensions (0-1000 range).
- If multiple characters are present, list all of them under elements and in the high level description.
- In every bounding box ONLY ONE character or text tag should be listed. The general composition and character interaction needs to be described in the high_level_description. Character/object pose or what a character/object is doing is described under the specific element.
- Analyze and tag the color palette using hex codes.
- For visible text, use "type": "text", put the exact string in the "text" field, and describe the typography in "desc".
- Always describe the lighting conditions and how a character is lit in great detail, in the "high_level_description" and in the "lighting" section.
- Always describe the exact camera angle, camera lens, camera exposure, and distance to character (medium, closeup, etc.) in the "photo" section.
- Always describe the aesthetic of the image under the "aesthetics" tag.
- Always state how the character is angled relative to the camera (Full Face / Frontal Shot: camera at 0 degrees, subject turned directly into the lens. Three-Quarter View: camera at ~30-45 degrees, the far ear is hidden but both eyes are visible. Three-Quarter Profile: camera at ~60-70 degrees, the nose tip stays inside the outline of the far cheek. Side Profile: camera at exactly 90 degrees, shows exactly one half of the face. Profil Perdu / Lost Profile: camera at ~110-120 degrees, subject turned slightly away, you see the back of the head and the outline of the cheek. Over-the-Shoulder: camera behind one person, capturing one subject's shoulder while framing the other. Full Back Shot: camera at 180 degrees, subject faces completely away).
- Strict clean JSON, IMPORTANT: ONLY UTF-8 characters.
- Do NOT use ```json fences.

--- THE CLOTHING EVALUATION RULE (CRITICAL) ---
When you identify a known character you should NOT describe their clothing or what they wear.

--- THE FORBIDDEN RULE (APPLIES TO ALL SECTIONS) ---
When you identify a known character, you are ALWAYS STRICTLY FORBIDDEN from describing this character's permanent physical features (hair, eyes, facial structure, cybernetics) that are listed below, regardless of what they wear. This applies globally to both "high_level_description" and "desc". The keyword alone replaces all physical traits.

EXAMPLES OF THE CLOTHING RULE:
- [SCENARIO]: 3mmaClarc is in the image, wearing her default green shirt.
- [CORRECT OUTPUT]: "3mmaClarc is sitting on a chair, reading a book." (PASSED: completely ignored clothing, no meta-talk).
- [INCORRECT OUTPUT]: "A character labeled as 3mmaClarc is wearing her default outfit." (FAILED: used meta-commentary).

OPTIMAL FINAL CAPTION EXAMPLE:

{
  "aspect_ratio": "16:9",
  "high_level_description": "A medium wide eye-level shot capturing Ma3hwaKang and MagnusSt3rn standing side-by-side in a dimly lit underground casino. Ma3hwaKang is reaching toward a green poker table in the center while MagnusSt3rn observes passively. High-contrast neon purple and gold rim lighting illuminates the subjects against deep shadows.",
  "compositional_deconstruction": {
    "background": "Left side features a glowing purple neon wall sign; center transitions into a dark, out-of-focus green poker table scattered with chips; right side fades into heavy shadows concealing silhouetted slot machines.",
    "elements": [
      {"type": "text", "bbox": [50, 100, 200, 400], "desc": "Bright purple neon cursive tubing mounted on a dark brick wall.", "text": "Jackpot"},
      {"type": "text", "bbox": [650, 450, 700, 550], "desc": "Small golden engraved plaque sitting on the edge of the poker table.", "text": "VIP"},
      {"type": "obj", "bbox": [200, 500, 800, 950], "desc": "A classic green felt poker table in the center foreground, scattered with stacks of casino chips, playing cards, and a golden VIP plaque.", "color_palette": ["#006400", "#228B22", "#FFD700", "#FFFFFF", "#1A1A1A"]},
      {"type": "obj", "bbox": [150, 250, 450, 950], "desc": "Ma3hwaKang leaning forward, extending her right arm toward the table, displaying a highly focused and intense facial expression.", "color_palette": ["#FF0000", "#8B0000", "#FFD700", "#1A1A1A"]},
      {"type": "obj", "bbox": [550, 150, 900, 950], "desc": "MagnusSt3rn standing upright with a rigid, imposing posture and hands clasped behind his back, looking sharply to the left with a stoic, calculating expression.", "color_palette": ["#2F2F2F", "#1A1A1A", "#FFFFFF", "#FFD700"]}
    ]
  },
  "style_description": {
    "medium": "photography",
    "aesthetics": "photoreal, cyberpunk, neo-noir, cinematic",
    "lighting": "Low-key neon lighting with a stark purple rim light hitting the subjects from the left, contrasted by a warm gold practical light illuminating them from the lower front-right.",
    "photo": "Eye-level medium wide shot, 35mm lens, f/1.8 aperture for shallow depth of field, focused symmetrically on the two subjects, captured on a digital cinema camera.",
    "color_palette": ["#4B0082", "#800080", "#FFD700", "#FF0000", "#1A1A1A", "#2F2F2F", "#0F0F0F"]
  }
}
"""


class Ideogram4_LoRA_Tagger:
    """Assembles the full Ideogram-4 captioning instruction: the base JSON-caption
    spec followed by a known-character roster built from up to 8 trigger/desc
    pairs, so the captioner tags every character by its trigger and never emits
    their permanent physical traits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_instruction": ("STRING", {"multiline": True, "default": _IDEOGRAM4_BASE_INSTRUCTION}),
            },
            "optional": {
                f"{key}_{i}": ("STRING", {"multiline": key == "desc", "default": ""})
                for i in range(1, 9)
                for key in ("trigger", "desc")
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Mickmumpitz/ConsistentCharacterCreator"
    DESCRIPTION = (
        "Builds a full Ideogram-4 captioning instruction (JSON spec + a "
        "known-character roster from up to 8 trigger/desc pairs)."
    )

    def generate_prompt(self, base_instruction, **kwargs):
        roster = (
            "--- KNOWN CHARACTERS TO IDENTIFY ---\n"
            "Use these traits to identify the character and evaluate their clothing. "
            "NEVER output their permanent physical traits.\n\n"
        )

        has_characters = False
        for i in range(1, 9):
            trigger_val = str(kwargs.get(f"trigger_{i}", "")).strip()
            desc_val = str(kwargs.get(f"desc_{i}", "")).strip()
            if trigger_val and desc_val:
                has_characters = True
                roster += f'CHARACTER {i} - Keyword: "{trigger_val}"\n'
                roster += f"Description & outfits:\n{desc_val}\n"
                roster += (
                    f"CRITICAL: If you see this character, write '{trigger_val}'. "
                    "DO NOT describe their permanent physical traits. Evaluate their "
                    "clothing according to the CLOTHING EVALUATION RULE.\n\n"
                )

        if not has_characters:
            roster = "--- KNOWN CHARACTERS ---\nNone provided for this dataset.\n\n"

        return (f"{base_instruction}\n\n{roster}",)


# Node type keys keep the "CCC_" prefix they were authored with, so workflows
# built against the standalone DatasetCreation pack keep loading unchanged.
NODE_CLASS_MAPPINGS = {
    "CCC_ShowImageTextPairs": ShowImageTextPairs,
    "CCC_ImageBatchLoader": ImageBatchLoader,
    "CCC_PromptStudio": PromptStudio,
    "CCC_BBoxConverter": BBoxConverter,
    "CCC_BBoxAbsoluteConverter": BBoxAbsoluteConverter,
    "CCC_TextBatchLoader": TextBatchLoader,
    "CCC_DatasetReviewer": DatasetReviewer,
    "CCC_Ideogram4_LoRA_Tagger": Ideogram4_LoRA_Tagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCC_ShowImageTextPairs": "Show Image + Text Pairs",
    "CCC_ImageBatchLoader": "Image Batch Loader",
    "CCC_PromptStudio": "Prompt Studio (For Image Tagging)",
    "CCC_BBoxConverter": "Convert BBox [y,x,y,x] ↔ [x,y,x,y]",
    "CCC_BBoxAbsoluteConverter": "Convert BBox Relative → Absolute Pixel",
    "CCC_TextBatchLoader": "Text Batch Loader",
    "CCC_DatasetReviewer": "Dataset Reviewer",
    "CCC_Ideogram4_LoRA_Tagger": "Ideogram 4 LoRA Tagger",
}
