"""
Storyboard Grid: lays the shot images out as one storyboard sheet with a header,
shot titles and a per-shot "Timeline: Frame <start>-<end>" caption. Fonts scale
with the cell size and resolve a real TTF on Windows / Linux / macOS.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def _find_font_path(bold=False):
    key = f"path_bold_{bold}"
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates += [
            os.path.join(windir, "Fonts", "arialbd.ttf" if bold else "arial.ttf"),
            os.path.join(windir, "Fonts", "segoeuib.ttf" if bold else "segoeui.ttf"),
            os.path.join(windir, "Fonts", "calibrib.ttf" if bold else "calibri.ttf"),
        ]
    elif os.uname().sysname == "Darwin" if hasattr(os, "uname") else False:
        candidates += [
            f"/Library/Fonts/Arial{' Bold' if bold else ''}.ttf",
            f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
        ]
    else:
        candidates += [
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/TTF/DejaVuSans{'-Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        ]

    found = None
    for c in candidates:
        if os.path.isfile(c):
            found = c
            break

    if found is None:
        try:
            import matplotlib.font_manager as fm
            found = fm.findfont("DejaVu Sans" + (":bold" if bold else ""), fallback_to_default=True)
        except Exception:
            found = None

    _FONT_CACHE[key] = found
    return found


def _load_font(size, bold=False):
    path = _find_font_path(bold=bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # Last resort: fixed-size bitmap font. Newer Pillow (>=10.1) accepts a
    # size kwarg here too, so try that before giving up on scaling entirely.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _tensor_to_pil(img_tensor):
    arr = img_tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _fit_cover(img, target_w, target_h):
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class MickmumpitzStoryboardGrid:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("storyboard",)
    FUNCTION = "run"
    CATEGORY = "Mickmumpitz/Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "MyStoryboard"}),
                "shot_count": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "cell_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "cell_height": ("INT", {"default": 288, "min": 64, "max": 4096, "step": 8}),
                "cell_padding": ("INT", {"default": 24, "min": 0, "max": 200, "step": 1}),
                "images": ("IMAGE",),
                "frame_lengths": ("STRING", {"default": "24,24,24,24,24,24,24,24"}),
                "shot_names": ("STRING", {"default": "Shot 1,Shot 2,Shot 3,Shot 4,Shot 5,Shot 6,Shot 7,Shot 8"}),
            },
            "optional": {
                "auto_scale_text": ("BOOLEAN", {"default": True}),
                "text_scale_multiplier": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05}),
                "timeline_start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    def run(self, project_name, shot_count, columns, cell_width, cell_height,
             cell_padding, images, frame_lengths, shot_names,
             auto_scale_text=True, text_scale_multiplier=1.0, timeline_start_frame=0):

        names = [s.strip() for s in shot_names.split(",")] if shot_names else []
        lengths = [s.strip() for s in frame_lengths.split(",")] if frame_lengths else []

        n = min(shot_count, images.shape[0])
        names = (names + [""] * n)[:n]
        lengths = (lengths + [""] * n)[:n]

        captions = []
        cursor = timeline_start_frame
        for idx in range(n):
            raw = lengths[idx]
            length_val = int(raw) if raw.isdigit() else 0
            start = cursor
            end = cursor + max(length_val, 1) - 1
            captions.append(f"Timeline: Frame {start}-{end}")
            cursor = end + 1

        ref_dim = min(cell_width, cell_height)
        mult = text_scale_multiplier if auto_scale_text else 1.0

        if auto_scale_text:
            title_size = int(_clamp(ref_dim * 0.045, 14, 60) * mult)
            caption_size = int(_clamp(ref_dim * 0.038, 12, 50) * mult)
            header_size = int(_clamp(ref_dim * 0.06, 24, 90) * mult)
            frame_thickness = max(4, int(ref_dim * 0.01))
        else:
            title_size = int(20 * mult)
            caption_size = int(18 * mult)
            header_size = int(34 * mult)
            frame_thickness = 6

        title_h = title_size + 22
        caption_h = caption_size + 20
        header_h = header_size + 36
        gap = cell_padding

        font_header = _load_font(header_size, bold=True)
        font_title = _load_font(title_size, bold=True)
        font_caption = _load_font(caption_size, bold=False)

        cell_outer_w = cell_width + frame_thickness * 2
        cell_outer_h = cell_height + frame_thickness * 2 + title_h + caption_h

        cols = max(1, columns)
        rows = max(1, (n + cols - 1) // cols)

        sheet_w = cols * cell_outer_w + (cols + 1) * gap
        sheet_h = header_h + rows * cell_outer_h + (rows + 1) * gap

        sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)

        draw.text((gap, (header_h - header_size) // 2), project_name, fill=(255, 255, 255), font=font_header)

        for idx in range(n):
            row = idx // cols
            col = idx % cols
            x0 = gap + col * (cell_outer_w + gap)
            y0 = header_h + gap + row * (cell_outer_h + gap)

            draw.rectangle(
                [x0, y0, x0 + cell_outer_w, y0 + cell_outer_h],
                outline=(230, 230, 230), width=2, fill=(30, 30, 30)
            )

            title_text = names[idx] if idx < len(names) and names[idx] else f"Shot {idx + 1}"
            tw = draw.textlength(title_text, font=font_title)
            draw.text((x0 + (cell_outer_w - tw) / 2, y0 + (title_h - title_size) / 2),
                      title_text, fill=(255, 255, 255), font=font_title)

            img_x0 = x0 + frame_thickness
            img_y0 = y0 + title_h + frame_thickness
            frame_img = _tensor_to_pil(images[idx])
            frame_img = _fit_cover(frame_img, cell_width, cell_height)
            sheet.paste(frame_img, (img_x0, img_y0))

            draw.rectangle(
                [img_x0 - frame_thickness, img_y0 - frame_thickness,
                 img_x0 + cell_width + frame_thickness, img_y0 + cell_height + frame_thickness],
                outline=(255, 255, 255), width=frame_thickness
            )

            caption_text = captions[idx]
            cw = draw.textlength(caption_text, font=font_caption)
            cap_y = img_y0 + cell_height + frame_thickness + (caption_h - caption_size) / 2
            draw.text((x0 + (cell_outer_w - cw) / 2, cap_y),
                      caption_text, fill=(200, 200, 200), font=font_caption)

        return (_pil_to_tensor(sheet),)


NODE_CLASS_MAPPINGS = {
    "MickmumpitzStoryboardGrid": MickmumpitzStoryboardGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MickmumpitzStoryboardGrid": "Make Storyboard Grid",
}
