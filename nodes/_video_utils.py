import torch
import torch.nn.functional as F
from fractions import Fraction

from comfy_api.latest._input.basic_types import AudioInput
from comfy_api.latest._input.video_types import VideoInput
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents


RESOLUTION_MODES = [
    "letterbox_to_first",
    "letterbox_to_largest",
    "letterbox_to_smallest",
    "crop_to_first",
    "crop_to_largest",
    "crop_to_smallest",
    "stretch_to_first",
]


def concatenate_videos(
    videos: list[VideoInput],
    resolution_mode: str = "letterbox_to_first",
) -> VideoFromComponents:
    """Concatenate multiple VIDEO inputs end-to-end, preserving audio.

    When inputs have mismatched (H, W), each shot is resized/padded/cropped
    to a common resolution chosen by ``resolution_mode``.
    """
    if len(videos) == 0:
        raise ValueError("concatenate_videos: at least one video required.")

    if len(videos) == 1:
        return videos[0]

    all_components = [v.get_components() for v in videos]

    frame_rate = all_components[0].frame_rate

    target_h, target_w = _target_resolution(all_components, resolution_mode)
    fitted = [
        _fit_frames(c.images, target_h, target_w, resolution_mode)
        for c in all_components
    ]
    combined_images = torch.cat(fitted, dim=0)

    combined_audio = _concat_audio(all_components, frame_rate)

    return VideoFromComponents(VideoComponents(
        images=combined_images,
        audio=combined_audio,
        frame_rate=frame_rate,
    ))


def _target_resolution(components: list[VideoComponents], mode: str) -> tuple[int, int]:
    heights = [c.images.shape[1] for c in components]
    widths = [c.images.shape[2] for c in components]
    if mode.endswith("_first"):
        return heights[0], widths[0]
    if mode.endswith("_largest"):
        return max(heights), max(widths)
    if mode.endswith("_smallest"):
        return min(heights), min(widths)
    raise ValueError(f"Unknown resolution_mode: {mode}")


def _fit_frames(images: torch.Tensor, target_h: int, target_w: int, mode: str) -> torch.Tensor:
    """images: (N, H, W, C) float in [0,1]. Returns (N, target_h, target_w, C)."""
    _, h, w, _ = images.shape
    if h == target_h and w == target_w:
        return images

    if mode.startswith("stretch"):
        return _resize(images, target_h, target_w)

    if mode.startswith("letterbox"):
        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = _resize(images, new_h, new_w) if (new_h, new_w) != (h, w) else images
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        padded = F.pad(
            resized.permute(0, 3, 1, 2),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant", value=0.0,
        ).permute(0, 2, 3, 1)
        return padded.contiguous()

    if mode.startswith("crop"):
        scale = max(target_h / h, target_w / w)
        new_h = max(target_h, int(round(h * scale)))
        new_w = max(target_w, int(round(w * scale)))
        resized = _resize(images, new_h, new_w) if (new_h, new_w) != (h, w) else images
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        return resized[:, top:top + target_h, left:left + target_w, :].contiguous()

    raise ValueError(f"Unknown resolution_mode: {mode}")


def _resize(images: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    x = images.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _concat_audio(components: list[VideoComponents], frame_rate: Fraction):
    """Concatenate audio from all components, padding silent gaps for videos without audio."""
    has_any_audio = any(c.audio is not None for c in components)
    if not has_any_audio:
        return None

    target_sr = None
    for c in components:
        if c.audio is not None:
            target_sr = c.audio["sample_rate"]
            break

    target_channels = 1
    for c in components:
        if c.audio is not None:
            target_channels = max(target_channels, c.audio["waveform"].shape[1])

    waveforms = []
    for c in components:
        num_frames = c.images.shape[0]
        duration_seconds = float(num_frames / frame_rate)
        num_samples = int(duration_seconds * target_sr)

        if c.audio is not None:
            waveform = c.audio["waveform"]  # (1, C, T)
            if waveform.shape[1] < target_channels:
                waveform = waveform.expand(-1, target_channels, -1)
            if waveform.shape[2] > num_samples:
                waveform = waveform[:, :, :num_samples]
            elif waveform.shape[2] < num_samples:
                pad = torch.zeros(1, target_channels, num_samples - waveform.shape[2])
                waveform = torch.cat([waveform, pad], dim=2)
            waveforms.append(waveform)
        else:
            waveforms.append(torch.zeros(1, target_channels, num_samples))

    combined = torch.cat(waveforms, dim=2)
    return AudioInput(waveform=combined, sample_rate=target_sr)
