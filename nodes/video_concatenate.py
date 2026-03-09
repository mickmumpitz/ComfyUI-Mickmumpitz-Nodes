import torch
from fractions import Fraction

from comfy_api.latest._input.basic_types import AudioInput
from comfy_api.latest._input.video_types import VideoInput
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents


class VideoConcatenate:
    """Concatenate multiple VIDEO inputs end-to-end, preserving audio."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"video_{i}"] = ("VIDEO",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "concatenate"
    CATEGORY = "Mickmumpitz/Video"

    def concatenate(self, **kwargs):
        videos: list[VideoInput] = []
        for i in range(1, 11):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                videos.append(v)

        if len(videos) == 0:
            raise ValueError("VideoConcatenate: at least one video input must be connected.")

        if len(videos) == 1:
            return (videos[0],)

        # Decompose all videos into components
        all_components = [v.get_components() for v in videos]

        # Use frame rate of the first video
        frame_rate = all_components[0].frame_rate

        # Concatenate image frames
        all_images = [c.images for c in all_components]
        combined_images = torch.cat(all_images, dim=0)

        # Concatenate audio (resample to first video's sample rate if needed)
        combined_audio = self._concat_audio(all_components, frame_rate)

        return (VideoFromComponents(VideoComponents(
            images=combined_images,
            audio=combined_audio,
            frame_rate=frame_rate,
        )),)

    def _concat_audio(self, components: list[VideoComponents], frame_rate: Fraction):
        """Concatenate audio from all components, padding silent gaps for videos without audio."""
        has_any_audio = any(c.audio is not None for c in components)
        if not has_any_audio:
            return None

        # Determine target sample rate from first video that has audio
        target_sr = None
        for c in components:
            if c.audio is not None:
                target_sr = c.audio["sample_rate"]
                break

        waveforms = []
        target_channels = 1
        # Determine max channels across all audio tracks
        for c in components:
            if c.audio is not None:
                target_channels = max(target_channels, c.audio["waveform"].shape[1])

        for c in components:
            num_frames = c.images.shape[0]
            duration_seconds = float(num_frames / frame_rate)
            num_samples = int(duration_seconds * target_sr)

            if c.audio is not None:
                waveform = c.audio["waveform"]  # (1, C, T)
                # Upmix mono to stereo if needed
                if waveform.shape[1] < target_channels:
                    waveform = waveform.expand(-1, target_channels, -1)
                # Trim or pad to match video duration
                if waveform.shape[2] > num_samples:
                    waveform = waveform[:, :, :num_samples]
                elif waveform.shape[2] < num_samples:
                    pad = torch.zeros(1, target_channels, num_samples - waveform.shape[2])
                    waveform = torch.cat([waveform, pad], dim=2)
                waveforms.append(waveform)
            else:
                # Silent audio for this segment
                waveforms.append(torch.zeros(1, target_channels, num_samples))

        combined = torch.cat(waveforms, dim=2)
        return AudioInput(waveform=combined, sample_rate=target_sr)


NODE_CLASS_MAPPINGS = {
    "VideoConcatenate": VideoConcatenate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatenate": "Video Concatenate",
}
