"""Inject real frames into the video stream of a MiniMax H3 AV latent (img2img).

Adapted from ComfyUI-H3-FaceRefine by Carasibana (MIT, (c) 2026).
https://github.com/Carasibana/ComfyUI-H3-FaceRefine
Only this single node is used; the rest of that pack (face tracking / crop /
stitch, which pull in ultralytics + insightface + scipy) is not included. See
the pack's THIRD_PARTY_LICENSES for the full attribution.
"""

import torch

import comfy.nested_tensor


class H3InjectVideoLatent:
    """Replace the VIDEO stream of an H3 AV latent with real encoded frames (img2img seed).

    H3's own nodes always build a zeros latent - references are conditioning that is
    re-injected each step, never a starting point - so there is no stock video-to-video
    path. This encodes real frames into the video stream while leaving the audio stream
    intact, which turns SamplerCustomAdvanced + truncated sigmas into ordinary img2img.

    Pair with MiniMaxH3NativeAudioLock for the audio stream, and set strength with
    BasicScheduler's `denoise` - NOT with SplitSigmas. H3's flow-matching shift (12 by
    default) puts even the last split point of a short schedule at an effective sigma
    around 0.8, which rewrites the frame. `denoise` instead builds a longer full-range
    schedule and keeps only its lowest sigmas, so steps and strength stay independent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "images": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "Mickmumpitz/video"
    DESCRIPTION = "Encode real frames into the video stream of an H3 joint AV latent."

    def run(self, av_latent, images, vae):
        samples = av_latent.get("samples")
        if samples is None:
            raise KeyError('LATENT is missing "samples".')
        is_nested = isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(
            samples, "is_nested", False
        )
        if not is_nested:
            raise ValueError(
                "Expected a MiniMax H3 joint AV latent (NestedTensor). Feed the LATENT output "
                "of MiniMaxH3ReferenceToVideo / EmptyMiniMaxH3LatentAV."
            )

        members = list(samples.unbind())
        video_tmpl = members[0]

        encoded = vae.encode(images[..., :3])
        if encoded.ndim == 4:  # [B,C,H,W] -> [1,C,T,H,W]
            encoded = encoded.unsqueeze(0).movedim(1, 2)

        tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
        got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
        if (got_h, got_w) != (tgt_h, tgt_w):
            raise ValueError(
                f"Spatial latent mismatch: encoded {got_h}x{got_w} but the AV latent expects "
                f"{tgt_h}x{tgt_w}. The crop canvas and the H3 node's width/height must match "
                f"(both are pixels/16)."
            )
        note = ""
        if got_t != tgt_t:
            # H3 packs 17 pixel frames -> 5 latent frames; a frame count off the 17k+5
            # grid lands here. Trim or pad rather than fail, but say so loudly.
            if got_t > tgt_t:
                encoded = encoded[..., :tgt_t, :, :]
            else:
                pad = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
                encoded = torch.cat([encoded, pad], dim=-3)
            note = (f"  WARNING temporal mismatch: encoded t={got_t} vs latent t={tgt_t} "
                    f"-> {'trimmed' if got_t > tgt_t else 'padded'}. Frame count is probably "
                    f"off H3's 17k+5 grid.\n")

        members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))

        report = (
            f"injected video latent {tuple(encoded.shape)} into AV latent "
            f"(streams={len(members)})\n{note}"
            f"frames_in={images.shape[0]}  {images.shape[2]}x{images.shape[1]}px"
        )
        return (out, report)


NODE_CLASS_MAPPINGS = {
    "H3InjectVideoLatent": H3InjectVideoLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3InjectVideoLatent": "H3 Inject Video Latent (img2img)",
}
