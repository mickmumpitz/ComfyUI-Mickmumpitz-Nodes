"""
OpenPose Part Cropper
=====================

Select an OpenPose rig region (hands, feet, head, ...), crop it for a detailer
sampler, and stitch the result back into the original image. Also reconstructs
keypoints from an already-rendered OpenPose image when raw keypoints are absent.

Nodes
-----
* OpenPose Image To Keypoints  : rendered OpenPose image  -> POSE_KEYPOINT
* OpenPose Part Mask           : POSE_KEYPOINT + IMAGE     -> MASK (+ bbox)
* OpenPose Part Crop           : POSE_KEYPOINT + IMAGE     -> STITCH + crop + mask
* OpenPose Part Stitch         : STITCH + detailed IMAGE   -> IMAGE

All nodes live under the `Mickmumpitz/OpenPosePartCropper` category and keep the
node keys they were authored with, so workflows built against the original
standalone pack keep loading.
"""

from .nodes import (
    OpenPoseImageToKeypoints,
    OpenPosePartMask,
    OpenPosePartCrop,
    OpenPosePartStitch,
)

NODE_CLASS_MAPPINGS = {
    "OpenPoseImageToKeypoints": OpenPoseImageToKeypoints,
    "OpenPosePartMask": OpenPosePartMask,
    "OpenPosePartCrop": OpenPosePartCrop,
    "OpenPosePartStitch": OpenPosePartStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseImageToKeypoints": "OpenPose Image → Keypoints",
    "OpenPosePartMask": "OpenPose Part Mask",
    "OpenPosePartCrop": "OpenPose Part Crop",
    "OpenPosePartStitch": "OpenPose Part Stitch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
