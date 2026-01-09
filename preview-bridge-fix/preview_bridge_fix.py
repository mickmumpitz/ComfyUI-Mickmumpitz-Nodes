"""
Preview Bridge Node for ComfyUI

This node allows you to:
1. Preview an image with an integrated mask editor
2. Edit the mask visually via right-click menu
3. Output both the original image and the edited mask for further processing
"""

import torch
import numpy as np
from PIL import Image
import os
import random
import json
import folder_paths


class PreviewBridge:
    """
    A bridge node that enables mask editing in the preview while maintaining
    the ability to output both the image and the edited mask.
    
    Takes a single image input, displays it with the ability to edit masks,
    and outputs both the image and a mask tensor.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_preview_bridge_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for x in range(5))
        self.compress_level = 1
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The image to preview and edit. Right-click to open the mask editor."
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    OUTPUT_TOOLTIPS = (
        "The original input image.",
        "The mask (edited using the mask editor).",
    )
    OUTPUT_NODE = True
    FUNCTION = "preview_bridge"
    CATEGORY = "image"
    DESCRIPTION = "Preview an image with integrated mask editor. Right-click the preview to edit the mask, then output both the image and edited mask for further processing."
    
    def preview_bridge(self, image):
        """
        Process the image and save it for preview with mask editing capability.
        Generates a default mask from the image's alpha channel if available,
        otherwise creates a blank white mask.
        
        Args:
            image: Input image tensor (B, H, W, C)
        
        Returns:
            Tuple of (image, mask) tensors in the UI output dict
        """
        # Generate mask from image
        mask = self._generate_mask_from_image(image)
        
        # Save preview images to temporary directory for UI display
        preview_images = self._save_preview(image)
        
        # Return both image and mask as output, plus UI preview images
        return {
            "ui": {
                "images": preview_images
            },
            "result": (image, mask)
        }
    
    def _generate_mask_from_image(self, image):
        """
        Generate a mask from the image. If the image has an alpha channel,
        use it as the mask. Otherwise, create a white mask (all 1s).
        
        Args:
            image: Image tensor (B, H, W, C)
        
        Returns:
            Mask tensor (B, H, W)
        """
        batch_size = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]
        channels = image.shape[3]
        
        if channels == 4:
            # Image has alpha channel, use it as mask
            mask = image[:, :, :, 3]
        else:
            # Create white mask (all 1s)
            mask = torch.ones((batch_size, height, width), dtype=torch.float32, device=image.device)
        
        # Ensure mask is in the range [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def _save_preview(self, image):
        """
        Save preview images to temporary directory for UI rendering.
        """
        try:
            batch_size = image.shape[0]
            results = []
            
            for batch_idx in range(batch_size):
                # Convert image to PIL
                img_np = image[batch_idx].cpu().numpy()
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                # Handle different channel configurations
                if img_np.shape[2] >= 3:
                    # Use first 3 channels as RGB
                    pil_image = Image.fromarray(img_np[:, :, :3], mode='RGB')
                else:
                    # Grayscale image
                    pil_image = Image.fromarray(img_np[:, :, 0], mode='L').convert('RGB')
                
                # Create filename
                filename = f"preview_bridge_{batch_idx:03d}.png"
                temp_dir = folder_paths.get_temp_directory()
                filepath = os.path.join(temp_dir, filename)
                
                # Save the preview image
                pil_image.save(filepath, compress_level=self.compress_level)
                
                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                })
            
            return results
        except Exception as e:
            print(f"Warning: Could not save preview image: {e}")
            return []


NODE_CLASS_MAPPINGS = {
    "PreviewBridge": PreviewBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewBridge": "Preview Bridge",
}
