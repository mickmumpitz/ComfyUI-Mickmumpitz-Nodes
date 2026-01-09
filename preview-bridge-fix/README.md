# Preview Bridge Node

A custom ComfyUI node that displays an image preview with integrated mask editor capabilities.

## Features

- **Single Image Input**: Takes only an image input (no separate mask required)
- **Image Preview Widget**: Shows your image in a preview widget just like PreviewImage node
- **Mask Editor Integration**: Right-click on the preview to activate the mask editor
- **Dual Output**: Outputs both the original image and an editable mask
- **Smart Mask Generation**: Automatically extracts alpha channel if present, otherwise creates a blank white mask

## Installation

This node is a custom node for ComfyUI. It should be placed in your `ComfyUI/custom_nodes/` directory.

The directory structure should be:
```
ComfyUI/
└── custom_nodes/
    └── preview-bridge-fix/
        ├── __init__.py
        ├── preview_bridge_fix.py
        └── README.md
```

## Usage

1. Add the "Preview Bridge" node to your workflow
2. Connect an image to the `image` input
3. The node displays the image in a preview widget (like PreviewImage)
4. **Right-click on the preview image** to activate the mask editor
5. Edit the mask as needed in the mask editor
6. The node outputs:
   - **IMAGE**: The original input image
   - **MASK**: The edited mask (or auto-generated mask if not edited)

## Node Properties

### Inputs
- **image**: The image to preview and edit (supports any dimensions and channels)

### Outputs
- **IMAGE**: The original input image passed through unchanged
- **MASK**: The mask (automatically generated or edited via the mask editor)

## How It Works

The Preview Bridge node:
1. Takes an image as input
2. Automatically generates a mask:
   - Uses the alpha channel if the image is RGBA
   - Creates a white mask (all 1s) for RGB images
3. Saves the image for preview display with the preview widget interface
4. Allows mask editing through the standard ComfyUI mask editor (right-click menu)
5. Returns both the original image and the mask for downstream processing

## Technical Details

- Written for Python 3.8+
- Compatible with PyTorch and NumPy
- Automatically handles various image formats and dimensions
- Ensures mask values are clamped to [0, 1] range
- Uses temporary directory for preview storage (like PreviewImage)

## License

This custom node is provided as-is for use with ComfyUI.
