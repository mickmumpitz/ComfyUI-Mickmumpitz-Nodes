# ComfyUI MickMumpitz Nodes

A collection of custom nodes for ComfyUI by MickMumpitz.

## Nodes

### Preview Bridge

A bridge node that enables mask editing in previews while maintaining the ability to output both the image and the edited mask.

**Features:**
- **Single Image Input**: Takes only an image input (no separate mask required)
- **Image Preview Widget**: Shows your image in a preview widget just like PreviewImage node
- **Mask Editor Integration**: Right-click on the preview to activate the mask editor
- **Dual Output**: Outputs both the original image and an editable mask
- **Smart Mask Generation**: Automatically extracts alpha channel if present, otherwise creates a blank white mask

**Usage:**
1. Add the "Preview Bridge" node to your workflow
2. Connect an image to the `image` input
3. The node displays the image in a preview widget (like PreviewImage)
4. **Right-click on the preview image** to activate the mask editor
5. Edit the mask as needed in the mask editor
6. The node outputs:
   - **IMAGE**: The original input image
   - **MASK**: The edited mask (or auto-generated mask if not edited)

**How It Works:**
1. Takes an image as input
2. Automatically generates a mask:
   - Uses the alpha channel if the image is RGBA
   - Creates a white mask (all 1s) for RGB images
3. Saves the image for preview display with the preview widget interface
4. Allows mask editing through the standard ComfyUI mask editor (right-click menu)
5. Returns both the original image and the mask for downstream processing

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "MickMumpitz" or "Preview Bridge"
3. Click Install
4. Restart ComfyUI

### Manual Installation

1. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfyui-mickmumpitz-nodes.git
   ```

2. Restart ComfyUI

## Requirements

- ComfyUI
- Python 3.8+
- PyTorch (included with ComfyUI)
- NumPy (included with ComfyUI)
- Pillow (included with ComfyUI)

## License

MIT License

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.
