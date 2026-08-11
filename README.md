# ComfyUI Mickmumpitz Nodes

A collection of custom nodes for ComfyUI by Mickmumpitz.

## Node Categories

- **Iterative Video** — Nodes for multi-iteration video generation with frame accumulation, routing, and resume support
- **Video Context** — Context image and control frame extraction for video workflows
- **Video Utilities** — Video concatenation and related tools
- **String Batch** — Story/style selectors and string batch processing
- **Utilities** — Resolution pickers (including **MiniMax H3 Resolution**, which takes the aspect ratio from the reference clip because H3 reframes rather than letterboxes, and scales the canvas from there), preprocessing settings, execution gates, and other helper nodes
- **Consistent Character Creator** — LoRA dataset tools: batch loaders, the model-specific captioning Prompt Studio, the Ideogram 4 tagger, bbox converters, the interactive Dataset Reviewer, face-area batch routing, and a branch gate

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Mickmumpitz"
3. Click Install
4. Restart ComfyUI

### Manual Installation

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mickmumpitz/ComfyUI-Mickmumpitz-Nodes.git
```

### Optional dependencies

The Consistent Character Creator face-area nodes work without any extra install if you
wire an `UltralyticsDetectorProvider` node (ComfyUI Impact Subpack) into the splitter's
`bbox_detector` input, and `SEGS Filter by Relative Area` likewise consumes Impact's
`SEGS`. Only if you leave `bbox_detector` empty and let **Face Area Batch Splitter**
load its own model does it need `ultralytics` (`pip install ultralytics`) plus a face
model such as `face_yolov8m.pt` in `ComfyUI/models/ultralytics/bbox/`. Neither is
required for the pack to load.

## License

MIT License
