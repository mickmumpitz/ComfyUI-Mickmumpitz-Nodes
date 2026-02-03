# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node pack providing image processing nodes for ComfyUI. Currently ships one active node (`BatchColorCorrector`) and one disabled node (`PreviewBridge`).

## Development Setup

This is a ComfyUI plugin — it must live inside `ComfyUI/custom_nodes/` to function. Dependencies: `pip install -r requirements.txt` (numpy, Pillow, opencv-python). No test suite or linter is configured.

## Publishing

Publishing to the ComfyUI registry is automated via GitHub Actions (`.github/workflows/publish_action.yml`). It triggers on pushes to `main` that modify `pyproject.toml` or via manual `workflow_dispatch`. Bump the version in `pyproject.toml` to publish a new release.

## Architecture

### Node Registration

ComfyUI discovers nodes through exported dicts in `__init__.py`:
- Each node file in `nodes/` exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- `nodes/__init__.py` aggregates all node mappings
- Root `__init__.py` re-exports the aggregated mappings

To add a new node: create a file in `nodes/`, define the node class with ComfyUI's required class attributes (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`), export the two mapping dicts, and import/merge them in `nodes/__init__.py`.

### Data Conventions

- **Images**: `(B, H, W, C)` torch tensors, values 0–1
- **Masks**: `(B, H, W)` torch tensors, values 0–1
- GPU tensors are used throughout with CPU fallback; use `comfy.model_management` for device selection

### Key Files

- `nodes/batch_color_corrector.py` — GPU-accelerated batch color correction with 30 presets, AI scene analysis, reference-based color matching, and manual controls. Contains custom `rgb_to_hsv`/`hsv_to_rgb` torch implementations.
- `nodes/preview_bridge.py` — Image preview with mask editing (currently disabled in `nodes/__init__.py`).
- `install.py` — Called by ComfyUI Manager to install opencv-python if missing.
