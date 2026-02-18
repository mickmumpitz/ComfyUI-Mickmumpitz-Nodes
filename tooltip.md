# Iterative Video Nodes

These three nodes work together to generate long videos iteratively — processing one chunk at a time and looping automatically until the full video is done.

---

## Iter Video Router

Decides which image(s) to use as the starting frame(s) for the current iteration.

- **Iteration 0**: Uses `start_image` (your initial image).
- **Iteration 1+**: Pulls the last N frames from the in-memory frame buffer, where N is `num_start_frames`. This gives the video encoder more context for better consistency between iterations.

### Inputs

| Input | Description |
|---|---|
| `start_image` | The initial image for the very first iteration |
| `iteration` | Current iteration number (set automatically by the loop) |
| `session_id` | Must match FrameAccumulator's session_id (synced automatically by the loop) |
| `num_start_frames` | How many frames from the end of the previous iteration to pass as start frames (default: 1). Higher values (e.g. 4) give the encoder more context for better consistency. |
| `previous_frame_path` | Fallback path to saved last frame (set automatically, used when `num_start_frames` is 1) |

### Output

| Output | Description |
|---|---|
| `current_start` | The starting frame(s) for this iteration — 1 frame on iteration 0, up to `num_start_frames` on iteration 1+ |

---

## Control Image Slicer

Splits a full batch of control images (e.g. depth maps, pose frames) into per-iteration chunks.

If you have 400 control frames and generate 81 frames per iteration, this node picks the correct 81 for each iteration.

### Inputs

| Input | Description |
|---|---|
| `control_images` | The full batch of control frames for the entire video |
| `frames_per_iteration` | How many frames to extract per iteration (default: 81) |
| `iteration` | Current iteration number (set automatically by the loop) |
| `overlap_frames` | Number of frames to overlap between consecutive iterations for smoother transitions (default: 0) |
| `extend_mode` | What to do when running out of frames: `none`, `repeat_last`, or `loop` (default: `none`) |

### Slicing without overlap

| Iteration | Frames |
|---|---|
| 0 | 0–80 |
| 1 | 81–161 |
| 2 | 162–242 |

### Slicing with overlap (e.g. overlap_frames=4)

| Iteration | Frames |
|---|---|
| 0 | 0–80 |
| 1 | 77–157 |
| 2 | 154–234 |

The last 4 frames of one iteration are the first 4 of the next. Use `trim_first_n` on the FrameAccumulator to avoid duplicating those frames in the final output.

### Extend modes (when frames run out)

- **`none`** — Returns whatever frames are left (short batch). If completely past the end, returns just the last frame.
- **`repeat_last`** — Pads with copies of the last frame to always return exactly `frames_per_iteration` frames.
- **`loop`** — Wraps around to the beginning using modulo indexing.

### Outputs

| Output | Description |
|---|---|
| `control_slice` | The sliced control frames for this iteration |
| `has_frames` | `true` if real frames were available, `false` if it had to pad/extend |

---

## Frame Accumulator

Collects generated frames across all iterations, controls the auto-requeue loop, and outputs the complete video when done.

After each iteration it saves the last frame to disk (for IterVideoRouter to pick up), then triggers the next iteration automatically via server events. After the final iteration, it resets all widgets so the workflow is ready for the next run.

### Inputs

| Input | Description |
|---|---|
| `new_frames` | The frames generated in this iteration |
| `iteration` | Current iteration number (set automatically by the loop) |
| `total_iterations` | How many iterations to run in total (default: 5) |
| `session_id` | Unique ID for this generation run. Auto-incremented after each completed run. |
| `trim_first_n` | Remove this many frames from the start of each iteration (except iteration 0) to discard overlap (default: 0) |
| `save_intermediate` | Save individual frames to disk after each iteration for debugging/recovery (default: false) |

### Outputs

| Output | Description |
|---|---|
| `all_frames` | All accumulated frames from iteration 0 through the current iteration |
| `frame_count` | Total number of frames accumulated so far |
| `last_frame` | The last frame from this iteration (useful for preview or further processing) |

### Auto-reset behavior

When the final iteration completes, all nodes are automatically reset:
- `iteration` is set back to 0 on all three node types
- `session_id` is incremented on FrameAccumulator
- `previous_frame_path` is cleared on IterVideoRouter

You can just hit Queue again to start a fresh run.

---

## Data flow overview

```
Iteration 0:
  IterVideoRouter ──> uses start_image (iteration == 0)
  ControlImageSlicer ──> slices frames 0..80
  [video generation happens]
  FrameAccumulator ──> stores frames, saves last frame to disk
                   ──> triggers next iteration automatically

Iteration 1:
  IterVideoRouter ──> loads last frame from disk (iteration > 0)
  ControlImageSlicer ──> slices frames 81..161
  [video generation continues from last frame]
  FrameAccumulator ──> appends frames to buffer
                   ──> repeat...

Final iteration:
  FrameAccumulator ──> returns all accumulated frames
                   ──> resets widgets for next run
```
