# Iterative Video Generation - Research Findings

## ComfyUI Execution Model

ComfyUI evaluates workflows as a **directed acyclic graph (DAG)**. Each node executes once per workflow run. There is no way to call other nodes from within a node or create loops within a single execution.

**Iterative processing** (needed for generating long videos in chunks) requires a **queue-based loop**: each iteration is a full workflow execution that re-queues itself. State passes between executions via:
1. **Disk persistence** - saving images/state to ComfyUI's temp directory
2. **Server events** - Python sends `PromptServer.instance.send_sync()` events
3. **Client-side JS** - listens for events, updates widget values, and re-queues

## Queue-Based Loop Pattern

This pattern is established by [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) and is the standard way to achieve iterative execution in ComfyUI.

### How it works

1. A Python output node finishes its work and sends server events:
   - One event to update widget values (iteration counter, file paths)
   - One event to trigger re-queuing
2. A JS extension file (`web/js/*.js`) listens for these events:
   - Updates widget values on relevant nodes in the graph
   - Calls `app.queuePrompt(0, 1)` to queue the next execution
3. ComfyUI processes the next queue entry with updated widget values
4. The loop stops when the Python node doesn't send the re-queue event (final iteration)

### Key reference implementations

- **Impact Pack loop control**: `comfyui-impact-pack/modules/impact/logics.py` - `QueueTriggerCountdown`, `ConditionalStopIteration`
- **Impact Pack JS events**: `comfyui-impact-pack/js/common.js` and `js/impact-pack.js`
- **Impact Pack image sender/receiver**: `comfyui-impact-pack/modules/impact/impact_pack.py:2331-2412`

## Video Generation Context

### WanVideo continuation approach
[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) LongCat mode uses overlap-based extension in latent space. The last N frames of one generation become the first N frames of the next, providing visual continuity. This happens within their own node logic, not via ComfyUI's queue system.

### VideoHelperSuite conventions
[comfyui-videohelpersuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) uses standard `(B, H, W, C)` image tensors for batched video frames. Their `batched_nodes.py` provides frame slicing/combining utilities.

## Data Conventions

- **Images**: `(B, H, W, C)` torch tensors, values 0-1
- **Masks**: `(B, H, W)` torch tensors, values 0-1
- **JS extensions**: Set `WEB_DIRECTORY = "./web"` in `__init__.py` and include it in `__all__` for ComfyUI to discover and serve JS files

## Architecture Decisions

### Why self-contained (no Impact Pack dependency)?
- Impact Pack is a large dependency with many unrelated features
- The loop mechanism is ~40 lines of JS and ~20 lines of Python for server events
- Self-contained implementation avoids version coupling and install complexity

### Memory considerations
- Accumulated frames stored on CPU RAM: ~1000 frames at 720p = ~3.7GB
- `save_intermediate` option allows disk-based storage for very long videos
- Frame buffer cleaned up after final iteration

### Known limitations (v1)
- Single loop per workflow (JS updates ALL matching node types)
- In-memory buffer lost on crash (use `save_intermediate=True` for recovery)
- Session ID must be manually set to be unique across concurrent workflows
