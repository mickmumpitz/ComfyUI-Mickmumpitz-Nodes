# Idea: Mirrored / Linked Prompt Widgets Across Shots

## Concept

When duplicating a shot group, optionally keep certain widget values **linked** across shots so that editing one updates all linked copies. This would be useful for:

- Shared style prompts (e.g., a negative prompt or style suffix used across all shots)
- Shared camera/motion parameters
- Global quality settings

## Possible Implementation

- A "LinkedWidget" system where Set/Get nodes with a special naming convention (e.g., `Shared_style`) are kept in sync across shot groups.
- Or a "prompt template" node that injects a shared prefix/suffix into each shot's prompt while keeping the shot-specific part independent.
- Could use ComfyUI's existing Set/Get mechanism: shared widgets use Get nodes referencing a single Set node outside any shot group.

## Status

Not implemented. Currently all widgets are independent per shot (deep-copied on duplication).
