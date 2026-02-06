import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ITER_NODE_TYPES = [
    "IterVideoRouter",
    "IterationSwitch",
    "ControlImageSlicer",
    "FrameAccumulator",
];

function findNodesByType(type) {
    return app.graph._nodes.filter((n) => n.type === type);
}

function setWidgetValue(node, widgetName, value) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (widget) {
        widget.value = value;
    }
}

// Update iteration widgets and sync session_id on all relevant nodes
api.addEventListener("mmz-iter-update", ({ detail }) => {
    const { session_id, iteration, last_frame_path } = detail;

    for (const type of ITER_NODE_TYPES) {
        for (const node of findNodesByType(type)) {
            setWidgetValue(node, "iteration", iteration);
        }
    }

    for (const node of findNodesByType("IterVideoRouter")) {
        setWidgetValue(node, "session_id", session_id);
        setWidgetValue(node, "previous_frame_path", last_frame_path);
    }
});

// After final iteration, reset widgets so the workflow is ready for the next run
api.addEventListener("mmz-iter-reset", ({ detail }) => {
    const { session_id } = detail;

    for (const type of ITER_NODE_TYPES) {
        for (const node of findNodesByType(type)) {
            setWidgetValue(node, "iteration", 0);
        }
    }

    for (const node of findNodesByType("FrameAccumulator")) {
        setWidgetValue(node, "session_id", session_id);
    }

    for (const node of findNodesByType("IterVideoRouter")) {
        setWidgetValue(node, "session_id", session_id);
        setWidgetValue(node, "previous_frame_path", "");
    }
});

// Re-queue the workflow for next iteration
api.addEventListener("mmz-add-queue", () => {
    app.queuePrompt(0, 1);
});
