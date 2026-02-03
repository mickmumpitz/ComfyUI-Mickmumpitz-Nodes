import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ITER_NODE_TYPES = [
    "IterVideoRouter",
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

// Update iteration widgets and previous_frame_path on all relevant nodes
api.addEventListener("mmz-iter-update", ({ detail }) => {
    const { iteration, last_frame_path } = detail;

    for (const type of ITER_NODE_TYPES) {
        for (const node of findNodesByType(type)) {
            setWidgetValue(node, "iteration", iteration);
        }
    }

    for (const node of findNodesByType("IterVideoRouter")) {
        setWidgetValue(node, "previous_frame_path", last_frame_path);
    }
});

// Re-queue the workflow for next iteration
api.addEventListener("mmz-add-queue", () => {
    app.queuePrompt(0, 1);
});
