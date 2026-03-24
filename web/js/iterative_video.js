import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function findNodesByType(type) {
    return app.graph._nodes.filter((n) => n.type === type);
}

function setWidgetValue(node, widgetName, value) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (widget) {
        widget.value = value;
    }
}

// After final iteration, reset resume widget for next run
api.addEventListener("mmz-iter-reset", () => {
    for (const node of findNodesByType("FrameAccumulator")) {
        setWidgetValue(node, "resume_from_iteration", 0);
    }
});
