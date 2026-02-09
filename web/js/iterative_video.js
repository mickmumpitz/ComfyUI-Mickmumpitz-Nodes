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

function getWidgetValue(node, widgetName) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    return widget ? widget.value : undefined;
}

// Track whether a queue was triggered by the auto-requeue loop
let _isAutoRequeue = false;

app.registerExtension({
    name: "Mickmumpitz.IterativeVideo",
    async setup() {
        // Intercept queuePrompt to handle resume logic on manual queues
        const _origQueuePrompt = app.queuePrompt.bind(app);
        app.queuePrompt = async function (number, batchCount) {
            if (!_isAutoRequeue) {
                // Manual queue — always clear Python-side iteration state first
                try {
                    await api.fetchApi("/mmz-iter/reset-session", { method: "POST" });
                } catch (e) {
                    console.warn("[MMZ Iter] Failed to reset session:", e);
                }

                // Check if resume is requested
                let isResume = false;
                for (const node of findNodesByType("FrameAccumulator")) {
                    const resumeFrom = getWidgetValue(node, "resume_from_iteration");
                    if (resumeFrom !== undefined && resumeFrom >= 0) {
                        // Cosmetic: set iteration widgets for non-subgraph nodes
                        for (const type of ITER_NODE_TYPES) {
                            for (const n of findNodesByType(type)) {
                                setWidgetValue(n, "iteration", resumeFrom);
                            }
                        }
                        isResume = true;
                        break;
                    }
                }
                if (!isResume) {
                    // Fresh run: reset iteration widgets to 0 (cosmetic for non-subgraph)
                    for (const type of ITER_NODE_TYPES) {
                        for (const n of findNodesByType(type)) {
                            setWidgetValue(n, "iteration", 0);
                        }
                    }
                    for (const node of findNodesByType("IterVideoRouter")) {
                        setWidgetValue(node, "previous_frame_path", "");
                    }
                }
            }
            _isAutoRequeue = false;
            return _origQueuePrompt(number, batchCount);
        };
    },
});

// Update iteration widgets on all relevant nodes
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

// After final iteration, reset widgets for next run
api.addEventListener("mmz-iter-reset", () => {
    for (const type of ITER_NODE_TYPES) {
        for (const node of findNodesByType(type)) {
            setWidgetValue(node, "iteration", 0);
        }
    }

    for (const node of findNodesByType("FrameAccumulator")) {
        setWidgetValue(node, "resume_from_iteration", -1);
    }

    for (const node of findNodesByType("IterVideoRouter")) {
        setWidgetValue(node, "previous_frame_path", "");
    }
});

// Re-queue the workflow for next iteration
api.addEventListener("mmz-add-queue", async () => {
    try {
        await api.fetchApi("/mmz-iter/auto-requeue", { method: "POST" });
    } catch (e) {
        console.warn("[MMZ Iter] Failed to mark auto-requeue:", e);
    }
    _isAutoRequeue = true;
    app.queuePrompt(0, 1);
});
