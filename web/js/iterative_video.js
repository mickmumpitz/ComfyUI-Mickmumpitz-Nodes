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

// Track whether a queue was triggered by the auto-requeue loop
let _isAutoRequeue = false;

app.registerExtension({
    name: "Mickmumpitz.IterativeVideo",
    async setup() {
        // Intercept queuePrompt to clear Python-side state on manual queues
        const _origQueuePrompt = app.queuePrompt.bind(app);
        app.queuePrompt = async function (number, batchCount) {
            if (!_isAutoRequeue) {
                // Manual queue — clear Python-side iteration state
                try {
                    await api.fetchApi("/mmz-iter/reset-session", { method: "POST" });
                } catch (e) {
                    console.warn("[MMZ Iter] Failed to reset session:", e);
                }
            }
            _isAutoRequeue = false;
            return _origQueuePrompt(number, batchCount);
        };
    },
});

// After final iteration, reset resume widget for next run
api.addEventListener("mmz-iter-reset", () => {
    for (const node of findNodesByType("FrameAccumulator")) {
        setWidgetValue(node, "resume_from_iteration", 0);
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
