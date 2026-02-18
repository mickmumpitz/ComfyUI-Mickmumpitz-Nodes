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

// When true, the next POST /prompt will be tagged as an auto-requeue
// via extra_data so the Python prompt handler can identify it.
let _pendingAutoRequeue = false;

app.registerExtension({
    name: "Mickmumpitz.IterativeVideo",
    async setup() {
        // Intercept fetchApi to embed the auto-requeue marker directly
        // in the prompt JSON. This avoids global-flag race conditions
        // when multiple prompts are queued.
        const _origFetch = api.fetchApi.bind(api);
        api.fetchApi = async function (route, options = {}) {
            if (_pendingAutoRequeue && route === "/prompt" && options?.method === "POST") {
                _pendingAutoRequeue = false;
                try {
                    const body = JSON.parse(options.body);
                    body.extra_data = body.extra_data || {};
                    body.extra_data.mmz_auto_requeue = true;
                    options.body = JSON.stringify(body);
                } catch (e) {
                    console.warn("[MMZ Iter] Failed to mark auto-requeue:", e);
                }
            }
            return _origFetch(route, options);
        };
    },
});

// After final iteration, reset resume widget for next run
api.addEventListener("mmz-iter-reset", () => {
    for (const node of findNodesByType("FrameAccumulator")) {
        setWidgetValue(node, "resume_from_iteration", 0);
    }
});

// Re-queue at front of queue for next iteration.
// Front-of-queue (-1) ensures all iterations complete before the next job.
api.addEventListener("mmz-add-queue", () => {
    _pendingAutoRequeue = true;
    app.queuePrompt(-1, 1);
});
