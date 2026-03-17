import { app } from "../../../scripts/app.js";

const FIELD_NODES = {
    IterPromptBuilder: "string",
    IterSeedBatch: "seed",
};

app.registerExtension({
    name: "Mickmumpitz.IterFieldVisibility",

    nodeCreated(node) {
        const prefix = FIELD_NODES[node.comfyClass];
        if (prefix) setupVisibility(node, prefix);
    },

    async loadedGraphNode(node) {
        const prefix = FIELD_NODES[node.comfyClass];
        if (prefix) applyVisibility(node, prefix);
    },
});

function safeResize(node) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        node.setSize(sz);
    });
}

function setupVisibility(node, prefix) {
    const numFieldsWidget = node.widgets.find((w) => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const updateVisibility = () => {
        const n = numFieldsWidget.value;
        for (let i = 1; i <= 24; i++) {
            const w = node.widgets.find((w) => w.name === `${prefix}_${i}`);
            if (w) w.hidden = i > n;
        }
        safeResize(node);
    };

    const originalCallback = numFieldsWidget.callback;
    numFieldsWidget.callback = function (value) {
        if (originalCallback) originalCallback.call(this, value);
        updateVisibility();
    };

    updateVisibility();
}

function applyVisibility(node, prefix) {
    const numFieldsWidget = node.widgets?.find((w) => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const n = numFieldsWidget.value;
    for (let i = 1; i <= 24; i++) {
        const w = node.widgets.find((w) => w.name === `${prefix}_${i}`);
        if (w) w.hidden = i > n;
    }
    safeResize(node);
}
