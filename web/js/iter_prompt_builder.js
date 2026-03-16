import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mickmumpitz.IterPromptBuilder",

    nodeCreated(node) {
        if (node.comfyClass !== "IterPromptBuilder") return;
        setupVisibility(node);
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "IterPromptBuilder") return;
        applyVisibility(node);
    },
});

function setupVisibility(node) {
    const numFieldsWidget = node.widgets.find((w) => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const updateVisibility = () => {
        const n = numFieldsWidget.value;
        for (let i = 1; i <= 24; i++) {
            const w = node.widgets.find((w) => w.name === `string_${i}`);
            if (w) w.hidden = i > n;
        }
        requestAnimationFrame(() => {
            const sz = node.computeSize();
            sz[0] = Math.max(sz[0], node.size[0]);
            node.setSize(sz);
        });
    };

    const originalCallback = numFieldsWidget.callback;
    numFieldsWidget.callback = function (value) {
        if (originalCallback) originalCallback.call(this, value);
        updateVisibility();
    };

    updateVisibility();
}

function applyVisibility(node) {
    const numFieldsWidget = node.widgets?.find((w) => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const n = numFieldsWidget.value;
    for (let i = 1; i <= 24; i++) {
        const w = node.widgets.find((w) => w.name === `string_${i}`);
        if (w) w.hidden = i > n;
    }
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        node.setSize(sz);
    });
}
