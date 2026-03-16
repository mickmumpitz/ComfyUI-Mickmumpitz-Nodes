import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mickmumpitz.StringPack",

    nodeCreated(node) {
        if (node.comfyClass === "StringPack") {
            setupStringPackVisibility(node);
        } else if (node.comfyClass === "PromptStitcher") {
            setupPromptStitcherOutputs(node);
        }
    },

    async loadedGraphNode(node) {
        if (node.comfyClass === "StringPack") {
            applyStringPackVisibility(node);
        } else if (node.comfyClass === "PromptStitcher") {
            applyPromptStitcherOutputs(node);
        }
    },
});

/**
 * Resize the node to fit its computed size, but never shrink the width.
 * Uses requestAnimationFrame so the DOM has laid out first.
 */
function safeResize(node) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        node.setSize(sz);
    });
}

function setupStringPackVisibility(node) {
    const numFieldsWidget = node.widgets.find(w => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const updateVisibility = () => {
        const n = numFieldsWidget.value;
        for (let i = 1; i <= 24; i++) {
            const w = node.widgets.find(w => w.name === `string_${i}`);
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

function applyStringPackVisibility(node) {
    const numFieldsWidget = node.widgets?.find(w => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const n = numFieldsWidget.value;
    for (let i = 1; i <= 24; i++) {
        const w = node.widgets.find(w => w.name === `string_${i}`);
        if (w) w.hidden = i > n;
    }
    safeResize(node);
}

function setupPromptStitcherOutputs(node) {
    const numOutputsWidget = node.widgets.find(w => w.name === "num_outputs");
    if (!numOutputsWidget) return;

    const updateOutputs = () => {
        const n = numOutputsWidget.value;
        while (node.outputs.length > n) {
            node.removeOutput(node.outputs.length - 1);
        }
        while (node.outputs.length < n) {
            const idx = node.outputs.length;
            node.addOutput(`prompt_${idx + 1}`, "STRING");
        }
        safeResize(node);
    };

    const originalCallback = numOutputsWidget.callback;
    numOutputsWidget.callback = function (value) {
        if (originalCallback) originalCallback.call(this, value);
        updateOutputs();
    };

    updateOutputs();
}

function applyPromptStitcherOutputs(node) {
    const numOutputsWidget = node.widgets?.find(w => w.name === "num_outputs");
    if (!numOutputsWidget) return;

    const n = numOutputsWidget.value;
    while (node.outputs.length > n) {
        node.removeOutput(node.outputs.length - 1);
    }
    while (node.outputs.length < n) {
        const idx = node.outputs.length;
        node.addOutput(`prompt_${idx + 1}`, "STRING");
    }
    safeResize(node);
}
