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

function setupStringPackVisibility(node) {
    const numFieldsWidget = node.widgets.find(w => w.name === "num_fields");
    if (!numFieldsWidget) return;

    const updateVisibility = () => {
        const n = numFieldsWidget.value;
        for (let i = 1; i <= 24; i++) {
            const w = node.widgets.find(w => w.name === `string_${i}`);
            if (w) w.hidden = i > n;
        }
        node.setSize(node.computeSize());
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
    node.setSize(node.computeSize());
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
        node.setSize(node.computeSize());
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
    node.setSize(node.computeSize());
}
