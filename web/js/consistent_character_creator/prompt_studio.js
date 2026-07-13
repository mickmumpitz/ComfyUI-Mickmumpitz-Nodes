import { app } from "../../../../scripts/app.js";

// Prompt Studio: show only the prompt field for the selected model, hide the rest.
app.registerExtension({
    name: "comfy.DatasetCreation.PromptStudio",
    async nodeCreated(node) {
        if (node.comfyClass !== "CCC_PromptStudio") {
            return;
        }

        const comboWidget = node.widgets.find((w) => w.name === "model_selection");
        if (!comboWidget) return;

        // Vertical space taken by everything that is NOT the prompt textarea
        // (title bar + the model_selection combo + trigger_word + lora_type +
        // margins). The visible textarea fills whatever node height remains, so
        // it grows and shrinks as the user resizes the node.
        const RESERVED_HEIGHT = 135;
        const MIN_TEXT_HEIGHT = 80;

        const toggleWidgets = () => {
            const prefix = String(comboWidget.value).split(" ")[0]; // "01", "02", ...

            for (const w of node.widgets) {
                if (!w.name.startsWith("prompt_")) {
                    continue;
                }
                if (w.name.startsWith(`prompt_${prefix}_`)) {
                    // Show the selected prompt's textarea and let it fill the
                    // node: width follows the node, height takes the leftover.
                    w.type = "customtext";
                    w.computeSize = () => [
                        node.size[0],
                        Math.max(MIN_TEXT_HEIGHT, node.size[1] - RESERVED_HEIGHT),
                    ];
                    if (w.inputEl) w.inputEl.style.display = "block";
                } else {
                    // Hide the others.
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                    if (w.inputEl) w.inputEl.style.display = "none";
                }
            }

            // Grow the node to fit only when it is currently too short for the
            // textarea's minimum; never shrink a node the user made taller.
            const minHeight = RESERVED_HEIGHT + MIN_TEXT_HEIGHT;
            if (node.size[1] < minHeight) {
                node.setSize([node.size[0], minHeight]);
            }
            if (node.onResize) node.onResize(node.size);
            app.graph.setDirtyCanvas(true, true);
        };

        // Re-layout the textarea whenever the node is resized.
        const originalOnResize = node.onResize;
        node.onResize = function (size) {
            const r = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
            app.graph.setDirtyCanvas(true, true);
            return r;
        };

        const originalCallback = comboWidget.callback;
        comboWidget.callback = function () {
            toggleWidgets();
            if (originalCallback) return originalCallback.apply(this, arguments);
        };

        // Let ComfyUI build the textarea overlays first, then collapse them.
        setTimeout(toggleWidgets, 10);
    },
});
