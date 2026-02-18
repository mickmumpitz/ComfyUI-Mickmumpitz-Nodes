import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mickmumpitz.PreprocessSettings",

    nodeCreated(node) {
        if (node.comfyClass !== "PreprocessSettings") return;

        const activateWidget = node.widgets.find(w => w.name === "activate");
        const methodWidget = node.widgets.find(w => w.name === "method");

        if (!activateWidget || !methodWidget) return;

        // Function to update visibility
        const updateVisibility = () => {
            const isActive = activateWidget.value;
            methodWidget.hidden = !isActive;

            // Trigger node resize
            node.setSize(node.computeSize());
        };

        // Store original callback
        const originalCallback = activateWidget.callback;

        // Override callback
        activateWidget.callback = function(value) {
            if (originalCallback) {
                originalCallback.call(this, value);
            }
            updateVisibility();
        };

        // Initial state
        updateVisibility();
    },

    // Also handle when loading saved workflows
    async loadedGraphNode(node) {
        if (node.comfyClass !== "PreprocessSettings") return;

        const activateWidget = node.widgets?.find(w => w.name === "activate");
        const methodWidget = node.widgets?.find(w => w.name === "method");

        if (activateWidget && methodWidget) {
            methodWidget.hidden = !activateWidget.value;
            node.setSize(node.computeSize());
        }
    }
});
