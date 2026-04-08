import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mickmumpitz.WanResolutionPicker",

    nodeCreated(node) {
        if (node.comfyClass !== "WanResolutionPicker") return;

        const resolutionWidget = node.widgets.find(w => w.name === "resolution");
        const aspectWidget = node.widgets.find(w => w.name === "aspect_ratio");
        const widthWidget = node.widgets.find(w => w.name === "custom_width");
        const heightWidget = node.widgets.find(w => w.name === "custom_height");

        if (!resolutionWidget || !aspectWidget || !widthWidget || !heightWidget) return;

        const updateVisibility = () => {
            const isCustom = resolutionWidget.value === "Custom";
            const isSquarePreset = resolutionWidget.value === "1024p square";

            aspectWidget.hidden = isCustom || isSquarePreset;
            widthWidget.hidden = !isCustom;
            heightWidget.hidden = !isCustom;

            node.setSize(node.computeSize());
        };

        const originalCallback = resolutionWidget.callback;
        resolutionWidget.callback = function(value) {
            if (originalCallback) originalCallback.call(this, value);
            updateVisibility();
        };

        updateVisibility();
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "WanResolutionPicker") return;

        const resolutionWidget = node.widgets?.find(w => w.name === "resolution");
        const aspectWidget = node.widgets?.find(w => w.name === "aspect_ratio");
        const widthWidget = node.widgets?.find(w => w.name === "custom_width");
        const heightWidget = node.widgets?.find(w => w.name === "custom_height");

        if (resolutionWidget && aspectWidget && widthWidget && heightWidget) {
            const isCustom = resolutionWidget.value === "Custom";
            const isSquarePreset = resolutionWidget.value === "1024p square";

            aspectWidget.hidden = isCustom || isSquarePreset;
            widthWidget.hidden = !isCustom;
            heightWidget.hidden = !isCustom;

            // Preserve the saved size: only grow to the computed minimum if needed
            const savedSize = [node.size[0], node.size[1]];
            const minSize = node.computeSize();
            node.setSize([
                Math.max(savedSize[0], minSize[0]),
                Math.max(savedSize[1], minSize[1])
            ]);
        }
    }
});
