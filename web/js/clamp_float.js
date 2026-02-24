import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mickmumpitz.ClampFloat",

    nodeCreated(node) {
        if (node.comfyClass !== "ClampFloat") return;

        const valueWidget = node.widgets.find(w => w.name === "value");
        const minWidget = node.widgets.find(w => w.name === "min");
        const maxWidget = node.widgets.find(w => w.name === "max");

        if (!valueWidget || !minWidget || !maxWidget) return;

        const updateConstraints = () => {
            let lo = minWidget.value;
            let hi = maxWidget.value;
            if (lo > hi) [lo, hi] = [hi, lo];

            valueWidget.options.min = lo;
            valueWidget.options.max = hi;

            if (valueWidget.value < lo) valueWidget.value = lo;
            if (valueWidget.value > hi) valueWidget.value = hi;
        };

        const origMinCb = minWidget.callback;
        minWidget.callback = function (...args) {
            origMinCb?.apply(this, args);
            updateConstraints();
        };

        const origMaxCb = maxWidget.callback;
        maxWidget.callback = function (...args) {
            origMaxCb?.apply(this, args);
            updateConstraints();
        };

        // Apply initial constraints
        updateConstraints();
    },
});
