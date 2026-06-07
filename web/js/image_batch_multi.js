import { app } from "../../../scripts/app.js";

/**
 * Dynamic IMAGE inputs driven by an "inputcount" widget, with an
 * "Update inputs" button. Mirrors KJNodes' Image Batch Multi UX so the
 * node feels familiar, but the backend skips empty inputs instead of
 * padding them with black images.
 */
function setupDynamicImageInputs(node) {
    const prefix = "image_";
    const type = "IMAGE";
    const slotOptions = { shape: 7 }; // optional/grid slot shape

    const rebuild = () => {
        if (!node.inputs) node.inputs = [];
        const countW = node.widgets?.find(w => w.name === "inputcount");
        if (!countW) return;
        const target = countW.value;
        const current = node.inputs.filter(i => i.name?.startsWith(prefix)).length;
        if (target === current) return;
        if (target < current) {
            for (let i = 0; i < current - target; i++) {
                node.removeInput(node.inputs.length - 1);
            }
        } else {
            for (let i = current + 1; i <= target; i++) {
                node.addInput(`${prefix}${i}`, type, slotOptions);
            }
        }
    };

    node.addWidget("button", "Update inputs", null, rebuild);

    const countW = node.widgets?.find(w => w.name === "inputcount");
    if (countW) {
        const origCb = countW.callback;
        countW.callback = function (value, canvas) {
            const r = origCb ? origCb.apply(this, arguments) : undefined;
            if (!canvas) rebuild(); // bare = API reload; skip interactive scrub
            return r;
        };
    }
}

app.registerExtension({
    name: "Mickmumpitz.ImageBatchMultiSkipEmpty",

    // Fires on every instantiation, including nodes restored from a saved
    // graph. Saved input slots are restored from the graph JSON, so we only
    // need to wire up the button + widget callback here.
    nodeCreated(node) {
        if (node.comfyClass === "ImageBatchMultiSkipEmpty") {
            setupDynamicImageInputs(node);
        }
    },
});
