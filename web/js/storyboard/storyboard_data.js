// Make Storyboard Data: show only the per-shot widget rows up to shot_count.
// Hooks the shot_count callback and polls as a fallback, so dragging,
// programmatic changes and workflow-load restores are all caught.

import { app } from "../../../../scripts/app.js";

const MAX_SHOTS = 20;
const SUFFIXES = ["title", "prompt", "frames"];

app.registerExtension({
    name: "Mickmumpitz.StoryboardData",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MickmumpitzStoryboardData") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const getCountWidget = () => node.widgets?.find((w) => w.name === "shot_count");

            let lastCount = null;

            const applyVisibility = (count) => {
                for (let i = 1; i <= MAX_SHOTS; i++) {
                    SUFFIXES.forEach((suffix) => {
                        const w = node.widgets?.find((w) => w.name === `shot${i}_${suffix}`);
                        if (!w) return;
                        const shouldShow = i <= count;
                        if (w.hidden === !shouldShow) return; // no-op if already correct
                        w.hidden = !shouldShow;
                        w.computeSize = shouldShow ? undefined : () => [0, -4];
                    });
                }
                node.setDirtyCanvas(true, true);
                if (node.computeSize) {
                    const newSize = node.computeSize();
                    node.size = [Math.max(node.size[0], newSize[0]), newSize[1]];
                }
            };

            const updateVisibility = () => {
                const countWidget = getCountWidget();
                const count = countWidget ? countWidget.value : MAX_SHOTS;
                if (count === lastCount) return;
                lastCount = count;
                applyVisibility(count);
            };

            // 1) React immediately on explicit widget interaction.
            const countWidget = getCountWidget();
            if (countWidget) {
                const originalCallback = countWidget.callback;
                countWidget.callback = function () {
                    const result = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                    updateVisibility();
                    return result;
                };
            }

            // 2) Robust fallback: poll the value so dragging, programmatic
            //    changes, and workflow-load restores are all caught too.
            const poll = setInterval(() => {
                if (!node.graph) {
                    clearInterval(poll);
                    return;
                }
                updateVisibility();
            }, 250);

            setTimeout(updateVisibility, 50);
            return r;
        };
    },
});
