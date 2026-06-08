import { app } from "../../../scripts/app.js";

const NODE_COLLECT = "ImageCollect";
const NODE_UNPACK = "ImageCollectUnpack";
const NAME_WIDGET = "name";
const SOURCE_WIDGET = "source";

app.registerExtension({
    name: "Mickmumpitz.ImageCollect",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_COLLECT) {
            patchCollect(nodeType);
        } else if (nodeData.name === NODE_UNPACK) {
            patchUnpack(nodeType);
        }
    },
});

function safeResize(node) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        node.setSize(sz);
    });
}

function getWidgetValue(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function* iterAllGraphs(rootGraph) {
    if (!rootGraph) return;
    const seen = new WeakSet();
    const stack = [rootGraph];
    while (stack.length) {
        const g = stack.pop();
        if (!g || seen.has(g)) continue;
        seen.add(g);
        yield g;
        for (const n of g._nodes || []) {
            const sub = n.subgraph || n.graph;
            if (sub && sub !== g && !seen.has(sub)) stack.push(sub);
        }
    }
}

function findAllNodesByClass(className) {
    const out = [];
    for (const g of iterAllGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            if ((n.comfyClass || n.type) === className) out.push(n);
        }
    }
    return out;
}

function collectCollectionNames() {
    const names = new Set();
    for (const n of findAllNodesByClass(NODE_COLLECT)) {
        const v = getWidgetValue(n, NAME_WIDGET);
        if (v) names.add(v);
    }
    return [...names].sort();
}

function refreshAllUnpacks() {
    // Combo `values` is a live getter, so the dropdown is always current; this
    // just nudges a redraw after a name changes.
    for (const u of findAllNodesByClass(NODE_UNPACK)) u.setDirtyCanvas?.(true, true);
}

// ----------------------------------------------------------------- Collect ---

function installNameWidgetCallback(node) {
    const w = node.widgets?.find(w => w.name === NAME_WIDGET);
    if (!w || w._mmzCollectNameHooked) return;
    w._mmzCollectNameHooked = true;
    const orig = w.callback;
    w.callback = function () {
        if (orig) orig.apply(this, arguments);
        refreshAllUnpacks();
    };
}

function patchCollect(nodeType) {
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        origCreated?.apply(this, arguments);
        installNameWidgetCallback(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        origConfigure?.apply(this, arguments);
        const self = this;
        requestAnimationFrame(() => installNameWidgetCallback(self));
    };
}

// ------------------------------------------------------------------ Unpack ---

function removeInjectedInputSlots(node) {
    if (!node.inputs) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        if (/^input_\d+$/.test(node.inputs[i].name)) node.removeInput(i);
    }
}

function installSourceCombo(node, initialValue) {
    const existing = node.widgets?.find(w => w.name === SOURCE_WIDGET);
    const value = initialValue != null ? initialValue : (existing?.value || "");
    if (existing) {
        const idx = node.widgets.indexOf(existing);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }
    const comboOptions = {};
    Object.defineProperty(comboOptions, "values", {
        get: () => {
            const names = collectCollectionNames();
            return names.length > 0 ? names : [""];
        },
        enumerable: true,
        configurable: true,
    });
    const w = node.addWidget("combo", SOURCE_WIDGET, value, () => {
        node.setDirtyCanvas?.(true, true);
    }, comboOptions);
    w.serialize = true;
    return w;
}

function patchUnpack(nodeType) {
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        origCreated?.apply(this, arguments);
        removeInjectedInputSlots(this);
        installSourceCombo(this, "");
        safeResize(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        origConfigure?.apply(this, arguments);
        const saved = getWidgetValue(this, SOURCE_WIDGET) || "";
        removeInjectedInputSlots(this);
        installSourceCombo(this, saved);
        safeResize(this);
    };

    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
        origMenu?.apply(this, arguments);
        const self = this;
        options.unshift({
            content: "Refresh collection names",
            callback: () => self.setDirtyCanvas?.(true, true),
        });
    };
}
