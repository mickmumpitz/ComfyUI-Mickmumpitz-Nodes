import { app } from "../../../scripts/app.js";

const MAX_SLOTS = 32;
const NODE_BUNDLE = "Bundle";
const NODE_UNBUNDLE = "Unbundle";
const NODE_UNBUNDLE_BY_NAME = "UnbundleByName";
const META_WIDGET = "bundle_meta";
const SAVED_META_WIDGET = "saved_meta";
const SLOT_NAMES_WIDGET = "slot_names";
const NAME_WIDGET = "name";
const SOURCE_BUNDLE_WIDGET = "source_bundle";
const REROUTE_TYPES = new Set(["Reroute", "Reroute (rgthree)", "ReroutePrimitive"]);
const GETNODE_TYPES = new Set(["GetNode"]);
const SETNODE_TYPES = new Set(["SetNode"]);

app.registerExtension({
    name: "Mickmumpitz.Bundle",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === NODE_BUNDLE) {
            patchBundle(nodeType);
        } else if (nodeData.name === NODE_UNBUNDLE) {
            patchUnbundle(nodeType);
        } else if (nodeData.name === NODE_UNBUNDLE_BY_NAME) {
            patchUnbundleByName(nodeType);
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

function hideWidget(node, name) {
    const w = node.widgets?.find(w => w.name === name);
    if (!w) return;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
}

function getWidgetValue(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function setWidgetValue(node, name, value) {
    const w = node.widgets?.find(w => w.name === name);
    if (w) w.value = value;
}

function safeParse(str) {
    try {
        const parsed = JSON.parse(str || "{}");
        return {
            names: Array.isArray(parsed.names) ? parsed.names : [],
            types: Array.isArray(parsed.types) ? parsed.types : [],
            sources: Array.isArray(parsed.sources) ? parsed.sources : [],
        };
    } catch (e) {
        return { names: [], types: [], sources: [] };
    }
}

function readBundleMeta(node) {
    return safeParse(getWidgetValue(node, META_WIDGET));
}

function readSlotNames(node) {
    try {
        const obj = JSON.parse(getWidgetValue(node, SLOT_NAMES_WIDGET) || "{}");
        return (obj && typeof obj === "object") ? obj : {};
    } catch (e) {
        return {};
    }
}

function writeSlotName(node, slotInputName, customName) {
    const obj = readSlotNames(node);
    if (customName && customName.length > 0) {
        obj[slotInputName] = customName;
    } else {
        delete obj[slotInputName];
    }
    setWidgetValue(node, SLOT_NAMES_WIDGET, JSON.stringify(obj));
}

function applyCustomSlotLabels(node) {
    const custom = readSlotNames(node);
    for (const inp of node.inputs || []) {
        const c = custom[inp.name];
        if (c) inp.label = c;
    }
}

function uniqueCustomName(node, slotInputName, desired) {
    const custom = readSlotNames(node);
    const taken = new Set();
    for (const [k, v] of Object.entries(custom)) {
        if (k !== slotInputName) taken.add(v);
    }
    for (const inp of node.inputs || []) {
        if (inp.name === slotInputName) continue;
        if (custom[inp.name]) continue;
        const lbl = inp.label || inp.name;
        if (lbl) taken.add(lbl);
    }
    if (!taken.has(desired)) return desired;
    let i = 2;
    while (taken.has(`${desired}_${i}`)) i++;
    return `${desired}_${i}`;
}

function readSavedMeta(node) {
    return safeParse(getWidgetValue(node, SAVED_META_WIDGET));
}

function rebuildBundleMeta(node) {
    const custom = readSlotNames(node);
    const names = [];
    const types = [];
    const sources = [];
    for (const inp of node.inputs || []) {
        if (inp.link != null) {
            names.push(custom[inp.name] || inp.label || inp.name);
            types.push(inp.type != null ? String(inp.type) : "*");
            sources.push(inp.name);
        }
    }
    setWidgetValue(node, META_WIDGET, JSON.stringify({ names, types, sources }));
}

function firstUnusedSlotName(node) {
    const used = new Set((node.inputs || []).map(i => i.name));
    for (let i = 1; i <= MAX_SLOTS; i++) {
        const n = `input_${i}`;
        if (!used.has(n)) return n;
    }
    return null;
}

function ensureTrailingEmpty(node) {
    const inputs = node.inputs || [];
    const emptyIdx = [];
    for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].link == null) emptyIdx.push(i);
    }
    if (emptyIdx.length > 1) {
        for (let j = emptyIdx.length - 2; j >= 0; j--) {
            node.removeInput(emptyIdx[j]);
        }
    } else if (emptyIdx.length === 0) {
        const nextName = firstUnusedSlotName(node);
        if (nextName) node.addInput(nextName, "*");
    }
    const fresh = node.inputs || [];
    if (fresh.length > 0) {
        const last = fresh[fresh.length - 1];
        if (last.link == null) {
            last.type = "*";
            last.label = undefined;
        }
    }
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

function findAllNodesByClass(classNames) {
    const set = classNames instanceof Set ? classNames : new Set(classNames);
    const out = [];
    for (const g of iterAllGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            const k = n.comfyClass || n.type;
            if (set.has(k)) out.push(n);
        }
    }
    return out;
}

function resolveLinkOrigin(graph, link) {
    if (!link) return null;
    const origin = graph?.getNodeById?.(link.origin_id);
    if (origin) return { node: origin, slot: link.origin_slot, graph };
    if (typeof link.resolve === "function") {
        try {
            const r = link.resolve(graph);
            const src = r?.output;
            if (src && src.origin_id != null) {
                const resolvedNode = graph?.getNodeById?.(src.origin_id)
                    || app.graph?.getNodeById?.(src.origin_id);
                if (resolvedNode) {
                    return { node: resolvedNode, slot: src.origin_slot ?? link.origin_slot, graph };
                }
            }
        } catch (e) { /* older LiteGraph */ }
    }
    return null;
}

function resolveSourceBundle(startNode, bundleInputSlot) {
    const visited = new Set();
    let node = startNode;
    let slot = bundleInputSlot;
    let graph = startNode.graph || app.graph;
    let guard = 0;
    while (node && guard++ < 64) {
        const key = `${graph?.id ?? ""}:${node.id}:${slot}`;
        if (visited.has(key)) return null;
        visited.add(key);

        const link = node.inputs?.[slot]?.link;
        if (link == null) return null;
        const linkInfo = graph?.links?.[link] ?? graph?._links?.get?.(link);
        if (!linkInfo) return null;
        const origin = resolveLinkOrigin(graph, linkInfo);
        if (!origin) return null;

        const originType = origin.node.comfyClass || origin.node.type;
        if (originType === NODE_BUNDLE) return origin.node;

        if (REROUTE_TYPES.has(originType)) {
            node = origin.node;
            slot = 0;
            graph = origin.node.graph || graph;
            continue;
        }

        if (GETNODE_TYPES.has(originType)) {
            const name = origin.node.widgets?.[0]?.value;
            if (!name) return null;
            const setter = findSetNodeByName(name);
            if (!setter) return null;
            node = setter;
            slot = 0;
            graph = setter.graph || graph;
            continue;
        }

        return null;
    }
    return null;
}

function findSetNodeByName(name) {
    for (const g of iterAllGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            const t = n.type || n.comfyClass;
            if (SETNODE_TYPES.has(t) && n.widgets?.[0]?.value === name) {
                return n;
            }
        }
    }
    return null;
}

function rebuildOutputsFromMeta(node, meta) {
    const wantCount = meta.names.length;
    for (let i = 0; i < wantCount; i++) {
        const wantName = meta.names[i] || `output_${i + 1}`;
        const wantType = meta.types[i] || "*";
        if (i < node.outputs.length) {
            const out = node.outputs[i];
            const typeChanged = out.type !== wantType && out.type !== "*" && wantType !== "*";
            if (typeChanged && out.links && out.links.length > 0) {
                const graph = node.graph || app.graph;
                for (const linkId of [...out.links]) {
                    const link = graph?.links?.[linkId];
                    if (link) {
                        const target = graph?.getNodeById?.(link.target_id);
                        if (target) node.disconnectOutput(i, target);
                    }
                }
            }
            out.type = wantType;
            out.name = wantName;
            out.label = wantName;
        } else {
            node.addOutput(wantName, wantType);
        }
    }
    while (node.outputs.length > wantCount) {
        node.removeOutput(node.outputs.length - 1);
    }
    safeResize(node);
}

function applyUnbundleOutputs(node) {
    let srcBundle = resolveSourceBundle(node, 0);
    let meta;
    if (srcBundle) {
        meta = readBundleMeta(srcBundle);
        setWidgetValue(node, SAVED_META_WIDGET, JSON.stringify(meta));
    } else {
        meta = readSavedMeta(node);
    }
    rebuildOutputsFromMeta(node, meta);
}

function findBundleByName(name) {
    if (!name) return null;
    for (const g of iterAllGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            if ((n.comfyClass || n.type) !== NODE_BUNDLE) continue;
            if (getWidgetValue(n, NAME_WIDGET) === name) return n;
        }
    }
    return null;
}

function collectBundleNames() {
    const names = new Set();
    for (const b of findAllNodesByClass([NODE_BUNDLE])) {
        const n = getWidgetValue(b, NAME_WIDGET);
        if (n) names.add(n);
    }
    return [...names].sort();
}

function uniqueBundleName(excludeNode) {
    const taken = new Set();
    for (const b of findAllNodesByClass([NODE_BUNDLE])) {
        if (b === excludeNode) continue;
        const n = getWidgetValue(b, NAME_WIDGET);
        if (n) taken.add(n);
    }
    for (let i = 1; i <= 9999; i++) {
        const candidate = `bundle_${i}`;
        if (!taken.has(candidate)) return candidate;
    }
    return `bundle_${Date.now()}`;
}

function applyUnbundleByNameOutputs(node) {
    const selected = getWidgetValue(node, SOURCE_BUNDLE_WIDGET);
    const srcBundle = findBundleByName(selected);
    let meta;
    if (srcBundle) {
        meta = readBundleMeta(srcBundle);
        setWidgetValue(node, SAVED_META_WIDGET, JSON.stringify(meta));
    } else {
        meta = readSavedMeta(node);
    }
    rebuildOutputsFromMeta(node, meta);
}

function refreshAllUnbundles() {
    for (const u of findAllNodesByClass([NODE_UNBUNDLE])) applyUnbundleOutputs(u);
    for (const u of findAllNodesByClass([NODE_UNBUNDLE_BY_NAME])) applyUnbundleByNameOutputs(u);
}

function installNameWidgetCallback(node) {
    const w = node.widgets?.find(w => w.name === NAME_WIDGET);
    if (!w || w._mmzBundleNameHooked) return;
    w._mmzBundleNameHooked = true;
    const orig = w.callback;
    w.callback = function (value) {
        if (orig) orig.apply(this, arguments);
        refreshAllUnbundles();
    };
}

function patchBundle(nodeType) {
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        origCreated?.apply(this, arguments);
        hideWidget(this, META_WIDGET);
        hideWidget(this, SLOT_NAMES_WIDGET);
        if (this.inputs) {
            for (let i = this.inputs.length - 1; i >= 0; i--) {
                this.removeInput(i);
            }
        }
        this.addInput("input_1", "*");
        const nameWidget = this.widgets?.find(w => w.name === NAME_WIDGET);
        if (nameWidget && !nameWidget.value) {
            nameWidget.value = uniqueBundleName(this);
        }
        installNameWidgetCallback(this);
        rebuildBundleMeta(this);
        safeResize(this);
    };

    const origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, linkInfo) {
        origConn?.apply(this, arguments);
        if (type !== LiteGraph.INPUT) return;
        const slot = this.inputs?.[slotIndex];
        if (!slot) return;

        if (isConnected && linkInfo) {
            const graph = this.graph || app.graph;
            const origin = resolveLinkOrigin(graph, linkInfo);
            const originOut = origin?.node?.outputs?.[origin.slot];
            if (originOut) {
                slot.type = originOut.type || "*";
                slot.label = originOut.label || originOut.name || String(originOut.type || "*");
            }
        } else {
            slot.type = "*";
            slot.label = undefined;
            writeSlotName(this, slot.name, null);
        }

        applyCustomSlotLabels(this);
        ensureTrailingEmpty(this);
        applyCustomSlotLabels(this);
        rebuildBundleMeta(this);
        refreshAllUnbundles();
        safeResize(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        origConfigure?.apply(this, arguments);
        hideWidget(this, META_WIDGET);
        hideWidget(this, SLOT_NAMES_WIDGET);
        const self = this;
        requestAnimationFrame(() => {
            installNameWidgetCallback(self);
            ensureTrailingEmpty(self);
            applyCustomSlotLabels(self);
            rebuildBundleMeta(self);
            refreshAllUnbundles();
            safeResize(self);
        });
    };

    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
        origMenu?.apply(this, arguments);
        options.unshift({
            content: "Refresh downstream Unbundles",
            callback: () => {
                rebuildBundleMeta(this);
                refreshAllUnbundles();
            },
        });
    };

    nodeType.prototype.getSlotMenuOptions = function (slotInfo) {
        if (!slotInfo || slotInfo.input == null) return null;
        const slotIdx = slotInfo.slot;
        const inp = this.inputs?.[slotIdx];
        if (!inp || inp.link == null) return null;
        const self = this;
        const current = readSlotNames(this)[inp.name] || inp.label || inp.name;
        return [
            {
                content: "Rename slot…",
                callback: () => {
                    const canvas = app.canvas || LGraphCanvas.active_canvas;
                    const commit = (value) => {
                        const trimmed = (value == null ? "" : String(value)).trim();
                        if (!trimmed) {
                            writeSlotName(self, inp.name, null);
                            inp.label = undefined;
                        } else {
                            const unique = uniqueCustomName(self, inp.name, trimmed);
                            writeSlotName(self, inp.name, unique);
                            inp.label = unique;
                        }
                        rebuildBundleMeta(self);
                        refreshAllUnbundles();
                        self.setDirtyCanvas?.(true, true);
                    };
                    if (canvas?.prompt) {
                        canvas.prompt("Slot name", current, commit, { target: canvas.canvas });
                    } else {
                        const v = window.prompt("Slot name", current);
                        if (v !== null) commit(v);
                    }
                },
            },
            {
                content: "Reset slot name",
                callback: () => {
                    writeSlotName(self, inp.name, null);
                    inp.label = undefined;
                    const graph = self.graph || app.graph;
                    const link = graph?.links?.[inp.link];
                    if (link) {
                        const originNode = graph?.getNodeById?.(link.origin_id);
                        const originOut = originNode?.outputs?.[link.origin_slot];
                        if (originOut) {
                            inp.label = originOut.label || originOut.name || String(originOut.type || "*");
                        }
                    }
                    rebuildBundleMeta(self);
                    refreshAllUnbundles();
                    self.setDirtyCanvas?.(true, true);
                },
            },
            {
                content: "Disconnect",
                callback: () => {
                    self.disconnectInput(slotIdx);
                },
            },
        ];
    };
}

function patchUnbundle(nodeType) {
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        origCreated?.apply(this, arguments);
        hideWidget(this, SAVED_META_WIDGET);
        while ((this.outputs?.length || 0) > 0) {
            this.removeOutput(this.outputs.length - 1);
        }
        safeResize(this);
    };

    const origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slotIndex) {
        origConn?.apply(this, arguments);
        if (type !== LiteGraph.INPUT) return;
        if (slotIndex !== 0) return;
        applyUnbundleOutputs(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        origConfigure?.apply(this, arguments);
        hideWidget(this, SAVED_META_WIDGET);
        const self = this;
        requestAnimationFrame(() => {
            if (self.graph) applyUnbundleOutputs(self);
        });
    };

    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
        origMenu?.apply(this, arguments);
        const self = this;
        const bundles = findAllNodesByClass([NODE_BUNDLE]);
        options.unshift({
            content: "Refresh outputs from source",
            callback: () => applyUnbundleOutputs(self),
        });
        if (bundles.length > 0) {
            options.unshift({
                content: "Pick source Bundle",
                has_submenu: true,
                submenu: {
                    options: bundles.map(b => ({
                        content: `${b.title || "Bundle"} id:${b.id}`,
                        callback: () => {
                            const meta = readBundleMeta(b);
                            setWidgetValue(self, SAVED_META_WIDGET, JSON.stringify(meta));
                            applyUnbundleOutputs(self);
                        },
                    })),
                },
            });
        }
    };
}

function installSourceBundleCombo(node, initialValue) {
    const existing = node.widgets?.find(w => w.name === SOURCE_BUNDLE_WIDGET);
    const value = initialValue != null ? initialValue : (existing?.value || "");
    if (existing) {
        const idx = node.widgets.indexOf(existing);
        if (idx >= 0) node.widgets.splice(idx, 1);
    }
    const comboOptions = {};
    Object.defineProperty(comboOptions, "values", {
        get: () => {
            const names = collectBundleNames();
            return names.length > 0 ? names : [""];
        },
        enumerable: true,
        configurable: true,
    });
    const w = node.addWidget("combo", SOURCE_BUNDLE_WIDGET, value, () => {
        applyUnbundleByNameOutputs(node);
    }, comboOptions);
    w.serialize = true;
    return w;
}

function removeBundleInputSlot(node) {
    if (!node.inputs) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        if (node.inputs[i].name === "bundle") node.removeInput(i);
    }
}

function patchUnbundleByName(nodeType) {
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        origCreated?.apply(this, arguments);
        removeBundleInputSlot(this);
        hideWidget(this, SAVED_META_WIDGET);
        installSourceBundleCombo(this, "");
        while ((this.outputs?.length || 0) > 0) {
            this.removeOutput(this.outputs.length - 1);
        }
        safeResize(this);
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        origConfigure?.apply(this, arguments);
        const savedValue = getWidgetValue(this, SOURCE_BUNDLE_WIDGET) || "";
        removeBundleInputSlot(this);
        hideWidget(this, SAVED_META_WIDGET);
        installSourceBundleCombo(this, savedValue);
        const self = this;
        requestAnimationFrame(() => {
            if (self.graph) applyUnbundleByNameOutputs(self);
        });
    };

    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
        origMenu?.apply(this, arguments);
        const self = this;
        options.unshift({
            content: "Refresh outputs from source",
            callback: () => applyUnbundleByNameOutputs(self),
        });
    };
}
