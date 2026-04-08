import { app } from "../../../scripts/app.js";

const NODE_TYPE = "MickmumpitzShotDuplicator";
const NODE_TITLE = "Shot Duplicator";
const CATEGORY = "Mickmumpitz/Shot";
const SHOT_PATTERN = /^SHOT\s+(\d+)$/i;

app.registerExtension({
    name: "Mickmumpitz.ShotDuplicator",

    registerCustomNodes() {
        class ShotDuplicatorNode extends LGraphNode {
            static title = NODE_TITLE;
            static type = NODE_TYPE;

            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.serialize_widgets = true;
                this.size = [260, 120];
                this.color = "#7155be";
                this.bgcolor = "#4a3580";

                this._sourceWidget = this.addWidget(
                    "combo",
                    "source_group",
                    "",
                    () => {},
                    { values: [] }
                );

                this.addWidget("button", "Add Shot", null, () => {
                    this._duplicateShot();
                });

                this._refreshGroups();
            }

            // ---------------------------------------------------------------
            // Refresh the dropdown with current SHOT groups
            // ---------------------------------------------------------------
            _refreshGroups() {
                const groups = this._getShotGroups();
                const names = groups.map((g) => g.title);
                this._sourceWidget.options.values = names;
                if (names.length > 0 && !names.includes(this._sourceWidget.value)) {
                    this._sourceWidget.value = names[0];
                }
            }

            _getShotGroups() {
                return (app.graph._groups || [])
                    .filter((g) => SHOT_PATTERN.test(g.title))
                    .sort((a, b) => {
                        const na = parseInt(a.title.match(SHOT_PATTERN)[1], 10);
                        const nb = parseInt(b.title.match(SHOT_PATTERN)[1], 10);
                        return na - nb;
                    });
            }

            _getNextShotNumber() {
                const groups = this._getShotGroups();
                if (groups.length === 0) return 1;
                const nums = groups.map((g) =>
                    parseInt(g.title.match(SHOT_PATTERN)[1], 10)
                );
                return Math.max(...nums) + 1;
            }

            // ---------------------------------------------------------------
            // Main duplication logic.
            //
            // Uses LiteGraph's native clipboard pair — `canvas._serializeItems`
            // + `canvas._deserializeItems` — which is the same path litegraph
            // itself uses for alt-drag cloning and copy-paste. It handles
            // everything we need automatically:
            //   - nodes, groups, reroutes, links
            //   - subgraphs: fresh UUIDs + createSubgraph + configure, giving
            //     each pasted SubgraphNode its own independent inner graph
            //
            // After the paste, we only do shot-specific post-processing:
            // rename the new group title, Set/Get channels, ShotVideoOutput
            // shot_number, and Save* filename_prefix on the top-level items.
            // ---------------------------------------------------------------
            _duplicateShot() {
                this._refreshGroups();

                const sourceName = this._sourceWidget.value;
                if (!sourceName) {
                    alert("No SHOT group selected.");
                    return;
                }

                const sourceGroup = (app.graph._groups || []).find(
                    (g) => g.title === sourceName
                );
                if (!sourceGroup) {
                    alert(`Group "${sourceName}" not found.`);
                    return;
                }

                if (
                    !app.canvas ||
                    typeof app.canvas._serializeItems !== "function" ||
                    typeof app.canvas._deserializeItems !== "function"
                ) {
                    alert(
                        "Shot Duplicator: canvas clipboard APIs are unavailable."
                    );
                    return;
                }

                const match = sourceName.match(SHOT_PATTERN);
                const sourceNum = parseInt(match[1], 10);
                const newNum = this._getNextShotNumber();
                const oldSuffix = String(sourceNum).padStart(2, "0");
                const newSuffix = String(newNum).padStart(2, "0");

                // Let litegraph compute which items are inside the group
                // (nodes via centre-point, sub-groups wholly contained,
                // reroutes by centre).
                sourceGroup.recomputeInsideNodes();
                const items = new Set([
                    sourceGroup,
                    ...(sourceGroup._children || []),
                ]);

                // Snapshot Set node channel values BEFORE paste, keyed by
                // source node id. During `_deserializeItems`, KJNodes'
                // `onConnectionsChange` fires validateName which auto-
                // appends `_0` to collide-detected channel names — making
                // the post-paste widget value unreliable. We'll use this
                // pre-paste snapshot to recompute the correct new value
                // after paste by matching on the preserved source id.
                //
                // GetNodes are intentionally NOT snapshotted or remapped:
                // they are consumer references (often to external / shared
                // Set nodes like SETTINGS outputs) and must keep their
                // original channel name so the user's inputs stay wired
                // to the same source across shots.
                const sourceChannelValues = new Map();
                for (const child of sourceGroup._children || []) {
                    if (
                        child?.type === "SetNode" &&
                        child.widgets?.[0] &&
                        typeof child.widgets[0].value === "string"
                    ) {
                        sourceChannelValues.set(
                            child.id,
                            child.widgets[0].value
                        );
                    }
                }

                const serialized = app.canvas._serializeItems(items);

                // Place the new group at the end of the SHOT chain — i.e.
                // to the right of whichever existing SHOT group has the
                // largest right edge. Avoids pasting on top of a sibling
                // shot when the selected source isn't the last one.
                const allShotGroups = this._getShotGroups();
                let anchorGroup = sourceGroup;
                let maxRight = -Infinity;
                for (const g of allShotGroups) {
                    const [x, , w] = g._bounding;
                    if (x + w > maxRight) {
                        maxRight = x + w;
                        anchorGroup = g;
                    }
                }
                const [, anchorY] = anchorGroup._bounding;
                const result = app.canvas._deserializeItems(serialized, {
                    position: [maxRight + 40, anchorY],
                });

                if (!result || !result.created || !result.created.length) {
                    alert("Shot Duplicator: paste failed.");
                    return;
                }

                // Set/Get channel names first — uses the pre-paste
                // snapshot, bypassing any validateName auto-suffixing.
                this._remapSetGetChannels(
                    result.nodes,
                    sourceChannelValues,
                    oldSuffix,
                    newSuffix
                );

                // Rename the new group title + shot-specific widget values
                this._applyShotRenaming(result.created, oldSuffix, newSuffix);

                this._refreshGroups();
                app.graph.change();
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            }

            // ---------------------------------------------------------------
            // Helper: replace numeric suffix in a string
            // ---------------------------------------------------------------
            _replaceSuffix(str, oldSuffix, newSuffix) {
                const escaped = oldSuffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                const re = new RegExp(`(\\b|_)${escaped}(\\b|$)`, "g");
                return str.replace(re, `$1${newSuffix}$2`);
            }

            // ---------------------------------------------------------------
            // Remap KJNodes Set/Get channel values on the pasted nodes.
            //
            // Background: during `_deserializeItems`, `node.configure(info)`
            // iterates the pasted inputs and fires `onConnectionsChange`.
            // KJNodes' SetNode handler calls `validateName(graph)` in that
            // path, which — once `widgets[0].value` has been restored from
            // `widgets_values` — detects the clash with the source shot's
            // Set node of the same channel and auto-appends `_0`. So by
            // the time we see the pasted node, its widget value may be
            // e.g. `OUT_03_0` instead of the expected `OUT_03`, breaking
            // a simple suffix replace.
            //
            // Fix: use the pre-paste snapshot (`sourceChannelValues`) as
            // the source of truth. The clipboard preserves the source id,
            // and `_deserializeItems` returns `result.nodes` as a
            // `Map<oldId, newNode>`, so we can match each pasted Set/Get
            // back to its original and recompute the new channel name
            // deterministically, then overwrite whatever KJNodes set.
            // ---------------------------------------------------------------
            _remapSetGetChannels(
                pastedNodesMap,
                sourceChannelValues,
                oldSuffix,
                newSuffix
            ) {
                if (!pastedNodesMap || !sourceChannelValues) return;

                for (const [oldId, oldValue] of sourceChannelValues) {
                    const node = pastedNodesMap.get(oldId);
                    if (!node || !node.widgets?.[0]) continue;
                    if (typeof oldValue !== "string" || !oldValue) continue;

                    // Prefer suffix replace (e.g. `OUT_03` → `OUT_04`);
                    // fall back to appending the new shot suffix when
                    // the original value has no matching shot number.
                    let newValue = this._replaceSuffix(
                        oldValue,
                        oldSuffix,
                        newSuffix
                    );
                    if (newValue === oldValue) {
                        newValue = `${oldValue}_${newSuffix}`;
                    }

                    node.widgets[0].value = newValue;

                    // Keep KJNodes' internal bookkeeping consistent so
                    // `findGetters` / `update` don't operate on stale
                    // `previousName` state.
                    if (node.type === "SetNode" && node.properties) {
                        node.properties.previousName = newValue;
                    }

                    node.title =
                        (node.type === "SetNode" ? "Set_" : "Get_") + newValue;

                    if (Array.isArray(node.widgets_values)) {
                        node.widgets_values = node.widgets.map((w) => w.value);
                    }
                }
            }

            // ---------------------------------------------------------------
            // Shot-specific post-paste renaming: titles, Set/Get channels,
            // ShotVideoOutput shot_number, Save* filename_prefix. Operates
            // on the freshly pasted top-level items returned by
            // _deserializeItems.
            // ---------------------------------------------------------------
            _applyShotRenaming(createdItems, oldSuffix, newSuffix) {
                const newNumber = parseInt(newSuffix, 10);

                for (const item of createdItems) {
                    // Groups: rename the title only
                    if (item instanceof LiteGraph.LGraphGroup) {
                        if (item.title) {
                            item.title = this._replaceSuffix(
                                item.title,
                                oldSuffix,
                                newSuffix
                            );
                        }
                        continue;
                    }

                    // Nodes: title + label + widgets
                    if (!item) continue;

                    const t = item.type || "";

                    // Set/Get channel values, titles and previousName are
                    // handled in _remapSetGetChannels (runs before this)
                    // so the pasted node's widget value is already final.
                    // Skip them here to avoid double-processing.
                    if (t === "SetNode" || t === "GetNode") continue;

                    if (item.title) {
                        item.title = this._replaceSuffix(
                            item.title,
                            oldSuffix,
                            newSuffix
                        );
                    }

                    if (typeof item.properties?.text === "string") {
                        item.properties.text = this._replaceSuffix(
                            item.properties.text,
                            oldSuffix,
                            newSuffix
                        );
                    }

                    if (!item.widgets) continue;

                    const cc = item.comfyClass || "";
                    const isShotOutput =
                        t === "ShotVideoOutput" || cc === "ShotVideoOutput";
                    const isSaveNode =
                        cc === "SaveVideo" ||
                        cc === "SaveImage" ||
                        t === "SaveVideo" ||
                        t === "SaveImage";

                    for (const w of item.widgets) {
                        if (isShotOutput && w.name === "shot_number") {
                            w.value = newNumber;
                        }
                        if (
                            isSaveNode &&
                            w.name === "filename_prefix" &&
                            typeof w.value === "string"
                        ) {
                            w.value = this._replaceSuffix(
                                w.value,
                                oldSuffix,
                                newSuffix
                            );
                        }
                    }

                    if (Array.isArray(item.widgets_values)) {
                        item.widgets_values = item.widgets.map((w) => w.value);
                    }
                }
            }

            // Refresh groups on draw (keep dropdown current)
            onDrawForeground(ctx) {
                const groups = this._getShotGroups();
                const names = groups.map((g) => g.title);
                const current = this._sourceWidget.options.values;
                if (
                    names.length !== current.length ||
                    names.some((n, i) => n !== current[i])
                ) {
                    this._sourceWidget.options.values = names;
                    if (names.length > 0 && !names.includes(this._sourceWidget.value)) {
                        this._sourceWidget.value = names[0];
                    }
                }

                // Draw shot count
                const count = groups.length;
                ctx.save();
                ctx.font = "11px Arial";
                ctx.fillStyle = "#aaa";
                ctx.textAlign = "right";
                ctx.fillText(
                    `${count} shot${count !== 1 ? "s" : ""}`,
                    this.size[0] - 10,
                    this.size[1] - 6
                );
                ctx.restore();
            }
        }

        LiteGraph.registerNodeType(NODE_TYPE, ShotDuplicatorNode);
        ShotDuplicatorNode.category = CATEGORY;
    },

    // -------------------------------------------------------------------
    // Hide _dep_N input slots on ShotAssembler nodes
    // -------------------------------------------------------------------
    nodeCreated(node) {
        if (node.comfyClass === "ShotAssembler") {
            hideDependencySlots(node);
        }
    },

    async loadedGraphNode(node) {
        if (node.comfyClass === "ShotAssembler") {
            hideDependencySlots(node);
        }
    },
});

/**
 * Remove all _dep_N input slots from the ShotAssembler node UI.
 * These slots exist only for the prompt handler to inject execution
 * dependencies — they should never be visible or manually wired.
 */
function hideDependencySlots(node) {
    if (!node.inputs) return;
    // Remove inputs whose name starts with _dep_
    // Iterate backwards to avoid index shifting
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        if (node.inputs[i].name.startsWith("_dep_")) {
            node.removeInput(i);
        }
    }
    // Compact the node size
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        node.setSize([Math.max(sz[0], node.size[0]), sz[1]]);
    });
}
