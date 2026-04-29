import { app } from "../../../scripts/app.js";

const NODE_TYPE = "Node Bypasser";
const NODE_TITLE = "Node Bypasser";
const CATEGORY = "Mickmumpitz/Utils";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

const ACTIVE_COLOR = "#a93226";
const ACTIVE_BG = "#3d1715";

app.registerExtension({
    name: "Mickmumpitz.NodeBypasser",

    registerCustomNodes() {
        class NodeBypasser extends LGraphNode {
            static title = NODE_TITLE;
            static type = NODE_TYPE;
            static category = CATEGORY;

            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.serialize_widgets = true;

                // node -> original mode, so toggling off restores instead of
                // unconditionally setting mode=0. In-memory only; rebuilt on load.
                this._originalModes = new Map();

                this._patternWidget = this.addWidget(
                    "text",
                    "pattern",
                    "",
                    (v) => {
                        this._applyBypass();
                    }
                );

                this._toggleWidget = this.addWidget(
                    "toggle",
                    this._toggleLabel(""),
                    false,
                    (v) => {
                        this._applyBypass();
                    }
                );

                this.setSize(this.computeSize());
            }

            _findMatchingNodes(pattern) {
                if (!pattern) return [];
                const needle = pattern.toLowerCase();
                const graph = this.graph || app.graph;
                if (!graph || !graph._nodes) return [];
                return graph._nodes.filter(
                    (n) =>
                        n !== this &&
                        typeof n.title === "string" &&
                        n.title.toLowerCase().includes(needle)
                );
            }

            _toggleLabel(pattern) {
                const count = this._findMatchingNodes(pattern).length;
                return `bypass · ${count} match${count === 1 ? "" : "es"}`;
            }

            _applyBypass() {
                const pattern = this._patternWidget?.value || "";
                const active = !!this._toggleWidget?.value;

                const targets = new Set(
                    active && pattern ? this._findMatchingNodes(pattern) : []
                );

                for (const n of targets) {
                    if (!this._originalModes.has(n)) {
                        this._originalModes.set(n, n.mode ?? MODE_ALWAYS);
                    }
                    n.mode = MODE_BYPASS;
                }

                for (const [n, orig] of [...this._originalModes]) {
                    if (!targets.has(n)) {
                        n.mode = orig;
                        this._originalModes.delete(n);
                    }
                }

                if (this._toggleWidget) {
                    this._toggleWidget.name = this._toggleLabel(pattern);
                }

                if (active && pattern) {
                    this.color = ACTIVE_COLOR;
                    this.bgcolor = ACTIVE_BG;
                } else {
                    this.color = null;
                    this.bgcolor = null;
                }

                app.graph?.setDirtyCanvas(true, true);
            }

            onAdded() {
                this._applyBypass();
            }

            onConfigure() {
                // Saved nodes that we bypassed last session come back at
                // mode=4. Reset matching nodes that look like "ours" before
                // re-applying so toggling off later restores cleanly.
                this._originalModes = new Map();
                setTimeout(() => {
                    const pattern = this._patternWidget?.value || "";
                    const active = !!this._toggleWidget?.value;
                    if (active && pattern) {
                        for (const n of this._findMatchingNodes(pattern)) {
                            if (n.mode === MODE_BYPASS) n.mode = MODE_ALWAYS;
                        }
                    }
                    this._applyBypass();
                }, 0);
            }

            onRemoved() {
                for (const [n, orig] of this._originalModes) {
                    n.mode = orig;
                }
                this._originalModes.clear();
                app.graph?.setDirtyCanvas(true, true);
            }
        }

        LiteGraph.registerNodeType(NODE_TYPE, NodeBypasser);
        NodeBypasser.category = CATEGORY;
    },
});
