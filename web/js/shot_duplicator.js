import { app } from "../../../scripts/app.js";
import { getShotGroups, duplicateShotGroup } from "./shot_utils.js";

const NODE_TYPE = "MickmumpitzShotDuplicator";
const NODE_TITLE = "Shot Duplicator";
const CATEGORY = "Mickmumpitz/Shot";

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

            _refreshGroups() {
                const names = getShotGroups().map((g) => g.title);
                this._sourceWidget.options.values = names;
                if (names.length > 0 && !names.includes(this._sourceWidget.value)) {
                    this._sourceWidget.value = names[0];
                }
            }

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

                const newGroup = duplicateShotGroup(sourceGroup);
                if (!newGroup) {
                    alert("Shot Duplicator: paste failed.");
                    return;
                }

                this._refreshGroups();
            }

            // Refresh groups on draw (keep dropdown current)
            onDrawForeground(ctx) {
                const names = getShotGroups().map((g) => g.title);
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
                const count = names.length;
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
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        if (node.inputs[i].name.startsWith("_dep_")) {
            node.removeInput(i);
        }
    }
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        node.setSize([Math.max(sz[0], node.size[0]), sz[1]]);
    });
}

