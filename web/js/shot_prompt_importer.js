import { app } from "../../../scripts/app.js";
import { getShotGroups, duplicateShotGroup } from "./shot_utils.js";

const NODE_TYPE = "MickmumpitzShotPromptImporter";
const NODE_TITLE = "Shot Prompt Importer";
const CATEGORY = "Mickmumpitz/Shot";

// Subgraph-node titles to match inside each SHOT group, and which MD
// column their `prompt` widget is fed from.
const TITLE_TO_COLUMN = {
    "IMAGE GEN": "imagePrompt",
    "VIDEO GEN": "videoPrompt",
};

const ACCEPT_EXT_RE = /\.(md|markdown|txt)$/i;
const ACCEPT_ATTR = ".md,.markdown,.txt,text/markdown,text/plain";

// Layout
const PAD = 8;
const WIDGET_AREA_H = 30; // space reserved above the drop zone for the combo
const ZONE_H = 80;
const BUTTON_H = 26;
const STATUS_H = 18;

app.registerExtension({
    name: "Mickmumpitz.ShotPromptImporter",

    registerCustomNodes() {
        class ShotPromptImporterNode extends LGraphNode {
            static title = NODE_TITLE;
            static type = NODE_TYPE;

            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.serialize_widgets = true; // persist source_group choice
                this.size = [
                    340,
                    WIDGET_AREA_H + PAD + ZONE_H + PAD + BUTTON_H + PAD + STATUS_H,
                ];
                this.color = "#7155be";
                this.bgcolor = "#4a3580";

                this._sourceWidget = this.addWidget(
                    "combo",
                    "source_group",
                    "",
                    () => {},
                    { values: [] }
                );

                this._mdContent = "";
                this._fileName = "";
                this._status = "";
                this._dragHover = false;

                this._refreshGroups();
            }

            // --------------------------------------------------------------
            // Layout helpers — all custom UI lives below WIDGET_AREA_H
            // --------------------------------------------------------------
            _zoneRect() {
                return {
                    x: PAD,
                    y: WIDGET_AREA_H + PAD,
                    w: this.size[0] - PAD * 2,
                    h: ZONE_H,
                };
            }

            _buttonRect() {
                const z = this._zoneRect();
                return {
                    x: PAD,
                    y: z.y + z.h + PAD,
                    w: this.size[0] - PAD * 2,
                    h: BUTTON_H,
                };
            }

            _hit(rect, pos) {
                return (
                    pos[0] >= rect.x &&
                    pos[0] <= rect.x + rect.w &&
                    pos[1] >= rect.y &&
                    pos[1] <= rect.y + rect.h
                );
            }

            // --------------------------------------------------------------
            // Source group picker — keep values in sync with the graph
            // --------------------------------------------------------------
            _refreshGroups() {
                const names = getShotGroups().map((g) => g.title);
                this._sourceWidget.options.values = names;
                if (
                    names.length > 0 &&
                    !names.includes(this._sourceWidget.value)
                ) {
                    this._sourceWidget.value = names[0];
                }
            }

            _getSourceGroup() {
                const name = this._sourceWidget.value;
                if (!name) return null;
                return (
                    (app.graph._groups || []).find((g) => g.title === name) ||
                    null
                );
            }

            // --------------------------------------------------------------
            // File intake — triggered by drop, by file picker, or re-import
            // --------------------------------------------------------------
            async _loadFile(file) {
                if (!file) return;
                const name = file.name || "file";
                if (!ACCEPT_EXT_RE.test(name)) {
                    this._setStatus(`Unsupported file: ${name}`);
                    return;
                }
                try {
                    const text = await file.text();
                    this._mdContent = text;
                    this._fileName = name;
                    this._runImport();
                } catch (err) {
                    this._setStatus(`Read error: ${err.message || err}`);
                }
            }

            _openFilePicker() {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ACCEPT_ATTR;
                input.style.display = "none";
                input.addEventListener("change", async () => {
                    const file = input.files && input.files[0];
                    if (file) await this._loadFile(file);
                    input.remove();
                });
                document.body.appendChild(input);
                input.click();
            }

            // --------------------------------------------------------------
            // Drag and drop — LiteGraph dispatches these on the drop target
            // --------------------------------------------------------------
            onDragOver(e) {
                const items = e?.dataTransfer?.items;
                if (!items || items.length === 0) return false;
                for (const it of items) {
                    if (it.kind === "file") {
                        this._dragHover = true;
                        app.canvas?.setDirty(true, true);
                        return true;
                    }
                }
                return false;
            }

            onDragLeave() {
                if (this._dragHover) {
                    this._dragHover = false;
                    app.canvas?.setDirty(true, true);
                }
            }

            async onDragDrop(e) {
                this._dragHover = false;
                app.canvas?.setDirty(true, true);
                const file = e?.dataTransfer?.files?.[0];
                if (!file) return false;
                await this._loadFile(file);
                return true;
            }

            // Alternate single-file drop path used by some LiteGraph builds.
            async onDropFile(file) {
                await this._loadFile(file);
            }

            // --------------------------------------------------------------
            // Mouse — click the drop zone to browse, click button to re-run
            // --------------------------------------------------------------
            onMouseDown(e, pos) {
                if (this._hit(this._buttonRect(), pos)) {
                    if (!this._mdContent) {
                        this._setStatus("Drop or select a .md file first.");
                        return true;
                    }
                    this._runImport();
                    return true;
                }
                if (this._hit(this._zoneRect(), pos)) {
                    this._openFilePicker();
                    return true;
                }
                return false;
            }

            // --------------------------------------------------------------
            // Drawing
            // --------------------------------------------------------------
            onDrawForeground(ctx) {
                // Keep the picker list fresh without waiting for interaction
                const names = getShotGroups().map((g) => g.title);
                const current = this._sourceWidget.options.values;
                if (
                    names.length !== current.length ||
                    names.some((n, i) => n !== current[i])
                ) {
                    this._sourceWidget.options.values = names;
                    if (
                        names.length > 0 &&
                        !names.includes(this._sourceWidget.value)
                    ) {
                        this._sourceWidget.value = names[0];
                    }
                }

                const zone = this._zoneRect();
                const btn = this._buttonRect();

                // Drop zone — dashed border, hint text
                ctx.save();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = this._dragHover ? "#c5a8f0" : "#888";
                ctx.fillStyle = this._dragHover
                    ? "rgba(168,139,218,0.18)"
                    : "rgba(255,255,255,0.04)";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(zone.x, zone.y, zone.w, zone.h, 6);
                } else {
                    ctx.rect(zone.x, zone.y, zone.w, zone.h);
                }
                ctx.fill();
                ctx.stroke();
                ctx.restore();

                ctx.save();
                ctx.fillStyle = "#ddd";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                if (this._fileName) {
                    ctx.font = "bold 12px Arial";
                    ctx.fillText(
                        this._fileName,
                        zone.x + zone.w / 2,
                        zone.y + zone.h / 2 - 8
                    );
                    ctx.font = "11px Arial";
                    ctx.fillStyle = "#aaa";
                    ctx.fillText(
                        "drop another file or click to replace",
                        zone.x + zone.w / 2,
                        zone.y + zone.h / 2 + 10
                    );
                } else {
                    ctx.font = "12px Arial";
                    ctx.fillText(
                        "Drop .md file here",
                        zone.x + zone.w / 2,
                        zone.y + zone.h / 2 - 8
                    );
                    ctx.font = "11px Arial";
                    ctx.fillStyle = "#aaa";
                    ctx.fillText(
                        "or click to browse",
                        zone.x + zone.w / 2,
                        zone.y + zone.h / 2 + 10
                    );
                }
                ctx.restore();

                // Re-import button
                ctx.save();
                const hasContent = !!this._mdContent;
                ctx.fillStyle = hasContent ? "#5a4390" : "#3a2e5c";
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(btn.x, btn.y, btn.w, btn.h, 4);
                } else {
                    ctx.rect(btn.x, btn.y, btn.w, btn.h);
                }
                ctx.fill();
                ctx.fillStyle = hasContent ? "#ddd" : "#888";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("Re-import", btn.x + btn.w / 2, btn.y + btn.h / 2);
                ctx.restore();

                // Status line
                if (this._status) {
                    ctx.save();
                    ctx.font = "11px Arial";
                    ctx.fillStyle = "#cfcfcf";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "bottom";
                    ctx.beginPath();
                    ctx.rect(PAD, 0, this.size[0] - PAD * 2, this.size[1] - 2);
                    ctx.clip();
                    ctx.fillText(this._status, PAD, this.size[1] - 4);
                    ctx.restore();
                }
            }

            _setStatus(msg) {
                this._status = msg || "";
                app.canvas?.setDirty(true, true);
            }

            // --------------------------------------------------------------
            // Main import — parse cached MD and mutate the graph
            // --------------------------------------------------------------
            _runImport() {
                if (!this._mdContent) {
                    this._setStatus("Drop a .md file first.");
                    return;
                }

                const shots = parseShotTable(this._mdContent);
                if (shots.length === 0) {
                    this._setStatus("No shot rows found in MD table.");
                    return;
                }

                const sourceGroup = this._getSourceGroup();
                if (!sourceGroup) {
                    this._setStatus(
                        "Pick a source group to duplicate from."
                    );
                    return;
                }

                let groups = getShotGroups();
                // Always duplicate from the user-selected template — this
                // guarantees every new shot is a full copy of a known-good
                // group, not a copy of a possibly-empty group we just made.
                while (groups.length < shots.length) {
                    const newGroup = duplicateShotGroup(sourceGroup);
                    if (!newGroup) {
                        this._setStatus("Duplicate failed mid-import.");
                        return;
                    }
                    groups = getShotGroups();
                }

                // Active shots — un-mute and write prompts.
                let missingTargets = 0;
                for (let i = 0; i < shots.length; i++) {
                    setGroupMuted(groups[i], false);
                    missingTargets += applyPromptsToGroup(groups[i], shots[i]);
                }

                // Extra groups beyond the MD — mute.
                for (let i = shots.length; i < groups.length; i++) {
                    setGroupMuted(groups[i], true);
                }

                app.graph.change();
                app.canvas?.setDirty(true, true);

                const extra = Math.max(0, groups.length - shots.length);
                const parts = [
                    `Imported ${shots.length} shot${
                        shots.length !== 1 ? "s" : ""
                    }`,
                ];
                if (extra > 0) parts.push(`muted ${extra} extra`);
                if (missingTargets > 0) {
                    parts.push(`${missingTargets} target(s) missing`);
                }
                this._setStatus(parts.join(", "));
            }
        }

        LiteGraph.registerNodeType(NODE_TYPE, ShotPromptImporterNode);
        ShotPromptImporterNode.category = CATEGORY;
    },
});

// ---------------------------------------------------------------------------
// Markdown table parsing
// ---------------------------------------------------------------------------

function parseShotTable(md) {
    const rows = [];
    for (const line of md.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("|")) continue;

        let cells = trimmed.split("|").map((s) => s.trim());
        if (cells.length && cells[0] === "") cells.shift();
        if (cells.length && cells[cells.length - 1] === "") cells.pop();
        if (cells.length < 3) continue;
        if (cells.every((c) => /^:?-+:?$/.test(c))) continue;
        if (!/^\d+$/.test(cells[0])) continue;

        rows.push({
            shotNumber: parseInt(cells[0], 10),
            imagePrompt: cells[1],
            videoPrompt: cells[2],
        });
    }
    rows.sort((a, b) => a.shotNumber - b.shotNumber);
    return rows;
}

// ---------------------------------------------------------------------------
// Graph mutation
// ---------------------------------------------------------------------------

function findPromptWidget(node) {
    if (!node || !node.widgets) return null;
    const byName = (pred) =>
        node.widgets.find(
            (w) => typeof w.name === "string" && pred(w.name.toLowerCase())
        );
    return (
        byName((n) => n === "prompt") ||
        byName((n) => n.includes("prompt")) ||
        byName((n) => n === "text" || n === "value" || n === "string") ||
        node.widgets.find((w) => typeof w.value === "string") ||
        null
    );
}

function writeWidget(node, widget, value) {
    widget.value = value;
    if (Array.isArray(node.widgets_values)) {
        node.widgets_values = node.widgets.map((ww) => ww.value);
    }
    if (typeof widget.callback === "function") {
        try {
            widget.callback(value, app.canvas, node);
        } catch (_) {
            // widget callbacks sometimes expect a specific event
            // shape — the value assignment above is authoritative.
        }
    }
}

/**
 * Write `value` to a node's prompt field. Tries widgets first, then falls
 * back to following a named input link ("prompt", "positive", "text") into
 * the upstream node and writing that node's widget instead. This covers
 * subgraph nodes that expose the prompt as an input slot rather than a
 * promoted widget.
 */
function writePromptToNode(node, value) {
    const w = findPromptWidget(node);
    if (w) {
        writeWidget(node, w, value);
        return true;
    }

    const inputs = node.inputs || [];
    const candidateNames = ["prompt", "positive", "text", "string"];
    const inputIdx = inputs.findIndex((i) => {
        const n = typeof i.name === "string" ? i.name.toLowerCase() : "";
        return candidateNames.includes(n) || n.includes("prompt");
    });
    if (inputIdx < 0) return false;

    const link = inputs[inputIdx].link;
    if (link == null) return false;
    const linkInfo = app.graph.links?.[link];
    if (!linkInfo) return false;
    const source = app.graph.getNodeById(linkInfo.origin_id);
    if (!source) return false;

    const sw = findPromptWidget(source);
    if (!sw) return false;
    writeWidget(source, sw, value);
    return true;
}

function applyPromptsToGroup(group, shot) {
    group.recomputeInsideNodes();
    const wanted = new Set(Object.keys(TITLE_TO_COLUMN));
    for (const child of group._children || []) {
        if (!child || typeof child.title !== "string") continue;
        const key = child.title.trim().toUpperCase();
        const column = TITLE_TO_COLUMN[key];
        if (!column) continue;
        const value = shot[column];
        if (typeof value !== "string") continue;

        if (writePromptToNode(child, value)) {
            wanted.delete(key);
        } else {
            const names = (child.widgets || []).map((w) => w.name).join(", ");
            console.warn(
                `[ShotPromptImporter] ${group.title}: '${child.title}' has no writable prompt target (widgets: [${names}])`
            );
        }
    }
    return wanted.size;
}

function setGroupMuted(group, muted) {
    group.recomputeInsideNodes();
    const mode = muted ? LiteGraph.NEVER : LiteGraph.ALWAYS;
    for (const child of group._children || []) {
        if (child && typeof child.mode !== "undefined") {
            child.mode = mode;
        }
    }
}
