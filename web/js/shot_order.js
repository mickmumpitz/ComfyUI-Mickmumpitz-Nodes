import { app } from "../../../scripts/app.js";

const NODE_TYPE = "MickmumpitzShotOrder";
const NODE_TITLE = "Shot Order";
const CATEGORY = "Mickmumpitz/Shot";
const SHOT_PATTERN = /^SHOT\s+(\d+)$/i;

const ROW_HEIGHT = 28;
const HANDLE_WIDTH = 24;
const BUTTON_HEIGHT = 30;
const PADDING = 8;
const NODE_WIDTH = 240;
const GAP = 40; // matches Shot Duplicator spacing

app.registerExtension({
    name: "Mickmumpitz.ShotOrder",

    registerCustomNodes() {
        class ShotOrderNode extends LGraphNode {
            static title = NODE_TITLE;
            static type = NODE_TYPE;

            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.serialize_widgets = false;
                this.size = [NODE_WIDTH, 120];
                this.color = "#7155be";
                this.bgcolor = "#4a3580";

                this._shotOrder = []; // { title, groupRef }
                this._pendingDrag = null; // { rowIndex, startY } — set on mousedown
                this._dragState = null; // { dragIndex, currentY } — set once mouse moves
                this._lastGroupKey = ""; // for change detection
            }

            // ---------------------------------------------------------------
            // Discover SHOT groups sorted by shot number
            // ---------------------------------------------------------------
            _getShotGroups() {
                return (app.graph._groups || [])
                    .filter((g) => SHOT_PATTERN.test(g.title))
                    .sort((a, b) => {
                        const na = parseInt(a.title.match(SHOT_PATTERN)[1], 10);
                        const nb = parseInt(b.title.match(SHOT_PATTERN)[1], 10);
                        return na - nb;
                    });
            }

            // ---------------------------------------------------------------
            // Sync _shotOrder with graph — preserve user ordering, add/remove
            // ---------------------------------------------------------------
            _syncWithGraph() {
                const currentGroups = this._getShotGroups();
                const key = currentGroups.map((g) => g.title).join("|");
                if (key === this._lastGroupKey && this._shotOrder.length > 0) {
                    return; // no change
                }

                const currentByTitle = new Map();
                for (const g of currentGroups) {
                    currentByTitle.set(g.title, g);
                }

                // Remove entries for deleted groups, update refs
                const kept = [];
                for (const entry of this._shotOrder) {
                    const ref = currentByTitle.get(entry.title);
                    if (ref) {
                        entry.groupRef = ref;
                        kept.push(entry);
                        currentByTitle.delete(entry.title);
                    }
                }

                // Append new groups sorted by x position (spatial order)
                const newEntries = [...currentByTitle.entries()].map(
                    ([title, ref]) => ({ title, groupRef: ref })
                );
                newEntries.sort(
                    (a, b) =>
                        a.groupRef._bounding[0] - b.groupRef._bounding[0]
                );
                kept.push(...newEntries);

                this._shotOrder = kept;
                this._lastGroupKey = key;

                // Cancel drag if the dragged entry was removed
                if (
                    this._dragState &&
                    this._dragState.dragIndex >= this._shotOrder.length
                ) {
                    this._dragState = null;
                    this._pendingDrag = null;
                }
            }

            // ---------------------------------------------------------------
            // Layout helpers
            // ---------------------------------------------------------------
            _listStartY() {
                return PADDING;
            }

            _listEndY() {
                return PADDING + this._shotOrder.length * ROW_HEIGHT;
            }

            _buttonY() {
                return this._listEndY() + PADDING;
            }

            _computeHeight() {
                const listH = this._shotOrder.length * ROW_HEIGHT;
                return (
                    PADDING + listH + PADDING + BUTTON_HEIGHT + PADDING + 16
                );
            }

            // ---------------------------------------------------------------
            // Determine drop index from a node-local Y coordinate
            // ---------------------------------------------------------------
            _getDropIndex(localY) {
                const relY = localY - this._listStartY();
                let idx = Math.round(relY / ROW_HEIGHT);
                // Range [0, length]: 0 = before first, length = after last
                return Math.max(
                    0,
                    Math.min(this._shotOrder.length, idx)
                );
            }

            // ---------------------------------------------------------------
            // Drawing
            // ---------------------------------------------------------------
            onDrawForeground(ctx) {
                this._syncWithGraph();

                // Dynamic size
                this.size[0] = NODE_WIDTH;
                this.size[1] = Math.max(this._computeHeight(), 80);

                const count = this._shotOrder.length;
                const dragIdx =
                    this._dragState !== null
                        ? this._dragState.dragIndex
                        : -1;
                const dropIdx =
                    this._dragState !== null
                        ? this._getDropIndex(this._dragState.currentY)
                        : -1;

                ctx.save();

                // Draw rows
                for (let i = 0; i < count; i++) {
                    const entry = this._shotOrder[i];
                    const rowY = this._listStartY() + i * ROW_HEIGHT;

                    // Dimmed if this row is being dragged
                    const isDragged = i === dragIdx;
                    ctx.globalAlpha = isDragged ? 0.3 : 1.0;

                    // Row background
                    ctx.fillStyle =
                        i % 2 === 0
                            ? "rgba(255,255,255,0.04)"
                            : "rgba(0,0,0,0.06)";
                    ctx.fillRect(0, rowY, this.size[0], ROW_HEIGHT);

                    // Drag handle (hamburger icon)
                    ctx.fillStyle = "#888";
                    const hx = PADDING + HANDLE_WIDTH / 2;
                    const hy = rowY + ROW_HEIGHT / 2;
                    for (let l = -1; l <= 1; l++) {
                        ctx.fillRect(hx - 5, hy + l * 4 - 1, 10, 2);
                    }

                    // Shot label
                    ctx.fillStyle = "#ddd";
                    ctx.font = "13px Arial";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText(
                        entry.title,
                        HANDLE_WIDTH + PADDING + 4,
                        rowY + ROW_HEIGHT / 2
                    );

                    // Position number on the right
                    ctx.fillStyle = "#999";
                    ctx.font = "11px Arial";
                    ctx.textAlign = "right";
                    ctx.fillText(
                        `#${i + 1}`,
                        this.size[0] - PADDING,
                        rowY + ROW_HEIGHT / 2
                    );
                }

                ctx.globalAlpha = 1.0;

                // Drag feedback: insertion line + floating row
                if (this._dragState !== null && dragIdx >= 0) {
                    // Insertion line (skip if drop would be a no-op)
                    const isNoOp =
                        dropIdx === dragIdx || dropIdx === dragIdx + 1;
                    if (!isNoOp) {
                        const lineY =
                            this._listStartY() + dropIdx * ROW_HEIGHT;
                        ctx.fillStyle = "#a88bda";
                        ctx.fillRect(
                            PADDING,
                            lineY - 1,
                            this.size[0] - PADDING * 2,
                            2
                        );
                    }

                    // Floating row
                    const floatY = this._dragState.currentY - ROW_HEIGHT / 2;
                    ctx.fillStyle = "rgba(90, 67, 144, 0.85)";
                    ctx.fillRect(2, floatY, this.size[0] - 4, ROW_HEIGHT);

                    ctx.fillStyle = "#fff";
                    ctx.font = "bold 13px Arial";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText(
                        this._shotOrder[dragIdx].title,
                        HANDLE_WIDTH + PADDING + 4,
                        floatY + ROW_HEIGHT / 2
                    );
                }

                // Apply Order button
                const by = this._buttonY();
                const bx = PADDING;
                const bw = this.size[0] - PADDING * 2;
                ctx.fillStyle = "#5a4390";
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(bx, by, bw, BUTTON_HEIGHT, 4);
                } else {
                    ctx.rect(bx, by, bw, BUTTON_HEIGHT);
                }
                ctx.fill();

                ctx.fillStyle = "#ddd";
                ctx.font = "bold 13px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    "Apply Order",
                    this.size[0] / 2,
                    by + BUTTON_HEIGHT / 2
                );

                // Status text
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

            // ---------------------------------------------------------------
            // Mouse handling — two-phase drag:
            //   1. onMouseDown sets _pendingDrag (invisible)
            //   2. onMouseMove promotes to _dragState once mouse moves
            // A click-without-move never enters the visual drag state,
            // so it can never get stuck.
            // ---------------------------------------------------------------
            onMouseDown(e, pos, graphCanvas) {
                // Clean up any prior state
                this._pendingDrag = null;
                this._dragState = null;

                // pos is already in content coordinates (below title bar)
                const contentY = pos[1];

                // Check Apply button
                const by = this._buttonY();
                if (
                    contentY >= by &&
                    contentY <= by + BUTTON_HEIGHT &&
                    pos[0] >= PADDING &&
                    pos[0] <= this.size[0] - PADDING
                ) {
                    this._applyOrder();
                    return true;
                }

                // Check list rows — start a pending drag (not visible yet)
                const listStart = this._listStartY();
                const listEnd = this._listEndY();

                if (contentY >= listStart && contentY < listEnd) {
                    const rowIdx = Math.floor(
                        (contentY - listStart) / ROW_HEIGHT
                    );
                    if (
                        rowIdx >= 0 &&
                        rowIdx < this._shotOrder.length
                    ) {
                        this._pendingDrag = {
                            rowIndex: rowIdx,
                            startY: contentY,
                        };
                        return true; // capture input
                    }
                }

                return false;
            }

            onMouseMove(e, pos, graphCanvas) {
                // If the mouse button was released without onMouseUp
                // firing (LiteGraph quirk), clear stale drag state
                if (e.buttons === 0) {
                    if (this._pendingDrag || this._dragState) {
                        this._pendingDrag = null;
                        this._dragState = null;
                        app.canvas.setDirty(true, true);
                    }
                    return false;
                }

                const contentY = pos[1];

                // Promote pending → active once mouse moves enough
                if (this._pendingDrag && !this._dragState) {
                    const dy = Math.abs(
                        contentY - this._pendingDrag.startY
                    );
                    if (dy > 3) {
                        this._dragState = {
                            dragIndex: this._pendingDrag.rowIndex,
                            currentY: contentY,
                        };
                        this._pendingDrag = null;
                    }
                    app.canvas.setDirty(true, true);
                    return true;
                }

                if (!this._dragState) return false;
                this._dragState.currentY = contentY;
                app.canvas.setDirty(true, true);
                return true;
            }

            onMouseUp(e, pos, graphCanvas) {
                // Click without move — just clean up, nothing to do
                if (this._pendingDrag) {
                    this._pendingDrag = null;
                    return true;
                }

                if (!this._dragState) return false;

                const contentY = pos[1];
                const dropIdx = this._getDropIndex(contentY);
                const dragIdx = this._dragState.dragIndex;

                // dropIdx is an insertion point (0..length).
                // No-op when inserting right before or after
                // the dragged item's original position.
                if (
                    dropIdx !== dragIdx &&
                    dropIdx !== dragIdx + 1
                ) {
                    const [removed] = this._shotOrder.splice(
                        dragIdx,
                        1
                    );
                    const insertAt =
                        dropIdx > dragIdx
                            ? dropIdx - 1
                            : dropIdx;
                    this._shotOrder.splice(insertAt, 0, removed);
                }

                this._dragState = null;
                app.canvas.setDirty(true, true);
                return true;
            }

            // ---------------------------------------------------------------
            // Apply the current order: move groups + renumber shots
            // ---------------------------------------------------------------
            _applyOrder() {
                if (this._shotOrder.length < 2) return;

                // Phase 1: Snapshot all positions and children BEFORE moving
                const snapshots = this._shotOrder.map((entry) => {
                    const g = entry.groupRef;
                    g.recomputeInsideNodes();
                    return {
                        group: g,
                        origX: g._bounding[0],
                        origY: g._bounding[1],
                        width: g._bounding[2],
                        height: g._bounding[3],
                        children: [...(g._children || [])],
                    };
                });

                // Phase 2: Compute target positions
                // Sort by original X to find the positional slots
                const byX = [...snapshots].sort(
                    (a, b) => a.origX - b.origX
                );
                const startX = byX[0].origX;
                const baselineY = byX[0].origY;

                const targets = [];
                let currentX = startX;
                for (const snap of snapshots) {
                    targets.push({ x: currentX, y: baselineY });
                    currentX += snap.width + GAP;
                }

                // Phase 3: Move atomically
                for (let i = 0; i < snapshots.length; i++) {
                    const snap = snapshots[i];
                    const tgt = targets[i];
                    const dx = tgt.x - snap.origX;
                    const dy = tgt.y - snap.origY;

                    if (dx === 0 && dy === 0) continue;

                    snap.group._bounding[0] = tgt.x;
                    snap.group._bounding[1] = tgt.y;

                    for (const child of snap.children) {
                        if (child.pos) {
                            child.pos[0] += dx;
                            child.pos[1] += dy;
                        } else if (child._bounding) {
                            // sub-groups
                            child._bounding[0] += dx;
                            child._bounding[1] += dy;
                        }
                    }
                }

                // Recompute after all moves
                for (const snap of snapshots) {
                    snap.group.recomputeInsideNodes();
                }

                // Phase 4: Renumber ShotVideoOutput nodes
                for (let i = 0; i < snapshots.length; i++) {
                    const shotNumber = i + 1;
                    for (const child of snapshots[i].children) {
                        const cc = child.comfyClass || child.type || "";
                        if (cc === "ShotVideoOutput") {
                            const w = child.widgets?.find(
                                (w) => w.name === "shot_number"
                            );
                            if (w) {
                                w.value = shotNumber;
                            }
                            if (Array.isArray(child.widgets_values)) {
                                child.widgets_values = child.widgets.map(
                                    (w) => w.value
                                );
                            }
                        }
                    }
                }

                // Phase 5: Finalize
                app.graph.change();
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }

                // Force re-sync so the list picks up new positions
                this._lastGroupKey = "";
            }

            // ---------------------------------------------------------------
            // Cleanup on removal
            // ---------------------------------------------------------------
            onRemoved() {
                this._pendingDrag = null;
                this._dragState = null;
            }
        }

        LiteGraph.registerNodeType(NODE_TYPE, ShotOrderNode);
        ShotOrderNode.category = CATEGORY;
    },
});
