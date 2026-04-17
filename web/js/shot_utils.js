// Shared helpers for SHOT-group operations.
// Used by shot_duplicator.js and shot_prompt_importer.js.

import { app } from "../../../scripts/app.js";

export const SHOT_PATTERN = /^SHOT\s+(\d+)$/i;

export function getShotGroups() {
    return (app.graph._groups || [])
        .filter((g) => SHOT_PATTERN.test(g.title))
        .sort((a, b) => {
            const na = parseInt(a.title.match(SHOT_PATTERN)[1], 10);
            const nb = parseInt(b.title.match(SHOT_PATTERN)[1], 10);
            return na - nb;
        });
}

export function getNextShotNumber() {
    const groups = getShotGroups();
    if (groups.length === 0) return 1;
    const nums = groups.map((g) =>
        parseInt(g.title.match(SHOT_PATTERN)[1], 10)
    );
    return Math.max(...nums) + 1;
}

export function replaceSuffix(str, oldSuffix, newSuffix) {
    const escaped = oldSuffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(\\b|_)${escaped}(\\b|$)`, "g");
    return str.replace(re, `$1${newSuffix}$2`);
}

// Snapshot KJNodes Set channel values BEFORE paste, keyed by source node id.
// `validateName` on paste may auto-suffix to disambiguate, so we can't trust
// the post-paste widget value — we recompute from this snapshot.
function snapshotSourceChannels(sourceGroup) {
    const map = new Map();
    for (const child of sourceGroup._children || []) {
        if (
            child?.type === "SetNode" &&
            child.widgets?.[0] &&
            typeof child.widgets[0].value === "string"
        ) {
            map.set(child.id, child.widgets[0].value);
        }
    }
    return map;
}

function remapSetGetChannels(
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

        let newValue = replaceSuffix(oldValue, oldSuffix, newSuffix);
        if (newValue === oldValue) newValue = `${oldValue}_${newSuffix}`;

        node.widgets[0].value = newValue;
        if (node.type === "SetNode" && node.properties) {
            node.properties.previousName = newValue;
        }
        node.title = (node.type === "SetNode" ? "Set_" : "Get_") + newValue;
        if (Array.isArray(node.widgets_values)) {
            node.widgets_values = node.widgets.map((w) => w.value);
        }
    }
}

function applyShotRenaming(createdItems, oldSuffix, newSuffix) {
    const newNumber = parseInt(newSuffix, 10);
    for (const item of createdItems) {
        if (item instanceof LiteGraph.LGraphGroup) {
            if (item.title) {
                item.title = replaceSuffix(item.title, oldSuffix, newSuffix);
            }
            continue;
        }
        if (!item) continue;
        const t = item.type || "";
        // Set/Get handled in remapSetGetChannels (runs first)
        if (t === "SetNode" || t === "GetNode") continue;

        if (item.title) {
            item.title = replaceSuffix(item.title, oldSuffix, newSuffix);
        }
        if (typeof item.properties?.text === "string") {
            item.properties.text = replaceSuffix(
                item.properties.text,
                oldSuffix,
                newSuffix
            );
        }
        if (!item.widgets) continue;

        const cc = item.comfyClass || "";
        const isShotOutput = t === "ShotVideoOutput" || cc === "ShotVideoOutput";
        const isSaveNode =
            cc === "SaveVideo" ||
            cc === "SaveImage" ||
            t === "SaveVideo" ||
            t === "SaveImage";

        for (const w of item.widgets) {
            if (isShotOutput && w.name === "shot_number") w.value = newNumber;
            if (
                isSaveNode &&
                w.name === "filename_prefix" &&
                typeof w.value === "string"
            ) {
                w.value = replaceSuffix(w.value, oldSuffix, newSuffix);
            }
        }
        if (Array.isArray(item.widgets_values)) {
            item.widgets_values = item.widgets.map((w) => w.value);
        }
    }
}

/**
 * Duplicate a SHOT group using LiteGraph's clipboard APIs. Handles nested
 * nodes, reroutes, links, and subgraph nodes. After the paste, applies the
 * shot-specific renames (group title, Set/Get channels, ShotVideoOutput
 * shot_number, Save* filename_prefix).
 *
 * Returns the new LGraphGroup on success, or null on failure.
 */
export function duplicateShotGroup(sourceGroup) {
    if (!sourceGroup) return null;
    const match = sourceGroup.title && sourceGroup.title.match(SHOT_PATTERN);
    if (!match) return null;
    if (
        !app.canvas ||
        typeof app.canvas._serializeItems !== "function" ||
        typeof app.canvas._deserializeItems !== "function"
    ) {
        return null;
    }

    const sourceNum = parseInt(match[1], 10);
    const newNum = getNextShotNumber();
    const oldSuffix = String(sourceNum).padStart(2, "0");
    const newSuffix = String(newNum).padStart(2, "0");

    sourceGroup.recomputeInsideNodes();
    const items = new Set([sourceGroup, ...(sourceGroup._children || [])]);
    const sourceChannelValues = snapshotSourceChannels(sourceGroup);

    const serialized = app.canvas._serializeItems(items);

    // Place at right end of the SHOT chain so we don't overlap a sibling
    const allShotGroups = getShotGroups();
    let maxRight = -Infinity;
    let anchorGroup = sourceGroup;
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
    if (!result || !result.created || !result.created.length) return null;

    remapSetGetChannels(result.nodes, sourceChannelValues, oldSuffix, newSuffix);
    applyShotRenaming(result.created, oldSuffix, newSuffix);

    let newGroup = null;
    for (const item of result.created) {
        if (
            item instanceof LiteGraph.LGraphGroup &&
            SHOT_PATTERN.test(item.title || "")
        ) {
            newGroup = item;
            break;
        }
    }

    app.graph.change();
    app.canvas.setDirty(true, true);
    return newGroup;
}
