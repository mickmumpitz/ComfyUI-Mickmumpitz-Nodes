import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

console.log("[ShowImageTextPairs] extension loaded");

function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
    } else {
        fallbackCopy(text);
    }
}

function fallbackCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
        document.execCommand("copy");
    } catch (e) {
        /* ignore */
    }
    document.body.removeChild(ta);
}

function flashButton(btn, label) {
    const original = btn.textContent;
    btn.textContent = label;
    setTimeout(() => {
        btn.textContent = original;
    }, 900);
}

const BTN_CSS = `
    align-self: flex-start;
    cursor: pointer;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border-color, #444);
    background: var(--comfy-input-bg, #333);
    color: var(--input-text, #ddd);
`;

app.registerExtension({
    name: "comfy.ShowImageTextPairs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CCC_ShowImageTextPairs") {
            return;
        }
        console.log("[ShowImageTextPairs] registering node UI");

        function ensureWidget(node) {
            if (node._pairsList) return;

            const root = document.createElement("div");
            root.style.cssText =
                "display:flex;flex-direction:column;width:100%;height:100%;" +
                "box-sizing:border-box;font-family:sans-serif;font-size:12px;" +
                "color:var(--input-text,#ddd);";

            const header = document.createElement("div");
            header.style.cssText =
                "display:flex;align-items:center;justify-content:space-between;" +
                "gap:8px;padding:4px 2px;flex:0 0 auto;";

            const countLabel = document.createElement("span");
            countLabel.textContent = "No data — run the workflow.";
            countLabel.style.opacity = "0.7";

            const copyAllBtn = document.createElement("button");
            copyAllBtn.textContent = "Copy all";
            copyAllBtn.style.cssText = BTN_CSS;
            copyAllBtn.addEventListener("click", () => {
                const texts = (node._pairs || []).map((p) => p.caption);
                copyToClipboard(texts.join("\n\n"));
                flashButton(copyAllBtn, "Copied!");
            });

            header.appendChild(countLabel);
            header.appendChild(copyAllBtn);

            const list = document.createElement("div");
            list.style.cssText =
                "flex:1 1 auto;overflow-y:auto;display:flex;flex-direction:column;" +
                "gap:8px;padding:2px;min-height:60px;";

            root.appendChild(header);
            root.appendChild(list);

            node._pairsCountLabel = countLabel;
            node._pairsList = list;
            node._pairs = node._pairs || [];

            node.addDOMWidget("image_text_pairs", "div", root, { serialize: false });

            if (node.size[0] < 360) node.size[0] = 360;
            if (node.size[1] < 320) node.size[1] = 320;
        }

        function renderPairs(node) {
            ensureWidget(node);
            const list = node._pairsList;
            const countLabel = node._pairsCountLabel;
            const pairs = node._pairs || [];
            list.innerHTML = "";
            countLabel.textContent = pairs.length
                ? `${pairs.length} item${pairs.length === 1 ? "" : "s"}`
                : "No data — run the workflow.";

            pairs.forEach((pair, idx) => {
                const row = document.createElement("div");
                row.style.cssText =
                    "display:flex;gap:8px;align-items:flex-start;padding:6px;" +
                    "border:1px solid var(--border-color,#3a3a3a);border-radius:6px;" +
                    "background:var(--comfy-menu-bg,rgba(255,255,255,0.03));";

                const thumb = document.createElement("img");
                thumb.src = pair.url;
                thumb.style.cssText =
                    "width:120px;height:auto;max-height:160px;object-fit:contain;" +
                    "border-radius:4px;flex:0 0 auto;background:#00000033;";

                const right = document.createElement("div");
                right.style.cssText =
                    "display:flex;flex-direction:column;gap:4px;flex:1 1 auto;min-width:0;";

                const idxLabel = document.createElement("div");
                idxLabel.textContent = `#${idx + 1}`;
                idxLabel.style.cssText = "opacity:0.5;font-size:10px;";

                const textBox = document.createElement("div");
                textBox.textContent = pair.caption || "(no caption)";
                textBox.style.cssText =
                    "user-select:text;-webkit-user-select:text;cursor:text;" +
                    "white-space:pre-wrap;word-break:break-word;max-height:140px;" +
                    "overflow-y:auto;line-height:1.35;color:var(--input-text,#ddd);" +
                    "opacity:" + (pair.caption ? "1" : "0.5") + ";";

                const copyBtn = document.createElement("button");
                copyBtn.textContent = "Copy";
                copyBtn.style.cssText = BTN_CSS;
                copyBtn.addEventListener("click", () => {
                    copyToClipboard(pair.caption || "");
                    flashButton(copyBtn, "Copied!");
                });

                right.appendChild(idxLabel);
                right.appendChild(textBox);
                right.appendChild(copyBtn);

                if (pair.info) {
                    const infoBox = document.createElement("div");
                    infoBox.textContent = pair.info;
                    infoBox.style.cssText =
                        "user-select:text;-webkit-user-select:text;white-space:pre-wrap;" +
                        "font-family:monospace;font-size:11px;line-height:1.4;margin-top:4px;" +
                        "padding:4px 6px;border-radius:4px;border:1px dashed var(--border-color,#555);" +
                        "background:rgba(255,255,255,0.04);opacity:0.9;color:var(--input-text,#ddd);";
                    right.appendChild(infoBox);
                }

                row.appendChild(thumb);
                row.appendChild(right);
                list.appendChild(row);
            });
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            ensureWidget(this);
            renderPairs(this);
            return r;
        };

        // Repair stale/shifted widget values from workflows saved under an older
        // version of this node, where the widgets were ordered differently. A
        // shift can leave bucket_divisibility holding "" (or some other widget's
        // value), and ComfyUI core runs int(val) during prompt validation BEFORE
        // the node's Python runs — so int("") throws and the prompt is rejected
        // ("Failed to convert an input value to a INT value"). Clamp it back to a
        // valid integer on load so the bad value never reaches the backend.
        function repairBucketDivisibility(node) {
            const w = node.widgets?.find((w) => w.name === "bucket_divisibility");
            if (!w) return;
            const n = parseInt(w.value, 10);
            w.value = Number.isFinite(n) ? Math.min(256, Math.max(1, n)) : 64;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            repairBucketDivisibility(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            console.log("[ShowImageTextPairs] onExecuted", message);
            if (!message) return;

            const images = message.itp_images || message.images || [];
            const captions = message.captions || [];
            const boxinfo = message.boxinfo || [];

            this._pairs = images.map((img, i) => {
                const params = new URLSearchParams({
                    filename: img.filename,
                    subfolder: img.subfolder || "",
                    type: img.type || "temp",
                    rand: Math.random().toString(),
                });
                return {
                    url: api.apiURL(`/view?${params.toString()}`),
                    caption: captions[i] != null ? String(captions[i]) : "",
                    info: boxinfo[i] ? String(boxinfo[i]) : "",
                };
            });

            renderPairs(this);
        };
    },
});
