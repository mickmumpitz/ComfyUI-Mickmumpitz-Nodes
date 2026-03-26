import { app } from "../../../scripts/app.js";

const NODE_TYPE = "MickmumpitzLabel";
const NODE_TITLE = "Label (Mickmumpitz)";
const CATEGORY = "Mickmumpitz/Utils";

let labelClass = null;

app.registerExtension({
    name: "Mickmumpitz.Label",

    registerCustomNodes() {
        class MickmumpitzLabel extends LGraphNode {
            static title = NODE_TITLE;
            static type = NODE_TYPE;
            static title_mode = LiteGraph.NO_TITLE;
            static collapsable = false;

            // Property definitions for the properties panel
            static "@text" = { type: "string" };
            static "@fontSize" = { type: "number" };
            static "@fontFamily" = { type: "string" };
            static "@fontColor" = { type: "string" };
            static "@textAlign" = {
                type: "combo",
                values: ["left", "center", "right"],
            };
            static "@backgroundColor" = { type: "string" };
            static "@padding" = { type: "number" };
            static "@borderRadius" = { type: "number" };

            constructor(title) {
                super(title);
                this.isVirtualNode = true;
                this.resizable = false;
                this.serialize_widgets = true;

                this.properties["text"] = "Label";
                this.properties["fontSize"] = 24;
                this.properties["fontFamily"] = "Arial";
                this.properties["fontColor"] = "#ffffff";
                this.properties["textAlign"] = "left";
                this.properties["backgroundColor"] = "transparent";
                this.properties["padding"] = 8;
                this.properties["borderRadius"] = 0;

                this.color = "#fff0";
                this.bgcolor = "#fff0";
                this.size = [200, 40];
            }

            drawLabel(ctx) {
                this.flags = this.flags || {};
                this.flags.allow_interaction = !this.flags.pinned;
                ctx.save();

                this.color = "#fff0";
                this.bgcolor = "#fff0";

                const fontSize = Math.max(this.properties["fontSize"] || 24, 1);
                const fontFamily = this.properties["fontFamily"] || "Arial";
                const fontColor = this.properties["fontColor"] || "#ffffff";
                const backgroundColor = this.properties["backgroundColor"] || "";
                const padding = Number(this.properties["padding"]) || 0;
                const borderRadius = Number(this.properties["borderRadius"]) || 0;
                const textAlign = this.properties["textAlign"] || "left";

                ctx.font = `${fontSize}px ${fontFamily}`;

                const text = (this.properties["text"] ?? "Label")
                    .replace(/\\n/g, "\n")
                    .replace(/\n*$/, "");
                const lines = text.split("\n");

                const lineHeight = fontSize * 1.2;
                const maxWidth = Math.max(
                    ...lines.map((line) => ctx.measureText(line || " ").width),
                    20
                );

                this.size[0] = maxWidth + padding * 2;
                this.size[1] = lineHeight * lines.length + padding * 2;

                // Draw background
                if (backgroundColor && backgroundColor !== "transparent") {
                    ctx.beginPath();
                    if (ctx.roundRect) {
                        ctx.roundRect(0, 0, this.size[0], this.size[1], [
                            borderRadius,
                        ]);
                    } else {
                        ctx.rect(0, 0, this.size[0], this.size[1]);
                    }
                    ctx.fillStyle = backgroundColor;
                    ctx.fill();
                }

                // Text alignment
                let textX = padding;
                ctx.textAlign = "left";
                if (textAlign === "center") {
                    ctx.textAlign = "center";
                    textX = this.size[0] / 2;
                } else if (textAlign === "right") {
                    ctx.textAlign = "right";
                    textX = this.size[0] - padding;
                }

                // Draw text
                ctx.textBaseline = "top";
                ctx.fillStyle = fontColor;
                let y = padding;
                for (const line of lines) {
                    ctx.fillText(line || " ", textX, y);
                    y += lineHeight;
                }

                ctx.restore();
            }

            onDblClick(_event, _pos, _canvas) {
                LGraphCanvas.active_canvas?.showShowNodePanel?.(this);
            }

            inResizeCorner(_x, _y) {
                return false;
            }

            onShowCustomPanelInfo(panel) {
                panel
                    .querySelector('div.property[data-property="Mode"]')
                    ?.remove();
                panel
                    .querySelector('div.property[data-property="Color"]')
                    ?.remove();
            }
        }

        labelClass = MickmumpitzLabel;

        LiteGraph.registerNodeType(NODE_TYPE, MickmumpitzLabel);
        MickmumpitzLabel.category = CATEGORY;

        // Patch drawNode so the label renders with custom draw, not the standard node chrome
        const origDrawNode = LGraphCanvas.prototype.drawNode;
        LGraphCanvas.prototype.drawNode = function (node, ctx) {
            if (node.constructor === labelClass) {
                node.bgcolor = "transparent";
                node.color = "transparent";
                const result = origDrawNode.apply(this, arguments);
                node.drawLabel(ctx);
                return result;
            }
            return origDrawNode.apply(this, arguments);
        };
    },
});
