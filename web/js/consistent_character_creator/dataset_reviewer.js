import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const EXT_NAME = "comfy.ConsistentCharacterCreator.DatasetReviewer";
const NODE_CLASSES = ["CCC_DatasetReviewer"];

/* ── palette ──────────────────────────────────────────────────────────── */
const BOX_COLORS = [
  "#FF3B30", "#34C759", "#0A84FF", "#FF9F0A",
  "#BF5AF2", "#FFD60A", "#32D74B", "#FF6961",
];

function boxColor(i) { return BOX_COLORS[i % BOX_COLORS.length]; }

/* ── tiny helpers ─────────────────────────────────────────────────────── */
function el(tag, css, attrs) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (attrs) Object.assign(e, attrs);
  return e;
}
function btn(label, css, onClick) {
  const b = el("button",
    `cursor:pointer;padding:3px 10px;border-radius:4px;font-size:11px;` +
    `border:1px solid var(--border-color,#555);` +
    `background:var(--comfy-input-bg,#2a2a2a);color:var(--input-text,#ddd);` +
    (css || "")
  );
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

/* ═══════════════════════════════════════════════════════════════════════
   HIDDEN INPUT SYNC
   Keeps the auto-generated "edited_data_json" widget (rendered by ComfyUI
   above our custom DOM widget because it's a plain STRING input) hidden
   from the user, and always filled with the current state of the bottom
   textarea + bbox edits. This is the payload Python reads on execution.
══════════════════════════════════════════════════════════════════════════ */
function hideRawJsonWidget(node) {
  if (node._dsr_hiddenWidget) return;
  const w = node.widgets?.find(w => w.name === "edited_data_json");
  if (!w) return;

  // Collapse it visually so it takes no space in the node body
  w.computeSize = () => [0, -4];
  if (w.inputEl) {
    w.inputEl.style.display = "none";
  }
  node._dsr_hiddenWidget = w;
}

function syncHiddenWidget(node) {
  const dsr = node._dsr;
  const hiddenWidget = node._dsr_hiddenWidget;
  if (!hiddenWidget || !dsr) return;

  const payload = (dsr.items || []).map((it, i) => ({
    index: i,
    caption: it.edited_caption,
    boxes: (it.edited_boxes || []).map(b => ({
      element_index: b.element_index,
      bbox: relBoxToOriginal(b),
    })),
  }));

  hiddenWidget.value = JSON.stringify(payload);
}

/* Converts a box's 0-1 relative coords into yxyx_normalized (0-1000 scale).
   Output format is FIXED regardless of the node's bbox_format dropdown,
   since downstream tools (e.g. AI Toolkit) require yxyx_normalized. */
function relBoxToOriginal(box) {
  const rx1 = Math.min(box.rx1, box.rx2);
  const rx2 = Math.max(box.rx1, box.rx2);
  const ry1 = Math.min(box.ry1, box.ry2);
  const ry2 = Math.max(box.ry1, box.ry2);

  return [
    Math.round(ry1 * 1000),
    Math.round(rx1 * 1000),
    Math.round(ry2 * 1000),
    Math.round(rx2 * 1000),
  ];
}

/* ═══════════════════════════════════════════════════════════════════════
   MAIN EXTENSION
══════════════════════════════════════════════════════════════════════════ */
app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.includes(nodeData.name)) return;

    /* ── init state on node creation ─────────────────────────────────── */
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origCreated?.apply(this, arguments);
      this._dsr = {
        items: [],        // [{url, caption, boxes, edited_caption, edited_boxes}]
        index: 0,
        bboxFormat: "yxyx_normalized",
        dragging: null,   // {boxIdx, handle, startX, startY, origRx1,ry1,rx2,ry2, imgW, imgH}
      };
      buildWidget(this);
      hideRawJsonWidget(this);
    };

    /* ── restore state after graph load ─────────────────────────────── */
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      origConfigure?.apply(this, arguments);
      if (!this._dsr) {
        this._dsr = { items: [], index: 0, bboxFormat: "yxyx_normalized", dragging: null };
      }
      buildWidget(this);
      hideRawJsonWidget(this);
    };

    /* ── receive data from Python ────────────────────────────────────── */
    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      if (!message) return;

      const images       = message.dsr_images     || [];
      const captions     = message.captions        || [];
      const boxesPerImg  = message.boxes_per_image || [];
      const fmt          = (message.bbox_format    || ["yxyx_normalized"])[0];
      const dsr          = this._dsr;

      dsr.bboxFormat = fmt;

      // Merge: preserve any edits the user already made for this run
      const prevItems = dsr.items || [];

      dsr.items = images.map((img, i) => {
        const params = new URLSearchParams({
          filename: img.filename,
          subfolder: img.subfolder || "",
          type: img.type || "temp",
          _r: Math.random(),
        });
        const url            = api.apiURL(`/view?${params}`);
        const serverCaption  = captions[i] != null ? String(captions[i]) : "";
        const serverBoxes    = boxesPerImg[i] || [];

        // Keep prior edits if they exist for the same slot
        const prev = prevItems[i];
        return {
          url,
          caption:         serverCaption,
          boxes:           serverBoxes,
          edited_caption:  prev?.edited_caption ?? serverCaption,
          edited_boxes:    prev?.edited_boxes   ?? serverBoxes.map(b => ({ ...b })),
        };
      });

      // Clamp index
      if (dsr.index >= dsr.items.length) dsr.index = Math.max(0, dsr.items.length - 1);

      buildWidget(this);
      hideRawJsonWidget(this);
      renderSlide(this);
      syncHiddenWidget(this);
    };
  },
});

/* ═══════════════════════════════════════════════════════════════════════
   BUILD WIDGET (called once per node)
══════════════════════════════════════════════════════════════════════════ */
function buildWidget(node) {
  if (node._dsr_built) return;
  node._dsr_built = true;

  /* ── root ──────────────────────────────────────────────────────────── */
  const root = el("div",
    "display:flex;flex-direction:column;width:100%;height:100%;" +
    "box-sizing:border-box;font-family:sans-serif;font-size:12px;" +
    "color:var(--input-text,#ddd);gap:6px;padding:4px;"
  );

  /* ── top bar: counter + nav ────────────────────────────────────────── */
  const topBar = el("div",
    "display:flex;align-items:center;justify-content:space-between;" +
    "flex:0 0 auto;gap:6px;"
  );
  const counter = el("span", "opacity:.6;font-size:11px;");
  const navLeft  = btn("◀", "padding:2px 8px;", () => navigate(node, -1));
  const navRight = btn("▶", "padding:2px 8px;", () => navigate(node,  1));
  const navRow   = el("div", "display:flex;gap:4px;align-items:center;");
  navRow.appendChild(navLeft);
  navRow.appendChild(counter);
  navRow.appendChild(navRight);
  topBar.appendChild(navRow);
  const unloadBtn = btn("Unload all", "background:#5a1a1a;border-color:#a33;", () => unloadAll(node));
  topBar.appendChild(unloadBtn);
  const copyAllBtn = btn("Copy all captions", "", () => {
    const texts = (node._dsr.items || []).map(it => it.edited_caption);
    navigator.clipboard?.writeText(texts.join("\n\n")).catch(() => {});
  });
  topBar.appendChild(copyAllBtn);
  root.appendChild(topBar);

  /* ── image + canvas overlay container ─────────────────────────────── */
  const imgWrap = el("div",
    "position:relative;flex:0 0 auto;width:100%;background:#11111166;" +
    "border-radius:4px;overflow:hidden;border:1px solid var(--border-color,#444);"
  );
  const imgEl = el("img", "display:block;width:100%;height:auto;max-height:400px;object-fit:contain;");
  imgEl.alt = "dataset image";
  const canvas = el("canvas",
    "position:absolute;top:0;left:0;width:100%;height:100%;cursor:default;"
  );
  imgWrap.appendChild(imgEl);
  imgWrap.appendChild(canvas);
  root.appendChild(imgWrap);

  /* keep canvas pixels in sync with rendered size */
  const ro = new ResizeObserver(() => {
    canvas.width  = imgWrap.clientWidth;
    canvas.height = imgWrap.clientHeight;
    drawBoxes(node);
  });
  ro.observe(imgWrap);

  /* ── bbox info bar ─────────────────────────────────────────────────── */
  const bboxInfo = el("div",
    "font-size:10px;opacity:.6;white-space:pre-wrap;word-break:break-all;" +
    "max-height:54px;overflow-y:auto;flex:0 0 auto;"
  );
  root.appendChild(bboxInfo);

  /* ── apply bbox button ─────────────────────────────────────────────── */
  const applyBboxBtn = btn("Apply BBox edits", "margin-bottom:2px;", () => applyBboxEdits(node));
  root.appendChild(applyBboxBtn);

  /* ── caption textarea ─────────────────────────────────────────────── */
  const captionLabel = el("div", "font-size:10px;opacity:.5;flex:0 0 auto;", { textContent: "Caption" });
  root.appendChild(captionLabel);

  const textarea = el("textarea",
    "flex:1 1 auto;min-height:90px;width:100%;box-sizing:border-box;" +
    "resize:vertical;background:var(--comfy-input-bg,#2a2a2a);border:1px solid var(--border-color,#555);" +
    "border-radius:4px;color:var(--input-text,#ddd);padding:6px;font-size:11px;line-height:1.4;" +
    "font-family:monospace;"
  );
  textarea.spellcheck = false;
  textarea.addEventListener("input", () => {
    const dsr = node._dsr;
    const item = dsr.items[dsr.index];
    if (item) item.edited_caption = textarea.value;
    syncHiddenWidget(node);
  });
  root.appendChild(textarea);

  /* ── bottom bar: copy + reset ─────────────────────────────────────── */
  const bottomBar = el("div", "display:flex;gap:4px;flex:0 0 auto;");
  bottomBar.appendChild(btn("Copy caption", "", () => {
    const item = node._dsr.items[node._dsr.index];
    navigator.clipboard?.writeText(item?.edited_caption || "").catch(() => {});
  }));
  bottomBar.appendChild(btn("Reset this caption", "", () => {
    const dsr = node._dsr;
    const item = dsr.items[dsr.index];
    if (item) {
      item.edited_caption = item.caption;
      textarea.value = item.caption;
      syncHiddenWidget(node);
    }
  }));
  bottomBar.appendChild(btn("Reset BBoxes", "", () => {
    const dsr = node._dsr;
    const item = dsr.items[dsr.index];
    if (item) {
      item.edited_boxes = item.boxes.map(b => ({ ...b }));
      drawBoxes(node);
      updateBboxInfo(node);
      syncHiddenWidget(node);
    }
  }));
  root.appendChild(bottomBar);

  /* ── store refs ────────────────────────────────────────────────────── */
  node._dsr_refs = { root, imgEl, canvas, counter, textarea, bboxInfo };

  /* ── mouse/touch events on canvas for drag-to-edit ────────────────── */
  canvas.addEventListener("mousedown",  e => onMouseDown(e, node));
  canvas.addEventListener("mousemove",  e => onMouseMove(e, node));
  canvas.addEventListener("mouseup",    e => onMouseUp(e, node));
  canvas.addEventListener("mouseleave", e => onMouseUp(e, node));

  node.addDOMWidget("dsr_widget", "div", root, { serialize: false });

  if (node.size[0] < 420) node.size[0] = 420;
  if (node.size[1] < 620) node.size[1] = 620;
}

/* ═══════════════════════════════════════════════════════════════════════
   RENDER SLIDE
══════════════════════════════════════════════════════════════════════════ */
function renderSlide(node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  if (!refs) return;

  const { imgEl, counter, textarea, bboxInfo } = refs;
  const items = dsr.items;

  if (!items || items.length === 0) {
    counter.textContent = "No data — run the workflow.";
    imgEl.src = "";
    textarea.value = "";
    bboxInfo.textContent = "";
    return;
  }

  const i    = dsr.index;
  const item = items[i];
  counter.textContent = `${i + 1} / ${items.length}`;

  imgEl.onload = () => {
    // After image loads, sync canvas size, stash natural pixel size on boxes, draw boxes
    const wrap  = refs.canvas.parentElement;
    const canvas = refs.canvas;
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    (item.edited_boxes || []).forEach(b => {
      b._natW = imgEl.naturalWidth;
      b._natH = imgEl.naturalHeight;
    });
    drawBoxes(node);
  };
  imgEl.src = item.url;

  textarea.value = item.edited_caption;
  updateBboxInfo(node);
}

/* ═══════════════════════════════════════════════════════════════════════
   DRAW BOXES on canvas
══════════════════════════════════════════════════════════════════════════ */
const HANDLE_R = 6; // hit radius in px

function drawBoxes(node) {
  const dsr    = node._dsr;
  const refs   = node._dsr_refs;
  if (!refs) return;

  const { canvas, imgEl } = refs;
  const item = (dsr.items || [])[dsr.index];
  const ctx  = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!item) return;

  // Compute the actual rendered rect of the image inside the canvas
  const imgRect = getImageRect(imgEl, canvas);
  if (!imgRect) return;

  const boxes = item.edited_boxes || [];
  const { x: ix, y: iy, w: iw, h: ih } = imgRect;

  boxes.forEach((box, bi) => {
    const color = boxColor(bi);
    const px1 = ix + box.rx1 * iw;
    const py1 = iy + box.ry1 * ih;
    const px2 = ix + box.rx2 * iw;
    const py2 = iy + box.ry2 * ih;

    // box outline
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

    // subtle fill
    ctx.fillStyle = color + "22";
    ctx.fillRect(px1, py1, px2 - px1, py2 - py1);

    // label
    ctx.fillStyle = color;
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(`#${bi + 1} ${box.label}`, px1 + 3, py1 + 13);

    // resize handles at corners + edge-midpoints
    const handles = getHandlePositions(px1, py1, px2, py2);
    handles.forEach(([hx, hy]) => {
      ctx.fillStyle   = color;
      ctx.strokeStyle = "#000";
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.arc(hx, hy, HANDLE_R, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  });
}

/* 8 handle positions: corners + edge-midpoints
   Order: tl, tm, tr, ml, mr, bl, bm, br */
function getHandlePositions(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return [
    [x1, y1], [mx, y1], [x2, y1],
    [x1, my],            [x2, my],
    [x1, y2], [mx, y2], [x2, y2],
  ];
}

const HANDLE_KEYS = ["tl","tm","tr","ml","mr","bl","bm","br"];

/* Returns the rendered rect of <img> inside <canvas> accounting for object-fit:contain */
function getImageRect(imgEl, canvas) {
  const natW = imgEl.naturalWidth;
  const natH = imgEl.naturalHeight;
  if (!natW || !natH) return null;

  const cw = canvas.width;
  const ch = canvas.height;

  const scale = Math.min(cw / natW, ch / natH);
  const rw = natW * scale;
  const rh = natH * scale;
  return {
    x: (cw - rw) / 2,
    y: (ch - rh) / 2,
    w: rw,
    h: rh,
  };
}

/* ═══════════════════════════════════════════════════════════════════════
   MOUSE DRAG for bbox editing
   Two interaction modes:
     - Grabbing one of the 8 corner/edge handles resizes that edge.
     - Grabbing anywhere inside the box body (but not on a handle) moves
       the whole box, keeping its width/height fixed.
══════════════════════════════════════════════════════════════════════════ */
function getCanvasPos(e, canvas) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvas.width  / r.width),
    y: (e.clientY - r.top)  * (canvas.height / r.height),
  };
}

function hitTestHandles(mx, my, node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  const item = (dsr.items || [])[dsr.index];
  if (!item) return null;

  const imgRect = getImageRect(refs.imgEl, refs.canvas);
  if (!imgRect) return null;

  const { x: ix, y: iy, w: iw, h: ih } = imgRect;
  const boxes = item.edited_boxes || [];

  for (let bi = boxes.length - 1; bi >= 0; bi--) {
    const box = boxes[bi];
    const px1 = ix + box.rx1 * iw;
    const py1 = iy + box.ry1 * ih;
    const px2 = ix + box.rx2 * iw;
    const py2 = iy + box.ry2 * ih;

    const handles = getHandlePositions(px1, py1, px2, py2);
    for (let hi = 0; hi < handles.length; hi++) {
      const [hx, hy] = handles[hi];
      if (Math.hypot(mx - hx, my - hy) <= HANDLE_R + 2) {
        return { boxIdx: bi, handle: HANDLE_KEYS[hi], px1, py1, px2, py2 };
      }
    }
  }
  return null;
}

/* Checks if (mx, my) is inside a box's body, excluding the handle zones,
   so a click near an edge/corner still prioritizes resizing over moving. */
function hitTestBoxBody(mx, my, node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  const item = (dsr.items || [])[dsr.index];
  if (!item) return null;

  const imgRect = getImageRect(refs.imgEl, refs.canvas);
  if (!imgRect) return null;

  const { x: ix, y: iy, w: iw, h: ih } = imgRect;
  const boxes = item.edited_boxes || [];

  for (let bi = boxes.length - 1; bi >= 0; bi--) {
    const box = boxes[bi];
    const px1 = ix + box.rx1 * iw;
    const py1 = iy + box.ry1 * ih;
    const px2 = ix + box.rx2 * iw;
    const py2 = iy + box.ry2 * ih;

    const inner = HANDLE_R + 2;
    if (
      mx >= px1 + inner && mx <= px2 - inner &&
      my >= py1 + inner && my <= py2 - inner
    ) {
      return { boxIdx: bi };
    }
  }
  return null;
}

function onMouseDown(e, node) {
  e.preventDefault();
  const refs   = node._dsr_refs;
  const dsr    = node._dsr;
  const { x, y } = getCanvasPos(e, refs.canvas);

  // Priority 1: resize handles
  const hit = hitTestHandles(x, y, node);
  if (hit) {
    const item = dsr.items[dsr.index];
    const box  = item.edited_boxes[hit.boxIdx];

    dsr.dragging = {
      boxIdx: hit.boxIdx,
      handle: hit.handle,
      startX: x,
      startY: y,
      origRx1: box.rx1,
      origRy1: box.ry1,
      origRx2: box.rx2,
      origRy2: box.ry2,
    };
    refs.canvas.style.cursor = "crosshair";
    return;
  }

  // Priority 2: move the whole box
  const bodyHit = hitTestBoxBody(x, y, node);
  if (bodyHit) {
    const item = dsr.items[dsr.index];
    const box  = item.edited_boxes[bodyHit.boxIdx];

    dsr.dragging = {
      boxIdx: bodyHit.boxIdx,
      handle: "move",
      startX: x,
      startY: y,
      origRx1: box.rx1,
      origRy1: box.ry1,
      origRx2: box.rx2,
      origRy2: box.ry2,
    };
    refs.canvas.style.cursor = "move";
  }
}

function onMouseMove(e, node) {
  const refs = node._dsr_refs;
  const dsr  = node._dsr;
  const { x, y } = getCanvasPos(e, refs.canvas);

  // Update cursor when hovering (handle > body > default)
  if (!dsr.dragging) {
    const hit = hitTestHandles(x, y, node);
    if (hit) {
      refs.canvas.style.cursor = "crosshair";
    } else if (hitTestBoxBody(x, y, node)) {
      refs.canvas.style.cursor = "move";
    } else {
      refs.canvas.style.cursor = "default";
    }
    return;
  }

  e.preventDefault();
  const imgRect = getImageRect(refs.imgEl, refs.canvas);
  if (!imgRect) return;

  const dx = (x - dsr.dragging.startX) / imgRect.w;
  const dy = (y - dsr.dragging.startY) / imgRect.h;

  const item = dsr.items[dsr.index];
  const box  = item.edited_boxes[dsr.dragging.boxIdx];
  const { handle, origRx1, origRy1, origRx2, origRy2 } = dsr.dragging;

  let rx1 = origRx1, ry1 = origRy1, rx2 = origRx2, ry2 = origRy2;

  if (handle === "move") {
    // Translate the whole box, preserving width/height, clamped to 0-1
    const width  = origRx2 - origRx1;
    const height = origRy2 - origRy1;
    const nx1 = Math.max(0, Math.min(1 - width,  origRx1 + dx));
    const ny1 = Math.max(0, Math.min(1 - height, origRy1 + dy));
    rx1 = nx1;
    ry1 = ny1;
    rx2 = nx1 + width;
    ry2 = ny1 + height;
  } else if (handle === "tl") { rx1 += dx; ry1 += dy; }
  else if (handle === "tm") { ry1 += dy; }
  else if (handle === "tr") { rx2 += dx; ry1 += dy; }
  else if (handle === "ml") { rx1 += dx; }
  else if (handle === "mr") { rx2 += dx; }
  else if (handle === "bl") { rx1 += dx; ry2 += dy; }
  else if (handle === "bm") { ry2 += dy; }
  else if (handle === "br") { rx2 += dx; ry2 += dy; }

  // Clamp 0-1 (redundant for "move", but harmless for resize handles)
  box.rx1 = Math.max(0, Math.min(1, rx1));
  box.ry1 = Math.max(0, Math.min(1, ry1));
  box.rx2 = Math.max(0, Math.min(1, rx2));
  box.ry2 = Math.max(0, Math.min(1, ry2));

  drawBoxes(node);
  updateBboxInfo(node);
  syncHiddenWidget(node);
}

function onMouseUp(e, node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  dsr.dragging = null;
  if (refs) refs.canvas.style.cursor = "default";
}

/* ═══════════════════════════════════════════════════════════════════════
   APPLY BBOX EDITS → write back to caption JSON
   Converts rx/ry (0-1 relative) into yxyx_normalized (0-1000 scale).
   Output format is FIXED — no longer depends on the bbox_format dropdown.
══════════════════════════════════════════════════════════════════════════ */
function applyBboxEdits(node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  const item = (dsr.items || [])[dsr.index];
  if (!item) return;

  // Parse current edited caption
  let data = null;
  const s = item.edited_caption.trim();
  for (const cand of [s, s.slice(s.indexOf("{"), s.lastIndexOf("}") + 1)]) {
    try { const d = JSON.parse(cand); if (typeof d === "object" && d) { data = d; break; } }
    catch (_) {}
  }
  if (!data) {
    alert("Caption is not valid JSON — cannot patch bboxes automatically.");
    return;
  }

  let elements = [];
  const cd = data.compositional_deconstruction;
  if (cd && Array.isArray(cd.elements)) elements = cd.elements;
  else if (Array.isArray(data.elements))  elements = data.elements;

  item.edited_boxes.forEach((box) => {
    if (typeof box.element_index !== "number") return;
    const el = elements[box.element_index];
    if (!el || !Array.isArray(el.bbox)) return;

    const rx1 = Math.min(box.rx1, box.rx2);
    const rx2 = Math.max(box.rx1, box.rx2);
    const ry1 = Math.min(box.ry1, box.ry2);
    const ry2 = Math.max(box.ry1, box.ry2);

    const a  = Math.round(ry1 * 1000);
    const b  = Math.round(rx1 * 1000);
    const c  = Math.round(ry2 * 1000);
    const d2 = Math.round(rx2 * 1000);

    el.bbox = [a, b, c, d2];
  });

  item.edited_caption = JSON.stringify(data, null, 2);
  refs.textarea.value = item.edited_caption;
  syncHiddenWidget(node);
}

/* ── bbox info display ─────────────────────────────────────────────── */
function updateBboxInfo(node) {
  const dsr  = node._dsr;
  const refs = node._dsr_refs;
  if (!refs) return;
  const item = (dsr.items || [])[dsr.index];
  if (!item) { refs.bboxInfo.textContent = ""; return; }

  const lines = [`Format: ${dsr.bboxFormat}`];
  (item.edited_boxes || []).forEach((box, i) => {
    const r = (v) => v.toFixed(3);
    lines.push(
      `#${i + 1} ${box.label}: rx(${r(box.rx1)}→${r(box.rx2)}) ry(${r(box.ry1)}→${r(box.ry2)})  raw: [${box.raw_bbox?.join(", ")}]`
    );
  });
  refs.bboxInfo.textContent = lines.join("\n");
}

/* ═══════════════════════════════════════════════════════════════════════
   NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
function unloadAll(node) {
  const dsr = node._dsr;
  const refs = node._dsr_refs;
  if (!dsr) return;

  const ok = confirm("This removes all loaded images, captions and bbox edits. Continue?");
  if (!ok) return;

  dsr.items = [];
  dsr.index = 0;
  dsr.dragging = null;

  if (refs) {
    refs.imgEl.src = "";
    refs.textarea.value = "";
    refs.bboxInfo.textContent = "";
    refs.counter.textContent = "No data — run the workflow.";
    const ctx = refs.canvas.getContext("2d");
    ctx.clearRect(0, 0, refs.canvas.width, refs.canvas.height);
  }

  syncHiddenWidget(node);
}

function navigate(node, delta) {
  const dsr = node._dsr;
  if (!dsr || !dsr.items || dsr.items.length === 0) return;
  dsr.index = (dsr.index + delta + dsr.items.length) % dsr.items.length;
  renderSlide(node);
}
