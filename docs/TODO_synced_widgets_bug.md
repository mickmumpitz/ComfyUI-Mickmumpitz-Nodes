# Bug: Duplicated subgraph nodes have synced/mirrored widget values

## Status: FIXED

## Root cause (confirmed)

ComfyUI's `registerSubgraphNodeDef()` creates a node class whose constructor
captures a **single shared `Subgraph` (LGraph) instance** in a closure:

```javascript
// Inside registerSubgraphNodeDef(M, O, P):
const R = class extends SubgraphNode {
    constructor() {
        super(app.rootGraph, O, P);   // O is the SAME Subgraph for ALL instances
    }
};
LiteGraph.registerNodeType(O.id, R);
```

Every instance of the same subgraph type shares `this.subgraph = O`.

The exposed widgets on a SubgraphNode are **Proxy objects** (`proxyWidgets`)
created by `newProxyWidget(node, innerNodeId, widgetName)`. Each proxy's
getter/setter reads from and writes to the matching widget on an inner node
inside `node.subgraph`. Since all instances share the same `Subgraph`, all
proxy widgets read/write to the same inner nodes — hence the "sync".

## Fix applied

In `web/js/shot_duplicator.js`, `_cloneNode()` now detects SubgraphNodes
(nodes with a `.subgraph` that has a `.clone()` method) and deep-clones the
inner graph before calling `configure()`:

```javascript
if (srcNode.subgraph && typeof srcNode.subgraph.clone === 'function') {
    newNode.subgraph = srcNode.subgraph.clone();
}
newNode.configure(data);
```

`Subgraph.clone()` serialises the entire inner graph (via `asSerialisable()`)
and constructs a brand-new `Subgraph` instance from that data. When
`configure()` then triggers `onConfigure()`, the rebuilt proxy widgets bind
to the **cloned** inner graph instead of the shared one. Each shot now owns
its own copy of the inner graph with fully independent widget values.
