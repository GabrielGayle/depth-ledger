/**
 * Kitchen (node-graph) workspace — sibling view to Operator.
 * Owns the React Flow canvas + left rail; shared run/infer state stays in App.
 */
import type { ReactNode } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
  type OnEdgesChange,
  type OnNodesChange,
  type Connection,
  type OnSelectionChangeParams,
  type IsValidConnection,
} from "@xyflow/react";
import { bust, type Meta, type VideoRecipe } from "./api";
import { AspectPreview } from "./AspectPreview";
import { edgeTypes } from "./edges";
import { nodeTypes } from "./nodes";
import { WindowTuneModal } from "./windowTune";
import { useKitchen } from "./kitchenContext";

export type KitchenViewProps = {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (c: Connection) => void;
  isValidConnection: IsValidConnection;
  onSelectionChange: (p: OnSelectionChangeParams) => void;
  openEdgeBlend: (edgeId: string) => void;
  ingredientNodes: Node[];
  selectedId: string | null;
  selectedNode: Node | null;
  selectedEdgeId: string | null;
  selectIngredient: (id: string) => void;
  addNode: (
    type:
      | "look"
      | "scene"
      | "blend"
      | "relief"
      | "smooth"
      | "freq_fuse"
      | "arena"
      | "arena_out"
      | "video_in"
      | "video_out"
  ) => void;
  deleteNode: (id: string) => void;
  inspector: () => ReactNode;
  modalEdge: Edge | null;
  modalBlendable: boolean;
  modalData: { blend_mode?: string; opacity?: number };
  setEdgeModalId: (id: string | null) => void;
  patchEdgeData: (id: string, patch: Record<string, unknown>) => void;
  deleteEdge: (id: string) => void;
  lightbox: { url: string; title: string } | null;
  setLightbox: (v: { url: string; title: string } | null) => void;
  protectedTypes: Set<string>;
  meta: Meta | null;
  videoRecipes: VideoRecipe[];
  applyVideoRecipe: (recipe: VideoRecipe) => void;
};

export function KitchenView(props: KitchenViewProps) {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    isValidConnection,
    onSelectionChange,
    openEdgeBlend,
    ingredientNodes,
    selectedId,
    selectedNode,
    selectedEdgeId,
    selectIngredient,
    addNode,
    deleteNode,
    inspector,
    modalEdge,
    modalBlendable,
    modalData,
    setEdgeModalId,
    patchEdgeData,
    deleteEdge,
    lightbox,
    setLightbox,
    protectedTypes,
    meta,
    videoRecipes,
    applyVideoRecipe,
  } = props;
  const { rgbUrl, evalToken, nodePreviews, busy, openLightbox } = useKitchen();

  return (
    <div className="workspace">
      <aside className="rail">
        <h2>Video setup</h2>
        <div className="add-node-row video-recipe-row">
          <select
            className="nodrag nopan"
            defaultValue=""
            disabled={busy || !videoRecipes.length}
            onChange={(e) => {
              const id = e.target.value;
              e.target.value = "";
              const recipe = videoRecipes.find((r) => r.id === id);
              if (recipe) applyVideoRecipe(recipe);
            }}
            title="Load a built-in video kitchen graph"
          >
            <option value="" disabled>
              Load recipe…
            </option>
            {videoRecipes.map((r) => (
              <option key={r.id} value={r.id} title={r.summary}>
                {r.label}
              </option>
            ))}
          </select>
        </div>

        <h2>Original</h2>
        {rgbUrl ? (
          <button
            type="button"
            className="original-card"
            onClick={() => openLightbox(bust(rgbUrl, evalToken), "Original")}
          >
            <AspectPreview
              src={bust(rgbUrl, evalToken)}
              alt="Original"
              maxEdge={280}
            />
          </button>
        ) : (
          <p className="muted">
            Load via Are.na, Video In, or Generate to see the source RGB.
          </p>
        )}

        <h2 style={{ marginTop: "1.25rem" }}>Ingredients</h2>
        {ingredientNodes.length === 0 ? (
          <p className="muted">Looks and scenes appear after depth loads.</p>
        ) : (
          <div className="look-thumbs">
            {ingredientNodes.map((n) => {
              const d = n.data as Record<string, unknown>;
              const url = nodePreviews[n.id];
              const label =
                n.type === "scene"
                  ? "Scene"
                  : n.type === "relief"
                    ? "Relief"
                    : n.type === "blend"
                      ? "Blend"
                      : n.type === "smooth"
                        ? "Smooth"
                        : n.type === "freq_fuse"
                          ? "Freq Fuse"
                        : meta?.looks.find((l) => l.key === d.preset)?.label ||
                          String(d.preset || n.id);
              return (
                <button
                  type="button"
                  key={n.id}
                  className={`ingredient-card${
                    selectedId === n.id ? " active" : ""
                  }`}
                  onClick={() => selectIngredient(n.id)}
                >
                  {url ? (
                    <AspectPreview
                      src={bust(url, evalToken)}
                      alt={label}
                      maxEdge={200}
                    />
                  ) : (
                    <div className="ingredient-placeholder" />
                  )}
                  <span>{label}</span>
                </button>
              );
            })}
          </div>
        )}

        <h2 style={{ marginTop: "1.25rem" }}>Add node</h2>
        <div className="add-node-row">
          <button type="button" disabled={busy} onClick={() => addNode("arena")}>
            + Are.na
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => addNode("video_in")}
          >
            + Video In
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => addNode("video_out")}
          >
            + Video Out
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => addNode("arena_out")}
          >
            + Are.na Out
          </button>
          <button type="button" disabled={busy} onClick={() => addNode("look")}>
            + Look
          </button>
          <button type="button" disabled={busy} onClick={() => addNode("smooth")}>
            + Smooth
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => addNode("freq_fuse")}
          >
            + Freq Fuse
          </button>
          <button type="button" disabled={busy} onClick={() => addNode("scene")}>
            + Scene
          </button>
          <button type="button" disabled={busy} onClick={() => addNode("relief")}>
            + Relief
          </button>
          <button type="button" disabled={busy} onClick={() => addNode("blend")}>
            + Blend
          </button>
        </div>

        <h2 style={{ marginTop: "1.25rem" }}>
          {selectedNode
            ? `${selectedNode.type} · ${selectedNode.id}`
            : selectedEdgeId
              ? "Connection"
              : "Inspector"}
        </h2>
        {inspector()}
        {selectedNode && !protectedTypes.has(selectedNode.type || "") ? (
          <button
            type="button"
            className="danger"
            style={{ marginTop: "0.75rem" }}
            onClick={() => deleteNode(selectedNode.id)}
          >
            Delete node
          </button>
        ) : null}
      </aside>

      <div className="canvas-wrap">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onSelectionChange={onSelectionChange}
          onEdgeClick={(_e, edge) => openEdgeBlend(edge.id)}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ minZoom: 0.7, maxZoom: 1, padding: 0.08 }}
          edgesFocusable
          elementsSelectable
          deleteKeyCode={["Backspace", "Delete"]}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} color="#ccc" />
          <Controls />
          <MiniMap />
        </ReactFlow>

        {modalEdge ? (
          <div className="modal-backdrop" onClick={() => setEdgeModalId(null)}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
              <h3>{modalBlendable ? "Blend on connection" : "Connection"}</h3>
              <p className="muted">
                {modalEdge.source} → {modalEdge.target}
              </p>
              {modalBlendable ? (
                <>
                  <div className="field">
                    <label>Mode</label>
                    <select
                      value={modalData.blend_mode || "normal"}
                      onChange={(e) =>
                        patchEdgeData(modalEdge.id, {
                          blend_mode: e.target.value,
                        })
                      }
                    >
                      {(meta?.blend_modes || []).map((m) => (
                        <option key={m.key} value={m.key}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="field">
                    <label>Opacity</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={Number(modalData.opacity ?? 1)}
                      onChange={(e) =>
                        patchEdgeData(modalEdge.id, {
                          opacity: parseFloat(e.target.value),
                        })
                      }
                    />
                  </div>
                </>
              ) : (
                <p className="muted">This edge carries depth / structure only.</p>
              )}
              <div className="modal-actions">
                <button type="button" onClick={() => setEdgeModalId(null)}>
                  Close
                </button>
                <button
                  type="button"
                  className="danger"
                  onClick={() => {
                    deleteEdge(modalEdge.id);
                    setEdgeModalId(null);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {lightbox ? (
          <div
            className="modal-backdrop"
            onClick={() => setLightbox(null)}
          >
            <div
              className="lightbox-card"
              onClick={(e) => e.stopPropagation()}
            >
              <h3>{lightbox.title}</h3>
              <img src={lightbox.url} alt={lightbox.title} />
              <button type="button" onClick={() => setLightbox(null)}>
                Close
              </button>
            </div>
          </div>
        ) : null}

        <WindowTuneModal nodes={nodes} />
      </div>
    </div>
  );
}
