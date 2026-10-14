import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import { useKitchen } from "./kitchenContext";

export type BlendEdgeData = {
  blend_mode?: string;
  opacity?: number;
  /** True when this edge composites two image plates (not depth→look). */
  blendable?: boolean;
};

function modeLabel(key: string, modes: { key: string; label: string }[]) {
  return modes.find((m) => m.key === key)?.label || key;
}

export function BlendEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  data,
  selected,
}: EdgeProps) {
  const { meta, openEdgeBlend } = useKitchen();
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const d = (data || {}) as BlendEdgeData;
  const blendable = Boolean(d.blendable);
  const mode = d.blend_mode || "normal";
  const opacity = Number(d.opacity ?? 1);
  const label = blendable
    ? `${modeLabel(mode, meta?.blend_modes || [])} · ${Math.round(opacity * 100)}%`
    : "in";

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          strokeWidth: selected ? 3 : 2,
          stroke: selected ? "#0044ff" : style?.stroke,
        }}
      />
      <EdgeLabelRenderer>
        <button
          type="button"
          className={`edge-blend-pill nodrag nopan${selected ? " selected" : ""}${
            blendable ? " blendable" : ""
          }`}
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
          onClick={(e) => {
            e.stopPropagation();
            openEdgeBlend(id);
          }}
          title={
            blendable
              ? "Edit blend on this connection"
              : "Connection — click to disconnect"
          }
        >
          {label}
        </button>
      </EdgeLabelRenderer>
    </>
  );
}

export const edgeTypes = {
  blend: BlendEdge,
};
