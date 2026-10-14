import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { Node } from "@xyflow/react";
import { bust, fetchTunePreview, sampleDepthAt, type DepthSample } from "./api";
import { useKitchen } from "./kitchenContext";

type WindowTuneModalProps = {
  nodes: Node[];
};

type HitBox = { left: number; top: number; width: number; height: number };

function containRect(
  stageW: number,
  stageH: number,
  imgW: number,
  imgH: number
): HitBox {
  if (stageW <= 0 || stageH <= 0 || imgW <= 0 || imgH <= 0) {
    return { left: 0, top: 0, width: Math.max(stageW, 0), height: Math.max(stageH, 0) };
  }
  const stageAR = stageW / stageH;
  const imgAR = imgW / imgH;
  if (imgAR > stageAR) {
    const height = stageW / imgAR;
    return { left: 0, top: (stageH - height) / 2, width: stageW, height };
  }
  const width = stageH * imgAR;
  return { left: (stageW - width) / 2, top: 0, width, height: stageH };
}

export function WindowTuneModal({ nodes }: WindowTuneModalProps) {
  const {
    windowTune,
    meta,
    runId,
    frameIndex,
    rgbUrl,
    evalToken,
    nodePreviews,
    progress,
    frameWidth,
    frameHeight,
    stopWindowTune,
    setWindowTuneMode,
    applyWindowPct,
    commitWindowTune,
  } = useKitchen();

  const stageRef = useRef<HTMLDivElement>(null);
  const hitRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    startY: number;
    basePct: number;
    field: "lo_pct" | "hi_pct";
    dragging: boolean;
  } | null>(null);
  const [lastSample, setLastSample] = useState<DepthSample | null>(null);
  const [sampling, setSampling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mix, setMix] = useState(0);
  const [turboOn, setTurboOn] = useState(false);
  const [turboUrl, setTurboUrl] = useState<string | null>(null);
  const [turboLoading, setTurboLoading] = useState(false);
  const [imgNat, setImgNat] = useState<{ w: number; h: number } | null>(
    frameWidth && frameHeight ? { w: frameWidth, h: frameHeight } : null
  );
  const [hitBox, setHitBox] = useState<HitBox | null>(null);

  const tuneNode = windowTune
    ? nodes.find((n) => n.id === windowTune.nodeId)
    : null;
  const d = (tuneNode?.data || {}) as Record<string, unknown>;
  const preset = String(d.preset || "ledger");
  const label =
    meta?.looks.find((l) => l.key === preset)?.label || preset;
  const lo = Number(d.lo_pct ?? 2);
  const hi = Number(d.hi_pct ?? 98);

  const lookPreview = windowTune ? nodePreviews[windowTune.nodeId] : null;
  const originalUrl = rgbUrl ? bust(rgbUrl, evalToken) : null;
  const lookUrl = lookPreview ? bust(lookPreview, evalToken) : null;
  const renderUrl = turboOn && turboUrl ? turboUrl : lookUrl;

  useEffect(() => {
    if (!windowTune) {
      setLastSample(null);
      setError(null);
      setSampling(false);
      setMix(0);
      setTurboOn(false);
      setTurboUrl(null);
      setTurboLoading(false);
      setHitBox(null);
    }
  }, [windowTune]);

  useEffect(() => {
    if (frameWidth && frameHeight) {
      setImgNat({ w: frameWidth, h: frameHeight });
    }
  }, [frameWidth, frameHeight]);

  useEffect(() => {
    setTurboUrl(null);
  }, [frameIndex, windowTune?.nodeId]);

  useEffect(() => {
    const nodeId = windowTune?.nodeId;
    if (!nodeId || !runId || !turboOn || turboUrl) return;
    let cancelled = false;
    setTurboLoading(true);
    setError(null);
    fetchTunePreview(runId, {
      frame: frameIndex,
      nodeId,
    })
      .then((res) => {
        if (!cancelled) setTurboUrl(bust(res.url, Date.now()));
      })
      .catch((e: any) => {
        if (!cancelled) {
          setError(e.message || "Turbo preview failed");
          setTurboOn(false);
        }
      })
      .finally(() => {
        if (!cancelled) setTurboLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [windowTune?.nodeId, runId, turboOn, turboUrl, frameIndex]);

  useEffect(() => {
    if (!windowTune) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") stopWindowTune();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [windowTune, stopWindowTune]);

  const layoutHit = useCallback(() => {
    const stage = stageRef.current;
    if (!stage || !imgNat) {
      setHitBox(null);
      return;
    }
    setHitBox(
      containRect(stage.clientWidth, stage.clientHeight, imgNat.w, imgNat.h)
    );
  }, [imgNat]);

  useLayoutEffect(() => {
    layoutHit();
    const stage = stageRef.current;
    if (!stage) return;
    const ro = new ResizeObserver(() => layoutHit());
    ro.observe(stage);
    return () => ro.disconnect();
  }, [layoutHit, windowTune]);

  const onRgbLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      setImgNat({ w: img.naturalWidth, h: img.naturalHeight });
    }
  }, []);

  const coordsFromEvent = useCallback(
    (clientX: number, clientY: number) => {
      const el = hitRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const u = (clientX - rect.left) / rect.width;
      const v = (clientY - rect.top) / rect.height;
      if (u < 0 || u > 1 || v < 0 || v > 1) return null;
      return { u, v };
    },
    []
  );

  const applyPct = useCallback(
    (field: "lo_pct" | "hi_pct", pct: number) => {
      if (!windowTune) return;
      applyWindowPct(windowTune.nodeId, field, pct);
    },
    [applyWindowPct, windowTune]
  );

  const sampleAndApply = useCallback(
    async (u: number, v: number, shiftKey: boolean) => {
      if (!windowTune || !runId) return;
      const field: "lo_pct" | "hi_pct" =
        shiftKey || windowTune.mode === "far" ? "hi_pct" : "lo_pct";
      setSampling(true);
      setError(null);
      try {
        const sample = await sampleDepthAt(runId, u, v, frameIndex);
        setLastSample(sample);
        applyPct(field, sample.pct);
        dragRef.current = {
          startY: 0,
          basePct: sample.pct,
          field,
          dragging: false,
        };
      } catch (e: any) {
        setError(e.message || "Depth sample failed");
      } finally {
        setSampling(false);
      }
    },
    [applyPct, frameIndex, runId, windowTune]
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.preventDefault();
      const coords = coordsFromEvent(e.clientX, e.clientY);
      if (!coords) return;
      (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
      dragRef.current = {
        startY: e.clientY,
        basePct: 0,
        field: e.shiftKey || windowTune?.mode === "far" ? "hi_pct" : "lo_pct",
        dragging: false,
      };
      void sampleAndApply(coords.u, coords.v, e.shiftKey);
    },
    [coordsFromEvent, sampleAndApply, windowTune?.mode]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragRef.current || !windowTune) return;
      const dy = e.clientY - dragRef.current.startY;
      if (!dragRef.current.dragging && Math.abs(dy) > 4) {
        dragRef.current.dragging = true;
      }
      if (!dragRef.current.dragging) return;
      e.preventDefault();
      const nudged = dragRef.current.basePct - dy * 0.15;
      applyPct(dragRef.current.field, nudged);
    },
    [applyPct, windowTune]
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!dragRef.current) return;
      dragRef.current = null;
      try {
        (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      void commitWindowTune();
    },
    [commitWindowTune]
  );

  const onTurboClick = useCallback(() => {
    if (turboOn) {
      setTurboOn(false);
      return;
    }
    setTurboOn(true);
    setMix((m) => (m < 0.4 ? 0.75 : m));
  }, [turboOn]);

  if (!windowTune || !tuneNode) return null;

  const busy = progress === "evaluate" || sampling || turboLoading;
  const hitStyle: React.CSSProperties | undefined = hitBox
    ? {
        left: hitBox.left,
        top: hitBox.top,
        width: hitBox.width,
        height: hitBox.height,
      }
    : { inset: 0 };

  return (
    <div
      className="modal-backdrop window-tune-backdrop"
      onClick={() => stopWindowTune()}
    >
      <div
        className="window-tune-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="window-tune-bar">
          <strong>Tune · Look · {label}</strong>
          <button type="button" onClick={() => stopWindowTune()}>
            Close
          </button>
        </div>

        <div className="window-tune-toolbar nodrag nopan">
          <div className="window-tune-mode">
            <button
              type="button"
              className={windowTune.mode === "near" ? "active" : ""}
              onClick={() => setWindowTuneMode("near")}
            >
              Near
            </button>
            <button
              type="button"
              className={windowTune.mode === "far" ? "active" : ""}
              onClick={() => setWindowTuneMode("far")}
            >
              Far
            </button>
          </div>
          <label className="window-tune-mix">
            <span>Original</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={mix}
              disabled={!originalUrl && !renderUrl}
              onChange={(e) => setMix(parseFloat(e.target.value))}
            />
            <span>Render</span>
          </label>
          <div className="window-tune-mode">
            <button
              type="button"
              className={turboOn ? "active" : ""}
              disabled={!runId || turboLoading}
              onClick={onTurboClick}
              title="Full-size grayscale depth for picking"
            >
              {turboLoading ? "Turbo…" : "Turbo"}
            </button>
          </div>
        </div>

        <div className="window-tune-stage" ref={stageRef}>
          {originalUrl || renderUrl ? (
            <div className="window-tune-layers">
              {originalUrl ? (
                <img
                  src={originalUrl}
                  alt="Original"
                  draggable={false}
                  onLoad={onRgbLoad}
                />
              ) : null}
              {renderUrl ? (
                <img
                  src={renderUrl}
                  alt={turboOn ? "Depth (gray)" : label}
                  draggable={false}
                  style={{ opacity: mix }}
                />
              ) : null}
            </div>
          ) : (
            <div className="preview-empty">No preview available</div>
          )}
          <div
            ref={hitRef}
            className={`window-tune-hit nodrag nopan${busy ? " busy" : ""}`}
            style={hitStyle}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
          />
          {busy ? (
            <div className="window-tune-overlay">
              <div className="progress-bar" />
            </div>
          ) : null}
        </div>

        <div className="window-tune-readout">
          {lastSample ? (
            <span>
              {lastSample.pct.toFixed(1)}% · {lastSample.depth_m.toFixed(2)} m
            </span>
          ) : (
            <span className="muted">Click to sample depth</span>
          )}
          <span>
            Near {lo.toFixed(1)} · Far {hi.toFixed(1)}
          </span>
        </div>

        <p className="window-tune-hint muted">
          Turbo for full-size gray depth · mix against original · click Near ·
          Shift-click Far · drag to nudge · Esc to close
        </p>
        {error ? <p className="window-tune-error">{error}</p> : null}
      </div>
    </div>
  );
}
