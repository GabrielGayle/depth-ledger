import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  bust,
  clearDefaultOperator,
  evaluateOperator,
  exportOperatorVideo,
  fetchDefaultOperator,
  saveDefaultOperator,
  type Meta,
  type OperatorOscillator,
  type OperatorPatch,
} from "../api";
import { AspectPreview } from "../AspectPreview";
import { lookDefaults } from "../kitchenContext";

const OSC_IDS = ["A", "B", "C", "D"] as const;
type OscId = (typeof OSC_IDS)[number];

function defaultOsc(id: OscId): OperatorOscillator {
  const presets: Record<OscId, Partial<OperatorOscillator>> = {
    A: { on: true, level: 1, preset: "ledger", lo_pct: 2, hi_pct: 98, gamma: 0.85 },
    B: { on: true, level: 0.55, preset: "sockets", lo_pct: 1, hi_pct: 30, gamma: 1 },
    C: { on: false, level: 0.4, preset: "soft", lo_pct: 40, hi_pct: 92, gamma: 0.95 },
    D: { on: false, level: 0.35, preset: "punchy", lo_pct: 5, hi_pct: 90, gamma: 0.7 },
  };
  return {
    on: true,
    level: 1,
    preset: "ledger",
    lo_pct: 2,
    hi_pct: 98,
    gamma: 0.85,
    cmap: "inherit",
    ...presets[id],
  };
}

export function defaultOperatorPatch(): OperatorPatch {
  return {
    version: 1,
    algorithm: 1,
    oscillators: {
      A: defaultOsc("A"),
      B: defaultOsc("B"),
      C: defaultOsc("C"),
      D: defaultOsc("D"),
    },
    filter: { type: "off", cutoff: 0.55, drive: 0 },
    shaper: { curve: "linear", amount: 0 },
    envelope: {
      attack: 0.05,
      decay: 0.2,
      sustain: 0.75,
      release: 0.3,
      amount: 0,
      dest: ["level_A"],
    },
    lfo: {
      rate: 0.25,
      shape: "sine",
      amount: 0,
      sync: false,
      dest: ["filter_cutoff"],
    },
    global: { time: 1, spread: 1, voices: ["plate"] },
    t: null,
  };
}

export type OperatorViewProps = {
  meta: Meta | null;
  runId: string | null;
  rgbUrl: string | null;
  busy: boolean;
  frameIndex: number;
  frameCount: number;
  sampleFps: number;
  evalToken: number;
  onGoToFrame: (index: number) => Promise<void> | void;
  onStatus?: (msg: string) => void;
};

function Shell({
  title,
  active,
  onSelect,
  children,
}: {
  title: string;
  active?: boolean;
  onSelect?: () => void;
  children: ReactNode;
}) {
  return (
    <section className={`op-shell${active ? " active" : ""}`}>
      <button
        type="button"
        className="op-shell-tab"
        onClick={onSelect}
        disabled={!onSelect}
      >
        {title}
      </button>
      <div className="op-shell-body">{children}</div>
    </section>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="op-field">
      <label>{label}</label>
      {children}
    </div>
  );
}

/** Tiny Ableton-style algorithm glyphs */
function AlgoGlyph({ id }: { id: number }) {
  const layouts: Record<number, string[][]> = {
    1: [["D"], ["C"], ["B"], ["A"]],
    2: [["D", "C"], ["B"], ["A"]],
    3: [["D"], ["C"], ["B"], ["A"]],
    4: [["D"], ["C"], ["B", "A"]],
    5: [["D", "C"], ["B", "A"]],
    6: [["D"], ["C", "B", "A"]],
    7: [["D"], ["C"], ["B", "A"]],
    8: [["D", "B"], ["C", "A"]],
    9: [["D"], ["C", "B"], ["A"]],
    10: [["D", "C"], ["B"], ["A"]],
    11: [["A", "B", "C", "D"]],
  };
  const rows = layouts[id] || [["A"]];
  return (
    <div className="op-algo-glyph" aria-hidden>
      {rows.map((row, ri) => (
        <div key={ri} className="op-algo-row">
          {row.map((cell) => (
            <span key={cell} className={`op-algo-cell op-algo-${cell}`}>
              {cell}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

export function OperatorView({
  meta,
  runId,
  rgbUrl,
  busy,
  frameIndex,
  frameCount,
  sampleFps,
  evalToken,
  onGoToFrame,
  onStatus,
}: OperatorViewProps) {
  const [patch, setPatch] = useState<OperatorPatch>(() => defaultOperatorPatch());
  const [activeOsc, setActiveOsc] = useState<OscId>("A");
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [oscPreviews, setOscPreviews] = useState<Record<string, string>>({});
  const [voices, setVoices] = useState<
    Record<string, string | null | undefined>
  >({});
  const [cooking, setCooking] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [hasCustomDefault, setHasCustomDefault] = useState(false);
  const [playing, setPlaying] = useState(false);
  const patchRef = useRef(patch);
  patchRef.current = patch;
  const evalTimer = useRef<number | null>(null);
  const playTimer = useRef<number | null>(null);

  const algorithms = meta?.operator?.algorithms || [];
  const looks = meta?.looks || [];
  const cmaps = meta?.cmaps || ["gray", "magma", "inferno", "viridis", "turbo", "photo", "wash"];
  const modDests = meta?.operator?.mod_dests || [
    "level_A",
    "filter_cutoff",
    "spread",
  ];
  const voiceKeys = meta?.operator?.voices || ["plate", "ply", "vector", "ndi"];

  useEffect(() => {
    void (async () => {
      try {
        const d = await fetchDefaultOperator();
        setPatch(d.patch);
        setHasCustomDefault(d.has_custom);
      } catch {
        /* keep built-in default */
      }
    })();
  }, []);

  const cook = useCallback(
    async (next?: OperatorPatch) => {
      if (!runId) return;
      const body = next || patchRef.current;
      const withT: OperatorPatch = {
        ...body,
        t:
          frameCount > 1
            ? frameIndex / Math.max(frameCount - 1, 1)
            : body.t ?? null,
      };
      setCooking(true);
      try {
        const res = await evaluateOperator(runId, withT);
        setOutputUrl(res.output_url);
        setOscPreviews(res.osc_previews || {});
        setVoices(res.voices || {});
        onStatus?.(`Operator · ${Math.round(res.elapsed_ms)}ms`);
      } catch (e: any) {
        onStatus?.(e?.message || "Operator evaluate failed");
      } finally {
        setCooking(false);
      }
    },
    [runId, frameCount, frameIndex, onStatus]
  );

  const scheduleCook = useCallback(
    (next: OperatorPatch) => {
      setPatch(next);
      if (evalTimer.current) window.clearTimeout(evalTimer.current);
      evalTimer.current = window.setTimeout(() => {
        void cook(next);
      }, 150);
    },
    [cook]
  );

  useEffect(() => {
    if (runId) void cook();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, frameIndex]);

  useEffect(() => {
    return () => {
      if (evalTimer.current) window.clearTimeout(evalTimer.current);
      if (playTimer.current) window.clearInterval(playTimer.current);
    };
  }, []);

  useEffect(() => {
    if (!playing || frameCount <= 1) {
      if (playTimer.current) window.clearInterval(playTimer.current);
      playTimer.current = null;
      return;
    }
    const ms = 1000 / Math.max(sampleFps * (patch.global.time || 1), 1);
    playTimer.current = window.setInterval(() => {
      const next = (frameIndex + 1) % frameCount;
      void onGoToFrame(next);
    }, ms);
    return () => {
      if (playTimer.current) window.clearInterval(playTimer.current);
    };
  }, [
    playing,
    frameCount,
    frameIndex,
    sampleFps,
    patch.global.time,
    onGoToFrame,
  ]);

  const patchOsc = (id: OscId, partial: Partial<OperatorOscillator>) => {
    const osc = { ...patch.oscillators[id], ...partial };
    if (partial.preset) {
      const defaults = lookDefaults(meta as any, partial.preset);
      if (defaults) {
        osc.lo_pct = defaults.lo_pct;
        osc.hi_pct = defaults.hi_pct;
        osc.gamma = defaults.gamma;
        if (defaults.cmap === "wash" || defaults.blur_amount != null) {
          osc.blur_amount = defaults.blur_amount ?? 0.4;
        }
      }
    }
    scheduleCook({
      ...patch,
      oscillators: { ...patch.oscillators, [id]: osc },
    });
  };

  const osc = patch.oscillators[activeOsc];

  const toggleVoice = (key: string) => {
    const cur = new Set(patch.global.voices || []);
    if (cur.has(key)) cur.delete(key);
    else cur.add(key);
    if (!cur.has("plate")) cur.add("plate");
    scheduleCook({
      ...patch,
      global: { ...patch.global, voices: Array.from(cur) },
    });
  };

  const onExport = async () => {
    if (!runId || frameCount <= 1) return;
    onStatus?.("Operator export…");
    try {
      const res = await exportOperatorVideo(runId, patch);
      setExportUrl(res.export_url);
      onStatus?.(`Operator export ready (${res.frame_count} frames)`);
    } catch (e: any) {
      onStatus?.(e?.message || "Export failed");
    }
  };

  const onSaveDefault = async () => {
    try {
      await saveDefaultOperator({ ...patch, t: null });
      setHasCustomDefault(true);
      onStatus?.("Operator default saved");
    } catch (e: any) {
      onStatus?.(e?.message || "Save default failed");
    }
  };

  const onResetDefault = async () => {
    try {
      await clearDefaultOperator();
      const d = defaultOperatorPatch();
      setPatch(d);
      setHasCustomDefault(false);
      void cook(d);
      onStatus?.("Operator default reset");
    } catch (e: any) {
      onStatus?.(e?.message || "Reset failed");
    }
  };

  const streamUrl = runId
    ? bust(`/api/runs/${runId}/operator/stream`, evalToken + (outputUrl || "").length)
    : null;

  return (
    <div className="workspace operator-workspace">
      <aside className="rail rail-left op-rail">
        <h2>Source</h2>
        {rgbUrl ? (
          <AspectPreview src={bust(rgbUrl, evalToken)} alt="Original" maxEdge={240} />
        ) : (
          <p className="muted">Generate or load a run first.</p>
        )}

        {frameCount > 1 ? (
          <>
            <h2 style={{ marginTop: "1rem" }}>Transport</h2>
            <div className="timeline">
              <div className="timeline-meta">
                Frame {frameIndex + 1} / {frameCount}
                <span className="muted"> · {sampleFps.toFixed(1)} fps</span>
              </div>
              <input
                type="range"
                min={0}
                max={frameCount - 1}
                value={frameIndex}
                disabled={busy}
                onChange={(e) => {
                  setPlaying(false);
                  void onGoToFrame(parseInt(e.target.value, 10));
                }}
              />
              <div className="timeline-actions">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setPlaying((p) => !p)}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  disabled={busy || cooking}
                  onClick={() => void onExport()}
                >
                  Export MP4
                </button>
              </div>
              {exportUrl ? (
                <p className="muted" style={{ marginTop: "0.4rem" }}>
                  <a href={bust(exportUrl, evalToken)} download>
                    Download export
                  </a>
                </p>
              ) : null}
            </div>
          </>
        ) : null}

        <h2 style={{ marginTop: "1rem" }}>Defaults</h2>
        <div className="add-node-row">
          <button type="button" disabled={busy} onClick={() => void onSaveDefault()}>
            Save default
          </button>
          {hasCustomDefault ? (
            <button type="button" disabled={busy} onClick={() => void onResetDefault()}>
              Reset
            </button>
          ) : null}
        </div>

        <h2 style={{ marginTop: "1rem" }}>Voices</h2>
        <div className="op-voice-list">
          {voiceKeys.map((v) => (
            <label key={v} className="op-check">
              <input
                type="checkbox"
                checked={(patch.global.voices || []).includes(v)}
                disabled={v === "plate"}
                onChange={() => toggleVoice(v)}
              />
              {v}
              {v === "ndi" && meta?.operator?.ndi_available ? " · ready" : null}
            </label>
          ))}
        </div>
        <div className="op-voice-links muted">
          {voices.ply ? (
            <a href={bust(voices.ply, evalToken)} download>
              PLY
            </a>
          ) : null}
          {voices.vector ? (
            <a href={bust(voices.vector, evalToken)} download>
              Vector
            </a>
          ) : null}
          {voices.vector_npz ? (
            <a href={bust(String(voices.vector_npz), evalToken)} download>
              NPZ
            </a>
          ) : null}
          {streamUrl ? (
            <a href={streamUrl} target="_blank" rel="noreferrer">
              Stream
            </a>
          ) : null}
        </div>
      </aside>

      <div className="operator-stage">
        <div className="operator-main">
          <div className="op-osc-tabs">
            {OSC_IDS.map((id) => (
              <button
                key={id}
                type="button"
                className={`op-osc-tab${activeOsc === id ? " active" : ""}${
                  patch.oscillators[id].on ? "" : " off"
                }`}
                onClick={() => setActiveOsc(id)}
              >
                Osc {id}
              </button>
            ))}
          </div>

          <div className="op-center">
            <Shell title={`Oscillator ${activeOsc}`}>
              <div className="op-osc-grid">
                <Field label="On">
                  <input
                    type="checkbox"
                    checked={osc.on}
                    onChange={(e) => patchOsc(activeOsc, { on: e.target.checked })}
                  />
                </Field>
                <Field label={`Level ${osc.level.toFixed(2)}`}>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={osc.level}
                    onChange={(e) =>
                      patchOsc(activeOsc, { level: parseFloat(e.target.value) })
                    }
                  />
                </Field>
                <Field label="Look">
                  <select
                    value={osc.preset}
                    onChange={(e) => patchOsc(activeOsc, { preset: e.target.value })}
                  >
                    {looks.map((l) => (
                      <option key={l.key} value={l.key}>
                        {l.label}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label={`Near ${osc.lo_pct.toFixed(0)}`}>
                  <input
                    type="range"
                    min={1}
                    max={40}
                    step={0.5}
                    value={osc.lo_pct}
                    onChange={(e) =>
                      patchOsc(activeOsc, { lo_pct: parseFloat(e.target.value) })
                    }
                  />
                </Field>
                <Field label={`Far ${osc.hi_pct.toFixed(0)}`}>
                  <input
                    type="range"
                    min={20}
                    max={99}
                    step={0.5}
                    value={osc.hi_pct}
                    onChange={(e) =>
                      patchOsc(activeOsc, { hi_pct: parseFloat(e.target.value) })
                    }
                  />
                </Field>
                <Field label={`Gamma ${osc.gamma.toFixed(2)}`}>
                  <input
                    type="range"
                    min={0.4}
                    max={1.6}
                    step={0.01}
                    value={osc.gamma}
                    onChange={(e) =>
                      patchOsc(activeOsc, { gamma: parseFloat(e.target.value) })
                    }
                  />
                </Field>
                <Field label="Cmap">
                  <select
                    value={osc.cmap}
                    onChange={(e) => {
                      const next = e.target.value;
                      const patch: Partial<OperatorOscillator> = { cmap: next };
                      if (
                        next === "wash" ||
                        (next === "inherit" &&
                          lookDefaults(meta as any, osc.preset)?.cmap === "wash")
                      ) {
                        if (osc.blur_amount == null) {
                          patch.blur_amount =
                            lookDefaults(meta as any, osc.preset)?.blur_amount ??
                            0.4;
                        }
                      }
                      patchOsc(activeOsc, patch);
                    }}
                  >
                    <option value="inherit">inherit</option>
                    {cmaps.map((c) => (
                      <option key={c} value={c}>
                        {c === "wash" ? "wash (from image)" : c}
                      </option>
                    ))}
                  </select>
                </Field>
                {(osc.cmap === "wash" ||
                  (osc.cmap === "inherit" &&
                    lookDefaults(meta as any, osc.preset)?.cmap ===
                      "wash")) && (
                  <Field
                    label={`Wash blur ${(osc.blur_amount ?? 0.4).toFixed(2)}`}
                  >
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={osc.blur_amount ?? 0.4}
                      onChange={(e) =>
                        patchOsc(activeOsc, {
                          blur_amount: parseFloat(e.target.value),
                        })
                      }
                    />
                  </Field>
                )}
              </div>
              {oscPreviews[activeOsc] ? (
                <div className="op-osc-preview">
                  <AspectPreview
                    src={bust(oscPreviews[activeOsc], evalToken)}
                    alt={`Osc ${activeOsc}`}
                    maxEdge={180}
                  />
                </div>
              ) : null}
            </Shell>

            <Shell title="Algorithms">
              <div className="op-algo-row-pick">
                {(algorithms.length
                  ? algorithms
                  : Array.from({ length: 11 }, (_, i) => ({
                      id: i + 1,
                      label: String(i + 1),
                      desc: "",
                    }))
                ).map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    title={`${a.label}${a.desc ? ` — ${a.desc}` : ""}`}
                    className={`op-algo-btn${
                      patch.algorithm === a.id ? " active" : ""
                    }`}
                    onClick={() => scheduleCook({ ...patch, algorithm: a.id })}
                  >
                    <AlgoGlyph id={a.id} />
                    <span>{a.id}</span>
                  </button>
                ))}
              </div>
            </Shell>

            <div className="op-output-stage">
              <div className="op-output-bar">
                <strong>Master</strong>
                <span className="muted">
                  {cooking ? "cooking…" : runId ? "live" : "no run"}
                </span>
              </div>
              {outputUrl ? (
                <AspectPreview
                  src={bust(outputUrl, evalToken)}
                  alt="Operator output"
                  maxEdge={720}
                />
              ) : (
                <p className="muted">
                  {runId
                    ? "Tweaking cooks the Operator plate…"
                    : "Load depth via Generate to start Operator."}
                </p>
              )}
              <div className="op-osc-thumbs">
                {OSC_IDS.map((id) =>
                  oscPreviews[id] ? (
                    <button
                      key={id}
                      type="button"
                      className={activeOsc === id ? "active" : ""}
                      onClick={() => setActiveOsc(id)}
                    >
                      <AspectPreview
                        src={bust(oscPreviews[id], evalToken)}
                        alt={id}
                        maxEdge={72}
                      />
                      <span>{id}</span>
                    </button>
                  ) : null
                )}
              </div>
            </div>
          </div>

          <div className="op-side-shells">
            <Shell title="Filter">
              <Field label="Type">
                <select
                  value={patch.filter.type}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      filter: { ...patch.filter, type: e.target.value },
                    })
                  }
                >
                  {(meta?.operator?.filter_types || ["off", "lowpass", "highpass", "formant"]).map(
                    (t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    )
                  )}
                </select>
              </Field>
              <Field label={`Cutoff ${patch.filter.cutoff.toFixed(2)}`}>
                <input
                  type="range"
                  min={0.02}
                  max={1}
                  step={0.01}
                  value={patch.filter.cutoff}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      filter: {
                        ...patch.filter,
                        cutoff: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </Field>
              <Field label={`Drive ${patch.filter.drive.toFixed(2)}`}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={patch.filter.drive}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      filter: {
                        ...patch.filter,
                        drive: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </Field>
            </Shell>

            <Shell title="Shaper">
              <Field label="Curve">
                <select
                  value={patch.shaper.curve}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      shaper: { ...patch.shaper, curve: e.target.value },
                    })
                  }
                >
                  {(
                    meta?.operator?.shaper_curves || [
                      "linear",
                      "s_curve",
                      "quantize",
                      "expand_near",
                    ]
                  ).map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`Amount ${patch.shaper.amount.toFixed(2)}`}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={patch.shaper.amount}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      shaper: {
                        ...patch.shaper,
                        amount: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </Field>
            </Shell>

            <Shell title="Envelope">
              {(
                [
                  ["attack", patch.envelope.attack],
                  ["decay", patch.envelope.decay],
                  ["sustain", patch.envelope.sustain],
                  ["release", patch.envelope.release],
                  ["amount", patch.envelope.amount],
                ] as const
              ).map(([key, val]) => (
                <Field key={key} label={`${key} ${val.toFixed(2)}`}>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={val}
                    onChange={(e) =>
                      scheduleCook({
                        ...patch,
                        envelope: {
                          ...patch.envelope,
                          [key]: parseFloat(e.target.value),
                        },
                      })
                    }
                  />
                </Field>
              ))}
              <Field label="Dest">
                <select
                  value={patch.envelope.dest[0] || "level_A"}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      envelope: { ...patch.envelope, dest: [e.target.value] },
                    })
                  }
                >
                  {modDests.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </Field>
            </Shell>

            <Shell title="LFO">
              <Field label={`Rate ${patch.lfo.rate.toFixed(2)}`}>
                <input
                  type="range"
                  min={0.01}
                  max={2}
                  step={0.01}
                  value={patch.lfo.rate}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      lfo: { ...patch.lfo, rate: parseFloat(e.target.value) },
                    })
                  }
                />
              </Field>
              <Field label="Shape">
                <select
                  value={patch.lfo.shape}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      lfo: { ...patch.lfo, shape: e.target.value },
                    })
                  }
                >
                  {(meta?.operator?.lfo_shapes || ["sine", "triangle", "saw", "square"]).map(
                    (s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    )
                  )}
                </select>
              </Field>
              <Field label={`Amount ${patch.lfo.amount.toFixed(2)}`}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={patch.lfo.amount}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      lfo: { ...patch.lfo, amount: parseFloat(e.target.value) },
                    })
                  }
                />
              </Field>
              <Field label="Dest">
                <select
                  value={patch.lfo.dest[0] || "filter_cutoff"}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      lfo: { ...patch.lfo, dest: [e.target.value] },
                    })
                  }
                >
                  {modDests.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </Field>
            </Shell>

            <Shell title="Global">
              <Field label={`Time ${patch.global.time.toFixed(2)}`}>
                <input
                  type="range"
                  min={0.05}
                  max={4}
                  step={0.05}
                  value={patch.global.time}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      global: {
                        ...patch.global,
                        time: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </Field>
              <Field label={`Spread ${patch.global.spread.toFixed(2)}`}>
                <input
                  type="range"
                  min={0.05}
                  max={4}
                  step={0.05}
                  value={patch.global.spread}
                  onChange={(e) =>
                    scheduleCook({
                      ...patch,
                      global: {
                        ...patch.global,
                        spread: parseFloat(e.target.value),
                      },
                    })
                  }
                />
              </Field>
            </Shell>
          </div>
        </div>
      </div>
    </div>
  );
}
