/** Merge Are.na resolve/infer patches without losing channel context. */

export type ArenaNodeData = Record<string, unknown>;

export function arenaDataFromResolve(resolved: {
  kind?: string;
  input?: string;
  mode?: string;
  block?: unknown;
  channel?: unknown;
  imageable_count?: number;
  cached?: boolean;
}): ArenaNodeData {
  const isChannel = resolved.kind === "channel";
  return {
    url: resolved.input,
    kind: resolved.kind,
    mode: resolved.mode,
    status: "ready" as const,
    error: null,
    block: resolved.block,
    channel: resolved.channel,
    imageable_count: resolved.imageable_count,
    cached: resolved.cached ?? false,
    ...(isChannel ? { channel_url: resolved.input } : {}),
  };
}

/**
 * When the node already has a channel_url, keep channel mode after inferring
 * a single block (which would otherwise overwrite kind/url and hide Shuffle).
 */
export function mergeArenaNodeData(
  prev: ArenaNodeData | undefined | null,
  patch: ArenaNodeData
): ArenaNodeData {
  const prevData = prev || {};
  const channelUrl = String(
    prevData.channel_url ||
      (prevData.kind === "channel" ? prevData.url : "") ||
      ""
  ).trim();

  if (!channelUrl) {
    return { ...prevData, ...patch };
  }

  return {
    ...prevData,
    ...patch,
    channel_url: channelUrl,
    url: channelUrl,
    kind: "channel",
    channel: patch.channel ?? prevData.channel,
    imageable_count:
      patch.imageable_count ?? prevData.imageable_count,
    // Keep the newly resolved/inferred block from the patch
    block: patch.block ?? prevData.block,
    mode: patch.mode ?? prevData.mode,
    status: patch.status ?? prevData.status,
    error: patch.error ?? null,
  };
}
