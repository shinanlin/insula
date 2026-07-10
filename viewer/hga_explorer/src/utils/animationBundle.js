import { parseViewSelection } from './viewSelection.js';

export function animationLoadKey(viewSelection) {
  const { task, condition } = parseViewSelection(viewSelection);
  if (task === 'all') return `all|${condition}`;
  return `${task}|${condition}`;
}

export function bundleHasPlayableFrames(bundle) {
  return Boolean(
    bundle?.frames?.length
    && bundle.frames.some((frame) => Object.keys(frame.hgaByElectrodeId || {}).length > 0),
  );
}

export function expandCompactBundle(compact) {
  if (!compact?.times?.length) {
    return { frames: [], scale: compact?.scale ?? { vmin: 0, vmax: 1 } };
  }
  const { times, electrode_ids: electrodeIds, values, scale } = compact;
  const frames = times.map((time, frameIndex) => {
    const hgaByElectrodeId = {};
    electrodeIds.forEach((electrodeId, electrodeIndex) => {
      const value = values[frameIndex]?.[electrodeIndex];
      if (value != null && Number.isFinite(value)) {
        hgaByElectrodeId[electrodeId] = value;
      }
    });
    return { time, hgaByElectrodeId };
  });
  return { frames, scale: scale ?? { vmin: 0, vmax: 1 } };
}

function percentile95(values) {
  if (!values.length) return 1;
  const sorted = values.map((value) => Math.abs(value)).sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1)));
  return sorted[index] > 0 ? sorted[index] : 1;
}

export function mergeCompactAnimationBundles(compacts, electrodeFilterSet) {
  if (!compacts.length) {
    return { frames: [], scale: { vmin: 0, vmax: 1, method: 'p95_abs_sliding_window_gaussian' } };
  }
  const times = compacts[0].times;
  const frames = times.map((time, frameIndex) => {
    const hgaByElectrodeId = {};
    compacts.forEach((compact) => {
      compact.electrode_ids.forEach((electrodeId, electrodeIndex) => {
        if (!electrodeFilterSet.has(electrodeId)) return;
        const value = compact.values[frameIndex]?.[electrodeIndex];
        if (value != null && Number.isFinite(value)) {
          hgaByElectrodeId[electrodeId] = value;
        }
      });
    });
    return { time, hgaByElectrodeId };
  });
  const smoothedValues = frames.flatMap((frame) => Object.values(frame.hgaByElectrodeId));
  return {
    frames,
    scale: {
      vmin: 0,
      vmax: percentile95(smoothedValues),
      method: 'p95_abs_sliding_window_gaussian',
    },
  };
}

export function extractBundleForLoad(subjectPhasePayload, viewSelection) {
  const loadKey = animationLoadKey(viewSelection);
  return subjectPhasePayload?.bundles?.[loadKey] ?? null;
}

export function compactToFrameHgaValues(compact, electrodeIds) {
  if (!compact?.times?.length) return [];
  const indexById = new Map(compact.electrode_ids.map((id, index) => [id, index]));
  return compact.times.map((_, frameIndex) => electrodeIds.map((electrodeId) => {
    const electrodeIndex = indexById.get(electrodeId);
    if (electrodeIndex == null) return 0;
    return compact.values[frameIndex]?.[electrodeIndex] ?? 0;
  }));
}
