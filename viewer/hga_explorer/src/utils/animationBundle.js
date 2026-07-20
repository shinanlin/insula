import { buildViewSelection, parseViewSelection } from './viewSelection.js';
import { resolveTaskList } from '../constants/tasks.js';

export function animationLoadKey(viewSelection, metadata = null) {
  const { task, condition, modality } = parseViewSelection(viewSelection, metadata);
  return buildViewSelection(task, condition, modality, metadata);
}

export function animationLoadKeys(viewSelection, metadata = null) {
  const { task, condition, modality } = parseViewSelection(viewSelection, metadata);
  const keys = [buildViewSelection(task, condition, modality, metadata)];
  const legacyKey = `${task}|${condition}`;
  if (!keys.includes(legacyKey)) keys.push(legacyKey);
  if (modality) {
    const explicitModalityKey = `${task}|${condition}|${modality}`;
    if (!keys.includes(explicitModalityKey)) keys.push(explicitModalityKey);
  }
  return keys;
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

function bundleKeysForTaskCondition(task, condition, modality, metadata = null) {
  const keys = [buildViewSelection(task, condition, modality, metadata)];
  const legacyKey = `${task}|${condition}`;
  if (!keys.includes(legacyKey)) keys.push(legacyKey);
  if (modality) {
    const explicitModalityKey = `${task}|${condition}|${modality}`;
    if (!keys.includes(explicitModalityKey)) keys.push(explicitModalityKey);
  }
  return keys;
}

function taskHasCondition(metadata, task, condition) {
  const taskConditions = metadata?.conditions?.[task];
  return !taskConditions?.length || taskConditions.includes(condition);
}

function mergeCompactSelectionBundles(compacts) {
  const validCompacts = (compacts || []).filter((compact) => compact?.times?.length);
  if (!validCompacts.length) return null;
  if (validCompacts.length === 1) return validCompacts[0];

  const times = validCompacts[0].times;
  const electrodeIds = Array.from(new Set(
    validCompacts.flatMap((compact) => compact.electrode_ids || []),
  )).sort();

  const values = times.map((_, frameIndex) => (
    electrodeIds.map((electrodeId) => {
      let sum = 0;
      let count = 0;
      validCompacts.forEach((compact) => {
        const electrodeIndex = compact.electrode_ids?.indexOf(electrodeId) ?? -1;
        if (electrodeIndex < 0) return;
        const value = compact.values?.[frameIndex]?.[electrodeIndex];
        if (value != null && Number.isFinite(value)) {
          sum += value;
          count += 1;
        }
      });
      return count > 0 ? sum / count : null;
    })
  ));
  const finiteValues = values.flat().filter((value) => value != null && Number.isFinite(value));

  return {
    ...validCompacts[0],
    electrode_ids: electrodeIds,
    values,
    scale: {
      vmin: 0,
      vmax: percentile95(finiteValues),
      method: 'p95_abs_sliding_window_gaussian',
    },
  };
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

export function extractBundleForLoad(subjectPhasePayload, viewSelection, metadata = null) {
  const bundles = subjectPhasePayload?.bundles;
  if (!bundles) return null;
  const keys = animationLoadKeys(viewSelection, metadata);
  const matchedKey = keys.find((key) => bundles[key]);
  if (matchedKey) return bundles[matchedKey];

  const { task, condition, modality } = parseViewSelection(viewSelection, metadata);
  if (task !== 'all') return null;

  const taskCompacts = resolveTaskList(metadata)
    .filter((taskName) => taskHasCondition(metadata, taskName, condition))
    .map((taskName) => {
      const taskKeys = bundleKeysForTaskCondition(taskName, condition, modality, metadata);
      const taskKey = taskKeys.find((key) => bundles[key]);
      return taskKey ? bundles[taskKey] : null;
    })
    .filter((compact) => compact?.times?.length);

  return mergeCompactSelectionBundles(taskCompacts);
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
