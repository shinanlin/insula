import { PHASES } from '../constants/phases.js';
import { resolveTaskList } from '../constants/tasks.js';
import { effectiveModalityForTask } from './viewSelection.js';

function resolveConditionTrace(phaseMap, condition, taskName, modality, metadata) {
  const conditionMap = phaseMap?.[condition];
  if (!conditionMap) return null;
  if (conditionMap.time) return conditionMap;
  const mod = effectiveModalityForTask(taskName, modality, metadata);
  if (conditionMap[mod]?.time?.length) return conditionMap[mod];
  const nested = Object.values(conditionMap).find((value) => value?.time?.length);
  return nested ?? null;
}

function hasTraceForSelection(traces, electrodeId, task, condition, modality, metadata) {
  const electrodeTraces = traces?.[electrodeId];
  if (!electrodeTraces) return false;
  const taskKeys = task === 'all' ? resolveTaskList(metadata) : [task];
  return taskKeys.some((taskName) => {
    const phaseMap = electrodeTraces[taskName];
    if (!phaseMap) return false;
    return PHASES.some((phase) => {
      const trace = resolveConditionTrace(phaseMap[phase], condition, taskName, modality, metadata);
      return trace?.time?.length;
    });
  });
}

function isFiniteHga(value) {
  return value != null && Number.isFinite(value);
}

function hasPrecomputedHgaForTask(electrode, taskName, condition, modality, metadata) {
  const nestedModality = electrode?.hga_by_task_condition_modality?.[taskName]?.[condition];
  if (nestedModality) {
    const mod = effectiveModalityForTask(taskName, modality, metadata);
    if (isFiniteHga(nestedModality[mod])) return true;
  }

  const nested = electrode?.hga_by_task_condition?.[taskName]?.[condition];
  if (isFiniteHga(nested)) return true;

  if (condition === 'Repeat' && isFiniteHga(electrode?.hga_by_task?.[taskName])) {
    return true;
  }
  return false;
}

function hasPrecomputedHgaForSelection(electrode, task, condition, modality, metadata) {
  const taskKeys = task === 'all' ? resolveTaskList(metadata) : [task];
  return taskKeys.some((taskName) => (
    hasPrecomputedHgaForTask(electrode, taskName, condition, modality, metadata)
  ));
}

export function modalitiesForTask(metadata, task) {
  if (task === 'all') {
    const pnModalities = metadata?.modalities?.PictureNaming;
    return pnModalities?.length ? [...pnModalities] : [];
  }
  const taskModalities = metadata?.modalities?.[task];
  return taskModalities?.length ? [...taskModalities] : [];
}

export function electrodeCoversView(electrode, task, condition, traces = {}, modality = null, metadata = null) {
  const effectiveModality = modality ?? metadata?.default_modality ?? 'sound';
  return hasPrecomputedHgaForSelection(
    electrode,
    task,
    condition,
    effectiveModality,
    metadata,
  ) || hasTraceForSelection(
    traces,
    electrode.id,
    task,
    condition,
    effectiveModality,
    metadata,
  );
}

export function filterElectrodesForView(electrodes, task, condition, traces = {}, modality = null, metadata = null) {
  return (electrodes || []).filter((electrode) => (
    electrodeCoversView(electrode, task, condition, traces, modality, metadata)
  ));
}

export function electrodeCoversTask(electrode, task, traces = {}, metadata = null) {
  const conditions = conditionsForTask(metadata, task);
  return conditions.some((condition) => (
    electrodeCoversView(electrode, task, condition, traces, null, metadata)
  ));
}

export function filterElectrodesForTask(electrodes, task, traces = {}, metadata = null) {
  return (electrodes || []).filter((electrode) => electrodeCoversTask(electrode, task, traces, metadata));
}

export function conditionsForTask(metadata, task) {
  const byTask = metadata?.conditions || {};
  if (task === 'all') {
    const union = new Set();
    resolveTaskList(metadata).forEach((taskName) => {
      (byTask[taskName] || []).forEach((condition) => union.add(condition));
    });
    return [...union];
  }
  const taskConditions = byTask[task];
  if (taskConditions?.length) {
    return [...taskConditions];
  }
  return metadata?.default_condition ? [metadata.default_condition] : ['Repeat'];
}
