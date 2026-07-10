import { TASKS } from '../constants/tasks.js';
import { PHASES } from '../constants/phases.js';

function hasTraceForSelection(traces, electrodeId, task, condition) {
  const electrodeTraces = traces?.[electrodeId];
  if (!electrodeTraces) return false;
  const taskKeys = task === 'all' ? TASKS : [task];
  return taskKeys.some((taskName) => {
    const phaseMap = electrodeTraces[taskName];
    if (!phaseMap) return false;
    return PHASES.some((phase) => phaseMap[phase]?.[condition]?.time?.length);
  });
}

export function electrodeCoversView(electrode, task, condition, traces = {}) {
  if (condition !== 'Repeat') {
    return hasTraceForSelection(traces, electrode.id, task, condition);
  }
  if (task === 'all') {
    return TASKS.some((taskName) => electrode.hga_by_task?.[taskName] != null)
      || hasTraceForSelection(traces, electrode.id, 'all', condition);
  }
  return electrode.hga_by_task?.[task] != null
    || hasTraceForSelection(traces, electrode.id, task, condition);
}

export function filterElectrodesForView(electrodes, task, condition, traces = {}) {
  return (electrodes || []).filter((electrode) => electrodeCoversView(electrode, task, condition, traces));
}

export function electrodeCoversTask(electrode, task, traces = {}, metadata = null) {
  const conditions = conditionsForTask(metadata, task);
  return conditions.some((condition) => electrodeCoversView(electrode, task, condition, traces));
}

export function filterElectrodesForTask(electrodes, task, traces = {}, metadata = null) {
  return (electrodes || []).filter((electrode) => electrodeCoversTask(electrode, task, traces, metadata));
}

export function conditionsForTask(metadata, task) {
  const byTask = metadata?.conditions || {};
  if (task === 'all') {
    const union = new Set();
    TASKS.forEach((taskName) => {
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
