import { HGA_RADIUS_MIN, HGA_RADIUS_MAX } from '../constants/brain.js';
import { PHASES } from '../constants/phases.js';
import { TASKS } from '../constants/tasks.js';
import { resolvePhaseFlags } from './electrodeCoords.js';
import { interpolateTraceValue } from './traces.js';
import { parseViewSelection } from './viewSelection.js';

const DEFAULT_SIGNIFICANCE_WINDOWS = {
  stimulus: [0.0, 0.5],
  delay: [0.0, 0.5],
  go: [0.0, 0.5],
  response: [-0.5, 0.5],
};

function averageTaskValues(hgaByTask) {
  const values = Object.values(hgaByTask || {}).filter((value) => value != null && Number.isFinite(value));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function meanAbsInWindow(trace, t0, t1) {
  if (!trace?.time?.length || t1 <= t0) return null;
  const nSamples = Math.max(4, Math.ceil((t1 - t0) / 0.02));
  const samples = [];
  for (let i = 0; i <= nSamples; i += 1) {
    const t = t0 + (i / nSamples) * (t1 - t0);
    const value = interpolateTraceValue(trace, t);
    if (value != null && Number.isFinite(value)) samples.push(Math.abs(value));
  }
  if (!samples.length) return null;
  return samples.reduce((sum, value) => sum + value, 0) / samples.length;
}

function resolveHgaFromTraces(traces, electrode, task, condition, significanceWindows = DEFAULT_SIGNIFICANCE_WINDOWS) {
  const taskKeys = task === 'all' ? TASKS : [task];
  const values = [];
  taskKeys.forEach((taskName) => {
    const phaseMap = traces?.[electrode?.id]?.[taskName];
    if (!phaseMap) return;
    PHASES.forEach((phase) => {
      const phaseFlags = resolvePhaseFlags(electrode, task);
      if (phaseFlags && !phaseFlags[phase]) return;
      const trace = phaseMap[phase]?.[condition];
      const [t0, t1] = significanceWindows[phase] || DEFAULT_SIGNIFICANCE_WINDOWS[phase];
      const value = meanAbsInWindow(trace, t0, t1);
      if (value != null) values.push(value);
    });
  });
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function resolveHgaFromPrecomputed(electrode, task, condition) {
  const nested = electrode?.hga_by_task_condition;
  if (nested) {
    if (task === 'all') {
      const values = TASKS
        .map((taskName) => nested[taskName]?.[condition])
        .filter((value) => value != null && Number.isFinite(value));
      if (values.length) {
        return values.reduce((sum, value) => sum + value, 0) / values.length;
      }
    } else if (nested[task]?.[condition] != null) {
      return nested[task][condition];
    }
  }

  if (condition === 'Repeat') {
    if (task === 'all') {
      return averageTaskValues(electrode.hga_by_task);
    }
    return electrode.hga_by_task?.[task] ?? null;
  }
  return null;
}

export function resolveHgaMean(electrode, viewSelection, traces = null, significanceWindows = DEFAULT_SIGNIFICANCE_WINDOWS) {
  const { task, condition } = parseViewSelection(viewSelection);

  const fromPrecomputed = resolveHgaFromPrecomputed(electrode, task, condition);
  if (fromPrecomputed != null) return fromPrecomputed;

  if (traces) {
    const fromTraces = resolveHgaFromTraces(traces, electrode, task, condition, significanceWindows);
    if (fromTraces != null) return fromTraces;
  }

  if (task === 'all') {
    return averageTaskValues(electrode.hga_by_task);
  }
  return electrode.hga_by_task?.[task] ?? null;
}

export function hgaToRadius(hga, scale, { active, selected, hovered }) {
  if (hga == null || !scale?.vmax) {
    return active ? 2.4 : selected ? 1.8 : hovered ? 1.5 : 1.0;
  }
  const vmin = scale.vmin ?? 0;
  const vmax = scale.vmax ?? 1;
  const normalized = vmax > vmin
    ? Math.max(0, Math.min(1, (Math.abs(hga) - vmin) / (vmax - vmin)))
    : 0;
  const base = HGA_RADIUS_MIN + normalized * (HGA_RADIUS_MAX - HGA_RADIUS_MIN);
  if (active) return base + 0.6;
  if (selected) return base + 0.35;
  if (hovered) return base + 0.25;
  return base;
}
