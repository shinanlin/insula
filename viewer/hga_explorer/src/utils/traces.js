import { PHASES, phaseTimeStart } from '../constants/phases.js';
import { PHASE_TIME_END, PHASE_TIME_RANGES } from '../constants/loads.js';
import { hexToRgba, phaseColor } from '../constants/colors.js';
import { resolvePhaseFlags } from './electrodeCoords.js';
import { parseViewSelection } from './viewSelection.js';

export function interpolateTraceValue(trace, time) {
  if (!trace?.time?.length) return null;
  const times = trace.time;
  const values = trace.value ?? trace.y;
  if (time <= times[0]) return values[0];
  if (time >= times[times.length - 1]) return values[times.length - 1];
  for (let i = 0; i < times.length - 1; i += 1) {
    if (time >= times[i] && time <= times[i + 1]) {
      const span = times[i + 1] - times[i];
      if (span === 0) return values[i];
      const weight = (time - times[i]) / span;
      return values[i] + weight * (values[i + 1] - values[i]);
    }
  }
  return null;
}

function averageTraces(traceList) {
  const traces = (traceList || []).filter((trace) => trace?.time?.length);
  if (!traces.length) return null;
  if (traces.length === 1) {
    return { time: traces[0].time, value: traces[0].value };
  }
  const timeSet = new Set();
  traces.forEach((trace) => trace.time.forEach((time) => timeSet.add(time)));
  const times = Array.from(timeSet).sort((a, b) => a - b);
  const values = times.map((time) => {
    const samples = traces
      .map((trace) => interpolateTraceValue(trace, time))
      .filter((value) => value != null);
    if (!samples.length) return null;
    return samples.reduce((sum, value) => sum + value, 0) / samples.length;
  });
  return { time: times, value: values };
}

function makeTrace(electrode, phase) {
  const phaseIndex = PHASES.indexOf(phase);
  const end = PHASE_TIME_END[phase];
  const start = phaseTimeStart(phase);
  const n = Math.max(60, Math.round((end - start) * 40));
  const active = electrode.phase_flags?.[phase];
  const roiSeed = electrode.roi.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % 9;
  const x = Array.from({ length: n }, (_, i) => start + (i / (n - 1)) * (end - start));
  const y = x.map((t, i) => {
    const bump = active ? Math.exp(-Math.pow((t - end * 0.35) / Math.max(end * 0.22, 0.18), 2)) : 0.15;
    const oscillation = 0.12 * Math.sin(i / 7 + roiSeed + phaseIndex);
    const baseline = 0.04 * phaseIndex;
    return Number((baseline + (active ? 0.85 : 0.08) * bump + oscillation).toFixed(3));
  });
  return { x, y };
}

function shouldUseMockTraces(allowMock) {
  return allowMock === true;
}

function resolveElectrodeTaskTrace(electrodeTraces, task, phase, condition) {
  if (!electrodeTraces) return null;
  if (task === 'all') {
    const taskTraces = Object.values(electrodeTraces)
      .map((taskData) => taskData?.[phase]?.[condition])
      .filter((trace) => trace?.time?.length);
    return averageTraces(taskTraces);
  }
  return electrodeTraces[task]?.[phase]?.[condition] ?? null;
}

export function resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock = false) {
  const { task, condition } = parseViewSelection(viewSelection);
  const electrodeTraces = traces?.[electrode?.id];
  const resolved = resolveElectrodeTaskTrace(electrodeTraces, task, phase, condition);
  if (resolved) return resolved;
  if (!electrode || !shouldUseMockTraces(allowMock)) return null;
  const trace = makeTrace(electrode, phase);
  return { time: trace.x, value: trace.y };
}

function meanAndSem(samples) {
  if (samples.length === 0) return { mean: null, sem: null };
  if (samples.length === 1) return { mean: samples[0], sem: 0 };
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (samples.length - 1);
  return { mean, sem: Math.sqrt(variance) / Math.sqrt(samples.length) };
}

function electrodeHasPhaseTrace(traces, electrode, phase, viewSelection, allowMock = false) {
  const trace = resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock);
  return Boolean(trace?.time?.length);
}

export function electrodesActiveInPhase(electrodes, phase, traces = null, viewSelection = 'all|Repeat') {
  const { task } = parseViewSelection(viewSelection);
  return (electrodes || []).filter((electrode) => {
    if (traces && electrodeHasPhaseTrace(traces, electrode, phase, viewSelection)) {
      return true;
    }
    return resolvePhaseFlags(electrode, task)?.[phase];
  });
}

export function averageElectrodePhaseTraces(traces, electrodes, phase, viewSelection, allowMock = false) {
  const activeElectrodes = electrodesActiveInPhase(electrodes, phase, traces, viewSelection);
  const electrodeTraces = activeElectrodes
    .map((electrode) => resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock))
    .filter((trace) => trace?.time?.length);
  if (electrodeTraces.length === 0) return null;
  if (electrodeTraces.length === 1) {
    return { time: electrodeTraces[0].time, value: electrodeTraces[0].value, sem: null };
  }

  const timeSet = new Set();
  electrodeTraces.forEach((trace) => trace.time.forEach((time) => timeSet.add(time)));
  const times = Array.from(timeSet).sort((a, b) => a - b);
  const values = [];
  const sems = [];
  times.forEach((time) => {
    const samples = electrodeTraces
      .map((trace) => interpolateTraceValue(trace, time))
      .filter((value) => value != null);
    const stats = meanAndSem(samples);
    values.push(stats.mean);
    sems.push(stats.sem);
  });
  return { time: times, value: values, sem: sems };
}

export function resolvePanelPhaseTrace(traces, electrodes, phase, viewSelection, electrode, allowMock = false) {
  if (electrode) {
    return resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock);
  }
  return averageElectrodePhaseTraces(traces, electrodes, phase, viewSelection, allowMock);
}

export function clipTraceToPhaseWindow(trace, phase) {
  if (!trace?.x?.length) return { x: [], y: [], upper: [], lower: [], sem: [] };
  const { min, max } = PHASE_TIME_RANGES[phase];
  const clipped = { x: [], y: [], upper: [], lower: [], sem: [] };
  trace.x.forEach((time, index) => {
    if (time >= min && time <= max) {
      clipped.x.push(time);
      clipped.y.push(trace.y[index]);
      const sem = trace.sem?.[index] ?? null;
      if (sem != null) {
        clipped.sem.push(sem);
        clipped.upper.push(trace.y[index] + sem);
        clipped.lower.push(trace.y[index] - sem);
      }
    }
  });
  return clipped;
}

export function computeTraceYRange(trace) {
  if (!trace?.y?.length) return [-0.5, 1.5];
  let ymin = Infinity;
  let ymax = -Infinity;
  const consider = (value) => {
    if (value != null && Number.isFinite(value)) {
      ymin = Math.min(ymin, value);
      ymax = Math.max(ymax, value);
    }
  };
  trace.y.forEach(consider);
  trace.upper?.forEach(consider);
  trace.lower?.forEach(consider);
  if (!Number.isFinite(ymin)) return [-0.5, 1.5];
  const span = ymax - ymin;
  const pad = Math.max(span * 0.08, 0.08);
  return [ymin - pad, ymax + pad];
}

export function buildWaveformPlotData(trace, phase, isAggregate) {
  const color = phaseColor(phase);
  const traces = [];
  if (isAggregate && trace.upper.length > 0) {
    traces.push({
      x: [...trace.x, ...trace.x.slice().reverse()],
      y: [...trace.upper, ...trace.lower.slice().reverse()],
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(0,0,0,0)', width: 0 },
      fill: 'toself',
      fillcolor: hexToRgba(color, 0.22),
      hoverinfo: 'skip',
      showlegend: false,
    });
  }
  traces.push({
    x: trace.x,
    y: trace.y,
    type: 'scatter',
    mode: 'lines',
    line: { color, width: 2 },
    hovertemplate: isAggregate && trace.sem.length
      ? 't=%{x:.2f}s<br>mean=%{y:.2f}<extra></extra>'
      : 't=%{x:.2f}s<br>HGA=%{y:.2f}<extra></extra>',
    showlegend: false,
  });
  return traces;
}

export function buildPlaybackVLine(time) {
  if (time == null || !Number.isFinite(time)) return [];
  return [{
    type: 'line',
    xref: 'x',
    yref: 'paper',
    x0: time,
    x1: time,
    y0: 0,
    y1: 1,
    line: {
      color: '#334155',
      width: 1.5,
      dash: 'dot',
    },
    layer: 'above',
  }];
}

export function windowMean(trace, t0, t1) {
  if (!trace?.time?.length || t1 <= t0) return null;
  const nSamples = Math.max(4, Math.ceil((t1 - t0) / 0.015625));
  const samples = [];
  for (let i = 0; i <= nSamples; i += 1) {
    const t = t0 + (i / nSamples) * (t1 - t0);
    const value = interpolateTraceValue(trace, t);
    if (value != null && Number.isFinite(value)) samples.push(value);
  }
  if (!samples.length) return null;
  return samples.reduce((sum, value) => sum + value, 0) / samples.length;
}

export function causalWindowMeanForElectrode(traces, electrode, phase, viewSelection, time, windowSec, allowMock = false) {
  const trace = resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock);
  if (!trace) return null;
  return windowMean(trace, time, time + windowSec);
}
