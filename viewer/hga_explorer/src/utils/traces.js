import { PHASES, phaseTimeStart, phasesForTask } from '../constants/phases.js';
import { PHASE_TIME_END, PHASE_TIME_RANGES } from '../constants/loads.js';
import { conditionColor, hexToRgba, phaseColor } from '../constants/colors.js';
import { resolvePhaseFlags } from './electrodeCoords.js';
import { buildViewSelection, effectiveModalityForTask, parseViewSelection } from './viewSelection.js';
import { conditionsForTask } from './taskFilter.js';
import { resolveTaskList } from '../constants/tasks.js';

export function interpolateTraceValue(trace, time) {
  if (!trace?.time?.length) return null;
  const times = trace.time;
  const values = trace.value ?? trace.y;
  if (!values?.length) return null;
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

function sameTimeGrid(sourceTimes, targetTimes) {
  if (sourceTimes.length !== targetTimes.length) return false;
  return sourceTimes.every((time, index) => time === targetTimes[index]);
}

function interpolateTraceValues(trace, targetTimes) {
  const times = trace?.time;
  const values = trace?.value ?? trace?.y;
  if (!times?.length || !values?.length || !targetTimes?.length) return [];
  if (sameTimeGrid(times, targetTimes)) return [...values];

  const lastIndex = times.length - 1;
  let sourceIndex = 0;
  return targetTimes.map((time) => {
    if (time <= times[0]) return values[0];
    if (time >= times[lastIndex]) return values[lastIndex];

    while (sourceIndex < lastIndex - 1 && time > times[sourceIndex + 1]) {
      sourceIndex += 1;
    }

    const t0 = times[sourceIndex];
    const t1 = times[sourceIndex + 1];
    const y0 = values[sourceIndex];
    const y1 = values[sourceIndex + 1];
    const span = t1 - t0;
    if (span === 0) return y0;
    return y0 + ((time - t0) / span) * (y1 - y0);
  });
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
  const sums = Array(times.length).fill(0);
  const counts = Array(times.length).fill(0);
  traces.forEach((trace) => {
    interpolateTraceValues(trace, times).forEach((value, index) => {
      if (value != null && Number.isFinite(value)) {
        sums[index] += value;
        counts[index] += 1;
      }
    });
  });
  const values = sums.map((sum, index) => (
    counts[index] > 0 ? sum / counts[index] : null
  ));
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

function resolveConditionTrace(conditionMap, taskName, modality, metadata) {
  if (!conditionMap) return null;
  if (conditionMap.time) return conditionMap;
  const mod = effectiveModalityForTask(taskName, modality, metadata);
  if (conditionMap[mod]?.time?.length) return conditionMap[mod];
  const nested = Object.values(conditionMap).find((value) => value?.time?.length);
  return nested ?? null;
}

function resolveElectrodeTaskTrace(electrodeTraces, task, phase, condition, modality, metadata) {
  if (!electrodeTraces) return null;
  if (task === 'all') {
    const taskTraces = resolveTaskList(metadata)
      .map((taskName) => resolveConditionTrace(
        electrodeTraces[taskName]?.[phase]?.[condition],
        taskName,
        modality,
        metadata,
      ))
      .filter((trace) => trace?.time?.length);
    return averageTraces(taskTraces);
  }
  return resolveConditionTrace(
    electrodeTraces[task]?.[phase]?.[condition],
    task,
    modality,
    metadata,
  );
}

export function resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock = false, metadata = null) {
  const { task, condition, modality } = parseViewSelection(viewSelection, metadata);
  const electrodeTraces = traces?.[electrode?.id];
  const resolved = resolveElectrodeTaskTrace(
    electrodeTraces,
    task,
    phase,
    condition,
    modality,
    metadata,
  );
  if (resolved) return resolved;
  if (!electrode || !shouldUseMockTraces(allowMock)) return null;
  const trace = makeTrace(electrode, phase);
  return { time: trace.x, value: trace.y };
}

function electrodeHasPhaseTrace(traces, electrode, phase, viewSelection, allowMock = false, metadata = null) {
  const trace = resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock, metadata);
  return Boolean(trace?.time?.length);
}

export function electrodesActiveInPhase(electrodes, phase, traces = null, viewSelection = 'all|Repeat', metadata = null) {
  const { task } = parseViewSelection(viewSelection, metadata);
  return (electrodes || []).filter((electrode) => {
    if (traces && electrodeHasPhaseTrace(traces, electrode, phase, viewSelection, false, metadata)) {
      return true;
    }
    return resolvePhaseFlags(electrode, task)?.[phase];
  });
}

export function averageElectrodePhaseTraces(traces, electrodes, phase, viewSelection, allowMock = false, metadata = null) {
  const activeElectrodes = electrodesActiveInPhase(electrodes, phase, traces, viewSelection, metadata);
  const electrodeTraces = activeElectrodes
    .map((electrode) => resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock, metadata))
    .filter((trace) => trace?.time?.length);
  if (electrodeTraces.length === 0) return null;
  if (electrodeTraces.length === 1) {
    return { time: electrodeTraces[0].time, value: electrodeTraces[0].value, sem: null };
  }

  const timeSet = new Set();
  electrodeTraces.forEach((trace) => trace.time.forEach((time) => timeSet.add(time)));
  const times = Array.from(timeSet).sort((a, b) => a - b);
  const sums = Array(times.length).fill(0);
  const sumSquares = Array(times.length).fill(0);
  const counts = Array(times.length).fill(0);

  electrodeTraces.forEach((trace) => {
    interpolateTraceValues(trace, times).forEach((value, index) => {
      if (value != null && Number.isFinite(value)) {
        sums[index] += value;
        sumSquares[index] += value ** 2;
        counts[index] += 1;
      }
    });
  });

  const values = sums.map((sum, index) => (
    counts[index] > 0 ? sum / counts[index] : null
  ));
  const sems = counts.map((count, index) => {
    if (count === 0) return null;
    if (count === 1) return 0;
    const mean = sums[index] / count;
    const variance = (sumSquares[index] - count * mean ** 2) / (count - 1);
    return Math.sqrt(Math.max(variance, 0)) / Math.sqrt(count);
  });
  return { time: times, value: values, sem: sems };
}

export function resolvePanelPhaseTrace(traces, electrodes, phase, viewSelection, electrode, allowMock = false, metadata = null) {
  if (electrode) {
    return resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock, metadata);
  }
  return averageElectrodePhaseTraces(traces, electrodes, phase, viewSelection, allowMock, metadata);
}

function electrodeHasConditionPhaseTrace(
  traces,
  electrode,
  task,
  phase,
  condition,
  modality = null,
  allowMock = false,
  metadata = null,
) {
  const viewSelection = buildViewSelection(task, condition, modality, metadata);
  return Boolean(resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock, metadata)?.time?.length);
}

export function conditionsPresentInCohort(
  traces,
  electrodes,
  task,
  metadata,
  phase = null,
  electrode = null,
  allowMock = false,
  modality = null,
) {
  const canonical = conditionsForTask(metadata, task);
  const present = new Set();
  const scanElectrodes = electrode ? [electrode] : (electrodes || []);
  const phases = phase ? [phase] : phasesForTask(task, metadata);

  scanElectrodes.forEach((item) => {
    canonical.forEach((condition) => {
      const hasTrace = phases.some((phaseName) => (
        electrodeHasConditionPhaseTrace(
          traces,
          item,
          task,
          phaseName,
          condition,
          modality,
          allowMock,
          metadata,
        )
      ));
      if (hasTrace) present.add(condition);
    });
  });

  return canonical.filter((condition) => present.has(condition));
}

export function resolvePanelPhaseTracesByCondition(
  traces,
  electrodes,
  phase,
  task,
  metadata,
  electrode = null,
  allowMock = false,
  modality = null,
) {
  const conditions = conditionsPresentInCohort(
    traces,
    electrodes,
    task,
    metadata,
    phase,
    electrode,
    allowMock,
    modality,
  );

  return conditions.map((condition) => {
    const viewSelection = buildViewSelection(task, condition, modality, metadata);
    const resolved = resolvePanelPhaseTrace(
      traces,
      electrodes,
      phase,
      viewSelection,
      electrode,
      allowMock,
      metadata,
    );
    return {
      condition,
      resolved,
    };
  }).filter((entry) => entry.resolved?.time?.length);
}

export function clipResolvedTraceToPhaseWindow(resolved, phase) {
  if (!resolved) return { x: [], y: [], upper: [], lower: [], sem: [] };
  const rawTrace = {
    x: resolved.time,
    y: resolved.value,
    sem: resolved.sem ?? null,
  };
  return clipTraceToPhaseWindow(rawTrace, phase);
}

export function buildPhaseSeriesList(
  traces,
  electrodes,
  phase,
  task,
  metadata,
  electrode,
  allowMock,
  modality = null,
) {
  return resolvePanelPhaseTracesByCondition(
    traces,
    electrodes,
    phase,
    task,
    metadata,
    electrode,
    allowMock,
    modality,
  ).map(({ condition, resolved }) => ({
    condition,
    trace: clipResolvedTraceToPhaseWindow(resolved, phase),
  })).filter((entry) => entry.trace.x.length > 0);
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

export function computeMultiTraceYRange(seriesList) {
  if (!seriesList?.length) return [-0.5, 1.5];
  let ymin = Infinity;
  let ymax = -Infinity;
  seriesList.forEach(({ trace }) => {
    const [lo, hi] = computeTraceYRange(trace);
    ymin = Math.min(ymin, lo);
    ymax = Math.max(ymax, hi);
  });
  if (!Number.isFinite(ymin)) return [-0.5, 1.5];
  return [ymin, ymax];
}

export function buildMultiConditionWaveformPlotData(seriesList, isAggregate, showLegend = false) {
  const multiCondition = seriesList.length > 1;
  const plotTraces = [];

  seriesList.forEach(({ condition, trace }) => {
    const color = conditionColor(condition);
    if (isAggregate && trace.upper?.length > 0) {
      plotTraces.push({
        x: [...trace.x, ...trace.x.slice().reverse()],
        y: [...trace.upper, ...trace.lower.slice().reverse()],
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(0,0,0,0)', width: 0 },
        fill: 'toself',
        fillcolor: hexToRgba(color, 0.18),
        hoverinfo: 'skip',
        showlegend: false,
        legendgroup: condition,
      });
    }
    plotTraces.push({
      x: trace.x,
      y: trace.y,
      type: 'scatter',
      mode: 'lines',
      name: condition,
      line: { color, width: 2 },
      hovertemplate: isAggregate && trace.sem?.length
        ? `${condition}<br>t=%{x:.2f}s<br>mean=%{y:.2f}<extra></extra>`
        : `${condition}<br>t=%{x:.2f}s<br>HGA=%{y:.2f}<extra></extra>`,
      showlegend: showLegend && multiCondition,
      legendgroup: condition,
    });
  });

  return plotTraces;
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

export function causalWindowMeanForElectrode(
  traces,
  electrode,
  phase,
  viewSelection,
  time,
  windowSec,
  allowMock = false,
  metadata = null,
) {
  const trace = resolvePhaseTrace(traces, electrode, phase, viewSelection, allowMock, metadata);
  if (!trace) return null;
  return windowMean(trace, time, time + windowSec);
}
