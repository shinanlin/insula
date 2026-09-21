const DEFAULT_TRACE_COLOR = '#64748b';

export function computeTraceYRange(trace) {
  if (!trace?.y?.length) return [-0.5, 1.5];
  let ymin = Infinity;
  let ymax = -Infinity;
  trace.y.forEach((value) => {
    if (value != null && Number.isFinite(value)) {
      ymin = Math.min(ymin, value);
      ymax = Math.max(ymax, value);
    }
  });
  if (!Number.isFinite(ymin)) return [-0.5, 1.5];
  const span = ymax - ymin;
  const pad = Math.max(span * 0.08, 0.08);
  return [ymin - pad, ymax + pad];
}

export function buildWaveformPlotData(trace, lineColor = DEFAULT_TRACE_COLOR) {
  return [{
    x: trace.x,
    y: trace.y,
    type: 'scatter',
    mode: 'lines',
    line: { color: lineColor || DEFAULT_TRACE_COLOR, width: 2 },
    hovertemplate: 't=%{x:.2f}s<br>HGA=%{y:.2f}<extra></extra>',
    showlegend: false,
  }];
}

export function traceFromExport(exported) {
  if (!exported?.time?.length) return null;
  return {
    x: exported.time,
    y: exported.value,
  };
}
