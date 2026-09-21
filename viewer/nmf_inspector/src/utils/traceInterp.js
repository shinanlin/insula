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

export function windowMean(trace, t0, t1) {
  if (!trace?.time?.length || t1 <= t0) return null;
  const nSamples = Math.max(4, Math.ceil((t1 - t0) / 0.02));
  const samples = [];
  for (let i = 0; i <= nSamples; i += 1) {
    const t = t0 + (i / nSamples) * (t1 - t0);
    const value = interpolateTraceValue(trace, t);
    if (value != null && Number.isFinite(value)) samples.push(value);
  }
  if (!samples.length) return null;
  return samples.reduce((sum, value) => sum + value, 0) / samples.length;
}
