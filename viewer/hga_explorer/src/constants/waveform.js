export const WAVEFORM_Y_MIN = -0.5;
export const WAVEFORM_Y_MAX = 1.0;

/** Shared y-axis for all phase panels. */
export function resolveWaveformYRange() {
  return [WAVEFORM_Y_MIN, WAVEFORM_Y_MAX];
}
