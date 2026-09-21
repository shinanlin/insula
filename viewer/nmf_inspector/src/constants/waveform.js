export const PHASES = ['stimulus', 'delay', 'go', 'response'];

export const PHASE_LABELS = {
  stimulus: 'Stimulus',
  delay: 'Delay',
  go: 'Go',
  response: 'Response',
};

/** Export uses Title-case keys; explorer uses lowercase. */
export const EXPORT_PHASE_TO_KEY = {
  Stimulus: 'stimulus',
  Delay: 'delay',
  Go: 'go',
  Response: 'response',
};

export const PHASE_TO_EXPORT_KEY = Object.fromEntries(
  Object.entries(EXPORT_PHASE_TO_KEY).map(([exportKey, phaseKey]) => [phaseKey, exportKey]),
);

export function normalizePhaseKey(phase) {
  return EXPORT_PHASE_TO_KEY[phase] ?? String(phase || '').toLowerCase();
}

export const PHASE_TIME_START = Object.fromEntries(PHASES.map((phase) => [phase, -0.5]));
export const PHASE_TIME_END = Object.fromEntries(PHASES.map((phase) => [phase, 1.0]));
export const PHASE_TIME_RANGES = Object.fromEntries(
  PHASES.map((phase) => [phase, { min: PHASE_TIME_START[phase], max: PHASE_TIME_END[phase] }]),
);

export const WAVEFORM_PLOT_MIN_HEIGHT = 128;
