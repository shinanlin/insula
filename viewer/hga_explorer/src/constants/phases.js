export const PHASES = ['stimulus', 'delay', 'go', 'response'];

export const PHASE_LABELS = {
  stimulus: 'Stimulus',
  delay: 'Delay',
  go: 'Go',
  response: 'Response',
};

export const DEFAULT_VENN_PHASES = ['stimulus', 'delay', 'go', 'response'];

export const PHASE_TIME_START = Object.fromEntries(PHASES.map((phase) => [phase, -0.5]));

export const phaseTimeStart = (phase) => PHASE_TIME_START[phase] ?? -0.5;
