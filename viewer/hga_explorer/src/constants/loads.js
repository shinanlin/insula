import { PHASES, phaseTimeStart } from './phases.js';

export const PHASE_TIME_END = Object.fromEntries(PHASES.map((phase) => [phase, 1.0]));

export const PHASE_TIME_RANGES = Object.fromEntries(
  PHASES.map((phase) => [phase, { min: phaseTimeStart(phase), max: PHASE_TIME_END[phase] }]),
);
