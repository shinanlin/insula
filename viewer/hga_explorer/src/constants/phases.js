export const PHASES = ['stimulus', 'delay', 'go', 'response'];

export const PHASE_LABELS = {
  stimulus: 'Stimulus',
  delay: 'Delay',
  go: 'Go',
  response: 'Response',
};

export const DEFAULT_VENN_PHASES = ['stimulus', 'delay', 'go', 'response'];

/** Task-specific phase sets for Venn / waveform (not all tasks have 4 phases). */
const TASK_VENN_PHASES = {
  LexicalNoDelay: ['stimulus', 'response'],
  PictureNaming: ['stimulus', 'delay', 'go', 'response'],
};

export function vennPhasesForTask(task, metadata = null) {
  if (task === 'all') return DEFAULT_VENN_PHASES;
  const fromMeta = metadata?.phases_by_task?.[task];
  if (fromMeta?.length) return [...fromMeta];
  return TASK_VENN_PHASES[task] || DEFAULT_VENN_PHASES;
}

export function phasesForTask(task, metadata = null) {
  if (task === 'all') return PHASES;
  const venn = vennPhasesForTask(task, metadata);
  return PHASES.filter((phase) => venn.includes(phase));
}

export const PHASE_TIME_START = Object.fromEntries(PHASES.map((phase) => [phase, -0.5]));

export const phaseTimeStart = (phase) => PHASE_TIME_START[phase] ?? -0.5;
