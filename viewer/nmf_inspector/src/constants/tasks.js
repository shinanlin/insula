/** Tasks included in the NMF inspector (LexicalNoDelay excluded). */
export const INSPECTOR_TASKS = [
  'LexicalDelay',
  'PhonemeSequence',
  'PictureNaming',
  'SentenceRep',
];

export const TASK_LABELS = {
  LexicalDelay: 'Lexical (delay)',
  PhonemeSequence: 'Phoneme',
  PictureNaming: 'Picture naming',
  SentenceRep: 'Sentence rep',
};

export function resolveInspectorTasks(metadata) {
  const fromManifest = metadata?.tasks ?? INSPECTOR_TASKS;
  return INSPECTOR_TASKS.filter((task) => fromManifest.includes(task));
}

export function taskLabel(task) {
  return TASK_LABELS[task] ?? task;
}
