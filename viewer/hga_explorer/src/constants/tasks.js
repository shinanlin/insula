export const TASKS = [
  'PhonemeSequence',
  'LexicalDelay',
  'LexicalNoDelay',
  'PictureNaming',
];

export const TASK_OPTIONS = [...TASKS, 'all'];

export const TASK_LABELS = {
  PhonemeSequence: 'Phoneme',
  LexicalDelay: 'Lexical',
  LexicalNoDelay: 'Lexical (no delay)',
  PictureNaming: 'Picture naming',
  all: 'All',
};

export function resolveTaskList(metadata) {
  return metadata?.tasks?.length ? metadata.tasks : TASKS;
}

export function resolveTaskOptions(metadata) {
  return [...resolveTaskList(metadata), 'all'];
}
