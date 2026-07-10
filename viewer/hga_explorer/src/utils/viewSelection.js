import { DEFAULT_VIEW_SELECTION } from '../constants/selection.js';
import { TASK_LABELS } from '../constants/tasks.js';

export function buildViewSelection(task, condition) {
  return `${task}|${condition}`;
}

export function parseViewSelection(selection = DEFAULT_VIEW_SELECTION) {
  const value = selection || DEFAULT_VIEW_SELECTION;
  if (!value.includes('|')) {
    return { task: 'PhonemeSequencing', condition: value };
  }
  const [task, condition] = value.split('|');
  return { task, condition };
}

export function formatViewSelectionLabel(selection = DEFAULT_VIEW_SELECTION) {
  const { task, condition } = parseViewSelection(selection);
  if (task === 'all') return `all tasks · ${condition}`;
  const taskLabel = TASK_LABELS[task] ?? task;
  return `${taskLabel} · ${condition}`;
}
