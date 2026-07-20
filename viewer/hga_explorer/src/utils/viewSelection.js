import {
  DEFAULT_VIEW_CONDITION,
  DEFAULT_VIEW_MODALITY,
  DEFAULT_VIEW_SELECTION,
} from '../constants/selection.js';
import { TASK_LABELS } from '../constants/tasks.js';

export function taskHasModalities(task, metadata) {
  return (metadata?.modalities?.[task]?.length ?? 0) > 1;
}

export function defaultModalityForTask(task, metadata) {
  const taskModalities = metadata?.modalities?.[task];
  if (!taskModalities?.length) return metadata?.default_modality || DEFAULT_VIEW_MODALITY;
  const preferred = metadata?.default_modality || DEFAULT_VIEW_MODALITY;
  return taskModalities.includes(preferred) ? preferred : taskModalities[0];
}

export function buildViewSelection(task, condition, modality, metadata = null) {
  if (taskHasModalities(task, metadata)) {
    return `${task}|${condition}|${modality || defaultModalityForTask(task, metadata)}`;
  }
  if (task === 'all' && metadata?.modalities?.PictureNaming?.length > 1) {
    return `all|${condition}|${modality || metadata?.default_modality || DEFAULT_VIEW_MODALITY}`;
  }
  return `${task}|${condition}`;
}

export function parseViewSelection(selection = DEFAULT_VIEW_SELECTION, metadata = null) {
  const value = selection || DEFAULT_VIEW_SELECTION;
  if (!value.includes('|')) {
    return {
      task: 'PhonemeSequence',
      condition: value,
      modality: metadata?.default_modality || DEFAULT_VIEW_MODALITY,
    };
  }
  const parts = value.split('|');
  if (parts.length === 2) {
    const [task, condition] = parts;
    return {
      task,
      condition,
      modality: defaultModalityForTask(task, metadata),
    };
  }
  const [task, condition, modality] = parts;
  return { task, condition, modality };
}

export function formatViewSelectionLabel(selection = DEFAULT_VIEW_SELECTION, metadata = null) {
  const { task, condition, modality } = parseViewSelection(selection, metadata);
  if (task === 'all') {
    const modalitySuffix = metadata?.modalities?.PictureNaming?.length > 1
      ? ` · ${modality}`
      : '';
    return `all tasks · ${condition}${modalitySuffix}`;
  }
  const taskLabel = TASK_LABELS[task] ?? task;
  if (taskHasModalities(task, metadata)) {
    return `${taskLabel} · ${condition} · ${modality}`;
  }
  return `${taskLabel} · ${condition}`;
}

export function effectiveModalityForTask(taskName, modality, metadata) {
  if (taskName === 'PictureNaming' || taskHasModalities(taskName, metadata)) {
    return modality || defaultModalityForTask(taskName, metadata);
  }
  return defaultModalityForTask(taskName, metadata);
}
