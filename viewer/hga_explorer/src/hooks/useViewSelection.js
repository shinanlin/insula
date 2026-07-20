import { useEffect, useMemo, useState } from 'react';
import {
  DEFAULT_VIEW_CONDITION,
  DEFAULT_VIEW_MODALITY,
  DEFAULT_VIEW_TASK,
} from '../constants/selection.js';
import { conditionsForTask, modalitiesForTask } from '../utils/taskFilter.js';
import {
  buildViewSelection,
  defaultModalityForTask,
} from '../utils/viewSelection.js';

export default function useViewSelection(metadata) {
  const [selectedTask, setSelectedTask] = useState(DEFAULT_VIEW_TASK);
  const [selectedCondition, setSelectedCondition] = useState(DEFAULT_VIEW_CONDITION);
  const [selectedModality, setSelectedModality] = useState(DEFAULT_VIEW_MODALITY);

  const availableConditions = useMemo(
    () => conditionsForTask(metadata, selectedTask),
    [metadata, selectedTask],
  );

  const availableModalities = useMemo(
    () => modalitiesForTask(metadata, selectedTask),
    [metadata, selectedTask],
  );

  useEffect(() => {
    if (!availableConditions.length) return;
    if (!availableConditions.includes(selectedCondition)) {
      const fallback = metadata?.default_condition && availableConditions.includes(metadata.default_condition)
        ? metadata.default_condition
        : availableConditions[0];
      setSelectedCondition(fallback);
    }
  }, [availableConditions, selectedCondition, metadata?.default_condition]);

  useEffect(() => {
    if (availableModalities.length <= 1) return;
    if (!availableModalities.includes(selectedModality)) {
      const fallback = defaultModalityForTask(selectedTask, metadata);
      setSelectedModality(
        availableModalities.includes(fallback) ? fallback : availableModalities[0],
      );
    }
  }, [availableModalities, selectedModality, selectedTask, metadata]);

  const viewSelection = useMemo(
    () => buildViewSelection(selectedTask, selectedCondition, selectedModality, metadata),
    [selectedTask, selectedCondition, selectedModality, metadata],
  );

  const selectTask = (task) => {
    setSelectedTask(task);
    const nextConditions = conditionsForTask(metadata, task);
    setSelectedCondition((current) => (
      nextConditions.includes(current)
        ? current
        : (metadata?.default_condition && nextConditions.includes(metadata.default_condition)
          ? metadata.default_condition
          : nextConditions[0] || DEFAULT_VIEW_CONDITION)
    ));
    const nextModalities = modalitiesForTask(metadata, task);
    if (nextModalities.length > 1) {
      const fallback = defaultModalityForTask(task, metadata);
      setSelectedModality(
        nextModalities.includes(fallback) ? fallback : nextModalities[0],
      );
    }
  };

  const selectCondition = (condition) => {
    if (availableConditions.includes(condition)) {
      setSelectedCondition(condition);
    }
  };

  const selectModality = (modality) => {
    if (availableModalities.includes(modality)) {
      setSelectedModality(modality);
    }
  };

  return {
    selectedTask,
    selectedCondition,
    selectedModality,
    availableConditions,
    availableModalities,
    viewSelection,
    selectTask,
    selectCondition,
    selectModality,
  };
}
