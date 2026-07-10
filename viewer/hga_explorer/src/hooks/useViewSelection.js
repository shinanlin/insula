import { useEffect, useMemo, useState } from 'react';
import {
  DEFAULT_VIEW_CONDITION,
  DEFAULT_VIEW_TASK,
} from '../constants/selection.js';
import { conditionsForTask } from '../utils/taskFilter.js';
import { buildViewSelection } from '../utils/viewSelection.js';

export default function useViewSelection(metadata) {
  const [selectedTask, setSelectedTask] = useState(DEFAULT_VIEW_TASK);
  const [selectedCondition, setSelectedCondition] = useState(DEFAULT_VIEW_CONDITION);

  const availableConditions = useMemo(
    () => conditionsForTask(metadata, selectedTask),
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

  const viewSelection = useMemo(
    () => buildViewSelection(selectedTask, selectedCondition),
    [selectedTask, selectedCondition],
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
  };

  const selectCondition = (condition) => {
    if (availableConditions.includes(condition)) {
      setSelectedCondition(condition);
    }
  };

  return {
    selectedTask,
    selectedCondition,
    availableConditions,
    viewSelection,
    selectTask,
    selectCondition,
  };
}
