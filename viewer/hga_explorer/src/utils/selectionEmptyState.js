export function getSelectionEmptyState({
  selectedSubjectCount = 0,
  selectedRegionCount = 0,
  availableRoiCount = 0,
  enabledRoiCount = 0,
  visibleElectrodeCount = 0,
}) {
  if (selectedSubjectCount === 0) {
    return {
      code: 'no_subjects',
      title: 'No subjects selected',
      message: 'Select at least one subject in the left panel to load electrodes and traces.',
    };
  }

  if (selectedRegionCount === 0) {
    return {
      code: 'no_venn_region',
      title: 'No Venn region selected',
      message: 'Click a Venn component to choose which phase-overlap group to visualize.',
    };
  }

  if (availableRoiCount > 0 && enabledRoiCount === 0) {
    return {
      code: 'no_rois',
      title: 'All ROIs hidden',
      message: 'Enable at least one ROI in the right panel, or click Show all.',
    };
  }

  if (visibleElectrodeCount === 0) {
    return {
      code: 'no_electrodes',
      title: 'No electrodes in selection',
      message: 'Try a different Venn region, subject set, or ROI filter.',
    };
  }

  return null;
}

export function isSelectionEmpty(emptyState) {
  return emptyState != null;
}
