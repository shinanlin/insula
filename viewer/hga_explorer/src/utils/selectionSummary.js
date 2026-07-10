export function buildSelectionSummary(selectedRegions, electrodeCount) {
  if (!selectedRegions?.length) {
    return {
      label: 'No Venn region selected',
      shortLabel: 'No Venn region selected',
      fullLabel: 'No Venn region selected',
      count: 0,
      regionCount: 0,
    };
  }

  const regionLabels = selectedRegions.map((region) => region.label);
  const fullLabel = regionLabels.join(' + ');
  const count = electrodeCount ?? 0;
  const regionCount = selectedRegions.length;
  const electrodeLabel = `${count} electrode${count === 1 ? '' : 's'}`;

  const shortLabel = regionCount === 1
    ? `${regionLabels[0]} · ${electrodeLabel}`
    : `${regionCount} regions · ${electrodeLabel}`;

  return {
    label: fullLabel,
    shortLabel,
    fullLabel,
    count,
    regionCount,
  };
}

export function formatWaveformTitle({
  summary,
  isSingleElectrode,
  electrode,
  conditionLabels = [],
  electrodeCount,
  insulaModeActive = false,
}) {
  const scopeLabel = insulaModeActive ? 'insula electrodes' : 'Selected region';
  const conditionSuffix = conditionLabels.length > 1
    ? 'all conditions'
    : (conditionLabels[0] || 'all conditions');

  if (isSingleElectrode) {
    const title = `${electrode.channel} · single electrode (${conditionSuffix})`;
    return {
      title,
      fullTitle: title,
    };
  }

  const shortLabel = insulaModeActive
    ? `Insula · ${electrodeCount} electrode${electrodeCount === 1 ? '' : 's'}`
    : (summary?.shortLabel || scopeLabel);
  const fullLabel = insulaModeActive
    ? `Insula electrodes (${electrodeCount})`
    : (summary?.fullLabel || shortLabel);
  const title = `${shortLabel} · mean ± SEM · ${conditionSuffix}`;
  const fullTitle = `${fullLabel} · mean ± SEM of ${electrodeCount} electrode${electrodeCount === 1 ? '' : 's'} · ${conditionSuffix}`;

  return { title, fullTitle };
}
