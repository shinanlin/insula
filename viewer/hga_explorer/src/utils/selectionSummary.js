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
  loadLabel,
  electrodeCount,
}) {
  if (isSingleElectrode) {
    return {
      title: `${electrode.channel} · single electrode (${loadLabel})`,
      fullTitle: `${electrode.channel} · single electrode (${loadLabel})`,
    };
  }

  const shortLabel = summary?.shortLabel || 'Selected region';
  const fullLabel = summary?.fullLabel || shortLabel;
  const title = `${shortLabel} · mean ± SEM (${loadLabel})`;
  const fullTitle = `${fullLabel} · mean ± SEM of ${electrodeCount} electrode${electrodeCount === 1 ? '' : 's'} (${loadLabel})`;

  return { title, fullTitle };
}
