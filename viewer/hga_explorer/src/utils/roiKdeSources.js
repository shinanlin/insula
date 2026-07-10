import { KDE_ELECTRODE_MODE_MAX } from '../constants/brain.js';

export function buildKdeSources(electrodes, hgaValues) {
  if (!electrodes.length) {
    return { mode: 'electrode', sources: [], label: 'No electrodes' };
  }

  if (electrodes.length <= KDE_ELECTRODE_MODE_MAX) {
    return {
      mode: 'electrode',
      sources: electrodes.map((electrode, index) => ({
        id: electrode.id,
        roi: electrode.roi,
        x: electrode.x,
        y: electrode.y,
        z: electrode.z,
        hga: hgaValues[index] ?? 0,
      })),
      label: 'Electrode KDE',
    };
  }

  const grouped = new Map();
  electrodes.forEach((electrode, index) => {
    const hga = hgaValues[index];
    if (hga == null || !Number.isFinite(hga)) return;
    if (!grouped.has(electrode.roi)) {
      grouped.set(electrode.roi, {
        roi: electrode.roi,
        xs: [],
        ys: [],
        zs: [],
        hgaValues: [],
      });
    }
    const bucket = grouped.get(electrode.roi);
    bucket.xs.push(electrode.x);
    bucket.ys.push(electrode.y);
    bucket.zs.push(electrode.z);
    bucket.hgaValues.push(Math.abs(hga));
  });

  const sources = [...grouped.values()].map((bucket) => ({
    id: `roi:${bucket.roi}`,
    roi: bucket.roi,
    x: bucket.xs.reduce((sum, value) => sum + value, 0) / bucket.xs.length,
    y: bucket.ys.reduce((sum, value) => sum + value, 0) / bucket.ys.length,
    z: bucket.zs.reduce((sum, value) => sum + value, 0) / bucket.zs.length,
    hga: bucket.hgaValues.reduce((sum, value) => sum + value, 0) / bucket.hgaValues.length,
  }));

  return {
    mode: 'roi',
    sources,
    label: `ROI KDE (${sources.length} regions)`,
  };
}

export function kdeSourceHgaValues(sources) {
  return sources.map((source) => source.hga ?? 0);
}

export function kdeSourcesForInfluenceMap(sources) {
  return sources.map((source) => ({
    x: source.x,
    y: source.y,
    z: source.z,
  }));
}
