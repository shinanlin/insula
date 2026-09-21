import { CLUSTERS } from '../constants.js';

/** top − second loading; clear winner when large. */
export const PURITY_PURE_MIN = 0.2;
export const PURITY_BORDERLINE_MIN = 0.1;

export const PURITY_TIERS = ['pure', 'borderline', 'mixed'];

export const PURITY_LABELS = {
  pure: 'pure',
  borderline: 'borderline',
  mixed: 'mixed',
};

export function loadingMargin(loadings) {
  if (!loadings) return null;
  const values = CLUSTERS
    .map((cluster) => loadings[cluster])
    .filter((value) => value != null && Number.isFinite(value))
    .sort((a, b) => b - a);
  if (values.length < 2) return values.length === 1 ? values[0] : null;
  return values[0] - values[1];
}

export function secondCluster(loadings) {
  if (!loadings) return null;
  const ranked = CLUSTERS
    .map((cluster) => ({ cluster, value: loadings[cluster] }))
    .filter((row) => row.value != null && Number.isFinite(row.value))
    .sort((a, b) => b.value - a.value);
  return ranked[1]?.cluster ?? null;
}

export function purityTier(margin) {
  if (margin == null || !Number.isFinite(margin)) return 'mixed';
  if (margin >= PURITY_PURE_MIN) return 'pure';
  if (margin >= PURITY_BORDERLINE_MIN) return 'borderline';
  return 'mixed';
}

export function electrodePurity(electrode) {
  const margin = loadingMargin(electrode?.loadings);
  return {
    margin,
    purity: purityTier(margin),
    secondCluster: secondCluster(electrode?.loadings),
  };
}
