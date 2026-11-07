export const ATLAS_REGION_COLORS = {
  Insula: '#6A5ACD',
  'Frontal operculum': '#2369BD',
  Other: '#777777',
};

export const MANUAL_LABELS = ['Operculum', 'Anterior Insula'];

export const MANUAL_LABEL_COLORS = {
  Operculum: '#EA580C',
  'Anterior Insula': '#DB2777',
};

export function anatomyContactColor(nativeRegion, manualLabel) {
  if (manualLabel && MANUAL_LABEL_COLORS[manualLabel]) {
    return MANUAL_LABEL_COLORS[manualLabel];
  }
  return ATLAS_REGION_COLORS[nativeRegion] || ATLAS_REGION_COLORS.Other;
}

export function shortAparc(name) {
  return (name || 'Unknown')
    .replace(/^ctx_[lr]h_/, '')
    .replace(/^(Left|Right)-Cerebral-/, '');
}
