/** Insula parcel patterns — keep in sync with export/insula_constants.py */

import { BRAIN_SPACES } from '../utils/electrodeCoords.js';

export const TEMPLATE_INSULA_MESH_URL = '/assets/cvs_avg35_insula_pial.glb';
export const TEMPLATE_INSULA_MASK_URL = '/assets/cvs_avg35_pial_insula_mask.json';
export const TEMPLATE_INSULA_META_URL = '/assets/cvs_avg35_insula.meta.json';

/** Back-compat aliases */
export const INSULA_MESH_URL = TEMPLATE_INSULA_MESH_URL;
export const INSULA_MASK_URL = TEMPLATE_INSULA_MASK_URL;
export const INSULA_META_URL = TEMPLATE_INSULA_META_URL;

/** fig2: Brain(..., alpha=0.05) */
export const INSULA_GHOST_OPACITY = 0.05;
/** fig2: add_label(..., alpha=0.6) */
export const INSULA_HIGHLIGHT_OPACITY = 0.6;

export const APARC_INSULA_ROIS = new Set(['INS', 'Insula']);

export const HAMMERS_INSULA_ROIS = new Set(['AIC', 'PIC']);

export const INSULA_PATTERNS = [
  'G_insular_short',
  'G_Ins_lg_and_S_cent_ins',
  'S_circular_insula_ant',
  'S_circular_insula_inf',
  'S_circular_insula_sup',
];

export function nativeInsulaMeshUrl(subject) {
  if (!subject) return null;
  return `/assets/native/${subject}_insula_pial.glb`;
}

export function nativeInsulaMaskUrl(subject) {
  if (!subject) return null;
  return `/assets/native/${subject}_pial_insula_mask.json`;
}

export function nativeInsulaMetaUrl(subject) {
  if (!subject) return null;
  return `/assets/native/${subject}_insula.meta.json`;
}

export function resolveInsulaAssets({ brainSpace = BRAIN_SPACES.template, subject = null } = {}) {
  if (brainSpace === BRAIN_SPACES.native && subject) {
    return {
      meshUrl: nativeInsulaMeshUrl(subject),
      maskUrl: nativeInsulaMaskUrl(subject),
      metaUrl: nativeInsulaMetaUrl(subject),
    };
  }
  return {
    meshUrl: TEMPLATE_INSULA_MESH_URL,
    maskUrl: TEMPLATE_INSULA_MASK_URL,
    metaUrl: TEMPLATE_INSULA_META_URL,
  };
}

export function isInsulaLabel(label) {
  const value = String(label || '').trim();
  if (!value) return false;
  return INSULA_PATTERNS.some((pattern) => value.includes(pattern));
}

export function electrodeInInsula(electrode, atlas = 'hammers') {
  if (!electrode) return false;
  if (atlas === 'hammers') {
    if (!HAMMERS_INSULA_ROIS.has(electrode.roi)) return false;
    if (electrode.mix) return false;
    return true;
  }
  if (APARC_INSULA_ROIS.has(electrode.roi)) return true;
  return isInsulaLabel(electrode.label);
}
