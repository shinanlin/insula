import { HGA_RADIUS_MAX, HGA_RADIUS_MIN } from '../constants/animation.js';

export function hgaToRadius(hga, scale, { active, selected, hovered }) {
  if (hga == null || !scale?.vmax) {
    return active ? 1.85 : selected ? 1.45 : hovered ? 1.2 : 0.95;
  }
  const vmin = scale.vmin ?? 0;
  const vmax = scale.vmax ?? 1;
  const normalized = vmax > vmin
    ? Math.max(0, Math.min(1, (Math.abs(hga) - vmin) / (vmax - vmin)))
    : 0;
  const compressed = Math.pow(normalized, 1.35);
  const base = HGA_RADIUS_MIN + compressed * (HGA_RADIUS_MAX - HGA_RADIUS_MIN);
  if (active) return base + 0.3;
  if (selected) return base + 0.18;
  if (hovered) return base + 0.12;
  return base;
}
