export function buildAnimationCacheKey(phase, selectedLoad, subjectsKey, electrodeSetKey) {
  return `${phase}|${selectedLoad}|${subjectsKey}|${electrodeSetKey}`;
}
