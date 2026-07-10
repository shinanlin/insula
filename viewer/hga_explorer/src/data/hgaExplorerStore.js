const MOCK_URL = '/data/hga_explorer_mock.json';
const TRACE_CACHE_MAX = 48;

class LruCache {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.map = new Map();
  }

  get(key) {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.maxSize) {
      const oldest = this.map.keys().next().value;
      this.map.delete(oldest);
    }
  }

  has(key) {
    return this.map.has(key);
  }
}

const traceCache = new LruCache(TRACE_CACHE_MAX);
const animationCache = new LruCache(TRACE_CACHE_MAX * 4);

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.json();
}

export async function loadViewerBootstrap({ onProgress } = {}) {
  const report = (stage, completed, total = 2) => {
    onProgress?.({ stage, completed, total });
  };

  try {
    report('manifest', 0);
    const manifest = await fetchJson('/data/manifest.json');
    report('manifest', 1);
    const electrodesPayload = await fetchJson(`/data/${manifest.files.electrodes}`);
    report('electrodes', 2);
    return {
      layout: manifest.layout || 'split',
      manifest,
      metadata: manifest.metadata,
      electrodes: electrodesPayload.electrodes,
      regions: electrodesPayload.regions || [],
      traces: {},
      dataSource: manifest.metadata?.source || 'export',
    };
  } catch {
    report('mock', 0);
    const payload = await fetchJson(MOCK_URL);
    report('mock', 2);
    return {
      layout: 'mock',
      manifest: null,
      metadata: payload.metadata,
      electrodes: payload.electrodes,
      regions: payload.regions || [],
      traces: payload.traces || {},
      dataSource: payload.metadata?.source || 'mock',
    };
  }
}

export async function loadSubjectTraces(manifest, subject) {
  if (!manifest?.files?.traces?.[subject]) return {};
  if (traceCache.has(subject)) return traceCache.get(subject);
  const payload = await fetchJson(`/data/${manifest.files.traces[subject]}`);
  const traces = payload.traces || {};
  traceCache.set(subject, traces);
  return traces;
}

export async function loadTracesForSubjects(manifest, subjects, existingTraces = {}, onProgress) {
  if (!manifest) {
    onProgress?.({ completed: 0, total: 0, progress: 1 });
    return existingTraces;
  }

  const merged = { ...existingTraces };
  const total = subjects.length;
  let completed = 0;

  const report = () => {
    onProgress?.({
      completed,
      total,
      progress: total > 0 ? completed / total : 1,
    });
  };

  report();

  if (!total) {
    return merged;
  }

  await Promise.all(subjects.map(async (subject) => {
    try {
      const subjectTraces = await loadSubjectTraces(manifest, subject);
      Object.assign(merged, subjectTraces);
    } finally {
      completed += 1;
      report();
    }
  }));

  return merged;
}

export async function loadSubjectPhaseAnimation(manifest, subject, phase) {
  const cacheKey = `${subject}:${phase}`;
  if (animationCache.has(cacheKey)) return animationCache.get(cacheKey);
  const path = manifest?.files?.animation?.[subject]?.[phase];
  if (!path) return null;
  const payload = await fetchJson(`/data/${path}`);
  animationCache.set(cacheKey, payload);
  return payload;
}

export function getManifestAnimationPath(manifest, subject, phase) {
  return manifest?.files?.animation?.[subject]?.[phase] ?? null;
}

export function clearTraceCache() {
  traceCache.map.clear();
  animationCache.map.clear();
}
