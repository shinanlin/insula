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
const atlasElectrodeCache = new Map();

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.json();
}

export function isMultiAtlasManifest(manifest) {
  return manifest?.version === 2 && manifest?.layout === 'split-multi-atlas';
}

export function resolveDefaultAtlas(manifest) {
  if (!manifest) return 'hammers';
  if (isMultiAtlasManifest(manifest)) {
    return manifest.default_atlas || manifest.atlases?.[0] || 'hammers';
  }
  return 'aparc2009s';
}

export function listAvailableAtlases(manifest) {
  if (!manifest) return [];
  if (isMultiAtlasManifest(manifest)) {
    return manifest.atlases || Object.keys(manifest.atlas || {});
  }
  return ['aparc2009s'];
}

export function atlasLabel(manifest, atlasId) {
  if (isMultiAtlasManifest(manifest)) {
    return manifest.atlas?.[atlasId]?.label || atlasId;
  }
  return atlasId === 'aparc2009s' ? 'APARC' : atlasId;
}

function resolveSharedFiles(manifest) {
  if (isMultiAtlasManifest(manifest)) {
    return manifest.shared?.files || {};
  }
  return manifest.files || {};
}

function resolveElectrodesPath(manifest, atlasId) {
  if (isMultiAtlasManifest(manifest)) {
    return manifest.atlas?.[atlasId]?.files?.electrodes;
  }
  return manifest.files?.electrodes;
}

function dataPath(relativePath) {
  if (!relativePath) return null;
  return `/data/${relativePath}`;
}

export async function loadAtlasElectrodes(manifest, atlasId) {
  if (!manifest || !atlasId) {
    throw new Error('Manifest and atlas id are required');
  }
  const cacheKey = `${manifest.version || 1}:${atlasId}`;
  if (atlasElectrodeCache.has(cacheKey)) {
    return atlasElectrodeCache.get(cacheKey);
  }
  const relativePath = resolveElectrodesPath(manifest, atlasId);
  if (!relativePath) {
    throw new Error(`No electrodes path for atlas ${atlasId}`);
  }
  const payload = await fetchJson(dataPath(relativePath));
  const result = {
    electrodes: payload.electrodes || [],
    regions: payload.regions || [],
    atlasMetadata: isMultiAtlasManifest(manifest)
      ? (manifest.atlas?.[atlasId]?.metadata || {})
      : {},
  };
  atlasElectrodeCache.set(cacheKey, result);
  return result;
}

export async function loadViewerBootstrap({ onProgress, atlasId = null } = {}) {
  const report = (stage, completed, total = 2) => {
    onProgress?.({ stage, completed, total });
  };

  try {
    report('manifest', 0);
    const manifest = await fetchJson('/data/manifest.json');
    report('manifest', 1);

    const selectedAtlas = atlasId || resolveDefaultAtlas(manifest);
    const electrodesPath = resolveElectrodesPath(manifest, selectedAtlas);
    const electrodesPayload = electrodesPath
      ? await fetchJson(dataPath(electrodesPath))
      : await fetchJson(`/data/${manifest.files?.electrodes || 'electrodes.json'}`);

    report('electrodes', 2);

    const layout = manifest.layout || 'split';
    const atlasMetadata = isMultiAtlasManifest(manifest)
      ? (manifest.atlas?.[selectedAtlas]?.metadata || {})
      : {};

    return {
      layout,
      manifest,
      metadata: {
        ...manifest.metadata,
        ...atlasMetadata,
        atlas: selectedAtlas,
      },
      electrodes: electrodesPayload.electrodes,
      regions: electrodesPayload.regions || [],
      traces: {},
      dataSource: manifest.metadata?.source || 'export',
      selectedAtlas,
      availableAtlases: listAvailableAtlases(manifest),
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
      selectedAtlas: 'hammers',
      availableAtlases: ['hammers'],
    };
  }
}

export async function loadSubjectTraces(manifest, subject) {
  const sharedFiles = resolveSharedFiles(manifest);
  const traceRel = sharedFiles?.traces?.[subject] || manifest?.files?.traces?.[subject];
  if (!traceRel) return {};
  if (traceCache.has(subject)) return traceCache.get(subject);
  const payload = await fetchJson(dataPath(traceRel));
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
  const sharedFiles = resolveSharedFiles(manifest);
  const animRel = sharedFiles?.animation?.[subject]?.[phase]
    || manifest?.files?.animation?.[subject]?.[phase];
  if (!animRel) return null;
  const payload = await fetchJson(dataPath(animRel));
  animationCache.set(cacheKey, payload);
  return payload;
}

export function getManifestAnimationPath(manifest, subject, phase) {
  const sharedFiles = resolveSharedFiles(manifest);
  return sharedFiles?.animation?.[subject]?.[phase]
    ?? manifest?.files?.animation?.[subject]?.[phase]
    ?? null;
}

export function clearTraceCache() {
  traceCache.map.clear();
  animationCache.map.clear();
}

export function clearAtlasElectrodeCache() {
  atlasElectrodeCache.clear();
}
