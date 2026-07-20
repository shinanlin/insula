import {
  animationLoadKey,
  expandCompactBundle,
  extractBundleForLoad,
  mergeCompactAnimationBundles,
} from '../utils/animationBundle.js';
import { loadSubjectPhaseAnimation } from '../data/hgaExplorerStore.js';

let mergeWorker = null;

function getMergeWorker() {
  if (typeof Worker === 'undefined') return null;
  if (!mergeWorker) {
    mergeWorker = new Worker(new URL('../workers/mergeAnimation.worker.js', import.meta.url), {
      type: 'module',
    });
  }
  return mergeWorker;
}

function mergeWithWorker(compacts, electrodeFilterSet) {
  const worker = getMergeWorker();
  if (!worker) {
    return Promise.resolve(mergeCompactAnimationBundles(compacts, electrodeFilterSet));
  }
  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleError);
      resolve(event.data);
    };
    const handleError = (error) => {
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleError);
      reject(error);
    };
    worker.addEventListener('message', handleMessage);
    worker.addEventListener('error', handleError);
    worker.postMessage({
      compacts,
      electrodeIds: [...electrodeFilterSet],
    });
  });
}

export async function fetchAndMergePhaseAnimation({
  manifest,
  subjects,
  phase,
  selectedLoad,
  metadata = null,
  electrodeFilterSet,
  onProgress,
}) {
  if (!manifest) return null;

  const compacts = [];
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

  await Promise.all(subjects.map(async (subject) => {
    try {
      const payload = await loadSubjectPhaseAnimation(manifest, subject, phase);
      const compact = extractBundleForLoad(payload, selectedLoad, metadata);
      if (compact) compacts.push(compact);
    } finally {
      completed += 1;
      report();
    }
  }));

  if (!compacts.length) return null;
  return mergeWithWorker(compacts, electrodeFilterSet);
}

export { animationLoadKey, expandCompactBundle, extractBundleForLoad, mergeCompactAnimationBundles };
