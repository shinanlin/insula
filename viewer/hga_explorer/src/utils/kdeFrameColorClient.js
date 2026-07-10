import { buildInfluenceMap, buildKdeFrameColorCache } from '../brainKde.js';

let worker = null;
let nextJobId = 0;

function getWorker() {
  if (typeof Worker === 'undefined') return null;
  if (!worker) {
    worker = new Worker(new URL('../workers/kdeFrameColor.worker.js', import.meta.url), {
      type: 'module',
    });
  }
  return worker;
}

export function buildKdeFrameColorsOffThread({
  positions,
  influencePoints,
  frameHgaValues,
  globalHgaMax,
  splitX,
  statsHemisphere,
  startIndex,
  fixedDensityRange = null,
  onFrameReady,
  onProgress,
}) {
  const kdeWorker = getWorker();
  const options = { statsHemisphere };
  const cacheOptions = {
    startIndex,
    onFrameReady,
    onProgress,
    chunkSize: 2,
    fixedDensityRange,
  };

  if (!kdeWorker) {
    const influenceMap = buildInfluenceMap(positions, influencePoints);
    return buildKdeFrameColorCache(
      influenceMap,
      frameHgaValues,
      globalHgaMax,
      positions,
      splitX,
      options,
      cacheOptions,
    ).then(({ fixedRange }) => ({ fixedRange }));
  }

  const id = nextJobId + 1;
  nextJobId = id;

  return new Promise((resolve, reject) => {
    let fixedRange = null;

    const handleMessage = (event) => {
      const message = event.data;
      if (message.id !== id) return;

      if (message.type === 'progress') {
        fixedRange = message.fixedRange ?? fixedRange;
        onProgress?.(message.done, message.total, message.fixedRange);
        return;
      }

      if (message.type === 'frame') {
        onFrameReady?.(message.index, new Float32Array(message.colors));
        return;
      }

      if (message.type === 'done') {
        kdeWorker.removeEventListener('message', handleMessage);
        kdeWorker.removeEventListener('error', handleError);
        resolve({ fixedRange });
        return;
      }

      if (message.type === 'error') {
        kdeWorker.removeEventListener('message', handleMessage);
        kdeWorker.removeEventListener('error', handleError);
        reject(new Error(message.message));
      }
    };

    const handleError = (error) => {
      kdeWorker.removeEventListener('message', handleMessage);
      kdeWorker.removeEventListener('error', handleError);
      reject(error);
    };

    kdeWorker.addEventListener('message', handleMessage);
    kdeWorker.addEventListener('error', handleError);

    kdeWorker.postMessage({
      type: 'build',
      id,
      positions: Float32Array.from(positions),
      influencePoints,
      frameHgaValues,
      globalHgaMax,
      splitX,
      statsHemisphere,
      startIndex,
      fixedDensityRange,
    });
  });
}
