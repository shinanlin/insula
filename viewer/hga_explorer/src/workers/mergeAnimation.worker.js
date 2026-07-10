import { mergeCompactAnimationBundles } from '../utils/animationBundle.js';

self.onmessage = (event) => {
  const { compacts, electrodeIds } = event.data;
  const result = mergeCompactAnimationBundles(compacts, new Set(electrodeIds));
  self.postMessage(result);
};
