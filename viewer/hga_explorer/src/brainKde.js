/** Gaussian KDE on pial mesh — matches univarite.ipynb (bandwidth=5mm, max_distance=10mm). */

export const KDE_BANDWIDTH = 5.0;
export const KDE_MAX_DISTANCE = 10.0;
export const KDE_DENSITY_OPACITY = 0.9;
export const KDE_PERCENTILE_MAX = 98;
export const KDE_COLORMAP_STEPS = 256;
/** Skip the whitest part of vlag so low density reads as pale pink, not blown-out white. */
export const KDE_COLORMAP_DISPLAY_MIN = 0.08;

// seaborn vlag peak red — same as notebooks/univarite.ipynb colormap='vlag'
export const PROJECT_RED = '#A9373B';

// seaborn.cm.vlag positive half (t=0.5..1.0), matches notebooks/univarite.ipynb
const VLAG_POSITIVE_LUT = [
  [0.980600, 0.961552, 0.958131], [0.981526, 0.959937, 0.956393], [0.981973, 0.957666, 0.953993], [0.981919, 0.954787, 0.950981],
  [0.981385, 0.951348, 0.947406], [0.980408, 0.947399, 0.943321], [0.979021, 0.943001, 0.938787], [0.977293, 0.938204, 0.933851],
  [0.975253, 0.933073, 0.928583], [0.972978, 0.927653, 0.923023], [0.970491, 0.922003, 0.917235], [0.967844, 0.916167, 0.911261],
  [0.965073, 0.910187, 0.905141], [0.962220, 0.904092, 0.898908], [0.959308, 0.897915, 0.892591], [0.956356, 0.891679, 0.886217],
  [0.953383, 0.885404, 0.879802], [0.950402, 0.879103, 0.873363], [0.947422, 0.872789, 0.866911], [0.944452, 0.866469, 0.860453],
  [0.941505, 0.860146, 0.853992], [0.938574, 0.853828, 0.847536], [0.935662, 0.847518, 0.841089], [0.932772, 0.841216, 0.834652],
  [0.929901, 0.834927, 0.828227], [0.927047, 0.828650, 0.821817], [0.924227, 0.822381, 0.815413], [0.921426, 0.816124, 0.809024],
  [0.918645, 0.809880, 0.802648], [0.915876, 0.803652, 0.796290], [0.913137, 0.797431, 0.789940], [0.910416, 0.791223, 0.783604],
  [0.907711, 0.785027, 0.777282], [0.905016, 0.778847, 0.770977], [0.902354, 0.772671, 0.764678], [0.899702, 0.766510, 0.758395],
  [0.897053, 0.760365, 0.752131], [0.894440, 0.754223, 0.745870], [0.891834, 0.748095, 0.739627], [0.889232, 0.741982, 0.733401],
  [0.886659, 0.735873, 0.727180], [0.884088, 0.729779, 0.720977], [0.881535, 0.723693, 0.714785], [0.878994, 0.717618, 0.708605],
  [0.876452, 0.711558, 0.702444], [0.873940, 0.705499, 0.696285], [0.871426, 0.699455, 0.690146], [0.868927, 0.693419, 0.684016],
  [0.866436, 0.687392, 0.677899], [0.863944, 0.681379, 0.671799], [0.861476, 0.675367, 0.665704], [0.858999, 0.669372, 0.659629],
  [0.856547, 0.663378, 0.653558], [0.854088, 0.657398, 0.647505], [0.851644, 0.651422, 0.641460], [0.849201, 0.645456, 0.635429],
  [0.846764, 0.639498, 0.629410], [0.844332, 0.633548, 0.623403], [0.841901, 0.627606, 0.617409], [0.839479, 0.621670, 0.611424],
  [0.837054, 0.615743, 0.605455], [0.834640, 0.609820, 0.599492], [0.832219, 0.603907, 0.593547], [0.829810, 0.597996, 0.587608],
  [0.827403, 0.592091, 0.581679], [0.824986, 0.586197, 0.575769], [0.822582, 0.580303, 0.569863], [0.820166, 0.574421, 0.563975],
  [0.817763, 0.568537, 0.558092], [0.815346, 0.562666, 0.552227], [0.812943, 0.556791, 0.546365], [0.810521, 0.550930, 0.540524],
  [0.808115, 0.545063, 0.534685], [0.805690, 0.539210, 0.528866], [0.803275, 0.533353, 0.523051], [0.800847, 0.527506, 0.517253],
  [0.798422, 0.521658, 0.511462], [0.795994, 0.515812, 0.505682], [0.793558, 0.509971, 0.499914], [0.791126, 0.504127, 0.494153],
  [0.788674, 0.498294, 0.488411], [0.786231, 0.492454, 0.482672], [0.783769, 0.486623, 0.476952], [0.781308, 0.480788, 0.471238],
  [0.778845, 0.474952, 0.465532], [0.776363, 0.469122, 0.459845], [0.773884, 0.463286, 0.454161], [0.771389, 0.457455, 0.448494],
  [0.768889, 0.451620, 0.442836], [0.766388, 0.445779, 0.437183], [0.763861, 0.439948, 0.431552], [0.761335, 0.434107, 0.425925],
  [0.758806, 0.428258, 0.420305], [0.756249, 0.422419, 0.414707], [0.753692, 0.416569, 0.409113], [0.751127, 0.410711, 0.403528],
  [0.748543, 0.404855, 0.397959], [0.745947, 0.398993, 0.392401], [0.743343, 0.393122, 0.386851], [0.740733, 0.387239, 0.381307],
  [0.738094, 0.381361, 0.375786], [0.735447, 0.375471, 0.370271], [0.732789, 0.369570, 0.364765], [0.730118, 0.363658, 0.359270],
  [0.727435, 0.357733, 0.353785], [0.724727, 0.351805, 0.348317], [0.722005, 0.345864, 0.342859], [0.719271, 0.339906, 0.337410],
  [0.716520, 0.333934, 0.331972], [0.713754, 0.327946, 0.326545], [0.710970, 0.321941, 0.321130], [0.708168, 0.315919, 0.315726],
  [0.705348, 0.309877, 0.310334], [0.702509, 0.303815, 0.304954], [0.699652, 0.297730, 0.299585], [0.696775, 0.291621, 0.294227],
  [0.693884, 0.285481, 0.288878], [0.690976, 0.279310, 0.283538], [0.688035, 0.273120, 0.278219], [0.685079, 0.266891, 0.272907],
  [0.682108, 0.260621, 0.267602], [0.679110, 0.254318, 0.262314], [0.676094, 0.247968, 0.257034], [0.673059, 0.241568, 0.251762],
  [0.670002, 0.235119, 0.246503], [0.666934, 0.228599, 0.241244], [0.663844, 0.222017, 0.235996], [0.660807, 0.215267, 0.230695],
];

function cellKey(ix, iy, iz) {
  return `${ix},${iy},${iz}`;
}

function buildVertexGrid(positions, cellSize) {
  const vertexCount = positions.length / 3;
  const grid = new Map();
  for (let i = 0; i < vertexCount; i += 1) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const ix = Math.floor(x / cellSize);
    const iy = Math.floor(y / cellSize);
    const iz = Math.floor(z / cellSize);
    const key = cellKey(ix, iy, iz);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }
  return grid;
}

function nearbyVertexIndices(grid, x, y, z, cellSize, maxDistance) {
  const cellRadius = Math.ceil(maxDistance / cellSize);
  const ix0 = Math.floor(x / cellSize);
  const iy0 = Math.floor(y / cellSize);
  const iz0 = Math.floor(z / cellSize);
  const indices = [];
  for (let dx = -cellRadius; dx <= cellRadius; dx += 1) {
    for (let dy = -cellRadius; dy <= cellRadius; dy += 1) {
      for (let dz = -cellRadius; dz <= cellRadius; dz += 1) {
        const bucket = grid.get(cellKey(ix0 + dx, iy0 + dy, iz0 + dz));
        if (bucket) indices.push(...bucket);
      }
    }
  }
  return indices;
}

/**
 * @param {Float32Array|number[]} positions flat xyz
 * @param {{x:number,y:number,z:number}[]} electrodes
 */
export function buildInfluenceMap(
  positions,
  electrodes,
  bandwidth = KDE_BANDWIDTH,
  maxDistance = KDE_MAX_DISTANCE,
) {
  const vertexCount = positions.length / 3;
  const cellSize = maxDistance;
  const grid = buildVertexGrid(positions, cellSize);
  const invTwoSigma2 = 1 / (2 * bandwidth * bandwidth);
  const contributions = electrodes.map(() => []);

  electrodes.forEach((electrode, electrodeIndex) => {
    const candidates = nearbyVertexIndices(
      grid,
      electrode.x,
      electrode.y,
      electrode.z,
      cellSize,
      maxDistance,
    );
    const seen = new Set();
    candidates.forEach((vertexIndex) => {
      if (seen.has(vertexIndex)) return;
      seen.add(vertexIndex);
      const dx = positions[vertexIndex * 3] - electrode.x;
      const dy = positions[vertexIndex * 3 + 1] - electrode.y;
      const dz = positions[vertexIndex * 3 + 2] - electrode.z;
      const dist2 = dx * dx + dy * dy + dz * dz;
      if (dist2 > maxDistance * maxDistance) return;
      const coeff = Math.exp(-dist2 * invTwoSigma2);
      if (coeff > 0) {
        contributions[electrodeIndex].push({ vertexIndex, coeff });
      }
    });
  });

  return { vertexCount, contributions };
}

export function normalizeHgaWeights(hgaValues, globalMax = null) {
  if (!hgaValues.length) return [];
  const absValues = hgaValues.map((value) => Math.abs(value ?? 0));
  const maxValue = globalMax ?? Math.max(...absValues, 1e-12);
  return absValues.map((value) => value / maxValue);
}

export function computeDensity(influenceMap, normalizedWeights) {
  const { vertexCount, contributions } = influenceMap;
  const density = new Float32Array(vertexCount);
  contributions.forEach((entries, electrodeIndex) => {
    const weight = normalizedWeights[electrodeIndex] ?? 0;
    if (weight <= 0) return;
    entries.forEach(({ vertexIndex, coeff }) => {
      density[vertexIndex] += coeff * weight;
    });
  });
  return density;
}

function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function colormapRgb(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const remapped = KDE_COLORMAP_DISPLAY_MIN + clamped * (1 - KDE_COLORMAP_DISPLAY_MIN);
  const last = VLAG_POSITIVE_LUT.length - 1;
  const scaled = remapped * last;
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, last);
  const frac = scaled - lo;
  const c0 = VLAG_POSITIVE_LUT[lo];
  const c1 = VLAG_POSITIVE_LUT[hi];
  return [
    c0[0] + (c1[0] - c0[0]) * frac,
    c0[1] + (c1[1] - c0[1]) * frac,
    c0[2] + (c1[2] - c0[2]) * frac,
  ];
}

function rgbToHex([r, g, b]) {
  const toByte = (value) => Math.round(Math.max(0, Math.min(1, value)) * 255);
  return `#${[toByte(r), toByte(g), toByte(b)].map((byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

/** CSS linear-gradient for the vlag positive-half colorbar (low → high, bottom → top). */
export function vlagPositiveCssGradient() {
  const sampleCount = 24;
  const last = VLAG_POSITIVE_LUT.length - 1;
  const stops = Array.from({ length: sampleCount }, (_, index) => {
    const t = index / (sampleCount - 1);
    const rgb = colormapRgb(t);
    return `${rgbToHex(rgb)} ${Math.round(t * 100)}%`;
  });
  return `linear-gradient(to top, ${stops.join(', ')})`;
}

/**
 * Density color range: fixed vmin at 0, vmax at p98 of visible density values.
 */
export function computeDensityRange(
  density,
  positions = null,
  splitX = null,
  options = {},
) {
  const { statsHemisphere = 'both' } = options;
  const valid = [];
  for (let i = 0; i < density.length; i += 1) {
    if (density[i] <= 0) continue;
    if (positions && !vertexInHemisphere(positions, i, splitX, statsHemisphere)) continue;
    valid.push(density[i]);
  }
  if (!valid.length) {
    return { vmin: 0, vmax: 1, hasData: false };
  }
  const vmin = 0;
  let vmax = percentile(valid, KDE_PERCENTILE_MAX);
  if (vmax <= vmin) vmax = vmin + 1e-12;
  return { vmin, vmax, hasData: true };
}

/**
 * Fixed KDE color scale across animation frames (vmin=0, vmax=p98 over pooled densities).
 */
export function computeGlobalKdeDensityRange(
  influenceMap,
  hgaValuesByFrame,
  globalHgaMax,
  positions = null,
  splitX = null,
  options = {},
) {
  if (!hgaValuesByFrame?.length || !globalHgaMax) {
    return { vmin: 0, vmax: 1, hasData: false };
  }

  const pooled = [];
  const maxSamples = 50000;
  hgaValuesByFrame.forEach((hgaValues) => {
    const weights = normalizeHgaWeights(hgaValues, globalHgaMax);
    const density = computeDensity(influenceMap, weights);
    for (let i = 0; i < density.length; i += 1) {
      if (density[i] <= 0) continue;
      if (positions && !vertexInHemisphere(positions, i, splitX, options.statsHemisphere ?? 'both')) {
        continue;
      }
      pooled.push(density[i]);
    }
  });

  if (!pooled.length) {
    return { vmin: 0, vmax: 1, hasData: false };
  }

  const stride = Math.max(1, Math.floor(pooled.length / maxSamples));
  const sampled = stride === 1 ? pooled : pooled.filter((_, index) => index % stride === 0);

  const vmin = 0;
  let vmax = percentile(sampled, KDE_PERCENTILE_MAX);
  if (vmax <= vmin) vmax = vmin + 1e-12;
  return { vmin, vmax, hasData: true };
}

function idleYield(timeout = 32) {
  return new Promise((resolve) => {
    if (typeof window !== 'undefined' && window.requestIdleCallback) {
      window.requestIdleCallback(() => resolve(), { timeout });
    } else {
      window.setTimeout(resolve, 0);
    }
  });
}

function buildPrioritizedFrameOrder(total, startIndex = 0) {
  const order = [];
  const seen = new Set();
  const push = (index) => {
    if (index < 0 || index >= total || seen.has(index)) return;
    seen.add(index);
    order.push(index);
  };
  push(startIndex);
  for (let offset = 1; offset < total; offset += 1) {
    push(startIndex + offset);
    push(startIndex - offset);
  }
  return order;
}

export async function estimateFixedDensityRangeAsync(
  influenceMap,
  hgaValuesByFrame,
  globalHgaMax,
  positions = null,
  splitX = null,
  options = {},
  sampleCount = 12,
) {
  if (!hgaValuesByFrame?.length || !globalHgaMax) {
    return { vmin: 0, vmax: 1, hasData: false };
  }

  const frameCount = hgaValuesByFrame.length;
  const indices = frameCount === 1
    ? [0]
    : Array.from(
      { length: Math.min(sampleCount, frameCount) },
      (_, index) => Math.round(index * (frameCount - 1) / (Math.min(sampleCount, frameCount) - 1)),
    );

  const pooled = [];
  for (const frameIndex of indices) {
    const weights = normalizeHgaWeights(hgaValuesByFrame[frameIndex], globalHgaMax);
    const density = computeDensity(influenceMap, weights);
    for (let i = 0; i < density.length; i += 1) {
      if (density[i] <= 0) continue;
      if (positions && !vertexInHemisphere(positions, i, splitX, options.statsHemisphere ?? 'both')) {
        continue;
      }
      pooled.push(density[i]);
    }
    await idleYield(16);
  }

  if (!pooled.length) {
    return { vmin: 0, vmax: 1, hasData: false };
  }

  const vmin = 0;
  let vmax = percentile(pooled, KDE_PERCENTILE_MAX);
  if (vmax <= vmin) vmax = vmin + 1e-12;
  return { vmin, vmax, hasData: true };
}

export function buildFrameVertexColors(
  influenceMap,
  hgaValues,
  globalHgaMax,
  positions,
  splitX,
  options = {},
  fixedRange = null,
) {
  const weights = normalizeHgaWeights(hgaValues, globalHgaMax);
  const density = computeDensity(influenceMap, weights);
  const range = fixedRange?.hasData
    ? fixedRange
    : computeDensityRange(
      density,
      positions,
      splitX,
      options,
    );
  const colors = densityToVertexColors(
    density,
    positions,
    splitX,
    {
      ...options,
      fixedRange: range,
    },
  );
  return { colors, range };
}

export function resolveDensityRange(autoRange, vmaxOverride) {
  if (vmaxOverride != null && Number.isFinite(vmaxOverride) && vmaxOverride > 0) {
    return { vmin: 0, vmax: vmaxOverride, hasData: true };
  }
  return autoRange;
}

export async function buildKdeFrameColorCache(
  influenceMap,
  hgaValuesByFrame,
  globalHgaMax,
  positions,
  splitX,
  options = {},
  {
    chunkSize = 2,
    startIndex = 0,
    onProgress,
    onFrameReady,
    fixedDensityRange = null,
  } = {},
) {
  if (!hgaValuesByFrame?.length) {
    return { cache: [], fixedRange: { vmin: 0, vmax: 1, hasData: false } };
  }

  await idleYield(0);
  onProgress?.(0, hgaValuesByFrame.length, null);

  const autoRange = await estimateFixedDensityRangeAsync(
    influenceMap,
    hgaValuesByFrame,
    globalHgaMax,
    positions,
    splitX,
    options,
  );
  const colorRange = fixedDensityRange?.hasData ? fixedDensityRange : autoRange;
  onProgress?.(0, hgaValuesByFrame.length, autoRange);

  const cache = new Array(hgaValuesByFrame.length);
  const order = buildPrioritizedFrameOrder(hgaValuesByFrame.length, startIndex);
  let done = 0;

  for (let index = 0; index < order.length; index += chunkSize) {
    const batch = order.slice(index, index + chunkSize);
    batch.forEach((frameIndex) => {
      cache[frameIndex] = buildFrameVertexColors(
        influenceMap,
        hgaValuesByFrame[frameIndex],
        globalHgaMax,
        positions,
        splitX,
        options,
        colorRange,
      ).colors;
      onFrameReady?.(frameIndex, cache[frameIndex]);
    });
    done += batch.length;
    onProgress?.(done, hgaValuesByFrame.length, autoRange);
    await idleYield(16);
  }

  return { cache, fixedRange: autoRange };
}

function vertexInHemisphere(positions, vertexIndex, splitX, hemisphereView) {
  if (hemisphereView === 'both' || splitX == null) return true;
  const x = positions[vertexIndex * 3];
  if (hemisphereView === 'left') return x <= splitX;
  return x > splitX;
}

/**
 * @param {Float32Array} density
 * @param {Float32Array|number[]|null} positions flat xyz
 * @param {number|null} splitX
 * @param {{ statsHemisphere?: string, maskColorsToHemisphere?: boolean }} [options]
 */
export function densityToVertexColors(
  density,
  positions = null,
  splitX = null,
  options = {},
) {
  const {
    statsHemisphere = 'both',
    maskColorsToHemisphere = false,
    fixedRange = null,
    insulaVertexMask = null,
  } = options;
  const vertexCount = density.length;
  const { vmin, vmax } = fixedRange?.hasData
    ? fixedRange
    : computeDensityRange(density, positions, splitX, options);

  const colors = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    if (insulaVertexMask && !insulaVertexMask[i]) {
      colors[i * 4 + 3] = 0;
      continue;
    }
    const value = density[i];
    if (value <= 0) {
      colors[i * 4 + 3] = 0;
      continue;
    }
    if (
      maskColorsToHemisphere
      && positions
      && !vertexInHemisphere(positions, i, splitX, statsHemisphere)
    ) {
      colors[i * 4 + 3] = 0;
      continue;
    }
    const norm = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin + 1e-12)));
    const [r, g, b] = colormapRgb(norm);
    colors[i * 4] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = 1;
  }
  return colors;
}

export function extractMeshPositions(root) {
  let mesh = null;
  root.traverse((child) => {
    if (!mesh && child.isMesh) mesh = child;
  });
  if (!mesh?.geometry?.attributes?.position) {
    return { positions: new Float32Array(0), vertexCount: 0 };
  }
  const attribute = mesh.geometry.attributes.position;
  const positions = attribute.array instanceof Float32Array
    ? attribute.array
    : Float32Array.from(attribute.array);
  return { positions, vertexCount: attribute.count };
}
