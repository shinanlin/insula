import { VENN_GAP, VENN_MIN_R, VENN_MAX_R } from '../constants/venn.js';
import { PHASE_LABELS } from '../constants/phases.js';

export function phaseUnionCounts(regions, vennPhases) {
  return vennPhases.map((phase) => (
    regions
      .filter((region) => region.phases_on.includes(phase))
      .reduce((sum, region) => sum + region.count, 0)
  ));
}

function scaleVennRadii(counts) {
  const maxCount = Math.max(...counts, 1);
  return counts.map((count) => {
    if (count <= 0) return VENN_MIN_R * 0.72;
    return VENN_MIN_R + Math.sqrt(count / maxCount) * (VENN_MAX_R - VENN_MIN_R);
  });
}

function regionInteriorMargin(x, y, circles, indices, gap = VENN_GAP) {
  let minMargin = Infinity;
  circles.forEach((circle, index) => {
    const dist = Math.hypot(x - circle.cx, y - circle.cy);
    if (indices.includes(index)) {
      minMargin = Math.min(minMargin, (circle.r - gap) - dist);
    } else {
      minMargin = Math.min(minMargin, dist - (circle.r + gap));
    }
  });
  return minMargin;
}

function regionSearchBounds(circles, indices) {
  const active = indices.map((index) => circles[index]);
  return {
    minX: Math.max(...active.map((circle) => circle.cx - circle.r)),
    maxX: Math.min(...active.map((circle) => circle.cx + circle.r)),
    minY: Math.max(...active.map((circle) => circle.cy - circle.r)),
    maxY: Math.min(...active.map((circle) => circle.cy + circle.r)),
  };
}

function labelSeparationPenalty(x, y, placed, minSeparation) {
  let penalty = 0;
  placed.forEach(({ countX, countY }) => {
    const dist = Math.hypot(x - countX, y - countY);
    if (dist < minSeparation) {
      penalty += (minSeparation - dist) ** 2;
    }
  });
  return penalty;
}

function searchBestLabelPoint(circles, indices, bounds, step, placed = [], minSeparation = 0) {
  let best = null;
  let bestScore = -Infinity;
  const collisionWeight = placed.length ? 2.8 : 0;
  for (let x = bounds.minX; x <= bounds.maxX; x += step) {
    for (let y = bounds.minY; y <= bounds.maxY; y += step) {
      const margin = regionInteriorMargin(x, y, circles, indices);
      if (margin <= 0) continue;
      const penalty = minSeparation > 0
        ? labelSeparationPenalty(x, y, placed, minSeparation)
        : 0;
      const score = margin - collisionWeight * penalty;
      if (score <= bestScore) continue;
      bestScore = score;
      best = { x, y, score, margin };
    }
  }
  return best;
}

function computeCountPosition(circles, indices, placed = [], minSeparation = 0) {
  const bounds = regionSearchBounds(circles, indices);
  if (bounds.minX >= bounds.maxX || bounds.minY >= bounds.maxY) {
    const active = indices.map((index) => circles[index]);
    return {
      x: active.reduce((sum, circle) => sum + circle.cx, 0) / active.length,
      y: active.reduce((sum, circle) => sum + circle.cy, 0) / active.length,
      hideCount: true,
    };
  }

  const span = Math.min(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  let coarseStep = Math.max(2, span / (circles.length >= 4 ? 32 : 24));
  let best = searchBestLabelPoint(circles, indices, bounds, coarseStep, placed, minSeparation);
  if (!best && coarseStep > 1) {
    best = searchBestLabelPoint(circles, indices, bounds, 1, placed, minSeparation);
  }

  if (!best) {
    const active = indices.map((index) => circles[index]);
    const cx = active.reduce((sum, circle) => sum + circle.cx, 0) / active.length;
    const cy = active.reduce((sum, circle) => sum + circle.cy, 0) / active.length;
    return { x: cx, y: cy, hideCount: true };
  }

  const pad = coarseStep * 1.25;
  const fineBounds = {
    minX: Math.max(bounds.minX, best.x - pad),
    maxX: Math.min(bounds.maxX, best.x + pad),
    minY: Math.max(bounds.minY, best.y - pad),
    maxY: Math.min(bounds.maxY, best.y + pad),
  };
  const refined = searchBestLabelPoint(circles, indices, fineBounds, 1, placed, minSeparation);
  const resolved = refined || best;
  const minReadableMargin = circles.length >= 4 ? 7 : 5;
  return {
    x: resolved.x,
    y: resolved.y,
    hideCount: resolved.margin < minReadableMargin,
  };
}

function indicesFromMask(mask, length) {
  return Array.from({ length }, (_, index) => index).filter((index) => mask & (1 << index));
}

function maskPopcount(mask) {
  let count = 0;
  let value = mask;
  while (value) {
    count += value & 1;
    value >>= 1;
  }
  return count;
}

function buildSubregions(vennPhases, circles) {
  const subregions = [];
  const masks = [];
  for (let mask = 1; mask < (1 << circles.length); mask += 1) {
    masks.push(mask);
  }
  masks.sort((a, b) => maskPopcount(a) - maskPopcount(b) || a - b);

  const placed = [];
  const minSeparation = circles.length >= 4 ? 16 : 18;

  masks.forEach((mask) => {
    const indices = indicesFromMask(mask, circles.length);
    const id = indices.map((index) => vennPhases[index]).join('_');
    const countPos = computeCountPosition(circles, indices, placed, minSeparation);
    const others = Array.from({ length: circles.length }, (_, index) => index).filter((index) => !indices.includes(index));
    const subregion = {
      id,
      clipId: `venn-clip-${indices.join('')}`,
      maskId: others.length ? `venn-mask-not-${others.join('')}` : null,
      countX: countPos.x,
      countY: countPos.y,
      hideCount: countPos.hideCount,
    };
    placed.push(subregion);
    subregions.push(subregion);
  });
  return subregions;
}

function labelBounds(circle, label, fontSize) {
  const charWidth = fontSize * 0.52;
  const labelWidth = label.length * charWidth;
  const labelHeight = fontSize * 1.15;
  const anchor = circle.labelAnchor || 'middle';
  let minX = circle.labelX;
  let maxX = circle.labelX;
  if (anchor === 'middle') {
    minX -= labelWidth / 2;
    maxX += labelWidth / 2;
  } else if (anchor === 'start') {
    maxX += labelWidth;
  } else if (anchor === 'end') {
    minX -= labelWidth;
  }
  return {
    minX,
    maxX,
    minY: circle.labelY - labelHeight,
    maxY: circle.labelY + labelHeight * 0.35,
  };
}

function finalizeVennConfig(circles, vennPhases) {
  const padding = 28;
  const labelPad = 34;
  const fontSize = circles.length >= 4 ? 17 : 20;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  circles.forEach((circle, index) => {
    minX = Math.min(minX, circle.cx - circle.r);
    maxX = Math.max(maxX, circle.cx + circle.r);
    minY = Math.min(minY, circle.cy - circle.r);
    maxY = Math.max(maxY, circle.cy + circle.r);

    const label = PHASE_LABELS[vennPhases[index]] ?? '';
    const bounds = labelBounds(circle, label, fontSize);
    minX = Math.min(minX, bounds.minX);
    maxX = Math.max(maxX, bounds.maxX);
    minY = Math.min(minY, bounds.minY);
    maxY = Math.max(maxY, bounds.maxY);
  });

  minX -= padding;
  maxX += padding;
  minY -= padding + labelPad;
  maxY += padding + labelPad;
  const width = maxX - minX;
  const height = maxY - minY;
  return {
    viewBox: `${minX} ${minY} ${width} ${height}`,
    width,
    height,
    minX,
    minY,
    circles,
    subregions: buildSubregions(vennPhases, circles),
  };
}

function buildTwoCircleLayout(vennPhases, phaseCounts, regionMap) {
  const [countA, countB] = phaseCounts;
  const [radiusA, radiusB] = scaleVennRadii(phaseCounts);
  const bothCount = regionMap.get(vennPhases.join('_'))?.count ?? 0;

  let separation = radiusA + radiusB + 14;
  if (bothCount > 0 && countA > 0 && countB > 0) {
    const overlapRatio = Math.min(bothCount / Math.min(countA, countB), 1);
    separation = radiusA + radiusB - overlapRatio * Math.min(radiusA, radiusB) * 0.82;
    separation = Math.max(separation, Math.abs(radiusA - radiusB) + 10);
  }

  const centerX = 195;
  const cy = 158;
  const circles = [
    {
      key: '0',
      cx: centerX - separation / 2,
      cy,
      r: radiusA,
      labelX: centerX - separation / 2,
      labelY: cy - radiusA - 18,
      labelAnchor: 'middle',
    },
    {
      key: '1',
      cx: centerX + separation / 2,
      cy,
      r: radiusB,
      labelX: centerX + separation / 2,
      labelY: cy - radiusB - 18,
      labelAnchor: 'middle',
    },
  ];
  return finalizeVennConfig(circles, vennPhases);
}

function buildThreeCircleLayout(vennPhases, phaseCounts, regionMap) {
  const [radiusA, radiusB, radiusC] = scaleVennRadii(phaseCounts);
  const avgRadius = (radiusA + radiusB + radiusC) / 3;
  let spread = 48 + (avgRadius - VENN_MIN_R) * 0.42;

  const pairIds = [
    [0, 1, vennPhases.slice(0, 2).join('_')],
    [0, 2, [vennPhases[0], vennPhases[2]].join('_')],
    [1, 2, [vennPhases[1], vennPhases[2]].join('_')],
  ];
  const overlapPressure = pairIds.reduce((sum, [, , id]) => {
    const count = regionMap.get(id)?.count ?? 0;
    return sum + (count > 0 ? 1 : 0);
  }, 0);
  spread -= overlapPressure * 4;

  const centerX = 210;
  const topY = 132;
  const bottomY = topY + spread * 1.08;
  const circles = [
    {
      key: '0',
      cx: centerX - spread,
      cy: topY,
      r: radiusA,
      labelX: centerX - spread,
      labelY: topY - radiusA - 16,
      labelAnchor: 'middle',
    },
    {
      key: '1',
      cx: centerX + spread,
      cy: topY,
      r: radiusB,
      labelX: centerX + spread,
      labelY: topY - radiusB - 16,
      labelAnchor: 'middle',
    },
    {
      key: '2',
      cx: centerX,
      cy: bottomY,
      r: radiusC,
      labelX: centerX,
      labelY: bottomY + radiusC + 24,
      labelAnchor: 'middle',
    },
  ];
  return finalizeVennConfig(circles, vennPhases);
}

function buildFourCircleLayout(vennPhases, phaseCounts, regionMap) {
  const radii = scaleVennRadii(phaseCounts).map((radius) => radius * 0.84);
  const [radiusA, radiusB, radiusC, radiusD] = radii;
  const avgRadius = radii.reduce((sum, radius) => sum + radius, 0) / radii.length;
  let spread = 36 + (avgRadius - VENN_MIN_R) * 0.34;

  const pairIds = [
    [0, 1, [vennPhases[0], vennPhases[1]].join('_')],
    [0, 2, [vennPhases[0], vennPhases[2]].join('_')],
    [0, 3, [vennPhases[0], vennPhases[3]].join('_')],
    [1, 2, [vennPhases[1], vennPhases[2]].join('_')],
    [1, 3, [vennPhases[1], vennPhases[3]].join('_')],
    [2, 3, [vennPhases[2], vennPhases[3]].join('_')],
  ];
  const overlapPressure = pairIds.reduce((sum, [, , id]) => {
    const count = regionMap.get(id)?.count ?? 0;
    return sum + (count > 0 ? 1 : 0);
  }, 0);
  spread -= overlapPressure * 2.2;

  const centerX = 210;
  const centerY = 170;
  const circles = [
    {
      key: '0',
      cx: centerX,
      cy: centerY - spread,
      r: radiusA,
      labelX: centerX,
      labelY: centerY - spread - radiusA - 14,
      labelAnchor: 'middle',
    },
    {
      key: '1',
      cx: centerX + spread,
      cy: centerY,
      r: radiusB,
      labelX: centerX + spread + radiusB + 12,
      labelY: centerY + 4,
      labelAnchor: 'start',
    },
    {
      key: '2',
      cx: centerX,
      cy: centerY + spread,
      r: radiusC,
      labelX: centerX,
      labelY: centerY + spread + radiusC + 20,
      labelAnchor: 'middle',
    },
    {
      key: '3',
      cx: centerX - spread,
      cy: centerY,
      r: radiusD,
      labelX: centerX - spread - radiusD - 12,
      labelY: centerY + 4,
      labelAnchor: 'end',
    },
  ];
  return finalizeVennConfig(circles, vennPhases);
}

export function buildVennConfig(vennPhases, regions) {
  const regionMap = new Map((regions || []).map((region) => [region.id, region]));
  const phaseCounts = phaseUnionCounts(regions || [], vennPhases);
  if (vennPhases.length === 2) {
    return buildTwoCircleLayout(vennPhases, phaseCounts, regionMap);
  }
  if (vennPhases.length === 4) {
    return buildFourCircleLayout(vennPhases, phaseCounts, regionMap);
  }
  return buildThreeCircleLayout(vennPhases, phaseCounts, regionMap);
}
