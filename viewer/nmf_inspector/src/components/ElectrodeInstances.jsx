import React, { useMemo } from 'react';
import {
  CLUSTER_COLORS,
  ELECTRODE_BASE_RADIUS,
  ELECTRODE_RENDER_ORDER,
  PURITY_OPACITY,
} from '../constants.js';
import { hgaToRadius } from '../utils/hgaRadius.js';
import { electrodePurity } from '../utils/purity.js';

export default function ElectrodeInstances({
  electrodes,
  clusterForElectrode,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
  isAnimating = false,
  animationScale = null,
  liveHgaByElectrodeId = null,
}) {
  const items = useMemo(() => electrodes.map((electrode) => {
    const selected = selectedId === electrode.id;
    const hovered = hoveredId === electrode.id;
    const cluster = clusterForElectrode(electrode);
    const { purity } = electrodePurity(electrode);
    const liveHga = liveHgaByElectrodeId?.[electrode.id];
    const radius = isAnimating
      ? hgaToRadius(liveHga ?? null, animationScale, { active: false, selected, hovered })
      : (selected ? 1.45 : hovered ? 1.2 : 1.0);
    const baseScale = purity === 'mixed' ? radius * 1.12 : radius;
    return {
      electrode,
      color: CLUSTER_COLORS[cluster] || '#94a3b8',
      scale: baseScale,
      purity,
      opacity: PURITY_OPACITY[purity] ?? 0.9,
      wireframe: purity === 'mixed',
    };
  }), [
    electrodes,
    clusterForElectrode,
    selectedId,
    hoveredId,
    isAnimating,
    animationScale,
    liveHgaByElectrodeId,
  ]);

  if (!items.length) return null;

  return (
    <group>
      {items.map(({ electrode, color, scale, opacity, wireframe }) => (
        <mesh
          key={electrode.id}
          position={[electrode.x, electrode.y, electrode.z]}
          scale={scale}
          renderOrder={ELECTRODE_RENDER_ORDER}
          onPointerMove={(event) => {
            event.stopPropagation();
            onHover(electrode.id);
            document.body.style.cursor = 'pointer';
          }}
          onPointerOut={() => {
            onHover(null);
            document.body.style.cursor = 'default';
          }}
          onClick={(event) => {
            event.stopPropagation();
            onSelect(electrode.id);
          }}
        >
          <sphereGeometry args={[ELECTRODE_BASE_RADIUS, 16, 12]} />
          <meshBasicMaterial
            color={color}
            toneMapped={false}
            transparent
            opacity={opacity}
            wireframe={wireframe}
            depthWrite={!wireframe}
          />
        </mesh>
      ))}
    </group>
  );
}
