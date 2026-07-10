import React from 'react';
import { Html } from '@react-three/drei';
import { ELECTRODE_BASE_RADIUS, ELECTRODE_RENDER_ORDER } from '../../constants/brain.js';
import { resolveHgaMean, hgaToRadius } from '../../utils/hga.js';
import { resolveBrainElectrodeColor } from '../../utils/electrodeColors.js';

export default function ElectrodePoint({
  electrode,
  vennPhases,
  selectedLoad,
  traces = null,
  significanceWindows = null,
  hgaScale,
  animationScale,
  liveHga,
  isAnimating,
  selected,
  dimmed,
  active,
  hovered,
  onHover,
  onSelect,
  colorByFunctional,
}) {
  const color = resolveBrainElectrodeColor({
    electrode,
    vennPhases,
    selected,
    colorByFunctional,
  });
  const scale = isAnimating && animationScale ? animationScale : hgaScale;
  const hgaMean = isAnimating
    ? (liveHga ?? null)
    : resolveHgaMean(electrode, selectedLoad, traces, significanceWindows);
  const radius = hgaToRadius(hgaMean, scale, { active, selected, hovered });
  const opacity = active || selected ? 0.9 : hovered ? 0.65 : dimmed ? 0.04 : 0.42;
  return (
    <group position={[electrode.x, electrode.y, electrode.z]}>
      <mesh
        renderOrder={ELECTRODE_RENDER_ORDER}
        scale={[radius, radius, radius]}
        onPointerOver={(event) => {
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
        <sphereGeometry args={[ELECTRODE_BASE_RADIUS, 24, 16]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          depthWrite
          emissive={active || hovered ? color : '#000000'}
          emissiveIntensity={active || hovered ? 0.35 : 0.15}
        />
      </mesh>
      {(active || hovered) && (
        <Html distanceFactor={8} className="tooltip">
          <strong>{electrode.channel}</strong>
          <span>{electrode.subject} · {electrode.roi} · {electrode.hemi}</span>
        </Html>
      )}
    </group>
  );
}
