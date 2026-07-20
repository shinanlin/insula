import React, { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { ELECTRODE_BASE_RADIUS, ELECTRODE_RENDER_ORDER } from '../../constants/brain.js';
import { resolveHgaMean, hgaToRadius } from '../../utils/hga.js';
import { resolveBrainElectrodeColor } from '../../utils/electrodeColors.js';

export default function ElectrodeInstances({
  electrodes,
  vennPhases,
  selectedLoad,
  traces = null,
  significanceWindows = null,
  metadata = null,
  hgaScale,
  animationScale,
  liveHgaByElectrodeId,
  isAnimating,
  selectedIds,
  selectedElectrodeId,
  hoveredId,
  colorByFunctional,
  onHover,
  onSelect,
}) {
  const meshRef = useRef(null);
  const tempObject = useMemo(() => new THREE.Object3D(), []);
  const scale = isAnimating && animationScale ? animationScale : hgaScale;

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || !electrodes.length) return;

    const color = new THREE.Color();
    electrodes.forEach((electrode, index) => {
      const selected = selectedIds.has(electrode.id);
      const active = selectedElectrodeId === electrode.id;
      const hovered = hoveredId === electrode.id;
      const liveHga = liveHgaByElectrodeId?.[electrode.id];
      const hgaMean = isAnimating
        ? (liveHga ?? null)
        : resolveHgaMean(electrode, selectedLoad, traces, significanceWindows, metadata);
      const radius = hgaToRadius(hgaMean, scale, { active, selected, hovered });

      tempObject.position.set(electrode.x, electrode.y, electrode.z);
      tempObject.scale.setScalar(radius);
      tempObject.updateMatrix();
      mesh.setMatrixAt(index, tempObject.matrix);

      color.set(resolveBrainElectrodeColor({
        electrode,
        vennPhases,
        selected,
        colorByFunctional,
      }));
      mesh.setColorAt(index, color);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [
    electrodes,
    vennPhases,
    selectedLoad,
    traces,
    significanceWindows,
    metadata,
    scale,
    liveHgaByElectrodeId,
    isAnimating,
    selectedIds,
    selectedElectrodeId,
    hoveredId,
    colorByFunctional,
    tempObject,
  ]);

  if (!electrodes.length) return null;

  return (
    <instancedMesh
      ref={meshRef}
      renderOrder={ELECTRODE_RENDER_ORDER}
      args={[undefined, undefined, electrodes.length]}
      onPointerMove={(event) => {
        event.stopPropagation();
        const index = event.instanceId;
        if (index == null) return;
        onHover(electrodes[index]?.id ?? null);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        onHover(null);
        document.body.style.cursor = 'default';
      }}
      onClick={(event) => {
        event.stopPropagation();
        const index = event.instanceId;
        if (index == null) return;
        onSelect(electrodes[index]?.id ?? null);
      }}
    >
      <sphereGeometry args={[ELECTRODE_BASE_RADIUS, 16, 12]} />
      <meshStandardMaterial vertexColors transparent opacity={0.85} depthWrite />
    </instancedMesh>
  );
}
