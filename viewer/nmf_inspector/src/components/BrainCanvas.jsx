import React, { Suspense } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import InsulaMesh from './InsulaMesh.jsx';
import ElectrodeInstances from './ElectrodeInstances.jsx';
import BrainSceneLighting from './BrainSceneLighting.jsx';
import { INSULA_CAMERA, INSULA_ORBIT_TARGET } from '../constants.js';

export default function BrainCanvas({
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
  return (
    <Canvas
      camera={{ position: INSULA_CAMERA.position, fov: INSULA_CAMERA.fov, up: [0, 0, 1] }}
      gl={{ localClippingEnabled: true, alpha: false, antialias: true }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.NoToneMapping;
      }}
    >
      <BrainSceneLighting mneStyle />
      <Suspense fallback={null}>
        <InsulaMesh useLitCortex />
        <ElectrodeInstances
          electrodes={electrodes}
          clusterForElectrode={clusterForElectrode}
          selectedId={selectedId}
          hoveredId={hoveredId}
          onHover={onHover}
          onSelect={onSelect}
          isAnimating={isAnimating}
          animationScale={animationScale}
          liveHgaByElectrodeId={liveHgaByElectrodeId}
        />
      </Suspense>
      <OrbitControls target={INSULA_ORBIT_TARGET} makeDefault />
    </Canvas>
  );
}
