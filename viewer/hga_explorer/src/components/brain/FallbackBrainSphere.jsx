import React, { useMemo } from 'react';
import {
  createBrainMaterial,
  hemisphereClippingPlanes,
} from '../../lib/brainMesh.js';

export default function FallbackBrainSphere({ opacity, hemisphereView }) {
  const material = useMemo(() => {
    const mat = createBrainMaterial(opacity, { forceSolid: opacity >= 0.95 });
    const planes = hemisphereClippingPlanes(hemisphereView);
    if (planes.length) {
      mat.clippingPlanes = planes;
      mat.clipIntersection = false;
    }
    return mat;
  }, [opacity, hemisphereView]);

  return (
    <mesh position={[0, 0, 8]} scale={[1.35, 1.0, 0.78]}>
      <sphereGeometry args={[62, 48, 24]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
