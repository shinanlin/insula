import React, { useEffect, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import {
  BRAIN_MESH_URL,
  BRAIN_HEMI_SPLIT_X,
  BRAIN_SOLID_OPACITY_THRESHOLD,
} from '../../constants/brain.js';
import {
  applyBrainMaterial,
  applyHemisphereClipping,
  prepareBrainWithHemispheres,
  setBrainHemisphereVisibility,
} from '../../lib/brainMesh.js';

export default function AverageBrainMesh({
  opacity,
  hemisphereView,
  useLitCortex = false,
  forceSolid = undefined,
}) {
  const { scene } = useGLTF(BRAIN_MESH_URL);
  const brain = useMemo(
    () => prepareBrainWithHemispheres(scene.clone(true), BRAIN_HEMI_SPLIT_X),
    [scene],
  );
  const lit = useLitCortex;
  const resolvedForceSolid = forceSolid ?? (opacity >= BRAIN_SOLID_OPACITY_THRESHOLD);

  useEffect(() => {
    applyBrainMaterial(brain, opacity, { forceSolid: resolvedForceSolid, lit });
    applyHemisphereClipping(brain, 'both');
    setBrainHemisphereVisibility(brain, hemisphereView);
  }, [brain, opacity, hemisphereView, lit, resolvedForceSolid]);

  return <primitive object={brain} />;
}
