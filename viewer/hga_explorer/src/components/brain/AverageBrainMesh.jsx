import React, { useEffect, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import { BRAIN_MESH_URL, BRAIN_HEMI_SPLIT_X } from '../../constants/brain.js';
import {
  applyBrainMaterial,
  applyHemisphereClipping,
  prepareBrainWithHemispheres,
  setBrainHemisphereVisibility,
} from '../../lib/brainMesh.js';

export default function AverageBrainMesh({ opacity, hemisphereView, useLitCortex = false }) {
  const { scene } = useGLTF(BRAIN_MESH_URL);
  const brain = useMemo(
    () => prepareBrainWithHemispheres(scene.clone(true), BRAIN_HEMI_SPLIT_X),
    [scene],
  );
  const lit = useLitCortex;

  useEffect(() => {
    applyBrainMaterial(brain, opacity, { forceSolid: opacity >= 0.95, lit });
    applyHemisphereClipping(brain, 'both');
    setBrainHemisphereVisibility(brain, hemisphereView);
  }, [brain, opacity, hemisphereView, lit]);

  return <primitive object={brain} />;
}
