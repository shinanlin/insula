import React, { useEffect, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import { TEMPLATE_INSULA_MESH_URL } from '../../constants/insula.js';
import { BRAIN_HEMI_SPLIT_X } from '../../constants/brain.js';
import {
  applyBrainMaterial,
  applyHemisphereClipping,
  prepareBrainWithHemispheres,
  setBrainHemisphereVisibility,
} from '../../lib/brainMesh.js';

export default function InsulaHighlightMesh({
  meshUrl = TEMPLATE_INSULA_MESH_URL,
  opacity,
  hemisphereView,
  useLitCortex = true,
}) {
  const { scene } = useGLTF(meshUrl);
  const brain = useMemo(
    () => prepareBrainWithHemispheres(scene.clone(true), BRAIN_HEMI_SPLIT_X),
    [scene],
  );

  useEffect(() => {
    applyBrainMaterial(brain, opacity, { forceSolid: false, lit: useLitCortex });
    applyHemisphereClipping(brain, 'both');
    setBrainHemisphereVisibility(brain, hemisphereView);
    brain.traverse((child) => {
      if (child.isMesh) {
        child.renderOrder = 2;
      }
    });
  }, [brain, opacity, hemisphereView, useLitCortex]);

  return <primitive object={brain} />;
}

useGLTF.preload(TEMPLATE_INSULA_MESH_URL);
