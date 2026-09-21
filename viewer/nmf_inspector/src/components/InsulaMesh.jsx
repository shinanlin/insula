import React, { useEffect, useMemo } from 'react';
import { useGLTF } from '@react-three/drei';
import { INSULA_MESH_URL } from '../constants.js';
import {
  applyBrainMaterial,
  applyHemisphereClipping,
  BRAIN_HEMI_SPLIT_X,
  INSULA_HIGHLIGHT_OPACITY,
  prepareBrainWithHemispheres,
  setBrainHemisphereVisibility,
} from '../lib/brainMesh.js';

/** Match hga_explorer InsulaHighlightMesh: lit cortex at INSULA_HIGHLIGHT_OPACITY. */
export default function InsulaMesh({
  meshUrl = INSULA_MESH_URL,
  opacity = INSULA_HIGHLIGHT_OPACITY,
  hemisphereView = 'both',
  useLitCortex = true,
}) {
  const { scene } = useGLTF(meshUrl);
  const brain = useMemo(
    () => prepareBrainWithHemispheres(scene.clone(true), BRAIN_HEMI_SPLIT_X),
    [scene],
  );

  useEffect(() => {
    applyBrainMaterial(brain, opacity, { forceSolid: false, lit: useLitCortex });
    applyHemisphereClipping(brain, hemisphereView);
    setBrainHemisphereVisibility(brain, hemisphereView);
    brain.traverse((child) => {
      if (child.isMesh) {
        child.renderOrder = 2;
      }
    });
  }, [brain, opacity, hemisphereView, useLitCortex]);

  return <primitive object={brain} />;
}

useGLTF.preload(INSULA_MESH_URL);
