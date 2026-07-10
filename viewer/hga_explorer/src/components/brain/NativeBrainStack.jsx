import React from 'react';
import {
  INSULA_GHOST_OPACITY,
  INSULA_HIGHLIGHT_OPACITY,
} from '../../constants/insula.js';
import SubjectBrainMesh from './SubjectBrainMesh.jsx';
import InsulaHighlightMesh from './InsulaHighlightMesh.jsx';

export default function NativeBrainStack({
  meshUrl,
  insulaMeshUrl,
  insulaMode,
  opacity,
  hemisphereView,
  useLitCortex = false,
}) {
  if (!insulaMode) {
    return (
      <SubjectBrainMesh
        meshUrl={meshUrl}
        opacity={opacity}
        hemisphereView={hemisphereView}
        useLitCortex={useLitCortex}
      />
    );
  }

  return (
    <>
      <SubjectBrainMesh
        meshUrl={meshUrl}
        opacity={INSULA_GHOST_OPACITY}
        hemisphereView={hemisphereView}
        useLitCortex={false}
        forceSolid={false}
      />
      <InsulaHighlightMesh
        meshUrl={insulaMeshUrl}
        opacity={INSULA_HIGHLIGHT_OPACITY}
        hemisphereView={hemisphereView}
        useLitCortex={useLitCortex}
      />
    </>
  );
}
