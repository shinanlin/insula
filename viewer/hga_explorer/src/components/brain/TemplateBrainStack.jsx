import React from 'react';
import {
  INSULA_GHOST_OPACITY,
  INSULA_HIGHLIGHT_OPACITY,
} from '../../constants/insula.js';
import AverageBrainMesh from './AverageBrainMesh.jsx';
import InsulaHighlightMesh from './InsulaHighlightMesh.jsx';

export default function TemplateBrainStack({
  insulaMode,
  opacity,
  hemisphereView,
  useLitCortex = false,
}) {
  if (!insulaMode) {
    return (
      <AverageBrainMesh
        opacity={opacity}
        hemisphereView={hemisphereView}
        useLitCortex={useLitCortex}
      />
    );
  }

  return (
    <>
      <AverageBrainMesh
        opacity={INSULA_GHOST_OPACITY}
        hemisphereView={hemisphereView}
        useLitCortex={false}
        forceSolid={false}
      />
      <InsulaHighlightMesh
        opacity={INSULA_HIGHLIGHT_OPACITY}
        hemisphereView={hemisphereView}
        useLitCortex={useLitCortex}
      />
    </>
  );
}
