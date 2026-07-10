import React from 'react';
import { Line } from '@react-three/drei';
import {
  ELECTRODE_BASE_RADIUS,
  ENDPOINT_MARKER_RADIUS,
  ELECTRODE_RENDER_ORDER,
} from '../../constants/brain.js';
import { resolveEndpointCoords } from '../../utils/electrodeCoords.js';

function EndpointMarker({
  position,
  color,
  active,
  onHover,
  onSelect,
}) {
  if (!position) return null;
  return (
    <group position={[position.x, position.y, position.z]}>
      <mesh
        renderOrder={ELECTRODE_RENDER_ORDER}
        scale={[ENDPOINT_MARKER_RADIUS, ENDPOINT_MARKER_RADIUS, ENDPOINT_MARKER_RADIUS]}
        onPointerOver={(event) => {
          event.stopPropagation();
          onHover?.();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          onHover?.(null);
          document.body.style.cursor = 'default';
        }}
        onClick={(event) => {
          event.stopPropagation();
          onSelect?.();
        }}
      >
        <sphereGeometry args={[ELECTRODE_BASE_RADIUS, 16, 12]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={active ? 0.95 : 0.72}
          depthWrite
          emissive={active ? color : '#000000'}
          emissiveIntensity={active ? 0.4 : 0}
        />
      </mesh>
    </group>
  );
}

export default function BipolarEndpoints({
  electrode,
  brainSpace,
  selectedEndpoint,
  onSelectEndpoint,
  onHoverEndpoint,
}) {
  if (!electrode) return null;

  const contact1 = resolveEndpointCoords(electrode, 'contact_1', brainSpace);
  const contact2 = resolveEndpointCoords(electrode, 'contact_2', brainSpace);
  if (!contact1 && !contact2) return null;

  const segmentPoints = contact1 && contact2
    ? [
      [contact1.x, contact1.y, contact1.z],
      [contact2.x, contact2.y, contact2.z],
    ]
    : null;

  return (
    <group>
      {segmentPoints && (
        <Line
          points={segmentPoints}
          color="#f59e0b"
          transparent
          opacity={0.55}
          lineWidth={1}
        />
      )}
      <EndpointMarker
        position={contact1}
        color="#f97316"
        active={selectedEndpoint === 'contact_1'}
        onHover={() => onHoverEndpoint?.('contact_1')}
        onSelect={() => onSelectEndpoint?.('contact_1')}
      />
      <EndpointMarker
        position={contact2}
        color="#38bdf8"
        active={selectedEndpoint === 'contact_2'}
        onHover={() => onHoverEndpoint?.('contact_2')}
        onSelect={() => onSelectEndpoint?.('contact_2')}
      />
    </group>
  );
}
