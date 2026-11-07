import React, { useMemo, useState } from 'react';
import { Html, Line } from '@react-three/drei';
import { ELECTRODE_RENDER_ORDER } from '../../constants/brain.js';
import { anatomyContactColor, shortAparc } from '../../constants/anatomyReview.js';

export default function AnatomyContacts({
  contacts,
  labelsByContact,
  selectedContact,
  onSelect,
}) {
  const [hovered, setHovered] = useState(null);
  const shafts = useMemo(() => {
    const groups = new Map();
    contacts.forEach((contact) => {
      const key = contact.shaft || contact.contact;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(contact);
    });
    return [...groups.entries()].map(([shaft, rows]) => ({
      shaft,
      rows: rows.slice().sort((a, b) => a.contact_number - b.contact_number),
    }));
  }, [contacts]);

  return (
    <group>
      {shafts.map(({ shaft, rows }) => (
        rows.length > 1 ? (
          <Line
            key={shaft}
            points={rows.map((contact) => [contact.x, contact.y, contact.z])}
            color="#777777"
            lineWidth={1.5}
            transparent
            opacity={0.9}
          />
        ) : null
      ))}
      {contacts.map((contact) => {
        const manual = labelsByContact[contact.contact] || null;
        const active = selectedContact === contact.contact;
        const hot = hovered === contact.contact;
        const color = anatomyContactColor(contact.native_region, manual);
        const radius = active ? 1.45 : manual ? 1.15 : 0.95;
        return (
          <group key={contact.contact} position={[contact.x, contact.y, contact.z]}>
            <mesh
              renderOrder={ELECTRODE_RENDER_ORDER}
              onPointerOver={(event) => {
                event.stopPropagation();
                setHovered(contact.contact);
                document.body.style.cursor = 'pointer';
              }}
              onPointerOut={() => {
                setHovered((current) => (current === contact.contact ? null : current));
                document.body.style.cursor = 'default';
              }}
              onClick={(event) => {
                event.stopPropagation();
                onSelect?.(contact.contact);
              }}
            >
              <sphereGeometry args={[2.4, 12, 8]} />
              <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            </mesh>
            <mesh renderOrder={ELECTRODE_RENDER_ORDER}>
              <sphereGeometry args={[radius, 18, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={active || hot ? color : '#000000'}
                emissiveIntensity={active ? 0.45 : hot ? 0.25 : 0}
              />
            </mesh>
            {active && (
              <mesh renderOrder={ELECTRODE_RENDER_ORDER + 1}>
                <sphereGeometry args={[radius + 0.55, 16, 12]} />
                <meshBasicMaterial color="#ffffff" wireframe />
              </mesh>
            )}
            {(active || hot) && (
              <Html distanceFactor={12} className="tooltip">
                <strong>{contact.contact}</strong>
                <span>
                  {contact.shaft}
                  {' · '}
                  {shortAparc(contact.native_aparc)}
                  {manual ? ` · ${manual}` : ''}
                </span>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}
