import React, { useMemo } from 'react';
import {
  ATLAS_REGION_COLORS,
  MANUAL_LABEL_COLORS,
  shortAparc,
} from '../../constants/anatomyReview.js';

function LegendSwatch({ color, label }) {
  return (
    <span className="anatomy-legend-item">
      <i style={{ background: color }} />
      {label}
    </span>
  );
}

export default function AnatomyReviewCard({
  subject,
  contacts,
  labelsByContact,
  selectedContact,
  reconstruction,
  coordinateSpace,
  meshNote,
  nativeMeshMissing,
  loading,
  saving,
  error,
  onSelectContact,
  onSetLabel,
}) {
  const contact = useMemo(
    () => contacts.find((row) => row.contact === selectedContact) || null,
    [contacts, selectedContact],
  );
  const shafts = useMemo(() => {
    const groups = new Map();
    contacts.forEach((row) => {
      const key = row.shaft || row.contact;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    });
    return [...groups.entries()].map(([shaft, rows]) => ({
      shaft,
      rows: rows.slice().sort((a, b) => a.contact_number - b.contact_number),
    }));
  }, [contacts]);

  const manual = contact ? labelsByContact[contact.contact] || null : null;
  const sliceUrl = subject && contact
    ? `/api/anatomy/slice.png?subject=${encodeURIComponent(subject)}&contact=${encodeURIComponent(contact.contact)}&v=${encodeURIComponent(manual || 'none')}`
    : null;

  return (
    <div className="selected-card anatomy-review-card">
      <div className="card-kicker">Anatomy review</div>
      <h3>{contact ? contact.contact : 'Choose a contact'}</h3>
      {loading && <p>Loading reconstructed contacts…</p>}
      {error && <p className="anatomy-error">{error}</p>}
      {nativeMeshMissing && (
        <p className="anatomy-note">
          Native mesh is missing for this subject. The MRI slice and contact coordinates are still available from the list below.
        </p>
      )}
      {meshNote && <p className="anatomy-note">{meshNote}</p>}
      <label className="anatomy-contact-picker">
        <span>Contact</span>
        <select
          value={contact?.contact || ''}
          onChange={(event) => onSelectContact?.(event.target.value || null)}
          disabled={!contacts.length}
        >
          <option value="">{contacts.length ? 'Click a contact on the brain' : 'No contacts loaded'}</option>
          {shafts.map(({ shaft, rows }) => (
            <optgroup key={shaft} label={shaft}>
              {rows.map((row) => (
                <option key={row.contact} value={row.contact}>
                  {row.contact}
                  {labelsByContact[row.contact] ? ` · ${labelsByContact[row.contact]}` : ''}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      {contact && (
        <>
          <p title={contact.native_aparc}>
            {subject}
            {reconstruction ? ` · ${reconstruction}` : ''}
            {' · '}
            {shortAparc(contact.native_aparc)}
          </p>
          <p className="endpoint-coords">
            {contact.x.toFixed(3)}, {contact.y.toFixed(3)}, {contact.z.toFixed(3)}
            {coordinateSpace ? ` ${coordinateSpace}` : ''}
          </p>
          <p className="anatomy-atlas-hint">
            Atlas hint: {contact.native_region}
            {manual ? ` · Manual label: ${manual}` : ' · No manual label'}
          </p>
          {sliceUrl && (
            <img
              className="anatomy-slice"
              src={sliceUrl}
              alt={`Native T1 through ${contact.contact}`}
            />
          )}
          <div className="anatomy-actions">
            <button
              type="button"
              className={`anatomy-label-btn operculum${manual === 'Operculum' ? ' active' : ''}`}
              disabled={saving}
              onClick={() => {
                if (manual !== 'Operculum') onSetLabel?.(contact.contact, 'Operculum');
              }}
            >
              Operculum
            </button>
            <button
              type="button"
              className={`anatomy-label-btn anterior${manual === 'Anterior Insula' ? ' active' : ''}`}
              disabled={saving}
              onClick={() => {
                if (manual !== 'Anterior Insula') onSetLabel?.(contact.contact, 'Anterior Insula');
              }}
            >
              Anterior Insula
            </button>
          </div>
          {manual && (
            <button
              type="button"
              className="anatomy-clear-btn"
              disabled={saving}
              onClick={() => onSetLabel?.(contact.contact, null)}
            >
              Clear label
            </button>
          )}
        </>
      )}
      {!contact && !loading && !error && (
        <p>Click a reconstructed contact. The native T1 slice opens here.</p>
      )}
      <div className="anatomy-legend">
        <LegendSwatch color={ATLAS_REGION_COLORS.Insula} label="Atlas insula" />
        <LegendSwatch color={ATLAS_REGION_COLORS['Frontal operculum']} label="Atlas operculum" />
        <LegendSwatch color={ATLAS_REGION_COLORS.Other} label="Other" />
        <LegendSwatch color={MANUAL_LABEL_COLORS.Operculum} label="Manual operculum" />
        <LegendSwatch color={MANUAL_LABEL_COLORS['Anterior Insula']} label="Manual anterior insula" />
      </div>
    </div>
  );
}
