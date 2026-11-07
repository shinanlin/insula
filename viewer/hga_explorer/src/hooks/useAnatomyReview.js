import { useCallback, useEffect, useMemo, useState } from 'react';

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Anatomy request failed (${response.status})`);
  }
  return body;
}

export default function useAnatomyReview(subject, enabled) {
  const [payload, setPayload] = useState(null);
  const [labels, setLabels] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!enabled || !subject) {
      setPayload(null);
      setLabels([]);
      setError(null);
      setLoading(false);
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    const subjectQuery = encodeURIComponent(subject);
    Promise.all([
      fetch(`/api/anatomy/contacts?subject=${subjectQuery}`).then(readJson),
      fetch(`/api/anatomy/labels?subject=${subjectQuery}`).then(readJson),
    ]).then(([contactsBody, labelsBody]) => {
      if (cancelled) return;
      setPayload(contactsBody);
      setLabels(labelsBody.labels || []);
      setLoading(false);
    }).catch((err) => {
      if (cancelled) return;
      setPayload(null);
      setLabels([]);
      setError(err.message || 'Anatomy review is unavailable');
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [subject, enabled]);

  const labelsByContact = useMemo(() => {
    const map = {};
    labels.forEach((row) => {
      if (row?.contact && row?.label) map[row.contact] = row.label;
    });
    return map;
  }, [labels]);

  const setLabel = useCallback(async (contact, label) => {
    if (!subject || !contact) return;
    setSaving(true);
    setError(null);
    try {
      const body = await readJson(await fetch('/api/anatomy/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, contact, label }),
      }));
      setLabels((current) => {
        const next = current.filter((row) => row.contact !== contact);
        if (body.label) next.push(body.label);
        return next;
      });
    } catch (err) {
      setError(err.message || 'Could not save the anatomy label');
    } finally {
      setSaving(false);
    }
  }, [subject]);

  return {
    contacts: payload?.contacts || [],
    reconstruction: payload?.reconstruction || null,
    coordinateSpace: payload?.coordinate_space || null,
    meshNote: payload?.mesh_note || null,
    labelsByContact,
    error,
    loading,
    saving,
    setLabel,
  };
}
