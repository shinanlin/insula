import { useEffect, useMemo, useState } from 'react';
import { BRAIN_SPACES, nativeMeshUrl } from '../utils/electrodeCoords.js';

export default function useBrainSpace({
  selectedSubjects,
  manifest,
}) {
  const [brainSpace, setBrainSpace] = useState(BRAIN_SPACES.template);
  const [nativeMeshAvailable, setNativeMeshAvailable] = useState({});

  const singleSubject = selectedSubjects.size === 1
    ? [...selectedSubjects][0]
    : null;

  const forcedTemplate = selectedSubjects.size !== 1;

  useEffect(() => {
    if (forcedTemplate && brainSpace !== BRAIN_SPACES.template) {
      setBrainSpace(BRAIN_SPACES.template);
    }
  }, [forcedTemplate, brainSpace]);

  useEffect(() => {
    const subjects = manifest?.subjects || manifest?.metadata?.subjects || [];
    if (!subjects.length) return undefined;

    let cancelled = false;
    const checks = subjects.map(async (subject) => {
      const url = nativeMeshUrl(subject);
      try {
        const response = await fetch(url, { method: 'HEAD' });
        return [subject, response.ok];
      } catch {
        return [subject, false];
      }
    });

    Promise.all(checks).then((entries) => {
      if (cancelled) return;
      setNativeMeshAvailable(Object.fromEntries(entries));
    });

    return () => {
      cancelled = true;
    };
  }, [manifest]);

  const nativeAvailableForSelection = Boolean(
    singleSubject && nativeMeshAvailable[singleSubject],
  );

  const activeBrainSpace = forcedTemplate ? BRAIN_SPACES.template : brainSpace;
  const activeNativeMeshUrl = activeBrainSpace === BRAIN_SPACES.native && singleSubject
    ? nativeMeshUrl(singleSubject)
    : null;

  const setBrainSpaceSafe = (nextSpace) => {
    if (nextSpace === BRAIN_SPACES.native && !nativeAvailableForSelection) return;
    if (forcedTemplate && nextSpace === BRAIN_SPACES.native) return;
    setBrainSpace(nextSpace);
  };

  const brainSpaceOptions = useMemo(() => ([
    {
      id: BRAIN_SPACES.template,
      label: 'Template',
      disabled: false,
      title: 'Average CVS template brain (multi-subject safe)',
    },
    {
      id: BRAIN_SPACES.native,
      label: 'Native',
      disabled: forcedTemplate || !nativeAvailableForSelection,
      title: forcedTemplate
        ? 'Native brain is available for a single selected subject only'
        : (!nativeAvailableForSelection
          ? 'Native mesh not exported for this subject'
          : 'Subject-native pial mesh'),
    },
  ]), [forcedTemplate, nativeAvailableForSelection]);

  return {
    brainSpace: activeBrainSpace,
    setBrainSpace: setBrainSpaceSafe,
    brainSpaceOptions,
    singleSubject,
    forcedTemplate,
    nativeMeshUrl: activeNativeMeshUrl,
    nativeMeshAvailable,
  };
}
