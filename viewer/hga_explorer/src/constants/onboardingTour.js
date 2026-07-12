export const TOUR_STORAGE_KEY = 'hga_explorer_tour_v1_completed';

export function isTourCompleted() {
  if (typeof window === 'undefined') return true;
  try {
    return window.localStorage.getItem(TOUR_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

export function markTourCompleted() {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(TOUR_STORAGE_KEY, '1');
  } catch {
    // ignore quota / private mode
  }
}

export function clearTourCompleted() {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(TOUR_STORAGE_KEY);
  } catch {
    // ignore
  }
}

function tourSelector(id) {
  return `[data-tour="${id}"]`;
}

const REQUIRED_TOUR_ANCHORS = [
  'tour-welcome',
  'task-selector',
  'venn-selector',
  'brain-controls',
  'waveform-panel',
];

export function areTourAnchorsReady() {
  if (typeof document === 'undefined') return false;
  return REQUIRED_TOUR_ANCHORS.every((id) => Boolean(document.querySelector(tourSelector(id))));
}

export function buildTourSteps() {
  return [
    {
      element: tourSelector('tour-welcome'),
      popover: {
        title: 'Welcome to the Insula HGA Explorer',
        description: 'Explore multi-task high-gamma activity across phase overlap regions, cortical maps, and time courses. Use Next to continue or Exit tour to skip.',
        side: 'over',
        align: 'center',
        popoverClass: 'phase-overlap-tour-popover phase-overlap-tour-welcome',
      },
    },
    {
      element: tourSelector('task-selector'),
      popover: {
        title: 'Task selector',
        description: 'Switch among Phoneme, Lexical, or All tasks. Waveforms show all conditions for the active task; the brain map uses the Map condition below.',
        side: 'bottom',
        align: 'start',
      },
    },
    {
      element: tourSelector('condition-selector'),
      popover: {
        title: 'Map condition',
        description: 'Choose Repeat or Decision (Lexical only). This controls brain sphere color, KDE, and animation playback. Waveforms always show all available conditions.',
        side: 'bottom',
        align: 'start',
      },
    },
    {
      element: tourSelector('venn-selector'),
      popover: {
        title: 'Phase overlap',
        description: 'Toggle stimulus, delay, go, and response phases, then click Venn regions to filter electrodes by overlap pattern.',
        side: 'right',
        align: 'start',
      },
    },
    {
      element: tourSelector('subject-filter'),
      popover: {
        title: 'Subject filter',
        description: 'Include or exclude subjects. Multi-subject viewing uses the template brain; native brain is available for a single subject when exported.',
        side: 'right',
        align: 'start',
      },
    },
    {
      element: tourSelector('brain-controls'),
      popover: {
        title: 'Brain view',
        description: 'Toggle template vs native brain (single subject), hemisphere, electrodes vs KDE, and click a midpoint to reveal bipolar endpoints.',
        side: 'bottom',
        align: 'start',
      },
    },
    {
      element: tourSelector('atlas-selector'),
      popover: {
        title: 'Parcellation atlas',
        description: 'Switch between APARC and Hammersmith parcellation. ROI labels and insula filtering update immediately; HGA traces are shared.',
        side: 'bottom',
        align: 'end',
      },
    },
    {
      element: tourSelector('roi-filter'),
      popover: {
        title: 'ROI filter',
        description: 'Click a bar to show or hide that ROI on the brain map.',
        side: 'left',
        align: 'start',
      },
    },
    {
      element: tourSelector('waveform-panel'),
      popover: {
        title: 'Time courses',
        description: 'Inspect four-phase HGA traces for the selected region or electrode. Each panel shows all available conditions; play animation synchronized with the Map condition on the brain.',
        side: 'top',
        align: 'center',
      },
    },
  ];
}

export function resolveTourSteps() {
  return buildTourSteps().filter((step) => {
    if (!step.element) return true;
    const selector = typeof step.element === 'string' ? step.element : null;
    if (!selector) return true;
    return Boolean(document.querySelector(selector));
  });
}
