import { useCallback, useEffect, useRef, useState } from 'react';
import { driver } from 'driver.js';
import {
  areTourAnchorsReady,
  isTourCompleted,
  markTourCompleted,
  resolveTourSteps,
} from '../constants/onboardingTour.js';

const AUTO_START_MAX_ATTEMPTS = 30;
const AUTO_START_RETRY_MS = 200;
const AUTO_START_INITIAL_DELAY_MS = 300;

function attachExitTourButton(popover, onExit) {
  const footer = popover.footer;
  if (!footer || footer.querySelector('.tour-popover-exit-btn')) return;

  const exitBtn = document.createElement('button');
  exitBtn.type = 'button';
  exitBtn.className = 'tour-popover-exit-btn';
  exitBtn.textContent = 'Exit tour';
  exitBtn.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    onExit();
  }, true);

  footer.insertBefore(exitBtn, popover.footerButtons);
}

function resetWelcomeAnchor() {
  const anchor = document.querySelector('[data-tour="tour-welcome"]');
  if (!anchor) return;

  anchor.style.position = '';
  anchor.style.left = '';
  anchor.style.top = '';
  anchor.style.width = '';
  anchor.style.height = '';
  anchor.style.transform = '';
  anchor.style.minHeight = '';
  anchor.style.maxWidth = '';
}

function syncWelcomeAnchorToPopover(popover, driverInstance) {
  if (driverInstance.getActiveIndex() !== 0 || !popover.wrapper) return;

  const anchor = document.querySelector('[data-tour="tour-welcome"]');
  if (!anchor) return;

  const applySync = () => {
    if (!driverInstance.isActive() || driverInstance.getActiveIndex() !== 0) return;

    const rect = popover.wrapper.getBoundingClientRect();
    anchor.style.position = 'fixed';
    anchor.style.left = `${Math.round(rect.left)}px`;
    anchor.style.top = `${Math.round(rect.top)}px`;
    anchor.style.width = `${Math.round(rect.width)}px`;
    anchor.style.height = `${Math.round(rect.height)}px`;
    anchor.style.transform = 'none';
    anchor.style.minHeight = '0';
    anchor.style.maxWidth = 'none';

    driverInstance.setConfig({
      ...driverInstance.getConfig(),
      stagePadding: 0,
    });
    driverInstance.refresh();
  };

  requestAnimationFrame(() => {
    requestAnimationFrame(applySync);
  });
}

function cleanupStaleDriverUi() {
  document.body.classList.remove('driver-active', 'driver-fade', 'driver-simple');
  document.querySelectorAll('.driver-overlay, .driver-overlay-animated').forEach((node) => {
    node.remove();
  });
  document.getElementById('driver-dummy-element')?.remove();
  document.querySelectorAll('.driver-active-element').forEach((node) => {
    node.classList.remove('driver-active-element', 'driver-no-interaction');
  });
}

export default function useOnboardingTour({ enabled = false } = {}) {
  const driverRef = useRef(null);
  const skipMarkOnDestroyRef = useRef(false);
  const exitTourRef = useRef(null);
  const startTourRef = useRef(null);
  const autoStartStartedRef = useRef(false);
  const [isTourActive, setIsTourActive] = useState(false);

  const destroyDriver = useCallback(({ skipMark = false } = {}) => {
    skipMarkOnDestroyRef.current = skipMark;
    if (!driverRef.current) {
      skipMarkOnDestroyRef.current = false;
      return;
    }
    driverRef.current.destroy();
  }, []);

  const exitTour = useCallback(() => {
    if (!driverRef.current) {
      setIsTourActive(false);
      markTourCompleted();
      cleanupStaleDriverUi();
      return;
    }
    destroyDriver({ skipMark: false });
  }, [destroyDriver]);

  exitTourRef.current = exitTour;

  const startTour = useCallback(({ force = false } = {}) => {
    if (!force && isTourCompleted()) return false;

    cleanupStaleDriverUi();
    destroyDriver({ skipMark: true });

    const steps = resolveTourSteps();
    if (!steps.length) return false;

    const driverObj = driver({
      animate: true,
      smoothScroll: true,
      allowClose: true,
      allowKeyboardControl: true,
      overlayOpacity: 0.55,
      stagePadding: 10,
      stageRadius: 12,
      showProgress: true,
      progressText: '{{current}} of {{total}}',
      nextBtnText: 'Next',
      prevBtnText: 'Back',
      doneBtnText: 'Done',
      showButtons: ['next', 'previous', 'close'],
      popoverClass: 'phase-overlap-tour-popover',
      steps,
      onPopoverRender: (popover, { driver: activeDriver }) => {
        attachExitTourButton(popover, () => exitTourRef.current?.());
        syncWelcomeAnchorToPopover(popover, activeDriver);
      },
      onHighlightStarted: (_element, _step, { driver: activeDriver }) => {
        setIsTourActive(true);
        if (activeDriver.getActiveIndex() !== 0) {
          resetWelcomeAnchor();
          activeDriver.setConfig({
            ...activeDriver.getConfig(),
            stagePadding: 10,
          });
        }
      },
      onHighlighted: (_element, _step, { driver: activeDriver }) => {
        if (activeDriver.getActiveIndex() !== 0) return;
        const popover = activeDriver.getState('popover');
        if (popover) syncWelcomeAnchorToPopover(popover, activeDriver);
      },
      onCloseClick: () => {
        exitTourRef.current?.();
      },
      onDestroyed: () => {
        if (!skipMarkOnDestroyRef.current) {
          markTourCompleted();
        }
        skipMarkOnDestroyRef.current = false;
        driverRef.current = null;
        setIsTourActive(false);
        cleanupStaleDriverUi();
      },
    });

    driverRef.current = driverObj;
    driverObj.drive();
    return true;
  }, [destroyDriver]);

  startTourRef.current = startTour;

  useEffect(() => {
    cleanupStaleDriverUi();
  }, []);

  useEffect(() => {
    if (!enabled || isTourCompleted() || autoStartStartedRef.current) return undefined;

    autoStartStartedRef.current = true;

    let cancelled = false;
    let retryTimer = null;
    let attempts = 0;

    const scheduleRetry = (delay = AUTO_START_RETRY_MS) => {
      if (cancelled || attempts >= AUTO_START_MAX_ATTEMPTS || isTourCompleted()) return;
      retryTimer = window.setTimeout(tryAutoStart, delay);
    };

    const tryAutoStart = () => {
      if (cancelled || isTourCompleted()) return;
      if (driverRef.current?.isActive()) return;

      attempts += 1;

      if (!areTourAnchorsReady()) {
        scheduleRetry();
        return;
      }

      const started = startTourRef.current?.({ force: false });
      if (started === false) {
        scheduleRetry();
        return;
      }

      window.setTimeout(() => {
        if (cancelled || isTourCompleted() || driverRef.current?.isActive()) return;
        scheduleRetry();
      }, AUTO_START_RETRY_MS);
    };

    const initialTimer = window.setTimeout(() => {
      if (cancelled || isTourCompleted()) return;
      requestAnimationFrame(() => {
        if (cancelled || isTourCompleted()) return;
        requestAnimationFrame(() => {
          if (cancelled || isTourCompleted()) return;
          tryAutoStart();
        });
      });
    }, AUTO_START_INITIAL_DELAY_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(initialTimer);
      if (retryTimer) window.clearTimeout(retryTimer);
    };
  }, [enabled]);

  useEffect(() => {
    if (!isTourActive) return undefined;

    const onKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      exitTourRef.current?.();
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isTourActive]);

  useEffect(() => () => {
    destroyDriver({ skipMark: true });
    driverRef.current = null;
    cleanupStaleDriverUi();
  }, [destroyDriver]);

  return { startTour, exitTour, isTourActive };
}
