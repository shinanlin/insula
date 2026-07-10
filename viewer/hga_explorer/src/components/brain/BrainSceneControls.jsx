import React, { useEffect, useMemo, useRef } from 'react';
import { OrbitControls } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import {
  BRAIN_CAMERA,
  BRAIN_ORBIT_TARGET,
  BRAIN_UP,
  INSULA_CAMERA,
  INSULA_ORBIT_TARGET,
  INSULA_ORBIT_DISTANCE,
  INSULA_ORBIT_AZIMUTH_DEG,
  INSULA_ORBIT_ELEVATION_DEG,
  INSULA_CAMERA_Z_OFFSET,
  brainCameraPosition,
} from '../../constants/brain.js';

function resolveCameraPreset(preset, insulaOrbitTarget) {
  if (preset === 'insula') {
    const target = insulaOrbitTarget ?? INSULA_ORBIT_TARGET;
    return {
      camera: {
        position: brainCameraPosition(
          target,
          INSULA_ORBIT_DISTANCE,
          INSULA_ORBIT_AZIMUTH_DEG,
          INSULA_ORBIT_ELEVATION_DEG,
          INSULA_CAMERA_Z_OFFSET,
        ),
        fov: INSULA_CAMERA.fov,
        up: BRAIN_UP,
      },
      target,
    };
  }
  return {
    camera: BRAIN_CAMERA,
    target: BRAIN_ORBIT_TARGET,
  };
}

function insulaTargetKey(insulaOrbitTarget) {
  if (!insulaOrbitTarget?.length) return '';
  return insulaOrbitTarget.map((value) => Number(value).toFixed(4)).join(',');
}

export default function BrainSceneControls({
  resetToken = 0,
  cameraPreset = 'default',
  insulaOrbitTarget = null,
}) {
  const camera = useThree((state) => state.camera);
  const controlsRef = useRef(null);
  const targetKey = insulaTargetKey(insulaOrbitTarget);
  const presetConfig = useMemo(
    () => resolveCameraPreset(cameraPreset, insulaOrbitTarget),
    [cameraPreset, targetKey, insulaOrbitTarget],
  );

  const applyPreset = () => {
    const { camera: presetCamera, target } = presetConfig;
    camera.up.set(...BRAIN_UP);
    camera.position.set(...presetCamera.position);
    if ('fov' in camera) {
      camera.fov = presetCamera.fov;
      camera.updateProjectionMatrix();
    }
    camera.lookAt(...target);
    if (controlsRef.current) {
      controlsRef.current.target.set(...target);
      controlsRef.current.update();
    }
  };

  useEffect(() => {
    applyPreset();
  }, [camera, cameraPreset, targetKey]);

  useEffect(() => {
    if (!resetToken) return;
    applyPreset();
  }, [resetToken, cameraPreset, targetKey]);

  return (
    <OrbitControls
      ref={controlsRef}
      target={presetConfig.target}
      enableDamping
      dampingFactor={0.08}
      rotateSpeed={0.85}
      minDistance={110}
      maxDistance={360}
      enablePan={false}
      minPolarAngle={0.12}
      maxPolarAngle={Math.PI - 0.12}
    />
  );
}
