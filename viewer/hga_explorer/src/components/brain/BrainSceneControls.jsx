import React, { useEffect, useRef } from 'react';
import { OrbitControls } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { BRAIN_ORBIT_TARGET, BRAIN_CAMERA, BRAIN_UP } from '../../constants/brain.js';

export default function BrainSceneControls({ resetToken = 0 }) {
  const camera = useThree((state) => state.camera);
  const controlsRef = useRef(null);

  useEffect(() => {
    camera.up.set(...BRAIN_UP);
    camera.position.set(...BRAIN_CAMERA.position);
    if ('fov' in camera) {
      camera.fov = BRAIN_CAMERA.fov;
      camera.updateProjectionMatrix();
    }
    camera.lookAt(...BRAIN_ORBIT_TARGET);
  }, [camera]);

  useEffect(() => {
    if (!resetToken) return;
    camera.up.set(...BRAIN_UP);
    camera.position.set(...BRAIN_CAMERA.position);
    if ('fov' in camera) {
      camera.fov = BRAIN_CAMERA.fov;
      camera.updateProjectionMatrix();
    }
    if (controlsRef.current) {
      controlsRef.current.target.set(...BRAIN_ORBIT_TARGET);
      controlsRef.current.update();
    }
    camera.lookAt(...BRAIN_ORBIT_TARGET);
  }, [resetToken, camera]);

  return (
    <OrbitControls
      ref={controlsRef}
      target={BRAIN_ORBIT_TARGET}
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
