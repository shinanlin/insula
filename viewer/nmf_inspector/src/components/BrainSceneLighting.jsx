import React from 'react';

const BRAIN_SCENE_BACKGROUND = '#ffffff';

/** Match hga_explorer BrainSceneLighting mneStyle preset. */
export default function BrainSceneLighting({ mneStyle = true }) {
  if (mneStyle) {
    return (
      <>
        <color attach="background" args={[BRAIN_SCENE_BACKGROUND]} />
        <ambientLight intensity={0.44} />
        <hemisphereLight args={['#ffffff', '#fafafa', 0.32]} />
        <directionalLight position={[80, 40, 140]} intensity={0.88} />
        <directionalLight position={[-60, 20, 100]} intensity={0.24} />
        <directionalLight position={[0, -120, 60]} intensity={0.1} />
      </>
    );
  }
  return (
    <>
      <color attach="background" args={[BRAIN_SCENE_BACKGROUND]} />
      <ambientLight intensity={0.95} />
      <hemisphereLight args={['#ffffff', '#ffffff', 0.55]} />
      <directionalLight position={[0, 60, 220]} intensity={0.42} />
      <directionalLight position={[0, -80, -120]} intensity={0.12} />
    </>
  );
}
