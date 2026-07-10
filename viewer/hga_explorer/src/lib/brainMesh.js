import * as THREE from 'three';
import {
  BRAIN_HEMI_SPLIT_X,
  BRAIN_MESH_RENDER_ORDER,
  BRAIN_SOLID_OPACITY_THRESHOLD,
  DEFAULT_BRAIN_COLOR,
  KDE_OVERLAY_HIGHLIGHT_CAP,
  KDE_OVERLAY_LIT_MIX,
  KDE_OVERLAY_SHADE_FLOOR,
} from '../constants/brain.js';

export function createBrainMaterial(opacity, { forceSolid = false, lit = false } = {}) {
  const transparent = opacity < 0.999;
  const depthWrite = forceSolid || opacity >= BRAIN_SOLID_OPACITY_THRESHOLD;
  if (lit) {
    return new THREE.MeshPhongMaterial({
      color: DEFAULT_BRAIN_COLOR,
      specular: 0x111111,
      shininess: 12,
      transparent,
      opacity,
      depthWrite,
      side: THREE.FrontSide,
      flatShading: false,
    });
  }
  return new THREE.MeshBasicMaterial({
    color: DEFAULT_BRAIN_COLOR,
    transparent,
    opacity,
    depthWrite,
    side: THREE.FrontSide,
  });
}

export function hemisphereClippingPlanes(hemisphereView) {
  if (hemisphereView === 'both') return [];
  if (hemisphereView === 'left') {
    return [new THREE.Plane(new THREE.Vector3(1, 0, 0), -BRAIN_HEMI_SPLIT_X)];
  }
  return [new THREE.Plane(new THREE.Vector3(-1, 0, 0), BRAIN_HEMI_SPLIT_X)];
}

export function applyHemisphereClipping(root, hemisphereView) {
  const planes = hemisphereClippingPlanes(hemisphereView);
  root.traverse((child) => {
    if (!child.isMesh) return;
    if (planes.length) {
      child.material.clippingPlanes = planes;
      child.material.clipIntersection = false;
    } else {
      child.material.clippingPlanes = null;
    }
    child.material.needsUpdate = true;
  });
}

function createKdeOverlayMaterial(planes) {
  const material = new THREE.MeshLambertMaterial({
    color: 0xffffff,
    vertexColors: true,
    transparent: true,
    opacity: 1,
    alphaTest: 0.01,
    depthWrite: true,
    depthTest: true,
    side: THREE.FrontSide,
    clippingPlanes: planes.length ? planes : null,
    clipIntersection: false,
  });

  const litMix = Math.max(0, Math.min(1, KDE_OVERLAY_LIT_MIX));
  const shadeFloor = Math.max(0.5, Math.min(0.98, KDE_OVERLAY_SHADE_FLOOR));
  const highlightCap = Math.max(0.75, Math.min(1, KDE_OVERLAY_HIGHLIGHT_CAP));
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;',
      `
        vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
        float wrapNdl = clamp( dot( normalize( normal ), normalize( vec3( 0.42, 0.24, 0.88 ) ) ) * 0.5 + 0.5, 0.0, 1.0 );
        vec3 softColor = diffuseColor.rgb * mix( ${shadeFloor.toFixed(4)}, 1.0, wrapNdl );
        outgoingLight = mix( softColor, outgoingLight, ${litMix.toFixed(4)} );
        float peak = max( max( outgoingLight.r, outgoingLight.g ), outgoingLight.b );
        outgoingLight *= min( 1.0, ${highlightCap.toFixed(4)} / max( peak, 1e-4 ) );
      `,
    );
  };
  material.customProgramCacheKey = () => (
    `kde-overlay-${litMix}-${shadeFloor}-${highlightCap}`
  );

  return material;
}

export function applyKdeOverlayMaterial(root, hemisphereView) {
  const planes = hemisphereClippingPlanes(hemisphereView);
  root.traverse((child) => {
    if (!child.isMesh || !child.userData.isKdeOverlay) return;
    child.material = createKdeOverlayMaterial(planes);
    child.renderOrder = 1;
  });
}

export function applyBrainMaterial(
  root,
  opacity,
  { forceSolid = false, lit = false, renderOrder = BRAIN_MESH_RENDER_ORDER } = {},
) {
  root.traverse((child) => {
    if (child.isMesh && !child.userData.isKdeOverlay) {
      child.material = createBrainMaterial(opacity, { forceSolid, lit });
      child.renderOrder = renderOrder;
    }
  });
}

export function normalizeElectrodeHemisphere(hemi) {
  const value = String(hemi || '').trim().toUpperCase();
  if (value === 'L' || value === 'LEFT') return 'left';
  if (value === 'R' || value === 'RIGHT') return 'right';
  return null;
}

export function electrodeMatchesHemisphere(electrode, hemisphereView) {
  if (hemisphereView === 'both') return true;
  const hemi = normalizeElectrodeHemisphere(electrode.hemi);
  if (hemi) return hemi === hemisphereView;
  return hemisphereView === 'left'
    ? electrode.x < BRAIN_HEMI_SPLIT_X
    : electrode.x >= BRAIN_HEMI_SPLIT_X;
}

function splitMeshIntoHemispheres(mesh, splitX) {
  const geometry = mesh.geometry;
  const position = geometry.attributes.position;
  const indexAttr = geometry.index;
  const faceCount = indexAttr ? indexAttr.count / 3 : position.count / 3;
  const leftIndices = [];
  const rightIndices = [];

  for (let face = 0; face < faceCount; face += 1) {
    let centroidX = 0;
    const vertexIndices = [];
    for (let corner = 0; corner < 3; corner += 1) {
      const vertexIndex = indexAttr ? indexAttr.getX(face * 3 + corner) : face * 3 + corner;
      vertexIndices.push(vertexIndex);
      centroidX += position.getX(vertexIndex);
    }
    centroidX /= 3;
    const bucket = centroidX <= splitX ? leftIndices : rightIndices;
    vertexIndices.forEach((vertexIndex) => bucket.push(vertexIndex));
  }

  const group = new THREE.Group();
  group.position.copy(mesh.position);
  group.quaternion.copy(mesh.quaternion);
  group.scale.copy(mesh.scale);

  [['left', leftIndices], ['right', rightIndices]].forEach(([hemisphereId, indices]) => {
    if (!indices.length) return;
    const hemiGeometry = geometry.clone();
    hemiGeometry.setIndex(indices);
    hemiGeometry.computeVertexNormals();
    const hemiMesh = new THREE.Mesh(hemiGeometry, mesh.material);
    hemiMesh.userData.hemisphere = hemisphereId;
    group.add(hemiMesh);
  });

  return group;
}

export function prepareBrainWithHemispheres(root, splitX) {
  const replacements = [];
  root.traverse((child) => {
    if (child.isMesh) replacements.push({ parent: child.parent, mesh: child });
  });
  replacements.forEach(({ parent, mesh }) => {
    const group = splitMeshIntoHemispheres(mesh, splitX);
    parent.add(group);
    parent.remove(mesh);
    mesh.geometry?.dispose();
  });
  return root;
}

export function setBrainHemisphereVisibility(root, hemisphereView) {
  root.traverse((child) => {
    if (!child.userData?.hemisphere) return;
    child.visible = hemisphereView === 'both' || child.userData.hemisphere === hemisphereView;
  });
}

export function prepareKdeOverlayBrain(scene, splitX) {
  const overlay = prepareBrainWithHemispheres(scene.clone(true), splitX);
  overlay.traverse((child) => {
    if (child.isMesh) {
      child.userData.isKdeOverlay = true;
    }
  });
  return overlay;
}

export function applyOverlayVertexColors(root, colors) {
  root.traverse((child) => {
    if (!child.isMesh || !child.userData.isKdeOverlay) return;
    const { geometry } = child;
    const vertexCount = geometry.attributes.position.count;
    let colorAttr = geometry.attributes.color;
    if (!colorAttr || colorAttr.count !== vertexCount) {
      colorAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 4), 4);
      geometry.setAttribute('color', colorAttr);
    }
    colorAttr.array.set(colors);
    colorAttr.needsUpdate = true;
  });
}
