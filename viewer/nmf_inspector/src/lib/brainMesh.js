import * as THREE from 'three';

export const BRAIN_MESH_CENTER = [0.803, -2.16, -3.09];
export const BRAIN_HEMI_SPLIT_X = BRAIN_MESH_CENTER[0];
export const BRAIN_SOLID_OPACITY_THRESHOLD = 0.95;
export const BRAIN_MESH_RENDER_ORDER = 0;
export const DEFAULT_BRAIN_COLOR = '#e6e6e6';
export const INSULA_HIGHLIGHT_OPACITY = 0.6;

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

export function applyBrainMaterial(
  root,
  opacity,
  { forceSolid = false, lit = false, renderOrder = BRAIN_MESH_RENDER_ORDER } = {},
) {
  root.traverse((child) => {
    if (child.isMesh) {
      child.material = createBrainMaterial(opacity, { forceSolid, lit });
      child.renderOrder = renderOrder;
    }
  });
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
