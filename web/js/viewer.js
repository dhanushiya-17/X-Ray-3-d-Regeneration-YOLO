// Three.js stage: one persistent canvas behind every page.
import * as THREE from "three";
import { OrbitControls } from "../vendor/three/OrbitControls.js";

const BONE = new THREE.Color("#e9e1d2");
const RAMP = [new THREE.Color("#ede3d1"), new THREE.Color("#f2b56b"), new THREE.Color("#e3572e")];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function decode(b64, Type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// Canonical SSM frame -> scene: long axis (x) becomes up, trochanter (+y) to screen left, AP (z) toward camera.
function toScene(positions) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    out[i] = -positions[i + 1];
    out[i + 1] = positions[i];
    out[i + 2] = positions[i + 2];
  }
  return out;
}

function rampColor(t, target) {
  const x = Math.min(Math.max(t, 0), 1) * 2;
  const i = Math.min(Math.floor(x), 1);
  return target.copy(RAMP[i]).lerp(RAMP[i + 1], x - i);
}

function fresnelMaterial(color, strength = 1) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uStrength: { value: strength }, uOpacity: { value: 1 } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uStrength; uniform float uOpacity; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2); gl_FragColor = vec4(uColor * (0.25 + 1.6*f) * uStrength, (0.06 + 0.9*f) * uOpacity); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
}

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 1, 20000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotateSpeed = 1.1;
    this.controls.autoRotate = !reduceMotion.matches;
    this.controls.enablePan = false;

    this.scene.add(new THREE.HemisphereLight(0xcfe6ff, 0x141b22, 0.75));
    const key = new THREE.DirectionalLight(0xfff1df, 2.4); key.position.set(400, 700, 900); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fd2ff, 2.0); rim.position.set(-700, 300, -800); this.scene.add(rim);
    const fill = new THREE.DirectionalLight(0xbfd8ee, 0.5); fill.position.set(-500, -200, 600); this.scene.add(fill);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.mode = "solid";        // solid | wireframe | xray
    this.colorBy = "bone";      // bone | spread
    this.ghostOn = false;
    this.primary = null;        // {mesh, geometry, spread, kind}
    this.ghost = null;
    this.center = new THREE.Vector3();
    this.radius = 250;
    this.fadeT = 1;
    this._tween = null;

    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    this._resize();
    this.renderer.setAnimationLoop(() => this._frame());
    reduceMotion.addEventListener?.("change", () => { if (reduceMotion.matches) this.setAutoRotate(false); });
  }

  _resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _buildGeometry(payload) {
    const pos = toScene(decode(payload.positions, Float32Array));
    const idx = decode(payload.indices, Uint32Array);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    const colors = new Float32Array(pos.length);
    const spread = payload.spread ? decode(payload.spread, Float32Array) : null;
    const c = new THREE.Color();
    for (let i = 0; i < pos.length / 3; i++) {
      (spread ? rampColor(spread[i] / 0.5, c) : c.copy(BONE)).toArray(colors, i * 3);
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    return { geometry: g, spread };
  }

  _materialFor(kind) {
    if (this.mode === "xray") return fresnelMaterial(kind === "ghost" ? "#7fb6e0" : "#bfe2ff", kind === "ghost" ? 0.55 : 1);
    if (kind === "ghost") return new THREE.MeshBasicMaterial({ color: 0x8fc8f0, wireframe: true, transparent: true, opacity: 0.12, depthWrite: false });
    const useSpread = this.colorBy === "spread" && this.primary?.spread;
    return new THREE.MeshPhysicalMaterial({
      color: useSpread ? 0xffffff : BONE, vertexColors: !!useSpread, roughness: 0.52, metalness: 0.0,
      clearcoat: 0.35, clearcoatRoughness: 0.45, sheen: 0.25, sheenColor: new THREE.Color("#cfe6ff"),
      wireframe: this.mode === "wireframe", transparent: true, opacity: 1,
    });
  }

  _refreshMaterials() {
    for (const item of [this.primary, this.ghost]) {
      if (!item) continue;
      item.mesh.material.dispose();
      item.mesh.material = this._materialFor(item.kind);
    }
    if (this.ghost) this.ghost.mesh.visible = this.ghostOn;
  }

  setMesh(payload, { kind = "result", animate = true } = {}) {
    const { geometry, spread } = this._buildGeometry(payload);
    const old = this.primary;
    this.primary = { geometry, spread, kind: "primary", source: kind, mesh: null };
    const mesh = new THREE.Mesh(geometry, this._materialFor("primary"));
    this.primary.mesh = mesh;
    this.root.add(mesh);
    const box = geometry.boundingBox;
    box.getCenter(this.center);
    this.radius = geometry.boundingSphere.radius;
    if (old) {
      this._fadeOut(old);
    } else {
      this.frame("ap", false);
    }
    if (animate && !reduceMotion.matches) {
      mesh.material.opacity = 0;
      this.fadeT = 0;
      this.fadeStart = performance.now();
    }
    if (this.ghost) this.ghost.mesh.position.set(0, 0, 0);
  }

  setGhost(payload) {
    if (this.ghost) { this.root.remove(this.ghost.mesh); this.ghost.geometry.dispose(); this.ghost.mesh.material.dispose(); }
    if (!payload) { this.ghost = null; return; }
    const { geometry } = this._buildGeometry(payload);
    const mesh = new THREE.Mesh(geometry, null);
    this.ghost = { geometry, kind: "ghost", mesh };
    mesh.material = this._materialFor("ghost");
    mesh.visible = this.ghostOn;
    mesh.renderOrder = 2;
    this.root.add(mesh);
  }

  _fadeOut(item) {
    const start = performance.now();
    const mat = item.mesh.material;
    const step = () => {
      const t = Math.min((performance.now() - start) / 700, 1);
      if ("opacity" in mat) mat.opacity = 1 - t;
      if (mat.uniforms) mat.uniforms.uOpacity.value = 1 - t;
      if (t < 1 && !reduceMotion.matches) requestAnimationFrame(step);
      else { this.root.remove(item.mesh); item.geometry.dispose(); mat.dispose(); }
    };
    step();
  }

  setMode(mode) { this.mode = mode; this._refreshMaterials(); }
  setColorBy(c) { this.colorBy = c; this._refreshMaterials(); }
  setGhostVisible(v) { this.ghostOn = v; if (this.ghost) this.ghost.mesh.visible = v; }
  setAutoRotate(v) { this.controls.autoRotate = v && !reduceMotion.matches; }
  get autoRotate() { return this.controls.autoRotate; }

  frame(view = "ap", animate = true) {
    const r = this.radius || 250;
    const d = (r / Math.sin((this.camera.fov * Math.PI) / 360)) * (window.innerWidth < 860 ? 1.25 : 1.55);
    const c = this.center.clone();
    const dirs = { ap: [0, 0.08, 1], lateral: [-1, 0.08, 0], axial: [0, 1, 0.0001], iso: [0.7, 0.35, 0.7] };
    const v = new THREE.Vector3(...(dirs[view] || dirs.ap)).normalize().multiplyScalar(d).add(c);
    this.controls.minDistance = r * 0.6;
    this.controls.maxDistance = d * 3;
    if (!animate || reduceMotion.matches) {
      this.camera.position.copy(v); this.controls.target.copy(c); this.controls.update(); return;
    }
    const p0 = this.camera.position.clone(), t0 = this.controls.target.clone(), start = performance.now();
    this._tween = () => {
      const k = Math.min((performance.now() - start) / 650, 1);
      const e = 1 - Math.pow(1 - k, 3);
      this.camera.position.lerpVectors(p0, v, e);
      this.controls.target.lerpVectors(t0, c, e);
      if (k >= 1) this._tween = null;
    };
  }

  screenshot() {
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL("image/png");
  }

  _frame() {
    if (this._tween) this._tween();
    if (this.fadeT < 1 && this.primary) {
      this.fadeT = Math.min((performance.now() - (this.fadeStart || 0)) / 800, 1);
      const m = this.primary.mesh.material;
      if ("opacity" in m && !m.uniforms) { m.opacity = this.fadeT; if (this.fadeT >= 1) m.transparent = false; m.needsUpdate = true; }
      if (m.uniforms) m.uniforms.uOpacity.value = this.fadeT;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
