import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const HOME_LAT = 26;
const HOME_LON = -12;
const Z_AXIS = new THREE.Vector3(0, 0, 1);
const Y_AXIS = new THREE.Vector3(0, 1, 0);

const LAYER_MODE_INDEX = {
  temperature: 0,
  anomaly: 1,
  wind: 2,
  rain: 3,
  pressure: 4,
};

export function latLonToVec3(lat, lon, radius = 1, out = new THREE.Vector3()) {
  const phi = lat * DEG2RAD;
  const lam = lon * DEG2RAD;
  const cosPhi = Math.cos(phi);
  return out.set(
    radius * cosPhi * Math.cos(lam),
    radius * Math.sin(phi),
    -radius * cosPhi * Math.sin(lam)
  );
}

export function vec3ToLatLon(vec) {
  const len = vec.length() || 1;
  const lat = Math.asin(Math.max(-1, Math.min(1, vec.y / len))) * RAD2DEG;
  const lon = Math.atan2(-vec.z, vec.x) * RAD2DEG;
  return { lat, lon };
}

export function parseHexRgba(hex) {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  const a = clean.length >= 8 ? parseInt(clean.slice(6, 8), 16) : 255;
  return [r, g, b, a];
}

export function sampleLayerColourAtValue(layerSpec, val) {
  const stops = layerSpec.stops;
  if (!stops || !stops.length) return [255, 255, 255, 255];
  const first = parseHexRgba(stops[0][1]);
  if (val <= stops[0][0]) return first;
  const last = parseHexRgba(stops[stops.length - 1][1]);
  if (val >= stops[stops.length - 1][0]) return last;
  for (let s = 0; s < stops.length - 1; s++) {
    const v0 = stops[s][0];
    const v1 = stops[s + 1][0];
    if (val >= v0 && val <= v1) {
      const c0 = parseHexRgba(stops[s][1]);
      const c1 = parseHexRgba(stops[s + 1][1]);
      const f = (val - v0) / Math.max(1e-6, v1 - v0);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
        Math.round(c0[3] + f * (c1[3] - c0[3])),
      ];
    }
  }
  return last;
}

function buildLutTexture(layerSpec) {
  const data = new Uint8Array(256 * 4);
  for (let i = 0; i < 255; i++) {
    const t = i / 254;
    const val = layerSpec.lo + t * (layerSpec.hi - layerSpec.lo);
    const c = sampleLayerColourAtValue(layerSpec, val);
    data[i * 4] = c[0];
    data[i * 4 + 1] = c[1];
    data[i * 4 + 2] = c[2];
    data[i * 4 + 3] = c[3];
  }
  data[255 * 4 + 3] = 0;
  const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

export function computeViewportCameraParams(width, height) {
  const aspect = Math.max(0.3, width / Math.max(1, height));
  const fov = aspect < 1.0 ? Math.min(54, 38 + (1.0 - aspect) * 28) : 38;
  const tanHalfX = aspect * Math.tan(fov * 0.5 * DEG2RAD);
  const targetWidthFrac = aspect < 1.0 ? 0.76 : 0.70;
  const ft = Math.max(0.12, targetWidthFrac * tanHalfX);
  const exactDist = Math.sqrt(1 + ft * ft) / ft;
  const initialDist = aspect < 1.0 ? Math.max(3.4, Math.min(6.0, exactDist)) : 2.75;
  const altitudeScale = aspect < 0.9 ? Math.min(2.15, 0.95 / Math.max(0.42, aspect)) : 1.0;
  return { aspect, fov, initialDist, altitudeScale };
}

function maxPixelRatioForViewport() {
  return window.innerWidth < 768 ? 1.5 : 2;
}

export class WeatherGlobe {
  constructor(stageEl, overlayEl, { onSpotClick, onPinClick, onUserInteract } = {}) {
    this.stageEl = stageEl;
    this.overlayEl = overlayEl;
    this.onSpotClick = onSpotClick;
    this.onPinClick = onPinClick;
    this.onUserInteract = onUserInteract;

    this.manifest = null;
    this.layerSpecs = new Map();
    this.layerLuts = new Map();
    this.layerTextures = new Map();
    this.layerBuffers = new Map();
    this.baseMapTexture = null;

    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    this.reducedMotion = Boolean(mq?.matches);
    mq?.addEventListener?.('change', e => {
      this.reducedMotion = Boolean(e.matches);
      if (this.reducedMotion) {
        this.autoSpin = false;
      }
    });

    this.currentLayer = 'temperature';
    this.transition = null;
    this.flight = null;
    this.tourControlled = false;
    this.autoSpin = !this.reducedMotion;

    this.focusRegion = null;
    this.focusCallout = null;
    this.searchMarker = null;

    this.hlElements = [];
    this.storyPinElements = [];
    this.cityElements = [];

    // Pre-allocated scratch structs to eliminate per-particle / per-frame GC churn
    this._windScratch = { u: 0, v: 0, speed: 0 };
    this._vHead = new THREE.Vector3();
    this._vTail = new THREE.Vector3();
    this._camDir = new THREE.Vector3();
    this._projVec = new THREE.Vector3();
    this._camRight = new THREE.Vector3();
    this._spinQuat = new THREE.Quaternion();
    this._flightQuat = new THREE.Quaternion();
    this._flightDir = new THREE.Vector3();
    this._flightLook = new THREE.Vector3();
    this._ringOutward = new THREE.Vector3();

    this._elapsedSec = 0;
    this._initScene();
    this._initParticles();
    this._initFocusSpotlight();
    this._initSearchMarker();
    this._bindEvents();
    this._lastFrame = performance.now();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _initScene() {
    this.scene = new THREE.Scene();
    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    this.camera = new THREE.PerspectiveCamera(vp.fov, vp.aspect, 0.05, 50);
    latLonToVec3(HOME_LAT, HOME_LON, vp.initialDist, this.camera.position);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatioForViewport()));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.stageEl.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.rotateSpeed = 0.55;
    this.controls.zoomSpeed = 0.75;
    this.controls.minDistance = 1.24;
    this.controls.maxDistance = 7.2;

    const starCount = 1200;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const u = Math.random() * 2 - 1;
      const theta = Math.random() * Math.PI * 2;
      const r = 16 + Math.random() * 10;
      const s = Math.sqrt(1 - u * u);
      starPos[i * 3] = r * s * Math.cos(theta);
      starPos[i * 3 + 1] = r * u;
      starPos[i * 3 + 2] = r * s * Math.sin(theta);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xbfd9f2, size: 0.045, transparent: true, opacity: 0.55 });
    this.scene.add(new THREE.Points(starGeo, starMat));

    this.earthGroup = new THREE.Group();
    this.scene.add(this.earthGroup);

    const blankTex = new THREE.DataTexture(new Uint8Array([9, 24, 44, 255]), 1, 1, THREE.RGBAFormat);
    blankTex.needsUpdate = true;

    this.globeUniforms = {
      uBaseMap: { value: blankTex },
      uTexA: { value: blankTex },
      uTexB: { value: blankTex },
      uLutA: { value: blankTex },
      uLutB: { value: blankTex },
      uChannelA: { value: 0 },
      uChannelB: { value: 0 },
      uSqrtA: { value: 0.0 },
      uSqrtB: { value: 0.0 },
      uMix: { value: 0.0 },
      uTime: { value: 0.0 },
      uLayerMode: { value: 0 },
      uGrid: { value: new THREE.Vector2(720, 361) },
      uSunDir: { value: new THREE.Vector3(-0.55, 0.45, 0.7).normalize() },
    };

    const globeMat = new THREE.ShaderMaterial({
      uniforms: this.globeUniforms,
      vertexShader: `
        varying vec3 vObjPos;
        varying vec3 vWorldNormal;
        void main() {
          vObjPos = position;
          vWorldNormal = normalize(mat3(modelMatrix) * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uBaseMap;
        uniform sampler2D uTexA;
        uniform sampler2D uTexB;
        uniform sampler2D uLutA;
        uniform sampler2D uLutB;
        uniform int uChannelA;
        uniform int uChannelB;
        uniform float uSqrtA;
        uniform float uSqrtB;
        uniform float uMix;
        uniform float uTime;
        uniform int uLayerMode;
        uniform vec2 uGrid;
        uniform vec3 uSunDir;

        varying vec3 vObjPos;
        varying vec3 vWorldNormal;

        vec4 sampleLayer(sampler2D fieldTex, sampler2D lutTex, int chan, float isSqrt, vec2 uvField, out float normVal) {
          vec4 raw = texture2D(fieldTex, uvField);
          float byteNorm = (chan == 2) ? raw.b : raw.r;
          if (byteNorm > 0.997) {
            normVal = 0.0;
            return vec4(0.0);
          }
          float t = clamp(byteNorm * (255.0 / 254.0), 0.0, 1.0);
          if (isSqrt > 0.5) {
            t = t * t;
          }
          normVal = t;
          float lutU = (t * 254.0 + 0.5) / 256.0;
          return texture2D(lutTex, vec2(lutU, 0.5));
        }

        void main() {
          vec3 nObj = normalize(vObjPos);
          float latDeg = degrees(asin(clamp(nObj.y, -1.0, 1.0)));
          float lonDeg = degrees(atan(-nObj.z, nObj.x));

          float u = fract((lonDeg + 180.0) / 360.0);
          float vTop = clamp((90.0 - latDeg) / 180.0, 0.0, 1.0);

          vec3 baseCol = texture2D(uBaseMap, vec2(u, vTop)).rgb;

          vec2 uvField = vec2(u + 0.5 / uGrid.x, (vTop * (uGrid.y - 1.0) + 0.5) / uGrid.y);
          float normA = 0.0;
          float normB = 0.0;
          vec4 colA = sampleLayer(uTexA, uLutA, uChannelA, uSqrtA, uvField, normA);
          vec4 colB = sampleLayer(uTexB, uLutB, uChannelB, uSqrtB, uvField, normB);
          vec4 weather = mix(colA, colB, uMix);
          float intensity = mix(normA, normB, uMix);

          if (uLayerMode == 3 && weather.a > 0.15) {
            float wave = 0.5 + 0.5 * sin(uTime * 4.4 - latDeg * 0.65 + lonDeg * 0.55);
            float heavyBoost = smoothstep(0.08, 0.45, intensity);
            weather.rgb += vec3(0.10, 0.22, 0.34) * wave * heavyBoost;
            weather.a = min(1.0, weather.a + 0.12 * wave * heavyBoost);
          } else if (uLayerMode == 1) {
            float extremeness = smoothstep(0.16, 0.38, abs(intensity - 0.5));
            float pulse = 0.5 + 0.5 * sin(uTime * 3.1 - latDeg * 0.25);
            weather.rgb *= (1.0 + 0.12 * pulse * extremeness);
          } else if (uLayerMode == 0) {
            float extremeness = smoothstep(0.24, 0.44, abs(intensity - 0.52));
            float pulse = 0.5 + 0.5 * sin(uTime * 2.6 + lonDeg * 0.2);
            weather.rgb *= (1.0 + 0.08 * pulse * extremeness);
          }

          vec3 nWorld = normalize(vWorldNormal);
          float diffuse = 0.80 + 0.24 * max(0.0, dot(nWorld, uSunDir));
          vec3 litBase = baseCol * diffuse;
          vec3 blended = mix(litBase, weather.rgb, weather.a * 0.86);

          float rim = pow(1.0 - max(0.0, dot(nWorld, normalize(cameraPosition - nObj))), 3.2);
          blended += vec3(0.14, 0.34, 0.58) * rim * 0.40;

          gl_FragColor = vec4(blended, 1.0);
        }
      `,
    });

    this.sphereMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), globeMat);
    this.earthGroup.add(this.sphereMesh);

    const haloMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vView;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vNormal = normalize(normalMatrix * normal);
          vView = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vView;
        void main() {
          float intensity = pow(clamp(0.72 - dot(vNormal, vView), 0.0, 1.0), 2.8);
          gl_FragColor = vec4(0.22, 0.56, 0.88, intensity * 0.46);
        }
      `,
    });
    this.earthGroup.add(new THREE.Mesh(new THREE.SphereGeometry(1.042, 80, 56), haloMat));

    this.isobarGroup = new THREE.Group();
    this.isobarGroup.visible = false;
    this.earthGroup.add(this.isobarGroup);
  }

  _initParticles() {
    this.globalParticleCount = 3000;
    this.regionalParticleCount = 1800;
    this.particleCount = this.globalParticleCount + this.regionalParticleCount;

    this.particles = new Float32Array(this.particleCount * 3);
    const positions = new Float32Array(this.particleCount * 6);
    const colors = new Float32Array(this.particleCount * 6);

    for (let i = 0; i < this.particleCount; i++) {
      this._respawnParticle(i, true);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setDrawRange(0, this.globalParticleCount * 2);

    this.particleMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.particleLines = new THREE.LineSegments(geo, this.particleMat);
    this.earthGroup.add(this.particleLines);
  }

  _initFocusSpotlight() {
    this.focusRingGroup = new THREE.Group();
    this.focusRingGroup.visible = false;

    const ringGeoA = new THREE.RingGeometry(0.012, 0.017, 48);
    const ringGeoB = new THREE.RingGeometry(0.012, 0.016, 48);
    this.focusRingMatA = new THREE.MeshBasicMaterial({
      color: 0xfde047,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    this.focusRingMatB = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    });
    this.focusRingA = new THREE.Mesh(ringGeoA, this.focusRingMatA);
    this.focusRingB = new THREE.Mesh(ringGeoB, this.focusRingMatB);
    this.focusDot = new THREE.Mesh(
      new THREE.CircleGeometry(0.0085, 24),
      new THREE.MeshBasicMaterial({ color: 0xe11d48, side: THREE.DoubleSide, depthWrite: false })
    );
    this.focusRingGroup.add(this.focusRingA, this.focusRingB, this.focusDot);
    this.earthGroup.add(this.focusRingGroup);

    const el = document.createElement('div');
    el.className = 'focus-callout';
    el.hidden = true;
    el.innerHTML =
      '<span class="focus-callout-dot"></span>' +
      '<span class="focus-callout-text">' +
      '<strong class="focus-callout-label"></strong>' +
      '<span class="focus-callout-metric"></span>' +
      '</span>';
    this.overlayEl.appendChild(el);
    this.focusCallout = {
      el,
      labelEl: el.querySelector('.focus-callout-label'),
      metricEl: el.querySelector('.focus-callout-metric'),
      targetVec: new THREE.Vector3(0, 0, 1.006),
      currentVec: new THREE.Vector3(0, 0, 1.006),
      active: false,
    };
  }

  _initSearchMarker() {
    const el = document.createElement('div');
    el.className = 'search-marker';
    el.hidden = true;
    el.innerHTML =
      '<span class="search-marker-pin"></span>' +
      '<span class="search-marker-label"></span>';
    this.overlayEl.appendChild(el);
    this.searchMarker = {
      el,
      labelEl: el.querySelector('.search-marker-label'),
      vec: new THREE.Vector3(0, 0, 1.008),
      active: false,
    };
  }

  setSearchMarker(lat, lon, label = '') {
    if (!this.searchMarker) return;
    latLonToVec3(lat, lon, 1.008, this.searchMarker.vec);
    this.searchMarker.labelEl.textContent = label || '';
    this.searchMarker.labelEl.hidden = !label;
    this.searchMarker.active = true;
  }

  clearSearchMarker() {
    if (!this.searchMarker) return;
    this.searchMarker.active = false;
    this.searchMarker.el.hidden = true;
    this.searchMarker.el.style.visibility = 'hidden';
  }

  _respawnParticle(i, randomAge = false) {
    const isRegional = i >= this.globalParticleCount && this.focusRegion;
    let lat;
    let lon;
    if (isRegional) {
      const spreadLat = 16;
      const cosC = Math.max(0.28, Math.cos(this.focusRegion.lat * DEG2RAD));
      const spreadLon = Math.min(42, 20 / cosC);
      lat = Math.max(-83, Math.min(83, this.focusRegion.lat + (Math.random() * 2 - 1) * spreadLat));
      lon = (((this.focusRegion.lon + (Math.random() * 2 - 1) * spreadLon + 180) % 360) + 360) % 360 - 180;
    } else {
      const u = (Math.random() * 2 - 1) * 0.985;
      lat = Math.asin(u) * RAD2DEG;
      lon = Math.random() * 360 - 180;
    }
    this.particles[i * 3] = lat;
    this.particles[i * 3 + 1] = lon;
    this.particles[i * 3 + 2] = randomAge ? Math.random() * 80 : 0;
  }

  setTourFocusCallout(cue) {
    if (!cue || cue.lat == null || cue.lon == null) {
      this.focusRegion = null;
      this.focusRingGroup.visible = false;
      if (this.focusCallout) {
        this.focusCallout.active = false;
        this.focusCallout.el.hidden = true;
        this.focusCallout.el.style.visibility = 'hidden';
      }
      return;
    }

    const prevRegion = this.focusRegion;
    this.focusRegion = { lat: cue.lat, lon: cue.lon };

    if (!prevRegion || Math.hypot(prevRegion.lat - cue.lat, prevRegion.lon - cue.lon) > 20) {
      for (let i = this.globalParticleCount; i < this.particleCount; i++) {
        this._respawnParticle(i, true);
      }
    }

    const vec = latLonToVec3(cue.lat, cue.lon, 1.005);
    if (!this.focusCallout.active || this.reducedMotion) {
      this.focusCallout.currentVec.copy(vec);
    }
    this.focusCallout.targetVec.copy(vec);
    this.focusCallout.active = true;
    this.focusCallout.labelEl.textContent = cue.label || '';
    this.focusCallout.metricEl.textContent = cue.metric || '';
    this.focusCallout.metricEl.hidden = !cue.metric;
    this.focusRingGroup.visible = true;
  }

  async loadBasemap(url = 'assets/basemap.webp') {
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.decoding = 'async';
      img.src = url;
      await img.decode();

      const baseTex = new THREE.Texture(img);
      baseTex.flipY = false;
      baseTex.wrapS = THREE.RepeatWrapping;
      baseTex.wrapT = THREE.ClampToEdgeWrapping;
      baseTex.minFilter = THREE.LinearFilter;
      baseTex.magFilter = THREE.LinearFilter;
      baseTex.needsUpdate = true;

      if (this.baseMapTexture) {
        this.baseMapTexture.dispose();
      }
      this.baseMapTexture = baseTex;
      this.globeUniforms.uBaseMap.value = baseTex;
    } catch (err) {
      console.warn('Basemap texture load failed:', err);
    }
  }

  async loadWeatherManifest(manifestUrl = 'data/manifest.json') {
    const resp = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!resp.ok) throw new Error('Unable to load weather manifest');
    const manifest = await resp.json();
    this.manifest = manifest;
    this.globeUniforms.uGrid.value.set(manifest.grid.nx, manifest.grid.ny);

    for (const spec of manifest.layers) {
      this.layerSpecs.set(spec.id, spec);
      const prevLut = this.layerLuts.get(spec.id);
      this.layerLuts.set(spec.id, buildLutTexture(spec));
      prevLut?.dispose();
    }

    const cacheBust = encodeURIComponent(manifest.generated_utc || 'latest');
    const results = await Promise.allSettled(
      manifest.layers.map(spec => this._loadLayerImage(spec.id, `data/layers/${spec.id}.webp?v=${cacheBust}`))
    );
    results.forEach((res, idx) => {
      if (res.status === 'rejected') {
        console.warn(`Layer ${manifest.layers[idx]?.id} failed to load:`, res.reason);
      }
    });

    this._buildIsobarsAndCentres(manifest.isobars || [], manifest.pressure_centres || []);
    if (!this.layerTextures.has(this.currentLayer)) {
      const firstAvailable = manifest.layers.find(spec => this.layerTextures.has(spec.id));
      if (firstAvailable) {
        this.currentLayer = firstAvailable.id;
      }
    }
    this.setLayer(this.currentLayer, true);
    return manifest;
  }

  async _loadLayerImage(layerId, url) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.src = url;
    await img.decode();

    const nx = img.naturalWidth;
    const ny = img.naturalHeight;
    const c = document.createElement('canvas');
    c.width = nx;
    c.height = ny;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const rgba = ctx.getImageData(0, 0, nx, ny).data;
    this.layerBuffers.set(layerId, { rgba, nx, ny });

    const tex = new THREE.DataTexture(new Uint8Array(rgba.buffer), nx, ny, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.flipY = false;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;

    const prevTex = this.layerTextures.get(layerId);
    this.layerTextures.set(layerId, tex);
    prevTex?.dispose();
  }

  _buildIsobarsAndCentres(isobars, centres) {
    while (this.isobarGroup.children.length) {
      const ch = this.isobarGroup.children.pop();
      ch.geometry?.dispose();
      ch.material?.dispose();
    }
    const pos = [];
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    for (const iso of isobars) {
      const p = iso.path || [];
      for (let i = 0; i < p.length - 2; i += 2) {
        if (Math.abs(p[i + 2] - p[i]) > 30) continue;
        latLonToVec3(p[i + 1], p[i], 1.0024, vA);
        latLonToVec3(p[i + 3], p[i + 2], 1.0024, vB);
        pos.push(vA.x, vA.y, vA.z, vB.x, vB.y, vB.z);
      }
    }
    if (pos.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      this.isobarMat = new THREE.LineBasicMaterial({ color: 0xe6f4ff, transparent: true, opacity: 0.56 });
      this.isobarGroup.add(new THREE.LineSegments(geo, this.isobarMat));
    }

    for (const item of this.hlElements) item.el.remove();
    this.hlElements = centres.map(c => {
      const el = document.createElement('div');
      el.className = `hl-badge ${c.kind === 'L' ? 'low' : 'high'}`;
      el.textContent = `${c.kind} ${c.hpa}`;
      el.hidden = true;
      el.style.visibility = 'hidden';
      this.overlayEl.appendChild(el);
      return { ...c, el, vec: latLonToVec3(c.lat, c.lon, 1.006) };
    });
  }

  /** Label major cities near the current story (dot + name), or clear them with an empty list. */
  setCityLabels(cities) {
    for (const c of this.cityElements) c.el.remove();
    this.cityElements = (cities || []).slice(0, 7).map(c => {
      const el = document.createElement('div');
      el.className = 'city-label';
      el.textContent = c.name;
      el.hidden = true;
      el.style.visibility = 'hidden';
      this.overlayEl.appendChild(el);
      return { el, vec: latLonToVec3(c.lat, c.lon, 1.004) };
    });
  }

  setStoryPins(stories, activeIndex = -1) {
    for (const p of this.storyPinElements) p.el.remove();
    this.storyPinElements = (stories || []).map((st, idx) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `story-pin${idx === activeIndex ? ' active' : ''}`;
      el.textContent = String(idx + 1);
      const place =
        st.country && st.country !== 'Global' ? st.country : st.region || st.headline || `Story ${idx + 1}`;
      el.setAttribute('aria-label', `Story ${idx + 1}: ${place}`);
      el.title = st.headline || `Story ${idx + 1}: ${place}`;
      el.onclick = e => {
        e.stopPropagation();
        this.onPinClick?.(idx);
      };
      this.overlayEl.appendChild(el);
      return {
        idx,
        el,
        vec: latLonToVec3(st.centre.lat, st.centre.lon, 1.012),
      };
    });
  }

  highlightStoryPin(activeIndex) {
    this.storyPinElements.forEach((p, idx) => {
      p.el.classList.toggle('active', idx === activeIndex);
    });
  }

  setLayer(layerId, immediate = false) {
    if (!this.layerSpecs.has(layerId)) return;
    const targetTex = this.layerTextures.get(layerId);
    const targetLut = this.layerLuts.get(layerId);
    const targetSpec = this.layerSpecs.get(layerId);
    if (!targetTex || !targetLut) return;

    const chan = layerId === 'wind' ? 2 : 0;
    const isSqrt = targetSpec.scale === 'sqrt' ? 1.0 : 0.0;

    if (immediate || this.reducedMotion || this.currentLayer === layerId || !this.layerTextures.has(this.currentLayer)) {
      this.globeUniforms.uTexA.value = targetTex;
      this.globeUniforms.uLutA.value = targetLut;
      this.globeUniforms.uChannelA.value = chan;
      this.globeUniforms.uSqrtA.value = isSqrt;
      this.globeUniforms.uTexB.value = targetTex;
      this.globeUniforms.uLutB.value = targetLut;
      this.globeUniforms.uChannelB.value = chan;
      this.globeUniforms.uSqrtB.value = isSqrt;
      this.globeUniforms.uMix.value = 0.0;
      this.transition = null;
    } else {
      const prevSpec = this.layerSpecs.get(this.currentLayer);
      this.globeUniforms.uTexA.value = this.layerTextures.get(this.currentLayer);
      this.globeUniforms.uLutA.value = this.layerLuts.get(this.currentLayer);
      this.globeUniforms.uChannelA.value = this.currentLayer === 'wind' ? 2 : 0;
      this.globeUniforms.uSqrtA.value = prevSpec.scale === 'sqrt' ? 1.0 : 0.0;

      this.globeUniforms.uTexB.value = targetTex;
      this.globeUniforms.uLutB.value = targetLut;
      this.globeUniforms.uChannelB.value = chan;
      this.globeUniforms.uSqrtB.value = isSqrt;
      this.globeUniforms.uMix.value = 0.0;
      this.transition = { start: performance.now(), duration: 420 };
    }

    this.currentLayer = layerId;
    this.globeUniforms.uLayerMode.value = LAYER_MODE_INDEX[layerId] ?? 0;
    this.isobarGroup.visible = layerId === 'pressure';

    if (layerId === 'wind') this.particleMat.opacity = 0.96;
    else if (layerId === 'rain') this.particleMat.opacity = 0.82;
    else if (layerId === 'pressure') this.particleMat.opacity = 0.72;
    else this.particleMat.opacity = 0.48;
  }

  _sampleScalarFast(layerId, lat, lon) {
    const buf = this.layerBuffers.get(layerId);
    const spec = this.layerSpecs.get(layerId);
    if (!buf || !spec) return null;
    const u = (((lon + 180) % 360) + 360) % 360 / 360;
    const v = Math.max(0, Math.min(1, (90 - lat) / 180));
    const ix = Math.floor(u * buf.nx) % buf.nx;
    const iy = Math.min(buf.ny - 1, Math.round(v * (buf.ny - 1)));
    const byte = buf.rgba[(iy * buf.nx + ix) * 4];
    if (byte >= 255) return null;
    let t = byte / 254;
    if (spec.scale === 'sqrt') t = t * t;
    return spec.lo + t * (spec.hi - spec.lo);
  }

  sampleAllAt(lat, lon) {
    if (!this.manifest) return null;
    const wind = this.sampleWindVector(lat, lon);
    return {
      temperature: this._sampleScalarFast('temperature', lat, lon),
      anomaly: this._sampleScalarFast('anomaly', lat, lon),
      rain: this._sampleScalarFast('rain', lat, lon),
      pressure: this._sampleScalarFast('pressure', lat, lon),
      windSpeed: wind ? wind.speed : null,
      windU: wind ? wind.u : 0,
      windV: wind ? wind.v : 0,
    };
  }

  sampleWindVector(lat, lon, out = this._windScratch) {
    const buf = this.layerBuffers.get('wind');
    const spec = this.layerSpecs.get('wind');
    if (!buf || !spec) return null;

    const nx = buf.nx;
    const ny = buf.ny;
    const rgba = buf.rgba;
    const u = (((lon + 180) % 360) + 360) % 360 / 360;
    const v = Math.max(0, Math.min(1, (90 - lat) / 180));
    const fx = u * nx;
    const fy = v * (ny - 1);
    const i0 = Math.floor(fx) % nx;
    const i1 = (i0 + 1) % nx;
    const j0 = Math.min(ny - 2, Math.floor(fy));
    const j1 = j0 + 1;
    const tx = fx - Math.floor(fx);
    const ty = fy - j0;
    const omtX = 1 - tx;
    const omtY = 1 - ty;

    const maxW = spec.hi;
    const span = (2 * maxW) / 254;
    const b00 = (j0 * nx + i0) * 4;
    const b10 = (j0 * nx + i1) * 4;
    const b01 = (j1 * nx + i0) * 4;
    const b11 = (j1 * nx + i1) * 4;

    const r00 = rgba[b00];
    const r10 = rgba[b10];
    const r01 = rgba[b01];
    const r11 = rgba[b11];
    const u00 = r00 >= 255 ? 0 : -maxW + r00 * span;
    const u10 = r10 >= 255 ? 0 : -maxW + r10 * span;
    const u01 = r01 >= 255 ? 0 : -maxW + r01 * span;
    const u11 = r11 >= 255 ? 0 : -maxW + r11 * span;
    const wu = (u00 * omtX + u10 * tx) * omtY + (u01 * omtX + u11 * tx) * ty;

    const g00 = rgba[b00 + 1];
    const g10 = rgba[b10 + 1];
    const g01 = rgba[b01 + 1];
    const g11 = rgba[b11 + 1];
    const v00 = g00 >= 255 ? 0 : -maxW + g00 * span;
    const v10 = g10 >= 255 ? 0 : -maxW + g10 * span;
    const v01 = g01 >= 255 ? 0 : -maxW + g01 * span;
    const v11 = g11 >= 255 ? 0 : -maxW + g11 * span;
    const wv = (v00 * omtX + v10 * tx) * omtY + (v01 * omtX + v11 * tx) * ty;

    out.u = wu;
    out.v = wv;
    out.speed = Math.hypot(wu, wv);
    return out;
  }

  getCurrentCameraGeo() {
    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    const rawDist = this.camera.position.length();
    const nominalDist = 1.0 + Math.max(0.15, (rawDist - 1.0) / vp.altitudeScale);
    const { lat, lon } = vec3ToLatLon(this.camera.position);
    return { lat, lon, distance: nominalDist };
  }

  flyTo(lat, lon, distance = 1.85, durationMs = 1800) {
    this.autoSpin = false;
    this.tourControlled = false;
    this.controls.target.set(0, 0, 0);

    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    const adjDistance = 1.0 + Math.max(0.24, distance - 1.0) * vp.altitudeScale;

    if (this.reducedMotion || durationMs <= 0) {
      this.flight = null;
      latLonToVec3(lat, lon, adjDistance, this.camera.position);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(0, 0, 0);
      this.controls.update();
      return;
    }

    const fromDir = this.camera.position.clone().normalize();
    const toDir = latLonToVec3(lat, lon, 1).normalize();
    const fromDist = this.camera.position.length();
    const angle = fromDir.angleTo(toDir);
    const peakBoost = Math.min(0.65, angle * 0.28);

    this.flight = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(Z_AXIS, fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(Z_AXIS, toDir),
      fromDist,
      toDist: adjDistance,
      fromLook: new THREE.Vector3(0, 0, 0),
      toLook: new THREE.Vector3(0, 0, 0),
      peakBoost,
      resumeAutoSpin: false,
      start: performance.now(),
      duration: durationMs,
    };
  }

  returnToHomeView(durationMs = 1650) {
    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    this.tourControlled = false;
    this.controls.enabled = true;
    this.autoSpin = false;
    this.setTourFocusCallout(null);
    this.controls.target.set(0, 0, 0);

    if (this.reducedMotion || durationMs <= 0) {
      this.flight = null;
      latLonToVec3(HOME_LAT, HOME_LON, vp.initialDist, this.camera.position);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(0, 0, 0);
      this.controls.update();
      return;
    }

    const fromDir = this.camera.position.clone().normalize();
    const toDir = latLonToVec3(HOME_LAT, HOME_LON, 1).normalize();
    const fromDist = this.camera.position.length();
    const angle = fromDir.angleTo(toDir);
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const fromLook = this.camera.position.clone().addScaledVector(fwd, fromDist);

    this.flight = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(Z_AXIS, fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(Z_AXIS, toDir),
      fromDist,
      toDist: vp.initialDist,
      fromLook,
      toLook: new THREE.Vector3(0, 0, 0),
      peakBoost: Math.min(0.35, angle * 0.16),
      resumeAutoSpin: !this.reducedMotion,
      start: performance.now(),
      duration: durationMs,
    };
  }

  setTourMode(enabled) {
    this.tourControlled = enabled;
    this.controls.enabled = !enabled;
    if (enabled) {
      this.autoSpin = false;
      this.flight = null;
    } else {
      this.returnToHomeView(this.reducedMotion ? 0 : 1650);
    }
  }

  setTourCameraPose(focusLat, focusLon, distance, bearingDeg = 0, tiltDeg = 18) {
    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    const isNarrowMobile = window.innerWidth <= 820;
    const effDist = 1.0 + Math.max(0.24, distance - 1.0) * vp.altitudeScale;

    const vFocus = latLonToVec3(focusLat, focusLon, 1.0);
    const upLocal = vFocus.clone().normalize();

    const lam = focusLon * DEG2RAD;
    const east = new THREE.Vector3(-Math.sin(lam), 0, -Math.cos(lam)).normalize();
    const north = new THREE.Vector3().crossVectors(upLocal, east).normalize();

    const bRad = bearingDeg * DEG2RAD;
    const tiltRad = tiltDeg * DEG2RAD;
    const backDir = north
      .clone()
      .multiplyScalar(-Math.cos(bRad))
      .addScaledVector(east, -Math.sin(bRad))
      .normalize();

    const camDir = upLocal
      .clone()
      .multiplyScalar(Math.cos(tiltRad))
      .addScaledVector(backDir, Math.sin(tiltRad))
      .normalize();

    this.camera.position.copy(camDir.multiplyScalar(effDist));
    this.camera.up.set(0, 1, 0);

    const tiltFrac = Math.min(1.0, tiltDeg / 14.5);
    const span = Math.max(0.15, effDist - 1.0);
    const eastShift = isNarrowMobile ? 0.0 : -0.065 * span * tiltFrac;
    const northShift = isNarrowMobile ? -0.035 * span * tiltFrac : -0.075 * span * tiltFrac;
    const lookTarget = vFocus
      .clone()
      .addScaledVector(north, northShift)
      .addScaledVector(east, eastShift)
      .multiplyScalar(1.0 - 0.14 * tiltFrac);
    this.camera.lookAt(lookTarget);
  }

  _bindEvents() {
    window.addEventListener('resize', () => {
      const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
      this.camera.fov = vp.fov;
      this.camera.aspect = vp.aspect;
      this.camera.updateProjectionMatrix();
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatioForViewport()));
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    let pointerDownPos = null;
    const dom = this.renderer.domElement;

    dom.addEventListener('pointerdown', e => {
      pointerDownPos = [e.clientX, e.clientY];
      this.autoSpin = false;
      this.flight = null;
      if (this.tourControlled) {
        this.onUserInteract?.();
      }
    });

    dom.addEventListener(
      'wheel',
      () => {
        this.autoSpin = false;
        this.flight = null;
      },
      { passive: true }
    );

    dom.addEventListener('pointerup', e => {
      if (!pointerDownPos) return;
      const moved = Math.hypot(e.clientX - pointerDownPos[0], e.clientY - pointerDownPos[1]);
      pointerDownPos = null;
      if (moved > 5 || this.tourControlled) return;

      const rect = dom.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, this.camera);
      const hit = raycaster.intersectObject(this.sphereMesh)[0];
      if (hit) {
        const localPt = this.earthGroup.worldToLocal(hit.point.clone());
        const { lat, lon } = vec3ToLatLon(localPt);
        this.onSpotClick?.(lat, lon);
      }
    });
  }

  _stepParticles(dt) {
    if (!this.layerBuffers.has('wind')) return;
    if (this.reducedMotion) {
      this.particleLines.geometry.setDrawRange(0, 0);
      return;
    }

    const posAttr = this.particleLines.geometry.getAttribute('position');
    const colAttr = this.particleLines.geometry.getAttribute('color');
    const posArr = posAttr.array;
    const colArr = colAttr.array;
    const vHead = this._vHead;
    const vTail = this._vTail;

    const layer = this.currentLayer;
    const modeSpeed = layer === 'wind' ? 1.55 : layer === 'rain' ? 1.2 : layer === 'pressure' ? 1.15 : 0.85;
    const speedScale = 0.26 * modeSpeed * dt * 60;

    const activeCount = this.focusRegion ? this.particleCount : this.globalParticleCount;
    this.particleLines.geometry.setDrawRange(0, activeCount * 2);

    for (let i = 0; i < activeCount; i++) {
      let lat = this.particles[i * 3];
      let lon = this.particles[i * 3 + 1];
      let age = this.particles[i * 3 + 2] + 1;

      const maxAge = i >= this.globalParticleCount ? 72 : 95;
      if (age > maxAge || Math.abs(lat) > 84) {
        this._respawnParticle(i, false);
        lat = this.particles[i * 3];
        lon = this.particles[i * 3 + 1];
        age = 0;
      }

      const w = this.sampleWindVector(lat, lon);
      const wu = w ? w.u : 0;
      const wv = w ? w.v : 0;
      const spd = w ? w.speed : 0;

      const cosLat = Math.max(0.18, Math.cos(lat * DEG2RAD));
      const dLon = (wu * speedScale * 0.08) / cosLat;
      const dLat = wv * speedScale * 0.08;

      const nextLat = Math.max(-85, Math.min(85, lat + dLat));
      const nextLon = (((lon + dLon + 180) % 360) + 360) % 360 - 180;

      this.particles[i * 3] = nextLat;
      this.particles[i * 3 + 1] = nextLon;
      this.particles[i * 3 + 2] = age;

      const tailMult = layer === 'wind' ? 0.095 : layer === 'rain' ? 0.075 : 0.062;
      const tailLen = Math.min(3.6, 0.2 + spd * tailMult);
      const invSpd = 1 / Math.max(1, spd);
      const tailLat = Math.max(-85, Math.min(85, nextLat - wv * invSpd * tailLen));
      const tailLon = nextLon - (wu * invSpd * tailLen) / cosLat;

      latLonToVec3(tailLat, tailLon, 1.0032, vTail);
      latLonToVec3(nextLat, nextLon, 1.0032, vHead);

      const base = i * 6;
      posArr[base] = vTail.x;
      posArr[base + 1] = vTail.y;
      posArr[base + 2] = vTail.z;
      posArr[base + 3] = vHead.x;
      posArr[base + 4] = vHead.y;
      posArr[base + 5] = vHead.z;

      let fade = Math.sin((age / maxAge) * Math.PI) * Math.min(1.0, spd / 7.5);
      if (i >= this.globalParticleCount) fade *= 1.22;

      let hr = 0.88;
      let hg = 0.97;
      let hb = 1.0;
      let tr = 0.22;
      let tg = 0.56;
      let tb = 0.78;

      if (layer === 'wind') {
        if (spd > 21) {
          hr = 1.0;
          hg = 0.93;
          hb = 0.45;
          tr = 0.92;
          tg = 0.44;
          tb = 0.18;
          fade = Math.min(1.0, fade * 1.3);
        }
      } else if (layer === 'rain') {
        const rainMm = this._sampleScalarFast('rain', nextLat, nextLon) || 0;
        fade *= Math.min(1.0, 0.16 + rainMm / 14.0);
        hr = 0.68;
        hg = 0.94;
        hb = 1.0;
        tr = 0.12;
        tg = 0.45;
        tb = 0.95;
      } else if (layer === 'anomaly' || layer === 'temperature') {
        const anom = this._sampleScalarFast('anomaly', nextLat, nextLon) || 0;
        if (anom > 2.5) {
          hr = 1.0;
          hg = 0.86;
          hb = 0.55;
          tr = 0.85;
          tg = 0.35;
          tb = 0.15;
        } else if (anom < -2.5) {
          hr = 0.68;
          hg = 0.92;
          hb = 1.0;
          tr = 0.18;
          tg = 0.48;
          tb = 0.92;
        }
      }

      colArr[base] = tr * fade;
      colArr[base + 1] = tg * fade;
      colArr[base + 2] = tb * fade;
      colArr[base + 3] = hr * fade;
      colArr[base + 4] = hg * fade;
      colArr[base + 5] = hb * fade;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  }

  _updateFocusRings(dt) {
    if (!this.focusCallout?.active) return;

    if (this.reducedMotion) {
      this.focusCallout.currentVec.copy(this.focusCallout.targetVec);
    } else {
      this.focusCallout.currentVec.lerp(this.focusCallout.targetVec, Math.min(1, dt * 2.8));
    }
    this.focusCallout.currentVec.normalize().multiplyScalar(1.0045);

    this.focusRingGroup.position.copy(this.focusCallout.currentVec);
    this._ringOutward.copy(this.focusCallout.currentVec).normalize();
    this.focusRingGroup.quaternion.setFromUnitVectors(Z_AXIS, this._ringOutward);

    if (this.reducedMotion) {
      this.focusRingA.scale.setScalar(1.3);
      this.focusRingB.scale.setScalar(1.8);
      this.focusRingMatA.opacity = 0.75;
      this.focusRingMatB.opacity = 0.45;
    } else {
      const pA = (this._elapsedSec * 0.62) % 1;
      const pB = (this._elapsedSec * 0.62 + 0.5) % 1;
      this.focusRingA.scale.setScalar(1.0 + pA * 3.8);
      this.focusRingB.scale.setScalar(1.0 + pB * 3.8);
      this.focusRingMatA.opacity = (1.0 - pA) * 0.85;
      this.focusRingMatB.opacity = (1.0 - pB) * 0.72;
    }
  }

  _updateOverlayProjections() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const isMobile = w <= 820;
    const camDir = this._camDir.copy(this.camera.position).normalize();
    const proj = this._projVec;

    const showHl = this.currentLayer === 'pressure';
    for (const item of this.hlElements) {
      if (!showHl || item.vec.dot(camDir) < 0.28) {
        item.el.hidden = true;
        item.el.style.visibility = 'hidden';
        continue;
      }
      proj.copy(item.vec).project(this.camera);
      const x = (proj.x * 0.5 + 0.5) * w;
      const y = (-proj.y * 0.5 + 0.5) * h;
      item.el.hidden = false;
      item.el.style.visibility = 'visible';
      item.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -50%)`;
    }

    for (const pin of this.storyPinElements) {
      if (this.tourControlled || pin.vec.dot(camDir) < 0.18) {
        pin.el.hidden = true;
        pin.el.style.visibility = 'hidden';
        continue;
      }
      proj.copy(pin.vec).project(this.camera);
      const x = (proj.x * 0.5 + 0.5) * w;
      const y = (-proj.y * 0.5 + 0.5) * h;
      pin.el.hidden = false;
      pin.el.style.visibility = 'visible';
      pin.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -50%)`;
    }

    let calloutBox = null;
    if (this.focusCallout?.active) {
      const fv = this.focusCallout.currentVec;
      const dot = (fv.x * camDir.x + fv.y * camDir.y + fv.z * camDir.z) / Math.max(1e-6, fv.length());
      if (dot < 0.2) {
        this.focusCallout.el.hidden = true;
        this.focusCallout.el.style.visibility = 'hidden';
      } else {
        proj.copy(fv).project(this.camera);
        const x = (proj.x * 0.5 + 0.5) * w;
        const y = (-proj.y * 0.5 + 0.5) * h;
        const offsetTransform = isMobile ? 'translate(-50%, -132%)' : 'translate(16px, -50%)';
        this.focusCallout.el.hidden = false;
        this.focusCallout.el.style.visibility = 'visible';
        this.focusCallout.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) ${offsetTransform}`;
        // Screen box of the callout (plus padding), so city labels never sit on top of it
        const cw = this.focusCallout.el.offsetWidth || 260;
        const ch = this.focusCallout.el.offsetHeight || 48;
        calloutBox = isMobile
          ? { x0: x - cw / 2 - 12, x1: x + cw / 2 + 12, y0: y - ch * 1.32 - 12, y1: y + 12 }
          : { x0: x - 12, x1: x + 16 + cw + 12, y0: y - ch / 2 - 10, y1: y + ch / 2 + 10 };
      }
    }

    // City labels: only on the visible hemisphere, and skip any that would overlap a label already
    // placed (cities are listed largest first, so the biggest names win)
    const placed = [];
    for (const city of this.cityElements) {
      let show = city.vec.dot(camDir) >= 0.3;
      let x = 0;
      let y = 0;
      if (show) {
        proj.copy(city.vec).project(this.camera);
        x = (proj.x * 0.5 + 0.5) * w;
        y = (-proj.y * 0.5 + 0.5) * h;
        const labelW = 12 + city.el.textContent.length * 8;
        const overlapsCallout =
          calloutBox && x + labelW > calloutBox.x0 && x < calloutBox.x1 && y + 10 > calloutBox.y0 && y - 10 < calloutBox.y1;
        show =
          x > 8 && x < w - 8 && y > 8 && y < h - 8 && !overlapsCallout &&
          !placed.some(p => Math.abs(p.x - x) < 110 && Math.abs(p.y - y) < 22);
      }
      if (!show) {
        city.el.hidden = true;
        city.el.style.visibility = 'hidden';
        continue;
      }
      placed.push({ x, y });
      city.el.hidden = false;
      city.el.style.visibility = 'visible';
      city.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-4px, -50%)`;
    }


    if (this.searchMarker?.active && !this.tourControlled) {
      if (this.searchMarker.vec.dot(camDir) < 0.18) {
        this.searchMarker.el.hidden = true;
        this.searchMarker.el.style.visibility = 'hidden';
      } else {
        proj.copy(this.searchMarker.vec).project(this.camera);
        const x = (proj.x * 0.5 + 0.5) * w;
        const y = (-proj.y * 0.5 + 0.5) * h;
        this.searchMarker.el.hidden = false;
        this.searchMarker.el.style.visibility = 'visible';
        this.searchMarker.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-7px, -50%)`;
      }
    }
  }

  _animate(now) {
    requestAnimationFrame(this._animate);
    const dt = Math.min(0.05, (now - this._lastFrame) / 1000);
    this._lastFrame = now;
    if (!this.reducedMotion) {
      this._elapsedSec += dt;
      this.globeUniforms.uTime.value = this._elapsedSec;
    }

    if (this.transition) {
      const t = Math.min(1, (now - this.transition.start) / this.transition.duration);
      this.globeUniforms.uMix.value = t * t * (3 - 2 * t);
      if (t >= 1) {
        this.globeUniforms.uTexA.value = this.globeUniforms.uTexB.value;
        this.globeUniforms.uLutA.value = this.globeUniforms.uLutB.value;
        this.globeUniforms.uChannelA.value = this.globeUniforms.uChannelB.value;
        this.globeUniforms.uSqrtA.value = this.globeUniforms.uSqrtB.value;
        this.globeUniforms.uMix.value = 0.0;
        this.transition = null;
      }
    }

    if (this.isobarMat && this.isobarGroup.visible && !this.reducedMotion) {
      this.isobarMat.opacity = 0.5 + 0.14 * Math.sin(this._elapsedSec * 2.4);
    }

    if (this.flight) {
      const rawT = Math.min(1, (now - this.flight.start) / this.flight.duration);
      const ease = rawT < 0.5 ? 4 * rawT * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 3) / 2;
      this._flightQuat.slerpQuaternions(this.flight.fromQuat, this.flight.toQuat, ease);
      this._flightDir.set(0, 0, 1).applyQuaternion(this._flightQuat);
      const arc = Math.sin(rawT * Math.PI) * this.flight.peakBoost;
      const dist = THREE.MathUtils.lerp(this.flight.fromDist, this.flight.toDist, ease) + arc;
      this.camera.position.copy(this._flightDir.multiplyScalar(dist));
      this.camera.up.set(0, 1, 0);
      this._flightLook.lerpVectors(this.flight.fromLook, this.flight.toLook, ease);
      this.camera.lookAt(this._flightLook);
      if (rawT >= 1) {
        const shouldSpin = this.flight.resumeAutoSpin && !this.reducedMotion;
        this.flight = null;
        this.controls.target.set(0, 0, 0);
        this.controls.update();
        if (shouldSpin) {
          this.autoSpin = true;
        }
      }
    } else if (!this.tourControlled) {
      if (this.autoSpin && !this.reducedMotion) {
        // Negative camera rotation around +Y makes Earth spin West-to-East (anticlockwise from above North Pole)
        this._spinQuat.setFromAxisAngle(Y_AXIS, -dt * 0.062);
        this.camera.position.applyQuaternion(this._spinQuat);
      }
      this.controls.update();
    }

    this._camRight.crossVectors(this.camera.up, this.camera.position).normalize();
    this.globeUniforms.uSunDir.value
      .copy(this.camera.position)
      .normalize()
      .addScaledVector(this._camRight, -0.45)
      .addScaledVector(this.camera.up, 0.35)
      .normalize();

    this._stepParticles(dt);
    this._updateFocusRings(dt);
    this._updateOverlayProjections();
    this.renderer.render(this.scene, this.camera);
  }
}
