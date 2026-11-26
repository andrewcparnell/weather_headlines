import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const HOME_LAT = 26;
const HOME_LON = -12;

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
  const n = vec.clone().normalize();
  const lat = Math.asin(Math.max(-1, Math.min(1, n.y))) * RAD2DEG;
  const lon = Math.atan2(-n.z, n.x) * RAD2DEG;
  return { lat, lon };
}

function parseHexRgba(hex) {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  const a = clean.length >= 8 ? parseInt(clean.slice(6, 8), 16) : 255;
  return [r, g, b, a];
}

function buildLutTexture(layerSpec) {
  const data = new Uint8Array(256 * 4);
  const stops = layerSpec.stops.map(([val, hex]) => ({ val, rgba: parseHexRgba(hex) }));
  for (let i = 0; i < 255; i++) {
    const t = i / 254;
    const val = layerSpec.lo + t * (layerSpec.hi - layerSpec.lo);
    let c = stops[0].rgba;
    if (val >= stops[stops.length - 1].val) {
      c = stops[stops.length - 1].rgba;
    } else {
      for (let s = 0; s < stops.length - 1; s++) {
        const s0 = stops[s];
        const s1 = stops[s + 1];
        if (val >= s0.val && val <= s1.val) {
          const f = (val - s0.val) / Math.max(1e-6, s1.val - s0.val);
          c = [
            Math.round(s0.rgba[0] + f * (s1.rgba[0] - s0.rgba[0])),
            Math.round(s0.rgba[1] + f * (s1.rgba[1] - s0.rgba[1])),
            Math.round(s0.rgba[2] + f * (s1.rgba[2] - s0.rgba[2])),
            Math.round(s0.rgba[3] + f * (s1.rgba[3] - s0.rgba[3])),
          ];
          break;
        }
      }
    }
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

function decodeDeltaRing(deltaArr, scale = 100) {
  const pts = [];
  let x = 0;
  let y = 0;
  for (let i = 0; i < deltaArr.length; i += 2) {
    x += deltaArr[i];
    y += deltaArr[i + 1];
    pts.push([x / scale, y / scale]);
  }
  return pts;
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

    this.currentLayer = 'temperature';
    this.transition = null;
    this.flight = null;
    this.tourControlled = false;
    this.autoSpin = true;

    // Regional focus for high-density storm particles & broadcast surface spotlight
    this.focusRegion = null; // { lat, lon }
    this.focusCallout = null; // { vec, el, labelEl, metricEl, visible }

    this.hlElements = [];
    this.storyPinElements = [];

    this._elapsedSec = 0;
    this._initScene();
    this._initParticles();
    this._initFocusSpotlight();
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
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
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

    // Subtle deep-space starfield
    const starCount = 1400;
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

    const blankTex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat);
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
        varying vec3 vViewPos;
        void main() {
          vObjPos = position;
          vWorldNormal = normalize(mat3(modelMatrix) * normal);
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vViewPos = -mvPos.xyz;
          gl_Position = projectionMatrix * mvPos;
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
        varying vec3 vViewPos;

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

          // Live meteorological shader animation tailored to active variable
          if (uLayerMode == 3 && weather.a > 0.15) {
            // Rainfall: animated radar wave ripple across active rainbands
            float wave = 0.5 + 0.5 * sin(uTime * 4.4 - latDeg * 0.65 + lonDeg * 0.55);
            float heavyBoost = smoothstep(0.08, 0.45, intensity);
            weather.rgb += vec3(0.10, 0.22, 0.34) * wave * heavyBoost;
            weather.a = min(1.0, weather.a + 0.12 * wave * heavyBoost);
          } else if (uLayerMode == 1) {
            // Temp Anomaly: living pulse on strong warm/cold departures
            float extremeness = smoothstep(0.16, 0.38, abs(intensity - 0.5));
            float pulse = 0.5 + 0.5 * sin(uTime * 3.1 - latDeg * 0.25);
            weather.rgb *= (1.0 + 0.14 * pulse * extremeness);
          } else if (uLayerMode == 0) {
            // Temperature: subtle thermal shimmer in extreme heat/cold zones
            float extremeness = smoothstep(0.24, 0.44, abs(intensity - 0.52));
            float pulse = 0.5 + 0.5 * sin(uTime * 2.6 + lonDeg * 0.2);
            weather.rgb *= (1.0 + 0.10 * pulse * extremeness);
          }

          vec3 nWorld = normalize(vWorldNormal);
          float diffuse = 0.78 + 0.26 * max(0.0, dot(nWorld, uSunDir));
          vec3 litBase = baseCol * diffuse;

          vec3 blended = mix(litBase, weather.rgb, weather.a * 0.89);

          float rim = pow(1.0 - max(0.0, dot(nWorld, normalize(cameraPosition - nObj))), 3.2);
          blended += vec3(0.14, 0.34, 0.58) * rim * 0.42;

          gl_FragColor = vec4(blended, 1.0);
        }
      `,
    });

    this.sphereMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 144, 108), globeMat);
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
          gl_FragColor = vec4(0.22, 0.56, 0.88, intensity * 0.48);
        }
      `,
    });
    this.earthGroup.add(new THREE.Mesh(new THREE.SphereGeometry(1.042, 96, 64), haloMat));

    this.isobarGroup = new THREE.Group();
    this.isobarGroup.visible = false;
    this.earthGroup.add(this.isobarGroup);
  }

  _initParticles() {
    // 3,000 global particles + 1,800 high-density regional storm particles during Tour mode!
    this.globalParticleCount = 3000;
    this.regionalParticleCount = 1800;
    this.particleCount = this.globalParticleCount + this.regionalParticleCount;

    this.particles = new Float32Array(this.particleCount * 3); // [lat, lon, age]
    const positions = new Float32Array(this.particleCount * 6);
    const colors = new Float32Array(this.particleCount * 6);

    for (let i = 0; i < this.particleCount; i++) {
      this._respawnParticle(i, true);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

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
    // Animated concentric surface radar rings + floating broadcast callout pin for Tour sentences
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

  _respawnParticle(i, randomAge = false) {
    const isRegional = i >= this.globalParticleCount && this.focusRegion;
    let lat;
    let lon;
    if (isRegional) {
      const spreadLat = 18;
      const cosC = Math.max(0.28, Math.cos(this.focusRegion.lat * DEG2RAD));
      const spreadLon = Math.min(48, 22 / cosC);
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
      }
      return;
    }

    const prevRegion = this.focusRegion;
    this.focusRegion = { lat: cue.lat, lon: cue.lon };

    // Immediately seed regional storm particles around the new focus if we jumped > 20 deg
    if (!prevRegion || Math.hypot(prevRegion.lat - cue.lat, prevRegion.lon - cue.lon) > 20) {
      for (let i = this.globalParticleCount; i < this.particleCount; i++) {
        this._respawnParticle(i, true);
      }
    }

    const vec = latLonToVec3(cue.lat, cue.lon, 1.005);
    if (!this.focusCallout.active) {
      this.focusCallout.currentVec.copy(vec);
    }
    this.focusCallout.targetVec.copy(vec);
    this.focusCallout.active = true;
    this.focusCallout.labelEl.textContent = cue.label || '';
    this.focusCallout.metricEl.textContent = cue.metric || '';
    this.focusCallout.metricEl.hidden = !cue.metric;
    this.focusRingGroup.visible = true;
  }

  async loadBasemap(url = 'assets/basemap.json') {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const topo = await resp.json();
    const scale = topo.scale || 100;

    const W = 2048;
    const H = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#091b32');
    grad.addColorStop(0.5, '#071426');
    grad.addColorStop(1, '#091a30');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#182b42';
    ctx.strokeStyle = '#294666';
    ctx.lineWidth = 0.8;

    const coastPositions = [];
    const vTmpA = new THREE.Vector3();
    const vTmpB = new THREE.Vector3();

    for (const encodedRing of topo.land || []) {
      const pts = decodeDeltaRing(encodedRing, scale);
      if (pts.length < 3) continue;

      let minLon = Infinity;
      let maxLon = -Infinity;
      let avgLat = 0;
      for (const [lo, la] of pts) {
        if (lo < minLon) minLon = lo;
        if (lo > maxLon) maxLon = lo;
        avgLat += la;
      }
      avgLat /= pts.length;
      const isAntarctica = avgLat < -62 && maxLon - minLon > 300;

      for (const shift of [-360, 0, 360]) {
        if (shift !== 0 && !isAntarctica && minLon >= -180 && maxLon <= 180) continue;
        ctx.beginPath();
        pts.forEach(([lo, la], idx) => {
          const px = ((lo + shift + 180) / 360) * W;
          const py = ((90 - la) / 180) * H;
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        if (isAntarctica && shift === 0) {
          const firstPx = ((pts[0][0] + 180) / 360) * W;
          const lastPx = ((pts[pts.length - 1][0] + 180) / 360) * W;
          ctx.lineTo(lastPx, H);
          ctx.lineTo(firstPx, H);
        }
        ctx.closePath();
        ctx.fill();
      }

      for (let i = 0; i < pts.length - 1; i++) {
        const [lo0, la0] = pts[i];
        const [lo1, la1] = pts[i + 1];
        if (isAntarctica && Math.abs(lo1 - lo0) > 40) continue;
        latLonToVec3(la0, lo0, 1.0018, vTmpA);
        latLonToVec3(la1, lo1, 1.0018, vTmpB);
        coastPositions.push(vTmpA.x, vTmpA.y, vTmpA.z, vTmpB.x, vTmpB.y, vTmpB.z);
      }
    }

    const baseTex = new THREE.CanvasTexture(canvas);
    baseTex.flipY = false;
    baseTex.wrapS = THREE.RepeatWrapping;
    baseTex.wrapT = THREE.ClampToEdgeWrapping;
    baseTex.minFilter = THREE.LinearFilter;
    baseTex.magFilter = THREE.LinearFilter;
    baseTex.needsUpdate = true;
    this.globeUniforms.uBaseMap.value = baseTex;

    const coastGeo = new THREE.BufferGeometry();
    coastGeo.setAttribute('position', new THREE.Float32BufferAttribute(coastPositions, 3));
    const coastMat = new THREE.LineBasicMaterial({
      color: 0xd8e8f8,
      transparent: true,
      opacity: 0.56,
    });
    this.earthGroup.add(new THREE.LineSegments(coastGeo, coastMat));

    const borderPositions = [];
    for (const encodedLine of topo.borders || []) {
      const pts = decodeDeltaRing(encodedLine, scale);
      for (let i = 0; i < pts.length - 1; i++) {
        latLonToVec3(pts[i][1], pts[i][0], 1.0015, vTmpA);
        latLonToVec3(pts[i + 1][1], pts[i + 1][0], 1.0015, vTmpB);
        borderPositions.push(vTmpA.x, vTmpA.y, vTmpA.z, vTmpB.x, vTmpB.y, vTmpB.z);
      }
    }
    const borderGeo = new THREE.BufferGeometry();
    borderGeo.setAttribute('position', new THREE.Float32BufferAttribute(borderPositions, 3));
    const borderMat = new THREE.LineBasicMaterial({
      color: 0x9ab6d4,
      transparent: true,
      opacity: 0.3,
    });
    this.earthGroup.add(new THREE.LineSegments(borderGeo, borderMat));
  }

  async loadWeatherManifest(manifestUrl = 'data/manifest.json') {
    const resp = await fetch(`${manifestUrl}?t=${Date.now()}`);
    if (!resp.ok) throw new Error('Unable to load weather manifest');
    const manifest = await resp.json();
    this.manifest = manifest;
    this.globeUniforms.uGrid.value.set(manifest.grid.nx, manifest.grid.ny);

    for (const spec of manifest.layers) {
      this.layerSpecs.set(spec.id, spec);
      this.layerLuts.set(spec.id, buildLutTexture(spec));
    }

    const cacheBust = encodeURIComponent(manifest.generated_utc || Date.now());
    await Promise.all(
      manifest.layers.map(spec => this._loadLayerImage(spec.id, `data/layers/${spec.id}.webp?v=${cacheBust}`))
    );

    this._buildIsobarsAndCentres(manifest.isobars || [], manifest.pressure_centres || []);
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
    this.layerTextures.set(layerId, tex);
  }

  _buildIsobarsAndCentres(isobars, centres) {
    while (this.isobarGroup.children.length) {
      const ch = this.isobarGroup.children.pop();
      ch.geometry?.dispose();
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
      this.overlayEl.appendChild(el);
      return { ...c, el, vec: latLonToVec3(c.lat, c.lon, 1.006) };
    });
  }

  setStoryPins(stories, activeIndex = -1) {
    for (const p of this.storyPinElements) p.el.remove();
    this.storyPinElements = (stories || []).map((st, idx) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `story-pin${idx === activeIndex ? ' active' : ''}`;
      el.textContent = String(idx + 1);
      el.title = st.headline;
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

    if (immediate || this.currentLayer === layerId) {
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
      this.transition = { start: performance.now(), duration: 480 };
    }

    this.currentLayer = layerId;
    this.globeUniforms.uLayerMode.value = LAYER_MODE_INDEX[layerId] ?? 0;

    // Show crisp isobars on Pressure AND subtly on Wind/Rain storms during Tour mode
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

  sampleWindVector(lat, lon) {
    const buf = this.layerBuffers.get('wind');
    const spec = this.layerSpecs.get('wind');
    if (!buf || !spec) return null;
    const u = (((lon + 180) % 360) + 360) % 360 / 360;
    const v = Math.max(0, Math.min(1, (90 - lat) / 180));
    const fx = u * buf.nx;
    const fy = v * (buf.ny - 1);
    const i0 = Math.floor(fx) % buf.nx;
    const i1 = (i0 + 1) % buf.nx;
    const j0 = Math.min(buf.ny - 2, Math.floor(fy));
    const j1 = j0 + 1;
    const tx = fx - Math.floor(fx);
    const ty = fy - j0;

    const decode = (idx, ch, lo, hi) => {
      const b = buf.rgba[idx * 4 + ch];
      return b >= 255 ? 0 : lo + (b / 254) * (hi - lo);
    };
    const maxW = spec.hi;
    const k00 = j0 * buf.nx + i0;
    const k10 = j0 * buf.nx + i1;
    const k01 = j1 * buf.nx + i0;
    const k11 = j1 * buf.nx + i1;

    const bilerp = (ch, lo, hi) => {
      const v00 = decode(k00, ch, lo, hi);
      const v10 = decode(k10, ch, lo, hi);
      const v01 = decode(k01, ch, lo, hi);
      const v11 = decode(k11, ch, lo, hi);
      return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
    };

    const wu = bilerp(0, -maxW, maxW);
    const wv = bilerp(1, -maxW, maxW);
    return { u: wu, v: wv, speed: Math.hypot(wu, wv) };
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
    const fromDir = this.camera.position.clone().normalize();
    const toDir = latLonToVec3(lat, lon, 1).normalize();
    const fromDist = this.camera.position.length();
    const angle = fromDir.angleTo(toDir);
    const peakBoost = Math.min(0.65, angle * 0.28);

    const vp = computeViewportCameraParams(window.innerWidth, window.innerHeight);
    const adjDistance = 1.0 + Math.max(0.24, distance - 1.0) * vp.altitudeScale;

    this.flight = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
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
    const fromDir = this.camera.position.clone().normalize();
    const toDir = latLonToVec3(HOME_LAT, HOME_LON, 1).normalize();
    const fromDist = this.camera.position.length();
    const angle = fromDir.angleTo(toDir);

    // Preserve exact optical axis on frame 1 so transitioning out of oblique tour view has zero jump
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const fromLook = this.camera.position.clone().addScaledVector(fwd, fromDist);

    this.tourControlled = false;
    this.controls.enabled = true;
    this.autoSpin = false;
    this.setTourFocusCallout(null);
    this.controls.target.set(0, 0, 0);

    this.flight = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
      fromDist,
      toDist: vp.initialDist,
      fromLook,
      toLook: new THREE.Vector3(0, 0, 0),
      peakBoost: Math.min(0.35, angle * 0.16),
      resumeAutoSpin: true,
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
      this.returnToHomeView(1650);
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

    // On desktop, nudge slightly right-of-centre to clear top-left script; on mobile, centre horizontally in the open middle window
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
    const posAttr = this.particleLines.geometry.getAttribute('position');
    const colAttr = this.particleLines.geometry.getAttribute('color');
    const posArr = posAttr.array;
    const colArr = colAttr.array;
    const vHead = new THREE.Vector3();
    const vTail = new THREE.Vector3();

    const layer = this.currentLayer;
    const modeSpeed = layer === 'wind' ? 1.55 : layer === 'rain' ? 1.2 : layer === 'pressure' ? 1.15 : 0.85;
    const speedScale = 0.26 * modeSpeed * dt * 60;

    const activeCount = this.focusRegion ? this.particleCount : this.globalParticleCount;

    for (let i = 0; i < this.particleCount; i++) {
      if (i >= activeCount) {
        const base = i * 6;
        posArr[base] = posArr[base + 1] = posArr[base + 2] = 0;
        posArr[base + 3] = posArr[base + 4] = posArr[base + 5] = 0;
        continue;
      }

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
      const tailLat = Math.max(-85, Math.min(85, nextLat - (wv / Math.max(1, spd)) * tailLen));
      const tailLon = nextLon - ((wu / Math.max(1, spd)) * tailLen) / cosLat;

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

      // Variable-specific colour & intensity modulation
      let hr = 0.88;
      let hg = 0.97;
      let hb = 1.0;
      let tr = 0.22;
      let tg = 0.56;
      let tb = 0.78;

      if (layer === 'wind') {
        if (spd > 21) {
          // Storm-force / gale streaks glow bright gold-white
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
        const rainWeight = Math.min(1.0, 0.16 + rainMm / 14.0);
        fade *= rainWeight;
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

    // Smoothly glide the surface spotlight & callout when transitioning between sentence locations
    this.focusCallout.currentVec.lerp(this.focusCallout.targetVec, Math.min(1, dt * 2.8));
    this.focusCallout.currentVec.normalize().multiplyScalar(1.0045);

    this.focusRingGroup.position.copy(this.focusCallout.currentVec);
    const outward = this.focusCallout.currentVec.clone().normalize();
    this.focusRingGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), outward);

    const pA = (this._elapsedSec * 0.62) % 1;
    const pB = (this._elapsedSec * 0.62 + 0.5) % 1;
    const sA = 1.0 + pA * 3.8;
    const sB = 1.0 + pB * 3.8;
    this.focusRingA.scale.setScalar(sA);
    this.focusRingB.scale.setScalar(sB);
    this.focusRingMatA.opacity = (1.0 - pA) * 0.85;
    this.focusRingMatB.opacity = (1.0 - pB) * 0.72;
  }

  _updateOverlayProjections() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const camDir = this.camera.position.clone().normalize();
    const proj = new THREE.Vector3();

    const showHl = this.currentLayer === 'pressure';
    for (const item of this.hlElements) {
      if (!showHl || item.vec.dot(camDir) < 0.28) {
        item.el.hidden = true;
        continue;
      }
      proj.copy(item.vec).project(this.camera);
      item.el.hidden = false;
      item.el.style.left = `${(proj.x * 0.5 + 0.5) * w}px`;
      item.el.style.top = `${(-proj.y * 0.5 + 0.5) * h}px`;
    }

    for (const pin of this.storyPinElements) {
      if (pin.vec.dot(camDir) < 0.18) {
        pin.el.hidden = true;
        continue;
      }
      proj.copy(pin.vec).project(this.camera);
      pin.el.hidden = false;
      pin.el.style.left = `${(proj.x * 0.5 + 0.5) * w}px`;
      pin.el.style.top = `${(-proj.y * 0.5 + 0.5) * h}px`;
    }

    if (this.focusCallout?.active) {
      const fv = this.focusCallout.currentVec;
      if (fv.clone().normalize().dot(camDir) < 0.2) {
        this.focusCallout.el.hidden = true;
      } else {
        proj.copy(fv).project(this.camera);
        this.focusCallout.el.hidden = false;
        this.focusCallout.el.style.left = `${(proj.x * 0.5 + 0.5) * w}px`;
        this.focusCallout.el.style.top = `${(-proj.y * 0.5 + 0.5) * h}px`;
      }
    }
  }

  _animate(now) {
    requestAnimationFrame(this._animate);
    const dt = Math.min(0.05, (now - this._lastFrame) / 1000);
    this._lastFrame = now;
    this._elapsedSec += dt;
    this.globeUniforms.uTime.value = this._elapsedSec;

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

    if (this.isobarMat && this.isobarGroup.visible) {
      this.isobarMat.opacity = 0.5 + 0.14 * Math.sin(this._elapsedSec * 2.4);
    }

    if (this.flight) {
      const rawT = Math.min(1, (now - this.flight.start) / this.flight.duration);
      const ease = rawT < 0.5 ? 4 * rawT * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 3) / 2;
      const q = new THREE.Quaternion().slerpQuaternions(this.flight.fromQuat, this.flight.toQuat, ease);
      const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
      const arc = Math.sin(rawT * Math.PI) * this.flight.peakBoost;
      const dist = THREE.MathUtils.lerp(this.flight.fromDist, this.flight.toDist, ease) + arc;
      this.camera.position.copy(dir.multiplyScalar(dist));
      this.camera.up.set(0, 1, 0);
      const lookPt = new THREE.Vector3().lerpVectors(this.flight.fromLook, this.flight.toLook, ease);
      this.camera.lookAt(lookPt);
      if (rawT >= 1) {
        const shouldSpin = this.flight.resumeAutoSpin;
        this.flight = null;
        this.controls.target.set(0, 0, 0);
        this.controls.update();
        if (shouldSpin) {
          this.autoSpin = true;
        }
      }
    } else if (!this.tourControlled) {
      if (this.autoSpin) {
        // Negative camera rotation around +Y makes Earth spin West-to-East (anticlockwise from above North Pole)
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dt * 0.062);
        this.camera.position.applyQuaternion(q);
      }
      this.controls.update();
    }

    const camRight = new THREE.Vector3().crossVectors(this.camera.up, this.camera.position).normalize();
    this.globeUniforms.uSunDir.value
      .copy(this.camera.position)
      .normalize()
      .addScaledVector(camRight, -0.45)
      .addScaledVector(this.camera.up, 0.35)
      .normalize();

    this._stepParticles(dt);
    this._updateFocusRings(dt);
    this._updateOverlayProjections();
    this.renderer.render(this.scene, this.camera);
  }
}
