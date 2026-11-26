import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js';

const FLIGHT_MS = 2500;
const TICKER_PX_PER_SEC = 92; // Comfortable BBC / Sky News broadcast ticker speed

const LAYER_LABELS = {
  temperature: '2 M TEMPERATURE',
  anomaly: 'TEMP ANOMALY VS NORMAL',
  wind: '10 M WINDS & GUSTS',
  rain: '24 H RAINFALL',
  pressure: 'SEA LEVEL PRESSURE',
};

function catmullRomVec3(points, t) {
  const n = points.length;
  if (n === 0) return new THREE.Vector3(0, 0, 2);
  if (n === 1) return points[0].clone();

  const scaled = Math.max(0, Math.min(1, t)) * (n - 1);
  const idx = Math.min(n - 2, Math.floor(scaled));
  const u = scaled - idx;

  const p0 = points[Math.max(0, idx - 1)];
  const p1 = points[idx];
  const p2 = points[idx + 1];
  const p3 = points[Math.min(n - 1, idx + 2)];

  const u2 = u * u;
  const u3 = u2 * u;

  return new THREE.Vector3(
    0.5 * (2 * p1.x + (-p0.x + p2.x) * u + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * u2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * u3),
    0.5 * (2 * p1.y + (-p0.y + p2.y) * u + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * u2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * u3),
    0.5 * (2 * p1.z + (-p0.z + p2.z) * u + (2 * p0.z - 5 * p1.z + 4 * p2.z - p3.z) * u2 + (-p0.z + 3 * p1.z - 3 * p2.z + p3.z) * u3)
  );
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function formatTickerHtml(rawTicker) {
  // Escape HTML first, then convert **highlighted phrases** and /// separators into broadcast markup
  const escaped = escapeHtml(rawTicker);
  const withHighlights = escaped.replace(/\*\*(.+?)\*\*/g, '<mark class="ticker-kw">$1</mark>');
  return withHighlights.replace(/\s*\/\/\/\s*/g, '<span class="ticker-sep" aria-hidden="true">◆</span>');
}

export class BroadcastTour {
  constructor(globe, dom, { onLayerSync } = {}) {
    this.globe = globe;
    this.dom = dom;
    this.onLayerSync = onLayerSync;

    this.stories = [];
    this.active = false;
    this.storyIndex = 0;
    this.beatIndex = 0;

    this.phase = 'idle'; // 'flight' | 'glide'
    this.phaseElapsed = 0;

    // Pixel-accurate smooth horizontal ticker state
    this.tickerCopyWidth = 2400;
    this.tickerStartX = 0;
    this.tickerCurrentX = 0;
    this.tickerTotalTravel = 2400;

    this.flightState = null;
    this.waypointVecs = [];
    this.waypointDists = [];

    this._lastTick = performance.now();
    this._bindControls();
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  setBulletin(bulletin) {
    this.stories = bulletin?.stories || [];
  }

  start(startIndex = 0) {
    if (!this.stories.length) return;
    this.active = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this.globe.setTourMode(true);
    this.goToStory(startIndex);
  }

  stop() {
    if (!this.active) return;
    this.active = false;
    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
  }

  goToStory(index) {
    if (!this.stories.length) return;
    this.storyIndex = ((index % this.stories.length) + this.stories.length) % this.stories.length;
    this.beatIndex = 0;

    const story = this.stories[this.storyIndex];
    this.globe.highlightStoryPin(this.storyIndex);

    const wps = story.camera_waypoints || [];
    this.waypointVecs = wps.map(w => latLonToVec3(w.lat, w.lon, 1).normalize());
    this.waypointDists = wps.map(w => w.distance || 1.85);

    const startWp = wps[0] || { lat: story.centre.lat, lon: story.centre.lon, distance: 1.85 };
    const camNow = this.globe.getCurrentCameraGeo();
    const fromDir = latLonToVec3(camNow.lat, camNow.lon, 1).normalize();
    const toDir = latLonToVec3(startWp.lat, startWp.lon, 1).normalize();
    const angle = fromDir.angleTo(toDir);

    this.flightState = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
      fromDist: camNow.distance,
      toDist: startWp.distance || 1.85,
      arcBoost: Math.min(0.82, 0.18 + angle * 0.34),
      duration: Math.max(1350, Math.min(FLIGHT_MS, 1050 + angle * 700)),
    };

    this.phase = 'flight';
    this.phaseElapsed = 0;

    const initialLayer = story.beats?.[0]?.layer || story.primary_layer;
    this._applyLayer(initialLayer, story);
    this._renderChyron(story);
  }

  _applyLayer(layerId, story) {
    this.globe.setLayer(layerId);
    this.onLayerSync?.(layerId);
    const statPart = story?.stats?.headline_metric ? `${story.stats.headline_metric}  ·  ` : '';
    this.dom.layerTag.textContent = `${statPart}${LAYER_LABELS[layerId] || layerId.toUpperCase()}`;
  }

  _renderChyron(story) {
    this.dom.counter.textContent = `HEADLINE ${this.storyIndex + 1} OF ${this.stories.length}`;
    this.dom.kicker.textContent = story.kicker || story.region || 'WORLD WEATHER';
    this.dom.headline.textContent = story.headline;
    this.dom.progressBar.style.width = '0%';

    // Build scrolling ticker with two copies so there is never empty dead space
    const rawTicker =
      story.ticker ||
      (story.beats || []).map(b => b.text).join('  ///  ');
    const htmlCopy = formatTickerHtml(rawTicker);
    this.dom.tickerTrack.innerHTML =
      `<div class="ticker-copy">${htmlCopy}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${htmlCopy}</div>`;

    // Start ticker immediately visible near the left side of the viewport so reading begins right away
    requestAnimationFrame(() => {
      const firstCopy = this.dom.tickerTrack.querySelector('.ticker-copy');
      const viewportW = this.dom.tickerViewport.clientWidth || window.innerWidth;
      this.tickerCopyWidth = Math.max(900, firstCopy ? firstCopy.offsetWidth : 2400);
      this.tickerStartX = Math.round(Math.min(260, viewportW * 0.22));
      this.tickerCurrentX = this.tickerStartX;
      this.tickerTotalTravel = this.tickerCopyWidth + this.tickerStartX;
      this.dom.tickerTrack.style.transform = `translate3d(${this.tickerCurrentX}px, 0, 0)`;
    });

    // Render direct, event-specific news links
    this.dom.newsLinks.replaceChildren();
    (story.news_links || []).forEach(link => {
      const a = document.createElement('a');
      a.className = 'news-article-link';
      a.href = link.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.innerHTML =
        `<span class="news-outlet-badge">${escapeHtml(link.source)}</span>` +
        `<span>${escapeHtml(link.title)}</span>` +
        `<span aria-hidden="true">↗</span>`;
      this.dom.newsLinks.appendChild(a);
    });
  }

  _bindControls() {
    this.dom.playTourBtn.onclick = () => this.start(0);
    this.dom.prevBtn.onclick = () => this.goToStory(this.storyIndex - 1);
    this.dom.nextBtn.onclick = () => this.goToStory(this.storyIndex + 1);
    this.dom.exitBtn.onclick = () => this.stop();

    document.addEventListener('keydown', e => {
      if (!this.active || document.activeElement?.tagName === 'INPUT') return;
      if (e.key === 'Escape') {
        this.stop();
      } else if (e.key === 'ArrowRight') {
        this.goToStory(this.storyIndex + 1);
      } else if (e.key === 'ArrowLeft') {
        this.goToStory(this.storyIndex - 1);
      }
    });
  }

  _tick(now) {
    requestAnimationFrame(this._tick);
    const dtSec = Math.min(0.08, (now - this._lastTick) / 1000);
    const dtMs = dtSec * 1000;
    this._lastTick = now;

    if (!this.active || !this.stories.length) return;

    const story = this.stories[this.storyIndex];

    // Scroll ticker continuously even during the opening 2s swoop so it feels like live television
    this.tickerCurrentX -= TICKER_PX_PER_SEC * dtSec;
    const travelled = this.tickerStartX - this.tickerCurrentX;
    // Wrap seamlessly if needed
    const displayX =
      this.tickerCurrentX <= -this.tickerCopyWidth
        ? this.tickerCurrentX + this.tickerCopyWidth
        : this.tickerCurrentX;
    this.dom.tickerTrack.style.transform = `translate3d(${displayX.toFixed(1)}px, 0, 0)`;

    const storyProgress = Math.max(0, Math.min(1, travelled / Math.max(1, this.tickerTotalTravel)));
    this.dom.progressBar.style.width = `${(storyProgress * 100).toFixed(1)}%`;

    if (this.phase === 'flight' && this.flightState) {
      this.phaseElapsed += dtMs;
      const t = Math.min(1, this.phaseElapsed / this.flightState.duration);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      const q = new THREE.Quaternion().slerpQuaternions(this.flightState.fromQuat, this.flightState.toQuat, ease);
      const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
      const dist =
        THREE.MathUtils.lerp(this.flightState.fromDist, this.flightState.toDist, ease) +
        Math.sin(t * Math.PI) * this.flightState.arcBoost;

      const { lat, lon } = vec3ToLatLon(dir);
      this.globe.setTourCameraPose(lat, lon, dist, story.centre.lat, story.centre.lon);

      if (t >= 1) {
        this.phase = 'glide';
      }
      return;
    }

    // Switch weather layers smoothly at each quarter of the story as the ticker scrolls
    const beats = story.beats || [];
    if (beats.length) {
      const targetBeat = Math.min(beats.length - 1, Math.floor(storyProgress * beats.length));
      if (targetBeat !== this.beatIndex) {
        this.beatIndex = targetBeat;
        const bLayer = beats[targetBeat]?.layer;
        if (bLayer) this._applyLayer(bLayer, story);
      }
    }

    // Once the entire event ticker has scrolled past, advance smoothly to the next global headline
    if (storyProgress >= 1) {
      this.goToStory(this.storyIndex + 1);
      return;
    }

    // Continuous 3D surface camera glide across the 4 waypoints synced to storyProgress
    const sPos = catmullRomVec3(this.waypointVecs, storyProgress).normalize();
    const sAhead = catmullRomVec3(this.waypointVecs, Math.min(1, storyProgress + 0.08)).normalize();

    const n = this.waypointDists.length;
    const dScaled = storyProgress * Math.max(1, n - 1);
    const dIdx = Math.min(Math.max(0, n - 2), Math.floor(dScaled));
    const dFrac = dScaled - dIdx;
    const dist =
      n >= 2
        ? THREE.MathUtils.lerp(this.waypointDists[dIdx], this.waypointDists[dIdx + 1], dFrac)
        : 1.82;

    const camGeo = vec3ToLatLon(sPos);
    const aheadGeo = vec3ToLatLon(sAhead);
    this.globe.setTourCameraPose(camGeo.lat, camGeo.lon, dist, aheadGeo.lat, aheadGeo.lon);
  }
}
