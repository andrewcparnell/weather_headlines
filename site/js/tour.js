import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js?v=20261126f';

const FLIGHT_MS = 2300;
const FALLBACK_STORY_SEC = 38;
const STORY_TICKER_SPEED = 86; // px/sec for main highlighted story ticker
const LINKS_TICKER_SPEED = 46; // px/sec slower independent CNN-style web links ticker

const LAYER_LABELS = {
  temperature: '2 M TEMP',
  anomaly: 'TEMP ANOMALY',
  wind: '10 M WINDS',
  rain: '24 H RAIN',
  pressure: 'PRESSURE',
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
  const escaped = escapeHtml(rawTicker);
  const withHighlights = escaped.replace(/\*\*(.+?)\*\*/g, '<mark class="ticker-kw">$1</mark>');
  return (
    withHighlights.replace(/\s*\/\/\/\s*/g, '<span class="ticker-sep" aria-hidden="true">◆</span>') +
    '<span class="ticker-sep" aria-hidden="true">◆</span>'
  );
}

function formatLinksCopyHtml(links) {
  return (links || [])
    .map(
      link =>
        `<a class="news-article-link" href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">` +
        `<span class="news-outlet-badge">${escapeHtml(link.source)}</span>` +
        `<span>${escapeHtml(link.title)}</span>` +
        `<span aria-hidden="true">↗</span>` +
        `</a><span class="links-sep" aria-hidden="true">•</span>`
    )
    .join('');
}

export class BroadcastTour {
  constructor(globe, dom, { onLayerSync } = {}) {
    this.globe = globe;
    this.dom = dom;
    this.onLayerSync = onLayerSync;

    this.stories = [];
    this.bulletinStamp = '';
    this.active = false;
    this._pendingStartIndex = null;
    this.muted = false;
    this.linksHovered = false;
    this.storyIndex = 0;
    this.beatIndex = 0;

    this.phase = 'idle'; // 'flight' | 'glide'
    this.phaseElapsed = 0;
    this.fallbackProgress = 0;
    this.audioEndedAt = null;

    // Independent continuous pixel offsets for the two CNN-style tickers
    this.storyOffsetPx = 0;
    this.storySingleWidth = 1800;
    this.linksOffsetPx = 0;
    this.linksSingleWidth = 1100;

    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.audioAvailable = false;

    this.audio.addEventListener('ended', () => {
      if (this.active) {
        this.audioEndedAt = performance.now();
      }
    });
    this.audio.addEventListener('error', () => {
      this.audioAvailable = false;
    });

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
    this.bulletinStamp = encodeURIComponent(bulletin?.generated_utc || Date.now());
    if (this._pendingStartIndex !== null && this.stories.length) {
      const idx = this._pendingStartIndex;
      this._pendingStartIndex = null;
      this.start(idx);
      return;
    }
    if (this.active && this.stories.length) {
      this.goToStory(this.storyIndex);
    }
  }

  start(startIndex = 0) {
    if (!this.stories.length) {
      this._pendingStartIndex = startIndex;
      return;
    }
    this._pendingStartIndex = null;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    this.active = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this.globe.setTourMode(true);
    this.goToStory(startIndex);
  }

  stop() {
    this.active = false;
    this._pendingStartIndex = null;
    this.audio.pause();
    this.audio.removeAttribute('src');
    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
  }

  toggleMute() {
    this.muted = !this.muted;
    this.audio.muted = this.muted;
    this.dom.muteBtn.setAttribute('aria-pressed', String(this.muted));
    this.dom.muteLabel.textContent = this.muted ? '🔇' : '🔊';
  }

  goToStory(index) {
    if (!this.stories.length) return;
    if (!this.active) {
      this.active = true;
      document.body.classList.add('tour-active');
      this.dom.playLauncher.hidden = true;
      this.dom.tourHud.hidden = false;
      this.globe.setTourMode(true);
    }

    this.storyIndex = ((index % this.stories.length) + this.stories.length) % this.stories.length;
    this.beatIndex = 0;
    this.fallbackProgress = 0;
    this.audioEndedAt = null;
    this.storyOffsetPx = 0;
    this.linksOffsetPx = 0;

    const story = this.stories[this.storyIndex];
    this.globe.highlightStoryPin(this.storyIndex);

    const audioUrl = story.audio_url || `data/audio/story-${this.storyIndex + 1}.mp3`;
    this.audio.pause();
    this.audio.src = `${audioUrl}?v=${this.bulletinStamp}`;
    this.audio.muted = this.muted;
    this.audioAvailable = true;
    this.audio.play().catch(() => {
      this.audioAvailable = false;
    });

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
      arcBoost: Math.min(0.8, 0.16 + angle * 0.32),
      duration: Math.max(1300, Math.min(FLIGHT_MS, 1000 + angle * 680)),
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
    const statPart = story?.stats?.headline_metric ? `${story.stats.headline_metric} · ` : '';
    this.dom.layerTag.textContent = `${statPart}${LAYER_LABELS[layerId] || layerId.toUpperCase()}`;
  }

  _renderChyron(story) {
    this.dom.counter.textContent = `${this.storyIndex + 1} / ${this.stories.length}`;
    this.dom.kicker.textContent = story.kicker || story.region || 'WORLD WEATHER';
    this.dom.headline.textContent = story.headline;
    this.dom.progressBar.style.width = '0%';

    // 1. Story ticker: repeat enough copies so the bar is 100% filled immediately and loops seamlessly
    const rawTicker = story.ticker || (story.beats || []).map(b => b.text).join('  ///  ');
    const storyHtml = formatTickerHtml(rawTicker);
    this.dom.tickerTrack.innerHTML =
      `<div class="ticker-copy">${storyHtml}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${storyHtml}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${storyHtml}</div>`;
    this.dom.tickerTrack.style.transform = 'translate3d(0px, 0, 0)';

    // 2. CNN-style bottom web links ticker: repeat 5 copies so even 1-2 short links fill the bar seamlessly
    const linksHtml = formatLinksCopyHtml(story.news_links || []);
    this.dom.newsLinks.innerHTML =
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>`;
    this.dom.newsLinks.style.transform = 'translate3d(0px, 0, 0)';

    requestAnimationFrame(() => {
      const firstStoryCopy = this.dom.tickerTrack.querySelector('.ticker-copy');
      if (firstStoryCopy && firstStoryCopy.offsetWidth > 100) {
        this.storySingleWidth = firstStoryCopy.offsetWidth;
      }
      const firstLinksCopy = this.dom.newsLinks.querySelector('.links-copy');
      if (firstLinksCopy && firstLinksCopy.offsetWidth > 80) {
        this.linksSingleWidth = firstLinksCopy.offsetWidth;
      }
    });
  }

  _bindControls() {
    this.dom.playTourBtn?.addEventListener('click', () => this.start(0));
    this.dom.prevBtn?.addEventListener('click', () => this.goToStory(this.storyIndex - 1));
    this.dom.nextBtn?.addEventListener('click', () => this.goToStory(this.storyIndex + 1));
    this.dom.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.dom.exitBtn?.addEventListener('click', () => this.stop());

    // Slow down links ticker gently on hover so clicking a moving news link is effortless
    this.dom.linksViewport?.addEventListener('mouseenter', () => {
      this.linksHovered = true;
    });
    this.dom.linksViewport?.addEventListener('mouseleave', () => {
      this.linksHovered = false;
    });

    document.addEventListener('keydown', e => {
      if (!this.active || document.activeElement?.tagName === 'INPUT') return;
      if (e.key === 'Escape') {
        this.stop();
      } else if (e.key === 'ArrowRight') {
        this.goToStory(this.storyIndex + 1);
      } else if (e.key === 'ArrowLeft') {
        this.goToStory(this.storyIndex - 1);
      } else if (e.key.toLowerCase() === 'm') {
        this.toggleMute();
      }
    });
  }

  _computeStoryProgress(dtSec) {
    const hasValidAudio =
      this.audioAvailable &&
      Number.isFinite(this.audio.duration) &&
      this.audio.duration > 3 &&
      !this.audio.paused;

    if (hasValidAudio) {
      this.fallbackProgress = Math.min(1, this.audio.currentTime / this.audio.duration);
      return this.fallbackProgress;
    }

    if (this.audioEndedAt !== null) {
      return 1;
    }

    this.fallbackProgress = Math.min(1, this.fallbackProgress + dtSec / FALLBACK_STORY_SEC);
    return this.fallbackProgress;
  }

  _tick(now) {
    requestAnimationFrame(this._tick);
    const dtSec = Math.min(0.08, (now - this._lastTick) / 1000);
    const dtMs = dtSec * 1000;
    this._lastTick = now;

    if (!this.active || !this.stories.length) return;

    const story = this.stories[this.storyIndex];

    // 1. Always scroll both tickers independently at smooth 60fps broadcast speeds!
    this.storyOffsetPx = (this.storyOffsetPx + STORY_TICKER_SPEED * dtSec) % Math.max(1, this.storySingleWidth);
    this.dom.tickerTrack.style.transform = `translate3d(${-this.storyOffsetPx.toFixed(1)}px, 0, 0)`;

    const linkSpeed = this.linksHovered ? LINKS_TICKER_SPEED * 0.2 : LINKS_TICKER_SPEED;
    this.linksOffsetPx = (this.linksOffsetPx + linkSpeed * dtSec) % Math.max(1, this.linksSingleWidth);
    this.dom.newsLinks.style.transform = `translate3d(${-this.linksOffsetPx.toFixed(1)}px, 0, 0)`;

    // 2. Story progress for progress bar, layer switching, and 3D camera glide
    const storyProgress = this._computeStoryProgress(dtSec);
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

    const beats = story.beats || [];
    if (beats.length) {
      const targetBeat = Math.min(beats.length - 1, Math.floor(storyProgress * beats.length));
      if (targetBeat !== this.beatIndex) {
        this.beatIndex = targetBeat;
        const bLayer = beats[targetBeat]?.layer;
        if (bLayer) this._applyLayer(bLayer, story);
      }
    }

    if (this.audioEndedAt !== null) {
      if (now - this.audioEndedAt >= 850) {
        this.goToStory(this.storyIndex + 1);
        return;
      }
    } else if (!this.audioAvailable && storyProgress >= 1) {
      this.goToStory(this.storyIndex + 1);
      return;
    }

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
