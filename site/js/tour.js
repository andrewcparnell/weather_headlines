import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js';

const FLIGHT_MS = 2700;
const BEAT_MS = 8600; // ~34.4s per 4-beat story so camera glides smoothly while viewer reads

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

export class BroadcastTour {
  constructor(globe, dom, { onLayerSync } = {}) {
    this.globe = globe;
    this.dom = dom;
    this.onLayerSync = onLayerSync;

    this.stories = [];
    this.active = false;
    this.playing = false;
    this.voiceEnabled = false;

    this.storyIndex = 0;
    this.beatIndex = 0;
    this.phase = 'idle'; // 'flight' | 'glide'
    this.phaseElapsed = 0;
    this.glideProgress = 0; // 0..1 across all 4 beats of the current story

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
    this._renderStepPills();
  }

  start(startIndex = 0) {
    if (!this.stories.length) return;
    this.active = true;
    this.playing = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this.globe.setTourMode(true);
    this._syncPlayIcon();
    this.goToStory(startIndex);
  }

  stop() {
    if (!this.active) return;
    this.active = false;
    this.playing = false;
    this._cancelSpeech();
    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
  }

  togglePlayPause() {
    if (!this.active) return;
    this.playing = !this.playing;
    this._syncPlayIcon();
    if (!this.playing) {
      this._cancelSpeech();
    } else if (this.voiceEnabled && this.phase === 'glide') {
      this._speakCurrentBeat();
    }
  }

  toggleVoice() {
    this.voiceEnabled = !this.voiceEnabled;
    this.dom.voiceBtn.setAttribute('aria-pressed', String(this.voiceEnabled));
    this.dom.voiceLabel.textContent = this.voiceEnabled ? '🔊 Voice' : '🔇 Voice';
    if (!this.voiceEnabled) {
      this._cancelSpeech();
    } else if (this.active && this.playing && this.phase === 'glide') {
      this._speakCurrentBeat();
    }
  }

  goToStory(index) {
    if (!this.stories.length) return;
    this._cancelSpeech();
    this.storyIndex = ((index % this.stories.length) + this.stories.length) % this.stories.length;
    this.beatIndex = 0;
    this.glideProgress = 0;

    const story = this.stories[this.storyIndex];
    this.globe.highlightStoryPin(this.storyIndex);

    // Prepare smooth 3D spline vectors for this story's surface waypoints
    const wps = story.camera_waypoints || [];
    this.waypointVecs = wps.map(w => latLonToVec3(w.lat, w.lon, 1).normalize());
    this.waypointDists = wps.map(w => w.distance || 1.9);

    const startWp = wps[0] || { lat: story.centre.lat, lon: story.centre.lon, distance: 1.95 };
    const camNow = this.globe.getCurrentCameraGeo();
    const fromDir = latLonToVec3(camNow.lat, camNow.lon, 1).normalize();
    const toDir = latLonToVec3(startWp.lat, startWp.lon, 1).normalize();
    const angle = fromDir.angleTo(toDir);

    this.flightState = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
      fromDist: camNow.distance,
      toDist: startWp.distance || 1.95,
      arcBoost: Math.min(0.85, 0.2 + angle * 0.36),
      duration: Math.max(1500, Math.min(FLIGHT_MS, 1100 + angle * 750)),
    };

    this.phase = 'flight';
    this.phaseElapsed = 0;

    const firstLayer = story.beats?.[0]?.layer || story.primary_layer;
    this._applyLayer(firstLayer);
    this._renderStoryCard(story, 0);
  }

  jumpToBeat(beatIdx) {
    const story = this.stories[this.storyIndex];
    if (!story || beatIdx < 0 || beatIdx >= story.beats.length) return;
    this._cancelSpeech();
    this.phase = 'glide';
    this.beatIndex = beatIdx;
    this.glideProgress = beatIdx / story.beats.length;
    this._applyBeat(story, beatIdx);
    if (this.playing && this.voiceEnabled) {
      this._speakCurrentBeat();
    }
  }

  _applyLayer(layerId) {
    this.globe.setLayer(layerId);
    this.onLayerSync?.(layerId);
  }

  _applyBeat(story, bIdx) {
    this.beatIndex = bIdx;
    const beat = story.beats[bIdx];
    if (!beat) return;
    if (beat.layer) this._applyLayer(beat.layer);
    this.dom.beatText.textContent = beat.text;
    [...this.dom.beatTabs.children].forEach((tab, idx) => {
      tab.classList.toggle('active', idx === bIdx);
      tab.setAttribute('aria-selected', String(idx === bIdx));
    });
  }

  _renderStepPills() {
    this.dom.steps.replaceChildren();
    this.stories.forEach((st, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'step-pill';
      btn.title = st.headline;
      btn.innerHTML = `<span class="step-fill"></span><span class="step-num">${idx + 1}</span>`;
      btn.onclick = () => {
        this.playing = true;
        this._syncPlayIcon();
        this.goToStory(idx);
      };
      this.dom.steps.appendChild(btn);
    });
  }

  _renderStoryCard(story, beatIdx) {
    this.dom.kicker.textContent = story.kicker || story.region || 'WORLD WEATHER';
    this.dom.headline.textContent = story.headline;

    const st = story.stats || {};
    this.dom.statMetric.textContent = st.headline_metric
      ? `${st.headline_metric}${st.metric_label ? ` · ${st.metric_label}` : ''}`
      : '';
    this.dom.statMetric.hidden = !this.dom.statMetric.textContent;

    this.dom.statRank.textContent = st.era5_rank ? `ERA5: ${st.era5_rank}` : '';
    this.dom.statRank.hidden = !this.dom.statRank.textContent;

    this.dom.statPersistence.textContent = st.persistence || '';
    this.dom.statPersistence.hidden = !this.dom.statPersistence.textContent;

    // 4 chapter tabs
    this.dom.beatTabs.replaceChildren();
    const shortLabels = ['1. Live Event', '2. Persistence', '3. Historical Rank', '4. Climate Link'];
    (story.beats || []).forEach((b, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.role = 'tab';
      btn.className = `beat-tab${idx === beatIdx ? ' active' : ''}`;
      btn.textContent = shortLabels[idx] || `${idx + 1}. ${b.tag}`;
      btn.title = b.title || b.tag;
      btn.onclick = () => this.jumpToBeat(idx);
      this.dom.beatTabs.appendChild(btn);
    });

    this._applyBeat(story, beatIdx);

    // International news pills
    this.dom.newsLinks.replaceChildren();
    (story.news_links || []).forEach(link => {
      const a = document.createElement('a');
      a.className = 'news-pill';
      a.href = link.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.innerHTML = `<strong>${this._esc(link.source)}</strong><span class="news-pill-title">${this._esc(link.title)}</span><span>↗</span>`;
      a.onclick = () => {
        // Pause gracefully while the user reads the news article in another tab
        if (this.playing) this.togglePlayPause();
      };
      this.dom.newsLinks.appendChild(a);
    });

    this._updateStepFill(0);
  }

  _updateStepFill(storyFraction) {
    [...this.dom.steps.children].forEach((pill, idx) => {
      pill.classList.toggle('active', idx === this.storyIndex);
      const fill = pill.querySelector('.step-fill');
      if (!fill) return;
      if (idx < this.storyIndex) fill.style.width = '100%';
      else if (idx > this.storyIndex) fill.style.width = '0%';
      else fill.style.width = `${Math.round(Math.max(0, Math.min(1, storyFraction)) * 100)}%`;
    });
  }

  _speakCurrentBeat() {
    if (!this.voiceEnabled || !('speechSynthesis' in window)) return;
    this._cancelSpeech();
    const story = this.stories[this.storyIndex];
    const beat = story?.beats?.[this.beatIndex];
    if (!beat) return;

    const textToRead = this.beatIndex === 0 ? `${story.headline}. ${beat.text}` : beat.text;
    const utter = new SpeechSynthesisUtterance(textToRead);
    const voices = window.speechSynthesis.getVoices();
    const gbVoice =
      voices.find(v => v.lang === 'en-GB' && /Daniel|Serena|Martha|Google UK/i.test(v.name)) ||
      voices.find(v => v.lang.startsWith('en-GB')) ||
      voices.find(v => v.lang.startsWith('en'));
    if (gbVoice) utter.voice = gbVoice;
    utter.rate = 1.02;
    this.currentUtterance = utter;
    window.speechSynthesis.speak(utter);
  }

  _cancelSpeech() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    this.currentUtterance = null;
  }

  _syncPlayIcon() {
    this.dom.playIcon.textContent = this.playing ? '❚❚' : '▶';
    this.dom.playPauseBtn.setAttribute('aria-label', this.playing ? 'Pause tour' : 'Resume tour');
  }

  _esc(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  _bindControls() {
    this.dom.playTourBtn.onclick = () => this.start(0);
    this.dom.prevBtn.onclick = () => {
      this.playing = true;
      this._syncPlayIcon();
      this.goToStory(this.storyIndex - 1);
    };
    this.dom.nextBtn.onclick = () => {
      this.playing = true;
      this._syncPlayIcon();
      this.goToStory(this.storyIndex + 1);
    };
    this.dom.playPauseBtn.onclick = () => this.togglePlayPause();
    this.dom.voiceBtn.onclick = () => this.toggleVoice();
    this.dom.exitBtn.onclick = () => this.stop();

    document.addEventListener('keydown', e => {
      if (!this.active || document.activeElement?.tagName === 'INPUT') return;
      if (e.key === 'Escape') {
        this.stop();
      } else if (e.key === ' ') {
        e.preventDefault();
        this.togglePlayPause();
      } else if (e.key === 'ArrowRight') {
        this.goToStory(this.storyIndex + 1);
      } else if (e.key === 'ArrowLeft') {
        this.goToStory(this.storyIndex - 1);
      }
    });
  }

  _tick(now) {
    requestAnimationFrame(this._tick);
    const dtMs = Math.min(80, now - this._lastTick);
    this._lastTick = now;

    if (!this.active || !this.stories.length) return;

    const story = this.stories[this.storyIndex];
    const numBeats = story.beats.length || 4;

    if (this.phase === 'flight' && this.flightState) {
      if (this.playing) this.phaseElapsed += dtMs;
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
        this.phaseElapsed = 0;
        if (this.playing && this.voiceEnabled) {
          this._speakCurrentBeat();
        }
      }
      return;
    }

    // Continuous surface glide across the 4 waypoints!
    // Even if paused, gentle micro-drift keeps the 3D scene alive; when playing, advance through all 4 beats.
    if (this.playing) {
      const totalStoryMs = numBeats * BEAT_MS;
      // If voiceover is enabled and speaking, hold beat switch until speech ends (with generous cap)
      const speechBusy =
        this.voiceEnabled && 'speechSynthesis' in window && window.speechSynthesis.speaking;
      const maxProgressForBeat = (this.beatIndex + 0.96) / numBeats;

      let nextProg = this.glideProgress + dtMs / totalStoryMs;
      if (speechBusy && nextProg > maxProgressForBeat) {
        nextProg = maxProgressForBeat;
      }
      this.glideProgress = nextProg;

      const targetBeat = Math.min(numBeats - 1, Math.floor(this.glideProgress * numBeats));
      if (targetBeat !== this.beatIndex && this.glideProgress < 1) {
        this._applyBeat(story, targetBeat);
        if (this.voiceEnabled) this._speakCurrentBeat();
      }

      this._updateStepFill(this.glideProgress);

      if (this.glideProgress >= 1) {
        this.goToStory(this.storyIndex + 1);
        return;
      }
    }

    // Evaluate smooth spherical spline position & look-ahead along waypoints
    const sPos = catmullRomVec3(this.waypointVecs, this.glideProgress).normalize();
    const sAhead = catmullRomVec3(this.waypointVecs, Math.min(1, this.glideProgress + 0.08)).normalize();

    const n = this.waypointDists.length;
    const dScaled = this.glideProgress * Math.max(1, n - 1);
    const dIdx = Math.min(Math.max(0, n - 2), Math.floor(dScaled));
    const dFrac = dScaled - dIdx;
    const dist =
      n >= 2
        ? THREE.MathUtils.lerp(this.waypointDists[dIdx], this.waypointDists[dIdx + 1], dFrac)
        : 1.9;

    const camGeo = vec3ToLatLon(sPos);
    const aheadGeo = vec3ToLatLon(sAhead);
    this.globe.setTourCameraPose(camGeo.lat, camGeo.lon, dist, aheadGeo.lat, aheadGeo.lon);
  }
}
