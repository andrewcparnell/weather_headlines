import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js?v=20261126r';

const STORY_FLIGHT_MS = 1900;
const FALLBACK_STORY_SEC = 47;
const SILENT_CLOCK_PACE = 0.80; // Gentle reading pace when presenter voice is switched off
const STORY_TICKER_SPEED = 46;  // px/sec calm, readable story ticker speed
const LINKS_TICKER_SPEED = 26;  // px/sec slower independent CNN-style web links ticker
const TOUR_TILT_DEG = 14.5;

const LAYER_LABELS = {
  temperature: '2 M TEMP',
  anomaly: 'TEMP ANOMALY',
  wind: '10 M WINDS',
  rain: '24 H RAINFALL',
  pressure: 'SEA-LEVEL PRESSURE',
};

function quinticEase(x) {
  const t = Math.max(0, Math.min(1, x));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function catmullRomScalar(v0, v1, v2, v3, u) {
  const u2 = u * u;
  const u3 = u2 * u;
  return (
    0.5 *
    (2 * v1 +
      (-v0 + v2) * u +
      (2 * v0 - 5 * v1 + 4 * v2 - v3) * u2 +
      (-v0 + 3 * v1 - 3 * v2 + v3) * u3)
  );
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function formatTickerClauseHtml(rawClause) {
  const escaped = escapeHtml(rawClause);
  const withHighlights = escaped.replace(/\*\*(.+?)\*\*/g, '<mark class="ticker-kw">$1</mark>');
  return withHighlights + '<span class="ticker-sep" aria-hidden="true">◆</span>';
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

function normaliseStoryCues(story) {
  const totalDur = Number(story.duration_sec) > 8 ? Number(story.duration_sec) : FALLBACK_STORY_SEC;
  const rawCues = Array.isArray(story.cues) && story.cues.length >= 2 ? story.cues : null;

  if (rawCues) {
    const hasExplicitTimes = rawCues.every(
      c => Number.isFinite(Number(c.start_sec)) && Number.isFinite(Number(c.end_sec))
    );
    const weights = rawCues.map(c => Math.max(15, String(c.ticker || c.spoken || '').length));
    const totalW = weights.reduce((a, b) => a + b, 0);
    let accW = 0;

    return rawCues.map((c, idx) => {
      const propStart = (accW / totalW) * totalDur;
      accW += weights[idx];
      const propEnd = idx === rawCues.length - 1 ? totalDur : (accW / totalW) * totalDur;
      const gentleBearing = -12 + (24 * idx) / Math.max(1, rawCues.length - 1);
      return {
        layer: c.layer || story.primary_layer || 'temperature',
        lat: Number(c.lat ?? story.centre.lat),
        lon: Number(c.lon ?? story.centre.lon),
        distance: Number(c.distance || 1.62),
        bearing: gentleBearing,
        label: c.label || story.kicker || story.region || 'WORLD WEATHER',
        metric: c.metric || story.stats?.headline_metric || '',
        startSec: hasExplicitTimes ? Number(c.start_sec) : propStart,
        endSec: hasExplicitTimes ? Number(c.end_sec) : propEnd,
        spoken: c.spoken || String(c.ticker || '').replace(/\*\*/g, ''),
        ticker: c.ticker || c.spoken || '',
      };
    });
  }

  const beats = story.beats || [];
  const wps = story.camera_waypoints || [];
  const count = Math.max(1, beats.length);
  return beats.map((b, idx) => {
    const wp = wps[Math.min(idx, Math.max(0, wps.length - 1))] || {
      lat: story.centre.lat,
      lon: story.centre.lon,
      distance: 1.64,
    };
    return {
      layer: b.layer || story.primary_layer || 'temperature',
      lat: wp.lat,
      lon: wp.lon,
      distance: wp.distance || 1.64,
      bearing: -12 + (24 * idx) / Math.max(1, count - 1),
      label: b.title || story.kicker || 'WORLD WEATHER',
      metric: story.stats?.headline_metric || b.tag || '',
      startSec: (idx / count) * totalDur,
      endSec: ((idx + 1) / count) * totalDur,
      spoken: String(b.text || '').replace(/\*\*/g, ''),
      ticker: b.text || '',
    };
  });
}

function buildCameraKnots(cues, totalDur) {
  const n = cues.length;
  return cues.map((c, i) => {
    let tSec;
    if (i === 0) tSec = 0;
    else if (i === n - 1) tSec = totalDur;
    else tSec = 0.5 * (c.startSec + c.endSec);

    return {
      tSec,
      vec: latLonToVec3(c.lat, c.lon, 1).normalize(),
      distance: c.distance,
      bearing: c.bearing,
    };
  });
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

    // Defaults: on-map script text ON, presenter voice OFF
    this.textEnabled = true;
    this.voiceEnabled = false;
    this.linksHovered = false;
    this.storyIndex = 0;
    this.cueIndex = -1;
    this.cues = [];
    this.cameraKnots = [];
    this.storyDurationSec = FALLBACK_STORY_SEC;
    this.storyElapsedSec = 0;

    this.phase = 'idle'; // 'flight' | 'cues'
    this.phaseElapsedMs = 0;
    this.storyFinishedAt = null;

    this.cueSpans = [];
    this.storySingleWidth = 1800;
    this.displayStoryPx = 0;
    this.linksOffsetPx = 0;
    this.linksSingleWidth = 1100;
    this.scriptWordEls = [];
    this.fastFlowSec = [];
    this.spokenCueFracs = [];

    // Continuous camera state (eliminates duplicate transitions or velocity snaps)
    this.flightState = null;
    this.currentFocusVec = new THREE.Vector3(0, 0, 1);
    this.currentDist = 1.85;
    this.currentBearing = 0;
    this.currentTilt = 0;

    // Short, soft felt-timpani transition beat played on launch and between stories
    this.stingAudio = new Audio('assets/bulletin-sting.wav');
    this.stingAudio.preload = 'auto';
    this.stingAudio.volume = 0.52;

    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.audio.muted = true;
    this.audioAvailable = false;

    this.audio.addEventListener('ended', () => {
      if (this.active && this.voiceEnabled && this.phase === 'cues') {
        this.storyFinishedAt = performance.now();
      }
    });
    this.audio.addEventListener('error', () => {
      this.audioAvailable = false;
    });

    this._lastTick = performance.now();
    this._syncVoiceButtonUi();
    this._bindControls();
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  _syncVoiceButtonUi() {
    if (this.dom.textBtn && this.dom.textLabel) {
      this.dom.textBtn.setAttribute('aria-pressed', String(this.textEnabled));
      this.dom.textBtn.classList.toggle('voice-on', this.textEnabled);
      this.dom.textLabel.textContent = this.textEnabled ? '📝 Text On' : '📝 Text Off';
    }
    if (this.dom.muteBtn && this.dom.muteLabel) {
      this.dom.muteBtn.setAttribute('aria-pressed', String(this.voiceEnabled));
      this.dom.muteBtn.classList.toggle('voice-on', this.voiceEnabled);
      this.dom.muteLabel.textContent = this.voiceEnabled ? '🔊 Voice On' : '🔇 Voice Off';
    }
    if (this.dom.scriptOverlay) {
      this.dom.scriptOverlay.hidden = !(this.active && this.textEnabled && this.phase === 'cues');
    }
  }

  toggleText() {
    this.textEnabled = !this.textEnabled;
    this._syncVoiceButtonUi();
  }

  _playSoftSting() {
    if (!this.voiceEnabled) return;
    this.stingAudio.currentTime = 0;
    this.stingAudio.play().catch(() => {});
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
      this.goToStory(this.storyIndex, { playSting: false });
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

    // Capture exact Explore camera position so frame 1 of the flight has zero jump
    const camNow = this.globe.getCurrentCameraGeo();
    this.currentFocusVec.copy(latLonToVec3(camNow.lat, camNow.lon, 1).normalize());
    this.currentDist = camNow.distance;
    this.currentBearing = 0;
    this.currentTilt = 0;

    this.active = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this._syncVoiceButtonUi();
    this.globe.setTourMode(true);
    this.goToStory(startIndex, { playSting: true });
  }

  stop() {
    this.active = false;
    this.phase = 'idle';
    this._pendingStartIndex = null;
    this.stingAudio.pause();
    this.audio.pause();
    this.audio.removeAttribute('src');
    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this._syncVoiceButtonUi();
    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
  }

  toggleMute() {
    this.voiceEnabled = !this.voiceEnabled;
    this.audio.muted = !this.voiceEnabled;
    if (!this.voiceEnabled) {
      this.stingAudio.pause();
    }
    this._syncVoiceButtonUi();

    if (this.voiceEnabled && this.active && this.phase === 'cues') {
      if (Number.isFinite(this.audio.duration) && this.audio.duration > 3) {
        const targetT = Math.min(this.audio.duration - 0.15, Math.max(0, this.storyElapsedSec));
        if (Math.abs(this.audio.currentTime - targetT) > 0.35) {
          this.audio.currentTime = targetT;
        }
      }
      this.audio.play().catch(() => {
        this.audioAvailable = false;
      });
    }
  }

  goToStory(index, { playSting = true } = {}) {
    if (!this.stories.length) return;
    if (!this.active) {
      this.start(index);
      return;
    }

    if (playSting) {
      this._playSoftSting();
    }

    this.storyIndex = ((index % this.stories.length) + this.stories.length) % this.stories.length;
    const story = this.stories[this.storyIndex];
    this.cues = normaliseStoryCues(story);
    this.storyDurationSec = this.cues[this.cues.length - 1]?.endSec || Number(story.duration_sec) || FALLBACK_STORY_SEC;
    this.cameraKnots = buildCameraKnots(this.cues, this.storyDurationSec);
    this.storyElapsedSec = 0;
    this.storyFinishedAt = null;
    this.cueIndex = -1;
    this.linksOffsetPx = 0;

    this.globe.highlightStoryPin(this.storyIndex);

    const audioUrl = story.audio_url || `data/audio/story-${this.storyIndex + 1}.mp3`;
    this.audio.pause();
    this.audio.src = `${audioUrl}?v=${this.bulletinStamp}`;
    this.audio.muted = !this.voiceEnabled;
    this.audio.currentTime = 0;
    this.audio.load();
    this.audioAvailable = true;

    // Single great-circle flight directly onto Knot 0 of the new story (never duplicated!)
    const startKnot = this.cameraKnots[0];
    const fromDir = this.currentFocusVec.clone().normalize();
    const toDir = startKnot.vec.clone().normalize();
    const angle = fromDir.angleTo(toDir);

    this.flightState = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
      fromDist: this.currentDist,
      toDist: startKnot.distance,
      fromBearing: this.currentBearing,
      toBearing: startKnot.bearing,
      fromTilt: this.currentTilt,
      toTilt: TOUR_TILT_DEG,
      arcBoost: Math.min(0.68, 0.12 + angle * 0.32),
      duration: Math.max(1350, Math.min(STORY_FLIGHT_MS, 1050 + angle * 550)),
    };

    this.phase = 'flight';
    this.phaseElapsedMs = 0;
    this._syncVoiceButtonUi();

    this._renderChyron(story);
    this._activateCue(0);
  }

  _beginCuesPhase() {
    this.phase = 'cues';
    this.storyElapsedSec = 0;
    this._syncVoiceButtonUi();
    this.audio.muted = !this.voiceEnabled;
    this.audio.play().catch(() => {
      this.audioAvailable = false;
    });
  }

  _activateCue(cueIdx) {
    if (cueIdx < 0 || cueIdx >= this.cues.length) return;
    this.cueIndex = cueIdx;
    const cue = this.cues[cueIdx];

    // Smoothly cross-fade weather variable & update read-only legend + strap badge
    this.globe.setLayer(cue.layer);
    this.onLayerSync?.(cue.layer);
    const metricPrefix = cue.metric ? `${cue.metric} · ` : '';
    this.dom.layerTag.textContent = `${metricPrefix}${LAYER_LABELS[cue.layer] || cue.layer.toUpperCase()}`;

    // Glide on-globe surface spotlight ring & location callout to this sentence's coordinates
    this.globe.setTourFocusCallout({
      lat: cue.lat,
      lon: cue.lon,
      label: cue.label,
      metric: `${cue.metric ? cue.metric + ' · ' : ''}${LAYER_LABELS[cue.layer] || cue.layer.toUpperCase()}`,
      layer: cue.layer,
    });

    this.dom.tickerTrack.querySelectorAll('.ticker-cue').forEach(el => {
      el.classList.toggle('active', Number(el.dataset.cue) === cueIdx);
    });

    this._prepareCueScriptWords(cue.spoken);
  }

  _prepareCueScriptWords(spokenText) {
    if (!this.dom.scriptText) return;
    const words = String(spokenText || '')
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (!words.length) {
      this.dom.scriptText.innerHTML = '';
      this.scriptWordEls = [];
      this.fastFlowSec = [];
      this.spokenCueFracs = [];
      return;
    }

    // 1. Voice OFF: quick 0.48-second animated word cascade, then hold steady until next point
    const flowWindowSec = Math.min(0.48, words.length * 0.022);
    this.fastFlowSec = words.map((_, idx) =>
      words.length <= 1 ? 0 : (idx / (words.length - 1)) * flowWindowSec
    );

    // 2. Voice ON: natural presenter speech cadence matching Sonia's spoken voice across the cue
    const weights = words.map(w => Math.max(2, w.length) + (/[.,;:]/.test(w) ? 3.5 : 0.8));
    const totalW = weights.reduce((a, b) => a + b, 0);
    let acc = 0;
    this.spokenCueFracs = weights.map(w => {
      const frac = (acc / totalW) * 0.86;
      acc += w;
      return frac;
    });

    this.dom.scriptText.innerHTML = words
      .map(w => `<span class="script-word">${escapeHtml(w)}</span>`)
      .join(' ');
    this.scriptWordEls = Array.from(this.dom.scriptText.querySelectorAll('.script-word'));
  }

  _updateScriptWordsProgress(cueElapsedSec, cueU) {
    if (!this.scriptWordEls.length) return;
    for (let i = 0; i < this.scriptWordEls.length; i++) {
      const visible = this.voiceEnabled
        ? cueU >= this.spokenCueFracs[i]
        : cueElapsedSec >= this.fastFlowSec[i];
      this.scriptWordEls[i].classList.toggle('spoken', visible);
    }
  }

  _sampleStorySpline(elapsedSec) {
    const knots = this.cameraKnots;
    const n = knots.length;
    if (n === 0) {
      return { vec: new THREE.Vector3(0, 0, 1), distance: 1.65, bearing: 0 };
    }
    if (n === 1 || elapsedSec <= knots[0].tSec) {
      return {
        vec: knots[0].vec.clone(),
        distance: knots[0].distance,
        bearing: knots[0].bearing,
      };
    }
    if (elapsedSec >= knots[n - 1].tSec) {
      return {
        vec: knots[n - 1].vec.clone(),
        distance: knots[n - 1].distance,
        bearing: knots[n - 1].bearing,
      };
    }

    let k = 0;
    for (let i = 0; i < n - 1; i++) {
      if (elapsedSec < knots[i + 1].tSec) {
        k = i;
        break;
      }
    }

    const k0 = knots[Math.max(0, k - 1)];
    const k1 = knots[k];
    const k2 = knots[k + 1];
    const k3 = knots[Math.min(n - 1, k + 2)];

    const span = Math.max(0.05, k2.tSec - k1.tSec);
    const u = Math.max(0, Math.min(1, (elapsedSec - k1.tSec) / span));

    const vec = new THREE.Vector3(
      catmullRomScalar(k0.vec.x, k1.vec.x, k2.vec.x, k3.vec.x, u),
      catmullRomScalar(k0.vec.y, k1.vec.y, k2.vec.y, k3.vec.y, u),
      catmullRomScalar(k0.vec.z, k1.vec.z, k2.vec.z, k3.vec.z, u)
    ).normalize();

    const distance = Math.max(
      1.36,
      Math.min(2.4, catmullRomScalar(k0.distance, k1.distance, k2.distance, k3.distance, u))
    );
    const bearing = catmullRomScalar(k0.bearing, k1.bearing, k2.bearing, k3.bearing, u);

    return { vec, distance, bearing };
  }

  _renderChyron(story) {
    this.dom.counter.textContent = `${this.storyIndex + 1} / ${this.stories.length}`;
    this.dom.kicker.textContent = story.kicker || story.region || 'WORLD WEATHER';
    this.dom.headline.textContent = story.headline;
    this.dom.progressBar.style.width = '0%';

    const copyInner = this.cues
      .map(
        (c, idx) =>
          `<span class="ticker-cue${idx === 0 ? ' active' : ''}" data-cue="${idx}">` +
          formatTickerClauseHtml(c.ticker) +
          `</span>`
      )
      .join('');

    // Repeat enough copies to span ultra-wide 4K monitors seamlessly
    this.dom.tickerTrack.innerHTML =
      `<div class="ticker-copy">${copyInner}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${copyInner}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${copyInner}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${copyInner}</div>`;

    const linksHtml = formatLinksCopyHtml(story.news_links || []);
    this.dom.newsLinks.innerHTML =
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>`;
    this.dom.newsLinks.style.transform = 'translate3d(0px, 0, 0)';

    this.cueSpans = [];
    this.displayStoryPx = -65;

    requestAnimationFrame(() => {
      const firstStoryCopy = this.dom.tickerTrack.querySelector('.ticker-copy');
      if (firstStoryCopy && firstStoryCopy.offsetWidth > 100) {
        this.storySingleWidth = firstStoryCopy.offsetWidth;
        const cueEls = Array.from(firstStoryCopy.querySelectorAll('.ticker-cue'));
        let runningLeft = 0;
        this.cueSpans = cueEls.map(el => {
          const w = Math.max(40, el.offsetWidth);
          const span = { leftPx: runningLeft, widthPx: w };
          runningLeft += w;
          return span;
        });
      }
      const firstLinksCopy = this.dom.newsLinks.querySelector('.links-copy');
      if (firstLinksCopy && firstLinksCopy.offsetWidth > 80) {
        this.linksSingleWidth = firstLinksCopy.offsetWidth;
      }
    });
  }

  _bindControls() {
    this.dom.playTourBtn?.addEventListener('click', () => this.start(0));
    this.dom.prevBtn?.addEventListener('click', () =>
      this.goToStory(this.storyIndex - 1, { playSting: true })
    );
    this.dom.nextBtn?.addEventListener('click', () =>
      this.goToStory(this.storyIndex + 1, { playSting: true })
    );
    this.dom.textBtn?.addEventListener('click', () => this.toggleText());
    this.dom.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.dom.exitBtn?.addEventListener('click', () => this.stop());

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
        this.goToStory(this.storyIndex + 1, { playSting: true });
      } else if (e.key === 'ArrowLeft') {
        this.goToStory(this.storyIndex - 1, { playSting: true });
      } else if (e.key.toLowerCase() === 'm') {
        this.toggleMute();
      } else if (e.key.toLowerCase() === 't') {
        this.toggleText();
      }
    });
  }

  _updateLinksTicker(dtSec) {
    const linkSpeed = this.linksHovered ? LINKS_TICKER_SPEED * 0.28 : LINKS_TICKER_SPEED;
    this.linksOffsetPx = (this.linksOffsetPx + linkSpeed * dtSec) % Math.max(1, this.linksSingleWidth);
    this.dom.newsLinks.style.transform = `translate3d(${-this.linksOffsetPx.toFixed(1)}px, 0, 0)`;
  }

  _applyStoryTickerOffset(rawPx) {
    const leadInPx = 28;
    const w = Math.max(1, this.storySingleWidth);
    const wrapped = (((rawPx - leadInPx) % w) + w) % w;
    this.dom.tickerTrack.style.transform = `translate3d(${-wrapped.toFixed(1)}px, 0, 0)`;
  }

  _tick(now) {
    requestAnimationFrame(this._tick);
    const dtSec = Math.min(0.08, (now - this._lastTick) / 1000);
    const dtMs = dtSec * 1000;
    this._lastTick = now;

    if (!this.active || !this.stories.length || !this.cues.length) return;

    this._updateLinksTicker(dtSec);

    // Phase 1: Single smooth great-circle flight onto Knot 0 of the story
    if (this.phase === 'flight' && this.flightState) {
      this.phaseElapsedMs += dtMs;
      const t = Math.min(1, this.phaseElapsedMs / this.flightState.duration);
      const ease = quinticEase(t);

      const q = new THREE.Quaternion().slerpQuaternions(this.flightState.fromQuat, this.flightState.toQuat, ease);
      const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
      const dist =
        THREE.MathUtils.lerp(this.flightState.fromDist, this.flightState.toDist, ease) +
        Math.sin(t * Math.PI) * this.flightState.arcBoost;
      const bearing = THREE.MathUtils.lerp(this.flightState.fromBearing, this.flightState.toBearing, ease);
      const tilt = THREE.MathUtils.lerp(this.flightState.fromTilt, this.flightState.toTilt, ease);

      this.currentFocusVec.copy(dir);
      this.currentDist = dist;
      this.currentBearing = bearing;
      this.currentTilt = tilt;

      const { lat, lon } = vec3ToLatLon(dir);
      this.globe.setTourCameraPose(lat, lon, dist, bearing, tilt);

      this.displayStoryPx = THREE.MathUtils.lerp(-65, 0, ease);
      this._applyStoryTickerOffset(this.displayStoryPx);
      this._updateScriptWordsProgress(-0.01, -0.01);

      if (t >= 1) {
        this._beginCuesPhase();
      }
      return;
    }

    // Phase 2: Continuous spline glide + sentence-synchronised variable switches & calm ticker scroll
    const useAudioClock =
      this.voiceEnabled &&
      this.audioAvailable &&
      !this.audio.paused &&
      Number.isFinite(this.audio.duration) &&
      this.audio.duration > 3;

    if (useAudioClock) {
      this.storyElapsedSec = this.audio.currentTime;
    } else if (this.storyFinishedAt === null) {
      this.storyElapsedSec += dtSec * SILENT_CLOCK_PACE;
      if (this.storyElapsedSec >= this.storyDurationSec) {
        this.storyElapsedSec = this.storyDurationSec;
        this.storyFinishedAt = now;
      }
    }

    const overallProgress = Math.min(1, this.storyElapsedSec / Math.max(1, this.storyDurationSec));
    this.dom.progressBar.style.width = `${(overallProgress * 100).toFixed(1)}%`;

    let targetCueIdx = this.cues.length - 1;
    for (let i = 0; i < this.cues.length; i++) {
      if (this.storyElapsedSec < this.cues[i].endSec || i === this.cues.length - 1) {
        targetCueIdx = i;
        break;
      }
    }

    if (targetCueIdx !== this.cueIndex) {
      this._activateCue(targetCueIdx);
    }

    const activeCue = this.cues[this.cueIndex];
    const cueDur = Math.max(0.4, activeCue.endSec - activeCue.startSec);
    const cueElapsedSec = Math.max(0, this.storyElapsedSec - activeCue.startSec);
    const cueU = Math.max(0, Math.min(1, cueElapsedSec / cueDur));

    this._updateScriptWordsProgress(cueElapsedSec, cueU);

    if (this.cueSpans.length === this.cues.length) {
      const span = this.cueSpans[this.cueIndex];
      const targetPx = span.leftPx + cueU * span.widthPx;
      if (Math.abs(targetPx - this.displayStoryPx) > this.storySingleWidth * 0.5) {
        this.displayStoryPx = targetPx;
      } else {
        this.displayStoryPx += (targetPx - this.displayStoryPx) * Math.min(1, dtSec * 6.5);
      }
    } else {
      this.displayStoryPx = (this.displayStoryPx + STORY_TICKER_SPEED * dtSec) % Math.max(1, this.storySingleWidth);
    }
    this._applyStoryTickerOffset(this.displayStoryPx);

    if (this.storyFinishedAt !== null && now - this.storyFinishedAt >= 800) {
      this.goToStory(this.storyIndex + 1, { playSting: true });
      return;
    }

    // Evaluate continuous Catmull-Rom spline + silky exponential damper (zero jumps at sentence boundaries!)
    const targetPose = this._sampleStorySpline(this.storyElapsedSec);
    const damp = 1 - Math.exp(-dtSec * 3.8);

    this.currentFocusVec.lerp(targetPose.vec, damp).normalize();
    this.currentDist = THREE.MathUtils.lerp(this.currentDist, targetPose.distance, damp);
    this.currentBearing = THREE.MathUtils.lerp(this.currentBearing, targetPose.bearing, damp);
    this.currentTilt = THREE.MathUtils.lerp(this.currentTilt, TOUR_TILT_DEG, damp);

    const { lat, lon } = vec3ToLatLon(this.currentFocusVec);
    this.globe.setTourCameraPose(lat, lon, this.currentDist, this.currentBearing, this.currentTilt);
  }
}
