import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js?v=c499fb13';

const STORY_FLIGHT_MS = 1850;
const FALLBACK_STORY_SEC = 47;
const SILENT_CLOCK_PACE = 0.82;
const STORY_TICKER_SPEED = 46; // px/sec smooth story findings scroll
const LINKS_TICKER_SPEED = 26; // px/sec independent CNN-style news links scroll
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

export function safeHttpUrl(rawUrl) {
  if (!rawUrl) return null;
  try {
    const parsed = new URL(String(rawUrl).trim(), window.location.origin);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.href;
    }
  } catch {
    // Reject malformed URLs
  }
  return null;
}

export function formatTickerClauseHtml(rawClause) {
  const escaped = escapeHtml(rawClause);
  const withHighlights = escaped.replace(/\*\*(.+?)\*\*/g, '<mark class="ticker-kw">$1</mark>');
  return withHighlights + '<span class="ticker-sep" aria-hidden="true">◆</span>';
}

function formatLinksCopyHtml(links) {
  const valid = (links || []).filter(lk => lk && safeHttpUrl(lk.url));
  return valid
    .map(link => {
      const href = safeHttpUrl(link.url);
      return (
        `<a class="news-article-link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">` +
        `<span class="news-outlet-badge">${escapeHtml(link.source)}</span>` +
        `<span>${escapeHtml(link.title)}</span>` +
        `<span aria-hidden="true">↗</span>` +
        `</a><span class="links-sep" aria-hidden="true">•</span>`
      );
    })
    .join('');
}

function normaliseStoryCues(story) {
  const totalDur = Number(story.duration_sec) > 8 ? Number(story.duration_sec) : FALLBACK_STORY_SEC;
  const rawCues = Array.isArray(story.cues) && story.cues.length >= 2 ? story.cues : null;

  if (rawCues) {
    const hasExplicitTimes = rawCues.every(
      c => Number.isFinite(Number(c.start_sec)) && Number.isFinite(Number(c.end_sec))
    );
    const weights = rawCues.map(c => Math.max(15, String(c.spoken || c.ticker || '').length));
    const totalW = weights.reduce((a, b) => a + b, 0);
    let accW = 0;

    return rawCues.map((c, idx) => {
      const propStart = (accW / totalW) * totalDur;
      accW += weights[idx];
      const propEnd = idx === rawCues.length - 1 ? totalDur : (accW / totalW) * totalDur;
      const gentleBearing = -10 + (20 * idx) / Math.max(1, rawCues.length - 1);
      return {
        layer: c.layer || story.primary_layer || 'temperature',
        lat: Number(c.lat ?? story.centre.lat),
        lon: Number(c.lon ?? story.centre.lon),
        distance: Number(c.distance || 1.62),
        bearing: Number.isFinite(Number(c.bearing)) ? Number(c.bearing) : gentleBearing,
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
      bearing: -10 + (20 * idx) / Math.max(1, count - 1),
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
  constructor(globe, dom, { onLayerSync, onBulletinApplied } = {}) {
    this.globe = globe;
    this.dom = dom;
    this.onLayerSync = onLayerSync;
    this.onBulletinApplied = onBulletinApplied;

    this.stories = [];
    this.bulletinStamp = '';
    this.bulletinFailed = false;
    this._queuedBulletin = null;
    this._pendingStartIndex = null;
    this._preTourLayer = null;
    this._hasCompletedTour = false;

    this.active = false;
    this.voiceEnabled = false;
    this.linksHovered = false;
    this.storyIndex = 0;
    this.cueIndex = -1;
    this.cues = [];
    this.cameraKnots = [];
    this.storyDurationSec = FALLBACK_STORY_SEC;
    this.storyElapsedSec = 0;

    this.phase = 'idle';
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

    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    this.reducedMotion = Boolean(mq?.matches);
    mq?.addEventListener?.('change', e => {
      this.reducedMotion = Boolean(e.matches);
    });

    this.flightState = null;
    this.currentFocusVec = new THREE.Vector3(0, 0, 1);
    this.currentDist = 1.85;
    this.currentBearing = 0;
    this.currentTilt = 0;
    this._flightQ = new THREE.Quaternion();
    this._flightDir = new THREE.Vector3();

    // Lazy-loaded audio objects: nothing is downloaded until Sound is switched on
    this.stingAudio = null;
    this.audio = new Audio();
    this.audio.preload = 'none';
    this.audio.muted = true;
    this.audioAvailable = false;
    this._loadedAudioSrc = '';
    this._pendingMetaHandler = null;

    this.audio.addEventListener('ended', () => {
      if (this.active && this.voiceEnabled && this.phase === 'cues') {
        this.storyFinishedAt = performance.now();
      }
    });
    this.audio.addEventListener('error', () => {
      this.audioAvailable = false;
    });

    this._lastTick = performance.now();
    this._syncSoundButtonUi();
    this._bindControls();
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  _clearPendingMetaListener() {
    if (this._pendingMetaHandler) {
      this.audio.removeEventListener('loadedmetadata', this._pendingMetaHandler);
      this._pendingMetaHandler = null;
    }
  }

  _syncSoundButtonUi() {
    if (this.dom.muteBtn && this.dom.muteLabel) {
      this.dom.muteBtn.setAttribute('aria-pressed', String(this.voiceEnabled));
      this.dom.muteBtn.classList.toggle('voice-on', this.voiceEnabled);
      this.dom.muteLabel.textContent = this.voiceEnabled ? 'Sound On' : 'Sound Off';
    }
    if (this.dom.iconSoundOff && this.dom.iconSoundOn) {
      this.dom.iconSoundOff.hidden = this.voiceEnabled;
      this.dom.iconSoundOn.hidden = !this.voiceEnabled;
    }
    if (this.dom.scriptOverlay) {
      this.dom.scriptOverlay.hidden = !(this.active && this.phase === 'cues');
    }
  }

  _showUnavailableNotice(message) {
    if (!this.dom.statusNotice) return;
    this.dom.statusNotice.textContent = message;
    this.dom.statusNotice.hidden = false;
  }

  _hideUnavailableNotice() {
    if (!this.dom.statusNotice) return;
    this.dom.statusNotice.hidden = true;
    this.dom.statusNotice.textContent = '';
  }

  setBulletinError() {
    this.bulletinFailed = true;
    this._pendingStartIndex = null;
  }

  _applyBulletinData(bulletin) {
    this.stories = bulletin?.stories || [];
    this.bulletinStamp = encodeURIComponent(bulletin?.generated_utc || 'latest');
    this.bulletinFailed = false;
    this._hideUnavailableNotice();
    this.onBulletinApplied?.(bulletin);
  }

  setBulletin(bulletin) {
    if (!bulletin || !Array.isArray(bulletin.stories) || !bulletin.stories.length) {
      return;
    }
    if (this.active) {
      this._queuedBulletin = bulletin;
      return;
    }
    this._applyBulletinData(bulletin);
    if (this._pendingStartIndex !== null && this.stories.length) {
      const idx = this._pendingStartIndex;
      this._pendingStartIndex = null;
      this.start(idx);
    }
  }

  _playSoftSting() {
    if (!this.voiceEnabled) return;
    if (!this.stingAudio) {
      this.stingAudio = new Audio('assets/bulletin-sting.mp4');
      this.stingAudio.preload = 'auto';
      this.stingAudio.volume = 0.5;
    }
    this.stingAudio.currentTime = 0;
    this.stingAudio.play().catch(() => {});
  }

  _storyAudioUrl(story, storyIndex) {
    const base = story.audio_url || `data/audio/story-${storyIndex + 1}.mp3`;
    return base.includes('?') ? base : `${base}?v=${this.bulletinStamp}`;
  }

  _ensureStoryAudioLoaded(story, storyIndex) {
    const targetSrc = this._storyAudioUrl(story, storyIndex);
    if (this._loadedAudioSrc !== targetSrc) {
      this._clearPendingMetaListener();
      this.audio.pause();
      this.audio.src = targetSrc;
      this._loadedAudioSrc = targetSrc;
      this.audio.load();
    }
    this.audio.muted = !this.voiceEnabled;
    this.audioAvailable = true;
  }

  start(startIndex = 0) {
    if (!this.stories.length) {
      this._pendingStartIndex = null;
      this._showUnavailableNotice('Headlines unavailable, please try again shortly');
      return;
    }
    this._pendingStartIndex = null;
    this._hideUnavailableNotice();

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    this._preTourLayer = this.globe.currentLayer;

    const camNow = this.globe.getCurrentCameraGeo();
    this.currentFocusVec.copy(latLonToVec3(camNow.lat, camNow.lon, 1).normalize());
    this.currentDist = camNow.distance;
    this.currentBearing = 0;
    this.currentTilt = 0;

    this.active = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this._syncSoundButtonUi();
    this.globe.setTourMode(true);
    this.goToStory(startIndex, { playSting: true });
    this.dom.nextBtn?.focus();
  }

  stop({ completed = false } = {}) {
    this.active = false;
    this.phase = 'idle';
    this._pendingStartIndex = null;
    this._clearPendingMetaListener();
    this.stingAudio?.pause();
    this.audio.pause();
    this.audio.removeAttribute('src');
    this._loadedAudioSrc = '';

    if (this._queuedBulletin) {
      const pending = this._queuedBulletin;
      this._queuedBulletin = null;
      this._applyBulletinData(pending);
    }

    if (completed) {
      this._hasCompletedTour = true;
    }
    if (this._hasCompletedTour) {
      if (this.dom.playCaption) {
        this.dom.playCaption.textContent = 'Replay headlines';
      }
      this.dom.playTourBtn?.setAttribute('aria-label', 'Replay headlines');
    }

    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this._syncSoundButtonUi();

    if (this._preTourLayer) {
      this.globe.setLayer(this._preTourLayer);
      this.onLayerSync?.(this._preTourLayer);
    }

    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
    this.dom.playTourBtn?.focus();
  }

  toggleMute() {
    this.voiceEnabled = !this.voiceEnabled;
    this.audio.muted = !this.voiceEnabled;
    this._clearPendingMetaListener();

    if (!this.voiceEnabled) {
      this.stingAudio?.pause();
      this.audio.pause();
      this._syncSoundButtonUi();
      return;
    }
    this._syncSoundButtonUi();

    if (!this.active || !this.stories[this.storyIndex]) return;

    const story = this.stories[this.storyIndex];
    this._ensureStoryAudioLoaded(story, this.storyIndex);

    const activeCueStart =
      this.phase === 'cues' && this.cueIndex >= 0 && this.cues[this.cueIndex]
        ? this.cues[this.cueIndex].startSec
        : 0;

    if (this.phase === 'cues') {
      this.storyElapsedSec = activeCueStart;
      this.storyFinishedAt = null;
      const expectedStoryIdx = this.storyIndex;
      const startFromCue = () => {
        this._pendingMetaHandler = null;
        if (!this.active || !this.voiceEnabled || this.storyIndex !== expectedStoryIdx) return;
        try {
          this.audio.currentTime = Math.min(
            Math.max(0, (this.audio.duration || activeCueStart + 1) - 0.15),
            Math.max(0, activeCueStart)
          );
        } catch (_) {}
        this.audio.play().catch(() => {
          this.audioAvailable = false;
        });
      };
      if (this.audio.readyState >= 1) {
        startFromCue();
      } else {
        this._pendingMetaHandler = startFromCue;
        this.audio.addEventListener('loadedmetadata', startFromCue, { once: true });
      }
    }
  }

  goToStory(index, { playSting = true } = {}) {
    if (this._queuedBulletin) {
      const pending = this._queuedBulletin;
      this._queuedBulletin = null;
      this._applyBulletinData(pending);
    }
    if (!this.stories.length) return;
    if (!this.active) {
      this.start(index);
      return;
    }

    if (index >= this.stories.length) {
      this.stop({ completed: true });
      return;
    }

    this._clearPendingMetaListener();
    this.audio.pause();

    if (playSting) {
      this._playSoftSting();
    }

    this.storyIndex = Math.max(0, index);
    const story = this.stories[this.storyIndex];
    this.cues = normaliseStoryCues(story);
    this.storyDurationSec =
      this.cues[this.cues.length - 1]?.endSec || Number(story.duration_sec) || FALLBACK_STORY_SEC;
    this.cameraKnots = buildCameraKnots(this.cues, this.storyDurationSec);
    this.storyElapsedSec = 0;
    this.storyFinishedAt = null;
    this.cueIndex = -1;
    this.linksOffsetPx = 0;

    this.globe.highlightStoryPin(this.storyIndex);

    if (this.voiceEnabled) {
      this._ensureStoryAudioLoaded(story, this.storyIndex);
      try {
        this.audio.currentTime = 0;
      } catch (_) {}
    } else {
      this.audioAvailable = false;
    }

    this._renderChyron(story);
    this._activateCue(0);

    const startKnot = this.cameraKnots[0];
    if (this.reducedMotion) {
      this.currentFocusVec.copy(startKnot.vec);
      this.currentDist = startKnot.distance;
      this.currentBearing = 0;
      this.currentTilt = TOUR_TILT_DEG;
      const { lat, lon } = vec3ToLatLon(startKnot.vec);
      this.globe.setTourCameraPose(lat, lon, startKnot.distance, 0, TOUR_TILT_DEG);
      this.flightState = null;
      this._beginCuesPhase();
      return;
    }

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
      duration: Math.max(1300, Math.min(STORY_FLIGHT_MS, 1000 + angle * 550)),
    };

    this.phase = 'flight';
    this.phaseElapsedMs = 0;
    this._syncSoundButtonUi();
  }

  _beginCuesPhase() {
    this.phase = 'cues';
    this.storyElapsedSec = 0;
    this._syncSoundButtonUi();
    this.audio.muted = !this.voiceEnabled;
    if (this.voiceEnabled) {
      try {
        this.audio.currentTime = 0;
      } catch (_) {}
      this.audio.play().catch(() => {
        this.audioAvailable = false;
      });
    }
  }

  _activateCue(cueIdx) {
    if (cueIdx < 0 || cueIdx >= this.cues.length) return;
    this.cueIndex = cueIdx;
    const cue = this.cues[cueIdx];

    this.globe.setLayer(cue.layer);
    this.onLayerSync?.(cue.layer);

    this.globe.setTourFocusCallout({
      lat: cue.lat,
      lon: cue.lon,
      label: cue.label,
      metric: `${cue.metric ? cue.metric + ' · ' : ''}${LAYER_LABELS[cue.layer] || cue.layer.toUpperCase()}`,
      layer: cue.layer,
    });

    if (this.dom.tickerTrack) {
      this.dom.tickerTrack.querySelectorAll('.ticker-cue').forEach(el => {
        el.classList.toggle('active', Number(el.dataset.cue) === cueIdx);
      });
    }

    if (this.reducedMotion) {
      this.currentFocusVec.copy(latLonToVec3(cue.lat, cue.lon, 1).normalize());
      this.currentDist = cue.distance;
      this.globe.setTourCameraPose(cue.lat, cue.lon, cue.distance, 0, TOUR_TILT_DEG);
    }

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

    const flowWindowSec = Math.min(0.45, words.length * 0.02);
    this.fastFlowSec = words.map((_, idx) =>
      words.length <= 1 ? 0 : (idx / (words.length - 1)) * flowWindowSec
    );

    const weights = words.map(w => Math.max(2, w.length) + (/[.,;:]/.test(w) ? 3.5 : 0.8));
    const totalW = weights.reduce((a, b) => a + b, 0);
    let acc = 0;
    this.spokenCueFracs = weights.map(w => {
      const frac = (acc / totalW) * 0.86;
      acc += w;
      return frac;
    });

    const initialClass = this.reducedMotion ? 'script-word spoken' : 'script-word';
    this.dom.scriptText.innerHTML = words
      .map(w => `<span class="${initialClass}">${escapeHtml(w)}</span>`)
      .join(' ');
    this.scriptWordEls = Array.from(this.dom.scriptText.querySelectorAll('.script-word'));
  }

  _updateScriptWordsProgress(cueElapsedSec, cueU) {
    if (!this.scriptWordEls.length || this.reducedMotion) return;
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
    if (this.dom.kicker) {
      this.dom.kicker.textContent = story.kicker || story.country || story.region || 'WORLD WEATHER';
    }
    this.dom.headline.textContent = story.headline || '';
    this.dom.progressBar.style.width = '0%';
    if (this.dom.prevBtn) {
      this.dom.prevBtn.disabled = this.storyIndex === 0;
    }

    const copyInner = this.cues
      .map(
        (c, idx) =>
          `<span class="ticker-cue${idx === 0 ? ' active' : ''}" data-cue="${idx}">` +
          formatTickerClauseHtml(c.ticker) +
          `</span>`
      )
      .join('');

    if (this.dom.tickerTrack) {
      this.dom.tickerTrack.innerHTML =
        `<div class="ticker-copy">${copyInner}</div>` +
        `<div class="ticker-copy" aria-hidden="true" inert>${copyInner}</div>` +
        `<div class="ticker-copy" aria-hidden="true" inert>${copyInner}</div>` +
        `<div class="ticker-copy" aria-hidden="true" inert>${copyInner}</div>`;
    }

    const linksHtml = formatLinksCopyHtml(story.news_links || []);
    if (this.dom.linksBar && this.dom.newsLinks) {
      if (linksHtml) {
        this.dom.linksBar.hidden = false;
        this.dom.newsLinks.innerHTML =
          `<div class="links-copy">${linksHtml}</div>` +
          `<div class="links-copy" aria-hidden="true" inert>${linksHtml}</div>` +
          `<div class="links-copy" aria-hidden="true" inert>${linksHtml}</div>` +
          `<div class="links-copy" aria-hidden="true" inert>${linksHtml}</div>` +
          `<div class="links-copy" aria-hidden="true" inert>${linksHtml}</div>`;
        this.dom.newsLinks.style.transform = 'translate3d(0px, 0, 0)';
      } else {
        this.dom.linksBar.hidden = true;
        this.dom.newsLinks.innerHTML = '';
      }
    }

    this.cueSpans = [];
    this.displayStoryPx = -65;

    requestAnimationFrame(() => {
      const firstStoryCopy = this.dom.tickerTrack?.querySelector('.ticker-copy');
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
      const firstLinksCopy = this.dom.newsLinks?.querySelector('.links-copy');
      if (firstLinksCopy && firstLinksCopy.offsetWidth > 80) {
        this.linksSingleWidth = firstLinksCopy.offsetWidth;
      }
    });
  }

  _bindControls() {
    this.dom.playTourBtn?.addEventListener('click', () => this.start(0));
    this.dom.prevBtn?.addEventListener('click', () => {
      if (this.storyIndex > 0) {
        this.goToStory(this.storyIndex - 1, { playSting: true });
      }
    });
    this.dom.nextBtn?.addEventListener('click', () =>
      this.goToStory(this.storyIndex + 1, { playSting: true })
    );
    this.dom.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.dom.exitBtn?.addEventListener('click', () => this.stop({ completed: false }));

    this.dom.linksViewport?.addEventListener('mouseenter', () => {
      this.linksHovered = true;
    });
    this.dom.linksViewport?.addEventListener('mouseleave', () => {
      this.linksHovered = false;
    });

    document.addEventListener('keydown', e => {
      if (!this.active || document.activeElement?.tagName === 'INPUT') return;
      if (e.key === 'Escape') {
        this.stop({ completed: false });
      } else if (e.key === 'ArrowRight') {
        this.goToStory(this.storyIndex + 1, { playSting: true });
      } else if (e.key === 'ArrowLeft' && this.storyIndex > 0) {
        this.goToStory(this.storyIndex - 1, { playSting: true });
      } else if (e.key.toLowerCase() === 'm') {
        this.toggleMute();
      }
    });
  }

  _updateLinksTicker(dtSec) {
    if (this.reducedMotion || !this.dom.newsLinks || this.dom.linksBar?.hidden) return;
    const linkSpeed = this.linksHovered ? LINKS_TICKER_SPEED * 0.22 : LINKS_TICKER_SPEED;
    this.linksOffsetPx = (this.linksOffsetPx + linkSpeed * dtSec) % Math.max(1, this.linksSingleWidth);
    this.dom.newsLinks.style.transform = `translate3d(${-this.linksOffsetPx.toFixed(1)}px, 0, 0)`;
  }

  _applyStoryTickerOffset(rawPx) {
    if (this.reducedMotion || !this.dom.tickerTrack) return;
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

    if (this.phase === 'flight' && this.flightState) {
      this.phaseElapsedMs += dtMs;
      const t = Math.min(1, this.phaseElapsedMs / this.flightState.duration);
      const ease = quinticEase(t);

      this._flightQ.slerpQuaternions(this.flightState.fromQuat, this.flightState.toQuat, ease);
      this._flightDir.set(0, 0, 1).applyQuaternion(this._flightQ).normalize();
      const dist =
        THREE.MathUtils.lerp(this.flightState.fromDist, this.flightState.toDist, ease) +
        Math.sin(t * Math.PI) * this.flightState.arcBoost;
      const bearing = THREE.MathUtils.lerp(this.flightState.fromBearing, this.flightState.toBearing, ease);
      const tilt = THREE.MathUtils.lerp(this.flightState.fromTilt, this.flightState.toTilt, ease);

      this.currentFocusVec.copy(this._flightDir);
      this.currentDist = dist;
      this.currentBearing = bearing;
      this.currentTilt = tilt;

      const { lat, lon } = vec3ToLatLon(this._flightDir);
      this.globe.setTourCameraPose(lat, lon, dist, bearing, tilt);

      this.displayStoryPx = THREE.MathUtils.lerp(-65, 0, ease);
      this._applyStoryTickerOffset(this.displayStoryPx);

      if (t >= 1) {
        this._beginCuesPhase();
      }
      return;
    }

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

    if (!this.reducedMotion) {
      if (this.cueSpans.length === this.cues.length) {
        const span = this.cueSpans[this.cueIndex];
        const targetPx = span.leftPx + cueU * span.widthPx;
        if (Math.abs(targetPx - this.displayStoryPx) > this.storySingleWidth * 0.5) {
          this.displayStoryPx = targetPx;
        } else {
          this.displayStoryPx += (targetPx - this.displayStoryPx) * Math.min(1, dtSec * 6.5);
        }
      } else {
        this.displayStoryPx =
          (this.displayStoryPx + STORY_TICKER_SPEED * dtSec) % Math.max(1, this.storySingleWidth);
      }
      this._applyStoryTickerOffset(this.displayStoryPx);
    }

    if (this.storyFinishedAt !== null && now - this.storyFinishedAt >= 800) {
      this.goToStory(this.storyIndex + 1, { playSting: true });
      return;
    }

    if (!this.reducedMotion) {
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
}
