import * as THREE from 'three';
import { latLonToVec3, vec3ToLatLon } from './globe.js?v=20261126g';

const OPENING_FLIGHT_MS = 1520;
const STORY_SWITCH_FLIGHT_MS = 1750;
const FALLBACK_STORY_SEC = 45;
const STORY_TICKER_SPEED = 96; // px/sec nominal story ticker speed
const LINKS_TICKER_SPEED = 46; // px/sec slower independent CNN-style web links ticker

const LAYER_LABELS = {
  temperature: '2 M TEMP',
  anomaly: 'TEMP ANOMALY',
  wind: '10 M WINDS',
  rain: '24 H RAINFALL',
  pressure: 'SEA-LEVEL PRESSURE',
};

function smoothstep01(x) {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
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
      return {
        layer: c.layer || story.primary_layer || 'temperature',
        lat: Number(c.lat ?? story.centre.lat),
        lon: Number(c.lon ?? story.centre.lon),
        distance: Number(c.distance || 1.58),
        bearing: Number(c.bearing ?? (idx % 2 === 0 ? -22 : 22)),
        label: c.label || story.kicker || story.region || 'WORLD WEATHER',
        metric: c.metric || story.stats?.headline_metric || '',
        startSec: hasExplicitTimes ? Number(c.start_sec) : propStart,
        endSec: hasExplicitTimes ? Number(c.end_sec) : propEnd,
        ticker: c.ticker || c.spoken || '',
      };
    });
  }

  // Fallback for legacy stories with only `beats` + `camera_waypoints`
  const beats = story.beats || [];
  const wps = story.camera_waypoints || [];
  const count = Math.max(1, beats.length);
  return beats.map((b, idx) => {
    const wp = wps[Math.min(idx, Math.max(0, wps.length - 1))] || {
      lat: story.centre.lat,
      lon: story.centre.lon,
      distance: 1.62,
    };
    return {
      layer: b.layer || story.primary_layer || 'temperature',
      lat: wp.lat,
      lon: wp.lon,
      distance: wp.distance || 1.62,
      bearing: idx % 2 === 0 ? -22 : 22,
      label: b.title || story.kicker || 'WORLD WEATHER',
      metric: story.stats?.headline_metric || b.tag || '',
      startSec: (idx / count) * totalDur,
      endSec: ((idx + 1) / count) * totalDur,
      ticker: b.text || '',
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

    // Presenter voice is switched OFF by default per user preference
    this.voiceEnabled = false;
    this.linksHovered = false;
    this.storyIndex = 0;
    this.cueIndex = -1;
    this.cues = [];
    this.storyDurationSec = FALLBACK_STORY_SEC;
    this.storyElapsedSec = 0;

    this.phase = 'idle'; // 'flight' | 'cues'
    this.phaseElapsedMs = 0;
    this.storyFinishedAt = null;

    // Ticker pixel layout & smooth scroll state
    this.cueSpans = [];
    this.storySingleWidth = 2200;
    this.displayStoryPx = 0;
    this.linksOffsetPx = 0;
    this.linksSingleWidth = 1100;

    // Smooth camera state across & within cues
    this.flightState = null;
    this.currentFocusVec = new THREE.Vector3(0, 0, 1);
    this.currentDist = 1.85;
    this.currentBearing = -20;
    this.cueCamTransition = null;

    // Opening urgent broadcast drumbeat sting
    this.stingAudio = new Audio('assets/bulletin-sting.wav');
    this.stingAudio.preload = 'auto';
    this.stingAudio.volume = 0.82;

    // Spoken British female presenter audio (muted while voiceEnabled === false)
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
    if (!this.dom.muteBtn || !this.dom.muteLabel) return;
    this.dom.muteBtn.setAttribute('aria-pressed', String(this.voiceEnabled));
    this.dom.muteBtn.classList.toggle('voice-on', this.voiceEnabled);
    this.dom.muteLabel.textContent = this.voiceEnabled ? '🔊 Voice On' : '🔇 Voice Off';
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
      this.goToStory(this.storyIndex, { withOpeningSting: false });
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

    // Play the short breaking-news drumbeat sting on launch!
    this.stingAudio.currentTime = 0;
    this.stingAudio.play().catch(() => {});

    this.active = true;
    document.body.classList.add('tour-active');
    this.dom.playLauncher.hidden = true;
    this.dom.tourHud.hidden = false;
    this.globe.setTourMode(true);
    this.goToStory(startIndex, { withOpeningSting: true });
  }

  stop() {
    this.active = false;
    this._pendingStartIndex = null;
    this.stingAudio.pause();
    this.audio.pause();
    this.audio.removeAttribute('src');
    document.body.classList.remove('tour-active');
    this.dom.tourHud.hidden = true;
    this.dom.playLauncher.hidden = false;
    this.globe.setTourMode(false);
    this.globe.highlightStoryPin(-1);
  }

  toggleMute() {
    this.voiceEnabled = !this.voiceEnabled;
    this.audio.muted = !this.voiceEnabled;
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

  goToStory(index, { withOpeningSting = false } = {}) {
    if (!this.stories.length) return;
    if (!this.active) {
      this.start(index);
      return;
    }

    this.storyIndex = ((index % this.stories.length) + this.stories.length) % this.stories.length;
    const story = this.stories[this.storyIndex];
    this.cues = normaliseStoryCues(story);
    this.storyDurationSec = this.cues[this.cues.length - 1]?.endSec || Number(story.duration_sec) || FALLBACK_STORY_SEC;
    this.storyElapsedSec = 0;
    this.storyFinishedAt = null;
    this.cueIndex = -1;
    this.linksOffsetPx = 0;

    this.globe.highlightStoryPin(this.storyIndex);

    // Prepare the story MP3 (starts speaking cleanly right as the camera flight arrives)
    const audioUrl = story.audio_url || `data/audio/story-${this.storyIndex + 1}.mp3`;
    this.audio.pause();
    this.audio.src = `${audioUrl}?v=${this.bulletinStamp}`;
    this.audio.muted = !this.voiceEnabled;
    this.audio.currentTime = 0;
    this.audio.load();
    this.audioAvailable = true;

    const firstCue = this.cues[0];
    const camNow = this.globe.getCurrentCameraGeo();
    const fromDir = this.phase === 'idle'
      ? latLonToVec3(camNow.lat, camNow.lon, 1).normalize()
      : this.currentFocusVec.clone().normalize();
    const toDir = latLonToVec3(firstCue.lat, firstCue.lon, 1).normalize();
    const angle = fromDir.angleTo(toDir);

    const flightDur = withOpeningSting
      ? OPENING_FLIGHT_MS
      : Math.max(1150, Math.min(STORY_SWITCH_FLIGHT_MS, 950 + angle * 620));

    this.flightState = {
      fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromDir),
      toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), toDir),
      fromDist: this.phase === 'idle' ? camNow.distance : this.currentDist,
      toDist: firstCue.distance,
      fromBearing: this.currentBearing,
      toBearing: firstCue.bearing - 10,
      arcBoost: Math.min(0.75, 0.16 + angle * 0.34),
      duration: flightDur,
    };

    this.phase = 'flight';
    this.phaseElapsedMs = 0;

    this._renderChyron(story);
    this._activateCue(0, story, { immediateCam: false });
  }

  _beginCuesPhase() {
    this.phase = 'cues';
    this.storyElapsedSec = 0;
    this.audio.muted = !this.voiceEnabled;
    // Play audio (muted when Voice Off, audible when Voice On) so hardware timing is always ready
    this.audio.play().catch(() => {
      this.audioAvailable = false;
    });
  }

  _activateCue(cueIdx, story, { immediateCam = false } = {}) {
    if (cueIdx < 0 || cueIdx >= this.cues.length) return;
    this.cueIndex = cueIdx;
    const cue = this.cues[cueIdx];

    // 1. Switch the live weather variable on the globe & update legend + strap metric
    this.globe.setLayer(cue.layer);
    this.onLayerSync?.(cue.layer);
    const metricPrefix = cue.metric ? `${cue.metric} · ` : '';
    this.dom.layerTag.textContent = `${metricPrefix}${LAYER_LABELS[cue.layer] || cue.layer.toUpperCase()}`;

    // 2. Move the animated surface spotlight rings & location callout onto this sentence's coordinates
    this.globe.setTourFocusCallout({
      lat: cue.lat,
      lon: cue.lon,
      label: cue.label,
      metric: `${cue.metric ? cue.metric + ' · ' : ''}${LAYER_LABELS[cue.layer] || cue.layer.toUpperCase()}`,
      layer: cue.layer,
    });

    // 3. Highlight the active sentence clause in the scrolling STORY ticker
    this.dom.tickerTrack.querySelectorAll('.ticker-cue').forEach(el => {
      el.classList.toggle('active', Number(el.dataset.cue) === cueIdx);
    });

    // 4. Set up smooth 3D camera sweep from previous focus onto this cue's location
    const targetVec = latLonToVec3(cue.lat, cue.lon, 1).normalize();
    if (immediateCam) {
      this.currentFocusVec.copy(targetVec);
      this.currentDist = cue.distance;
      this.currentBearing = cue.bearing;
      this.cueCamTransition = null;
    } else {
      const fromVec = this.currentFocusVec.clone().normalize();
      const hopAngle = fromVec.angleTo(targetVec);
      this.cueCamTransition = {
        fromQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), fromVec),
        toQuat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), targetVec),
        fromDist: this.currentDist,
        toDist: cue.distance,
        fromBearing: this.currentBearing,
        toBearing: cue.bearing - 11,
        hopArc: Math.min(0.28, hopAngle * 0.25),
        elapsedSec: 0,
        durationSec: Math.min(2.6, Math.max(1.5, 1.2 + hopAngle * 2.2)),
      };
    }
  }

  _renderChyron(story) {
    this.dom.counter.textContent = `${this.storyIndex + 1} / ${this.stories.length}`;
    this.dom.kicker.textContent = story.kicker || story.region || 'WORLD WEATHER';
    this.dom.headline.textContent = story.headline;
    this.dom.progressBar.style.width = '0%';

    // Render sentence cues as distinct inline spans so we can measure exact pixel boundaries
    const buildCopyInner = () =>
      this.cues
        .map(
          (c, idx) =>
            `<span class="ticker-cue${idx === 0 ? ' active' : ''}" data-cue="${idx}">` +
            formatTickerClauseHtml(c.ticker) +
            `</span>`
        )
        .join('');

    const copyInner = buildCopyInner();
    this.dom.tickerTrack.innerHTML =
      `<div class="ticker-copy">${copyInner}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${copyInner}</div>` +
      `<div class="ticker-copy" aria-hidden="true">${copyInner}</div>`;

    const linksHtml = formatLinksCopyHtml(story.news_links || []);
    this.dom.newsLinks.innerHTML =
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>` +
      `<div class="links-copy">${linksHtml}</div>`;
    this.dom.newsLinks.style.transform = 'translate3d(0px, 0, 0)';

    this.cueSpans = [];
    this.displayStoryPx = -120;

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
      this.goToStory(this.storyIndex - 1, { withOpeningSting: false })
    );
    this.dom.nextBtn?.addEventListener('click', () =>
      this.goToStory(this.storyIndex + 1, { withOpeningSting: false })
    );
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
        this.goToStory(this.storyIndex + 1, { withOpeningSting: false });
      } else if (e.key === 'ArrowLeft') {
        this.goToStory(this.storyIndex - 1, { withOpeningSting: false });
      } else if (e.key.toLowerCase() === 'm') {
        this.toggleMute();
      }
    });
  }

  _updateLinksTicker(dtSec) {
    const linkSpeed = this.linksHovered ? LINKS_TICKER_SPEED * 0.2 : LINKS_TICKER_SPEED;
    this.linksOffsetPx = (this.linksOffsetPx + linkSpeed * dtSec) % Math.max(1, this.linksSingleWidth);
    this.dom.newsLinks.style.transform = `translate3d(${-this.linksOffsetPx.toFixed(1)}px, 0, 0)`;
  }

  _applyStoryTickerOffset(rawPx) {
    const leadInPx = 22;
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

    const story = this.stories[this.storyIndex];

    // Always scroll the CNN-style bottom web links ticker at its independent speed
    this._updateLinksTicker(dtSec);

    // Phase 1: Dramatic swoop onto Cue 0 (accompanied by the opening drumbeat sting on launch)
    if (this.phase === 'flight' && this.flightState) {
      this.phaseElapsedMs += dtMs;
      const t = Math.min(1, this.phaseElapsedMs / this.flightState.duration);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      const q = new THREE.Quaternion().slerpQuaternions(this.flightState.fromQuat, this.flightState.toQuat, ease);
      const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
      const dist =
        THREE.MathUtils.lerp(this.flightState.fromDist, this.flightState.toDist, ease) +
        Math.sin(t * Math.PI) * this.flightState.arcBoost;
      const bearing = THREE.MathUtils.lerp(this.flightState.fromBearing, this.flightState.toBearing, ease);

      this.currentFocusVec.copy(dir);
      this.currentDist = dist;
      this.currentBearing = bearing;

      const { lat, lon } = vec3ToLatLon(dir);
      this.globe.setTourCameraPose(lat, lon, dist, bearing, 18.5);

      // Lead Cue 0 smoothly into the STORY reading window during the swoop so text never sits still
      this.displayStoryPx = THREE.MathUtils.lerp(-130, 0, ease);
      this._applyStoryTickerOffset(this.displayStoryPx);

      if (t >= 1) {
        this._beginCuesPhase();
      }
      return;
    }

    // Phase 2: Sentence-by-sentence synchronised playback (locked to voice when Voice On, or real-time clock when Voice Off)
    const useAudioClock =
      this.voiceEnabled &&
      this.audioAvailable &&
      !this.audio.paused &&
      Number.isFinite(this.audio.duration) &&
      this.audio.duration > 3;

    if (useAudioClock) {
      this.storyElapsedSec = this.audio.currentTime;
    } else if (this.storyFinishedAt === null) {
      this.storyElapsedSec += dtSec;
      if (this.storyElapsedSec >= this.storyDurationSec) {
        this.storyElapsedSec = this.storyDurationSec;
        this.storyFinishedAt = now;
      }
    }

    const overallProgress = Math.min(1, this.storyElapsedSec / Math.max(1, this.storyDurationSec));
    this.dom.progressBar.style.width = `${(overallProgress * 100).toFixed(1)}%`;

    // Find which sentence cue corresponds to `storyElapsedSec`
    let targetCueIdx = this.cues.length - 1;
    for (let i = 0; i < this.cues.length; i++) {
      if (this.storyElapsedSec < this.cues[i].endSec || i === this.cues.length - 1) {
        targetCueIdx = i;
        break;
      }
    }

    if (targetCueIdx !== this.cueIndex) {
      this._activateCue(targetCueIdx, story, { immediateCam: false });
    }

    const activeCue = this.cues[this.cueIndex];
    const cueDur = Math.max(0.4, activeCue.endSec - activeCue.startSec);
    const cueU = Math.max(0, Math.min(1, (this.storyElapsedSec - activeCue.startSec) / cueDur));

    // Synchronise the scrolling STORY ticker position directly with the active sentence cue!
    if (this.cueSpans.length === this.cues.length) {
      const span = this.cueSpans[this.cueIndex];
      const targetPx = span.leftPx + cueU * span.widthPx;
      if (Math.abs(targetPx - this.displayStoryPx) > this.storySingleWidth * 0.5) {
        this.displayStoryPx = targetPx;
      } else {
        this.displayStoryPx += (targetPx - this.displayStoryPx) * Math.min(1, dtSec * 12);
      }
    } else {
      this.displayStoryPx = (this.displayStoryPx + STORY_TICKER_SPEED * dtSec) % Math.max(1, this.storySingleWidth);
    }
    this._applyStoryTickerOffset(this.displayStoryPx);

    // Advance cleanly to the next story 0.75s after the final sentence finishes
    if (this.storyFinishedAt !== null && now - this.storyFinishedAt >= 750) {
      this.goToStory(this.storyIndex + 1, { withOpeningSting: false });
      return;
    }

    // Continuous 3D camera motion: smooth hop onto active cue + steady cinema orbit + drift toward next cue!
    const nextCue = this.cues[Math.min(this.cues.length - 1, this.cueIndex + 1)];
    const cueBaseVec = latLonToVec3(activeCue.lat, activeCue.lon, 1).normalize();
    const nextCueVec = latLonToVec3(nextCue.lat, nextCue.lon, 1).normalize();

    let focusVec;
    let dist;
    let bearing;

    if (this.cueCamTransition) {
      this.cueCamTransition.elapsedSec += dtSec;
      const trT = Math.min(1, this.cueCamTransition.elapsedSec / this.cueCamTransition.durationSec);
      const trEase = smoothstep01(trT);
      const q = new THREE.Quaternion().slerpQuaternions(
        this.cueCamTransition.fromQuat,
        this.cueCamTransition.toQuat,
        trEase
      );
      focusVec = new THREE.Vector3(0, 0, 1).applyQuaternion(q).normalize();
      dist =
        THREE.MathUtils.lerp(this.cueCamTransition.fromDist, this.cueCamTransition.toDist, trEase) +
        Math.sin(trT * Math.PI) * this.cueCamTransition.hopArc;
      bearing = THREE.MathUtils.lerp(this.cueCamTransition.fromBearing, activeCue.bearing, trEase);
      if (trT >= 1) {
        this.cueCamTransition = null;
      }
    } else {
      focusVec = cueBaseVec.clone();
      dist = activeCue.distance;
      bearing = activeCue.bearing;
    }

    // Keep the globe smoothly gliding & banking throughout every single second of the sentence
    const anticipation = smoothstep01((cueU - 0.35) / 0.65) * 0.28;
    focusVec.lerp(nextCueVec, anticipation).normalize();
    const liveBearing = bearing + (cueU - 0.3) * 22.0;
    const liveDist = dist - 0.065 * Math.sin(cueU * Math.PI);

    this.currentFocusVec.copy(focusVec);
    this.currentDist = liveDist;
    this.currentBearing = liveBearing;

    const { lat, lon } = vec3ToLatLon(focusVec);
    this.globe.setTourCameraPose(lat, lon, liveDist, liveBearing, 18.5);
  }
}
