import { WeatherGlobe, sampleLayerColourAtValue } from './globe.js?v=abec8c13';
import { LocationSearch } from './search.js?v=abec8c13';
import { BroadcastTour } from './tour.js?v=abec8c13';

const $ = id => document.getElementById(id);

let manifestData = null;
let reportCreatedIso = null;

function formatUtcStamp(dt) {
  return (
    dt.toLocaleString('en-GB', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'UTC',
    }) + ' UTC'
  );
}

function nextScheduledReportUtc(createdDt) {
  const afterMs = createdDt.getTime() + 20 * 60_000;
  const base = new Date(afterMs);
  const y = base.getUTCFullYear();
  const m = base.getUTCMonth();
  const d = base.getUTCDate();

  for (const dayOffset of [0, 1, 2]) {
    for (const hr of [1, 7, 13, 19]) {
      const slotMs = Date.UTC(y, m, d + dayOffset, hr, 30, 0);
      if (slotMs > afterMs) {
        return new Date(slotMs);
      }
    }
  }
  return new Date(afterMs + 6 * 3600_000);
}

function isNearScheduledReportWindow(nowMs = Date.now()) {
  const dt = new Date(nowMs);
  const minsOfDay = dt.getUTCHours() * 60 + dt.getUTCMinutes();
  // Scheduled cron runs at 01:25, 07:25, 13:25, 19:25 UTC -> live around :30 UTC
  for (const hr of [1, 7, 13, 19]) {
    const targetMins = hr * 60 + 30;
    if (Math.abs(minsOfDay - targetMins) <= 22) {
      return true;
    }
  }
  return false;
}

function updateReportScheduleUi(isoString) {
  if (isoString) {
    reportCreatedIso = isoString;
  }
  if (!reportCreatedIso) return;
  const createdDt = new Date(reportCreatedIso);
  if (Number.isNaN(createdDt.getTime())) return;

  $('liveTimestamp').textContent = `Report created: ${formatUtcStamp(createdDt)}`;

  const nextDt = nextScheduledReportUtc(createdDt);
  const diffMs = nextDt.getTime() - Date.now();
  if (diffMs <= 0) {
    $('nextReportTimestamp').textContent = 'Next report: Updating shortly...';
    return;
  }

  const minsLeft = Math.max(1, Math.round(diffMs / 60_000));
  const hrs = Math.floor(minsLeft / 60);
  const mins = minsLeft % 60;
  const rel = hrs > 0 ? `in ${hrs}h ${String(mins).padStart(2, '0')}m` : `in ${mins}m`;
  $('nextReportTimestamp').textContent = `Next report: ${formatUtcStamp(nextDt)} (${rel})`;
}

function formatTemp(c, isDiff = false) {
  if (c == null || Number.isNaN(c)) return '--';
  const sign = isDiff && c > 0 ? '+' : '';
  const rounded = Number.isInteger(c) ? String(c) : c.toFixed(1);
  return `${sign}${rounded} °C`;
}

function formatWind(ms) {
  if (ms == null || Number.isNaN(ms)) return '--';
  return `${Math.round(ms * 3.6)} km/h`;
}

export function buildLegendGradientCss(spec) {
  const sampleCount = 16;
  const parts = [];
  for (let i = 0; i <= sampleCount; i++) {
    const p = i / sampleCount;
    const norm = spec.scale === 'sqrt' ? p * p : p;
    const val = spec.lo + norm * (spec.hi - spec.lo);
    const [r, g, b] = sampleLayerColourAtValue(spec, val);
    parts.push(`rgb(${r}, ${g}, ${b}) ${(p * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

function updateLegend(layerId) {
  if (!manifestData) return;
  const spec = manifestData.layers.find(l => l.id === layerId);
  if (!spec) return;

  $('legendGradient').style.background = buildLegendGradientCss(spec);

  const midNorm = spec.scale === 'sqrt' ? 0.25 : 0.5;
  const midVal = spec.lo + midNorm * (spec.hi - spec.lo);

  if (layerId === 'temperature') {
    $('legendTitle').textContent = '2 m Temperature';
    $('legendMin').textContent = formatTemp(spec.lo);
    $('legendMid').textContent = formatTemp(midVal, true);
    $('legendMax').textContent = formatTemp(spec.hi);
  } else if (layerId === 'anomaly') {
    $('legendTitle').textContent = 'Temp Anomaly';
    $('legendMin').textContent = formatTemp(spec.lo, true);
    $('legendMid').textContent = '0 °C';
    $('legendMax').textContent = formatTemp(spec.hi, true);
  } else if (layerId === 'wind') {
    $('legendTitle').textContent = '10 m Wind Speed';
    $('legendMin').textContent = '0 km/h';
    $('legendMid').textContent = formatWind(midVal);
    $('legendMax').textContent = formatWind(spec.hi);
  } else if (layerId === 'rain') {
    $('legendTitle').textContent = '24 h Rainfall';
    $('legendMin').textContent = '0 mm';
    $('legendMid').textContent = `${Math.round(midVal)} mm`;
    $('legendMax').textContent = `${Math.round(spec.hi)} mm`;
  } else {
    $('legendTitle').textContent = 'Pressure (hPa)';
    $('legendMin').textContent = `${Math.round(spec.lo)}`;
    $('legendMid').textContent = `${Math.round(midVal)}`;
    $('legendMax').textContent = `${Math.round(spec.hi)} hPa`;
  }
}

function setActiveLayerButtons(layerId) {
  document.querySelectorAll('.layer-btn').forEach(btn => {
    const active = btn.dataset.layer === layerId;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-checked', String(active));
  });
  updateLegend(layerId);
}

function showSpotReadout(globe, { name, sub, lat, lon }) {
  const vals = globe.sampleAllAt(lat, lon);
  if (!vals) return;

  $('spotSub').textContent =
    sub || `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? 'E' : 'W'}`;
  $('spotTitle').textContent = name;
  $('spotTemp').textContent = formatTemp(vals.temperature);
  $('spotAnom').textContent = formatTemp(vals.anomaly, true);
  $('spotWind').textContent = formatWind(vals.windSpeed);
  $('spotRain').textContent = vals.rain != null ? `${vals.rain.toFixed(1)} mm` : '--';
  $('spotPressure').textContent = vals.pressure != null ? `${Math.round(vals.pressure)} hPa` : '--';
  $('spotCard').hidden = false;
}

async function init() {
  let tour = null;
  const layerRail = $('layerRail');
  const mobileLayersToggle = $('mobileLayersToggle');

  const closeMobileLayersMenu = () => {
    layerRail.classList.remove('layers-open');
    mobileLayersToggle.setAttribute('aria-expanded', 'false');
  };

  mobileLayersToggle.addEventListener('click', e => {
    e.stopPropagation();
    const isOpen = layerRail.classList.toggle('layers-open');
    mobileLayersToggle.setAttribute('aria-expanded', String(isOpen));
  });

  document.addEventListener('click', e => {
    if (!layerRail.contains(e.target)) {
      closeMobileLayersMenu();
    }
  });

  const globe = new WeatherGlobe($('globeStage'), $('globeOverlay'), {
    onSpotClick: (lat, lon) => {
      const ns = lat >= 0 ? 'N' : 'S';
      const ew = lon >= 0 ? 'E' : 'W';
      globe.clearSearchMarker();
      showSpotReadout(globe, {
        name: `${Math.abs(lat).toFixed(1)}°${ns}, ${Math.abs(lon).toFixed(1)}°${ew}`,
        sub: 'Forecast Point Readout',
        lat,
        lon,
      });
    },
    onPinClick: storyIdx => {
      $('spotCard').hidden = true;
      closeMobileLayersMenu();
      tour?.start(storyIdx);
    },
  });

  tour = new BroadcastTour(
    globe,
    {
      playLauncher: $('playLauncher'),
      playTourBtn: $('playTourBtn'),
      playCaption: $('playTourCaption'),
      statusNotice: $('tourStatusNotice'),
      tourHud: $('tourHud'),
      prevBtn: $('tourPrev'),
      nextBtn: $('tourNext'),
      muteBtn: $('tourMuteBtn'),
      muteLabel: $('tourMuteLabel'),
      iconSoundOff: $('iconSoundOff'),
      iconSoundOn: $('iconSoundOn'),
      exitBtn: $('tourExit'),
      counter: $('tourCounter'),
      kicker: $('tourKicker'),
      progressBar: $('tourProgressBar'),
      headline: $('tourHeadline'),
      tickerViewport: $('tickerViewport'),
      tickerTrack: $('tickerTrack'),
      linksBar: $('linksBar'),
      linksViewport: $('linksViewport'),
      newsLinks: $('tourNewsLinks'),
      scriptOverlay: $('voiceScriptOverlay'),
      scriptText: $('voiceScriptText'),
    },
    {
      onLayerSync: layerId => setActiveLayerButtons(layerId),
      onBulletinApplied: bulletin => {
        globe.setStoryPins(bulletin.stories, tour.active ? tour.storyIndex : -1);
      },
    }
  );

  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const layerId = btn.dataset.layer;
      globe.setLayer(layerId);
      setActiveLayerButtons(layerId);
      closeMobileLayersMenu();
    });
  });

  $('spotClose').addEventListener('click', () => {
    $('spotCard').hidden = true;
    globe.clearSearchMarker();
  });

  const search = new LocationSearch($('searchInput'), $('searchResults'), place => {
    if (tour.active) tour.stop({ completed: false });
    globe.setSearchMarker(place.lat, place.lon, place.name);
    globe.flyTo(place.lat, place.lon, place.distance || 1.6, 1750);
    showSpotReadout(globe, {
      name: place.name,
      sub: place.kind === 'Country' ? `Country · ${place.sub}` : place.sub,
      lat: place.lat,
      lon: place.lon,
    });
  });
  void search.load();
  void globe.loadBasemap();

  const fetchLatestBulletin = async (isPoll = false) => {
    if (isPoll && document.hidden) return;
    try {
      const r = await fetch('data/headlines.json', { cache: 'no-cache' });
      if (!r.ok) {
        if (!isPoll) tour.setBulletinError();
        return;
      }
      const bulletin = await r.json();
      if (!bulletin || !Array.isArray(bulletin.stories)) {
        if (!isPoll) tour.setBulletinError();
        return;
      }
      const changed = Boolean(bulletin.generated_utc && bulletin.generated_utc !== reportCreatedIso);
      if (!isPoll || changed) {
        tour.setBulletin(bulletin);
        if (!tour.active) {
          globe.setStoryPins(bulletin.stories, -1);
        }
        if (bulletin.generated_utc) {
          updateReportScheduleUi(bulletin.generated_utc);
        }
        if (isPoll && changed) {
          const manifest = await globe.loadWeatherManifest('data/manifest.json');
          manifestData = manifest;
          updateLegend(globe.currentLayer);
        }
      }
    } catch (err) {
      console.error('Failed loading headlines:', err);
      if (!isPoll) tour.setBulletinError();
    }
  };

  void fetchLatestBulletin(false);
  setInterval(() => {
    if (!document.hidden) updateReportScheduleUi();
  }, 30_000);

  // Poll every 60s around scheduled [1, 7, 13, 19]:30 UTC update windows, and every 5 min otherwise
  let pollTimerId = null;
  const scheduleNextPoll = () => {
    const delayMs = isNearScheduledReportWindow() ? 60_000 : 300_000;
    pollTimerId = setTimeout(async () => {
      await fetchLatestBulletin(true);
      scheduleNextPoll();
    }, delayMs);
  };
  scheduleNextPoll();

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      updateReportScheduleUi();
      if (isNearScheduledReportWindow()) {
        void fetchLatestBulletin(true);
      }
    }
  });

  try {
    const manifest = await globe.loadWeatherManifest('data/manifest.json');
    manifestData = manifest;
    updateLegend(globe.currentLayer);
    if (!reportCreatedIso && manifest.generated_utc) {
      updateReportScheduleUi(manifest.generated_utc);
    }
  } catch (err) {
    console.error(err);
    $('liveTimestamp').textContent = 'Weather data unavailable';
  }
}

init();
