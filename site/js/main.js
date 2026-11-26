import { WeatherGlobe } from './globe.js?v=20261126w';
import { LocationSearch } from './search.js?v=20261126w';
import { BroadcastTour } from './tour.js?v=20261126w';

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
  // Next 6-hourly cron slot (`25 1,7,13,19 * * *` -> live ~:30 UTC) strictly after `createdDt`
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
  return `${sign}${c.toFixed(1)} °C`;
}

function formatWind(ms) {
  if (ms == null || Number.isNaN(ms)) return '--';
  return `${Math.round(ms * 3.6)} km/h`;
}

function updateLegend(layerId) {
  if (!manifestData) return;
  const spec = manifestData.layers.find(l => l.id === layerId);
  if (!spec) return;

  $('legendTitle').textContent = spec.label;

  const cssStops = spec.stops
    .map(([val, hex]) => {
      const pct = Math.max(0, Math.min(100, ((val - spec.lo) / (spec.hi - spec.lo)) * 100));
      const rgbHex = hex.slice(0, 7);
      return `${rgbHex} ${pct.toFixed(1)}%`;
    })
    .join(', ');
  $('legendGradient').style.background = `linear-gradient(90deg, ${cssStops})`;

  const mid = 0.5 * (spec.lo + spec.hi);
  if (layerId === 'temperature') {
    $('legendMin').textContent = formatTemp(spec.lo);
    $('legendMid').textContent = '0 °C';
    $('legendMax').textContent = formatTemp(spec.hi);
  } else if (layerId === 'anomaly') {
    $('legendMin').textContent = formatTemp(spec.lo, true);
    $('legendMid').textContent = '0 °C';
    $('legendMax').textContent = formatTemp(spec.hi, true);
  } else if (layerId === 'wind') {
    $('legendMin').textContent = '0';
    $('legendMid').textContent = formatWind(20);
    $('legendMax').textContent = formatWind(spec.hi);
  } else if (layerId === 'rain') {
    $('legendMin').textContent = '0 mm';
    $('legendMid').textContent = '40 mm';
    $('legendMax').textContent = `${spec.hi} mm`;
  } else {
    $('legendMin').textContent = `${spec.lo} hPa`;
    $('legendMid').textContent = `${Math.round(mid)} hPa`;
    $('legendMax').textContent = `${spec.hi} hPa`;
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

  const globe = new WeatherGlobe($('globeStage'), $('globeOverlay'), {
    onSpotClick: (lat, lon) => {
      const ns = lat >= 0 ? 'N' : 'S';
      const ew = lon >= 0 ? 'E' : 'W';
      showSpotReadout(globe, {
        name: `${Math.abs(lat).toFixed(1)}°${ns}, ${Math.abs(lon).toFixed(1)}°${ew}`,
        sub: 'Surface Point Readout',
        lat,
        lon,
      });
    },
    onPinClick: storyIdx => {
      $('spotCard').hidden = true;
      tour?.start(storyIdx);
    },
  });

  tour = new BroadcastTour(
    globe,
    {
      playLauncher: $('playLauncher'),
      playTourBtn: $('playTourBtn'),
      tourHud: $('tourHud'),
      prevBtn: $('tourPrev'),
      nextBtn: $('tourNext'),
      textBtn: $('tourTextBtn'),
      textLabel: $('tourTextLabel'),
      muteBtn: $('tourMuteBtn'),
      muteLabel: $('tourMuteLabel'),
      exitBtn: $('tourExit'),
      counter: $('tourCounter'),
      kicker: $('tourKicker'),
      layerTag: $('tourLayerTag'),
      progressBar: $('tourProgressBar'),
      headline: $('tourHeadline'),
      tickerViewport: $('tickerViewport'),
      tickerTrack: $('tickerTrack'),
      linksViewport: $('linksViewport'),
      newsLinks: $('tourNewsLinks'),
      scriptOverlay: $('voiceScriptOverlay'),
      scriptText: $('voiceScriptText'),
    },
    {
      onLayerSync: layerId => setActiveLayerButtons(layerId),
    }
  );

  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const layerId = btn.dataset.layer;
      globe.setLayer(layerId);
      setActiveLayerButtons(layerId);
    });
  });

  $('spotClose').addEventListener('click', () => {
    $('spotCard').hidden = true;
  });

  const search = new LocationSearch($('searchInput'), $('searchResults'), place => {
    if (tour.active) tour.stop();
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
    try {
      const r = await fetch(`data/headlines.json?t=${Date.now()}`, { cache: 'no-store' });
      if (!r.ok) return;
      const bulletin = await r.json();
      if (!bulletin) return;
      const changed = Boolean(bulletin.generated_utc && bulletin.generated_utc !== reportCreatedIso);
      if (!isPoll || changed) {
        tour.setBulletin(bulletin);
        globe.setStoryPins(bulletin.stories, tour.active ? tour.storyIndex : -1);
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
    }
  };

  void fetchLatestBulletin(false);
  setInterval(() => updateReportScheduleUi(), 30_000);
  setInterval(() => void fetchLatestBulletin(true), 90_000);

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
