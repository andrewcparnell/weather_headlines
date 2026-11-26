import { WeatherGlobe } from './globe.js';
import { LocationSearch } from './search.js';
import { BroadcastTour } from './tour.js';

const $ = id => document.getElementById(id);

let imperialUnits = false; // false = °C & m/s, true = °F & km/h
let lastSpot = null; // { name, sub, lat, lon }
let manifestData = null;

function formatTemp(c, isDiff = false) {
  if (c == null || Number.isNaN(c)) return '--';
  if (imperialUnits) {
    const val = isDiff ? c * 1.8 : c * 1.8 + 32;
    const sign = isDiff && val > 0 ? '+' : '';
    return `${sign}${val.toFixed(1)} °F`;
  }
  const sign = isDiff && c > 0 ? '+' : '';
  return `${sign}${c.toFixed(1)} °C`;
}

function formatWind(ms) {
  if (ms == null || Number.isNaN(ms)) return '--';
  if (imperialUnits) {
    return `${Math.round(ms * 3.6)} km/h`;
  }
  return `${ms.toFixed(1)} m/s`;
}

function updateLegend(layerId) {
  if (!manifestData) return;
  const spec = manifestData.layers.find(l => l.id === layerId);
  if (!spec) return;

  $('legendTitle').textContent = spec.label;
  $('unitToggle').textContent = imperialUnits ? '°F · km/h' : '°C · m/s';

  const cssStops = spec.stops
    .map(([val, hex]) => {
      const pct = Math.max(0, Math.min(100, ((val - spec.lo) / (spec.hi - spec.lo)) * 100));
      // Force solid alpha in the legend bar so transparent dry/calm thresholds remain legible
      const rgbHex = hex.slice(0, 7);
      return `${rgbHex} ${pct.toFixed(1)}%`;
    })
    .join(', ');
  $('legendGradient').style.background = `linear-gradient(90deg, ${cssStops})`;

  const mid = 0.5 * (spec.lo + spec.hi);
  if (layerId === 'temperature') {
    $('legendMin').textContent = formatTemp(spec.lo);
    $('legendMid').textContent = formatTemp(0);
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
  lastSpot = { name, sub, lat, lon };
  const vals = globe.sampleAllAt(lat, lon);
  if (!vals) return;

  $('spotSub').textContent = sub || `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? 'E' : 'W'}`;
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
    onUserInteract: () => {
      // Allow manual drag during tour to exit camera lock gracefully
      tour?.stop();
    },
  });

  tour = new BroadcastTour(
    globe,
    {
      playLauncher: $('playLauncher'),
      playTourBtn: $('playTourBtn'),
      tourHud: $('tourHud'),
      steps: $('tourSteps'),
      prevBtn: $('tourPrev'),
      playPauseBtn: $('tourPlayPause'),
      playIcon: $('tourPlayIcon'),
      nextBtn: $('tourNext'),
      voiceBtn: $('tourVoiceBtn'),
      voiceLabel: $('tourVoiceLabel'),
      exitBtn: $('tourExit'),
      kicker: $('tourKicker'),
      statMetric: $('statMetric'),
      statRank: $('statRank'),
      statPersistence: $('statPersistence'),
      headline: $('tourHeadline'),
      beatTabs: $('beatTabs'),
      beatText: $('tourBeatText'),
      newsLinks: $('tourNewsLinks'),
    },
    {
      onLayerSync: layerId => setActiveLayerButtons(layerId),
    }
  );

  // Layer rail buttons
  document.querySelectorAll('.layer-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const layerId = btn.dataset.layer;
      globe.setLayer(layerId);
      setActiveLayerButtons(layerId);
    });
  });

  $('unitToggle').addEventListener('click', () => {
    imperialUnits = !imperialUnits;
    updateLegend(globe.currentLayer);
    if (lastSpot && !$('spotCard').hidden) {
      showSpotReadout(globe, lastSpot);
    }
  });

  $('spotClose').addEventListener('click', () => {
    $('spotCard').hidden = true;
  });

  // Search bar
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

  // Load basemap, live weather manifest, and 6-hourly headlines in parallel
  void globe.loadBasemap();

  try {
    const [manifest, bulletinResp] = await Promise.all([
      globe.loadWeatherManifest('data/manifest.json'),
      fetch(`data/headlines.json?t=${Date.now()}`),
    ]);
    manifestData = manifest;
    updateLegend(globe.currentLayer);

    const validDt = new Date(manifest.valid_time_utc);
    const fmt = validDt.toLocaleString('en-GB', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'UTC',
    });
    $('liveTimestamp').textContent = `Live ECMWF · ${fmt} UTC`;

    if (bulletinResp.ok) {
      const bulletin = await bulletinResp.json();
      tour.setBulletin(bulletin);
      globe.setStoryPins(bulletin.stories, -1);
    }
  } catch (err) {
    console.error(err);
    $('liveTimestamp').textContent = 'Weather data unavailable';
  }
}

init();
