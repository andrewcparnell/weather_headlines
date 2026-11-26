import { WeatherGlobe } from './globe.js?v=20261126j';
import { LocationSearch } from './search.js?v=20261126j';
import { BroadcastTour } from './tour.js?v=20261126j';

const $ = id => document.getElementById(id);

let manifestData = null;

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

  // Load headlines independently of WebP textures so the Play button is ready immediately!
  fetch(`data/headlines.json?t=${Date.now()}`, { cache: 'no-store' })
    .then(r => (r.ok ? r.json() : null))
    .then(bulletin => {
      if (!bulletin) return;
      tour.setBulletin(bulletin);
      globe.setStoryPins(bulletin.stories, -1);
    })
    .catch(err => console.error('Failed loading headlines:', err));

  try {
    const manifest = await globe.loadWeatherManifest('data/manifest.json');
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
  } catch (err) {
    console.error(err);
    $('liveTimestamp').textContent = 'Weather data unavailable';
  }
}

init();
