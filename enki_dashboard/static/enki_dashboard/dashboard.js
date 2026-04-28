const CONFIG = JSON.parse(document.getElementById('dashboard-config').textContent);

// ===== Map & tile layer =====
const map = L.map('map', { attributionControl: false })
  .setView([CONFIG.centerLat, CONFIG.centerLon], CONFIG.zoom);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
}).addTo(map);

// ===== Heatmap color ramp =====
const COLOR_STOPS = [
  [0.0, 255, 255, 191],
  [0.2, 253, 174,  97],
  [0.5, 215,  48,  39],
  [0.8, 127,   0,   0],
];
const lerp = (a, b, t) => a + (b - a) * t;

function colorByIntensity(i) {
  if (i < 0.05) return [0, 0, 0, 0];
  let lo = COLOR_STOPS[0], hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let k = 0; k < COLOR_STOPS.length - 1; k++) {
    if (i >= COLOR_STOPS[k][0] && i <= COLOR_STOPS[k + 1][0]) {
      lo = COLOR_STOPS[k]; hi = COLOR_STOPS[k + 1]; break;
    }
  }
  if (i > hi[0]) lo = hi;
  const span = hi[0] - lo[0];
  const t = span > 0 ? (i - lo[0]) / span : 0;
  return [
    Math.round(lerp(lo[1], hi[1], t)),
    Math.round(lerp(lo[2], hi[2], t)),
    Math.round(lerp(lo[3], hi[3], t)),
    Math.round(lerp(80, 230, Math.min(i, 1))),
  ];
}

// ===== Formatting (Italian numerals) =====
const fmtInt = n => new Intl.NumberFormat('it-IT').format(n);
const fmtPct = n => (n * 100).toFixed(1).replace('.', ',') + ' %';
const fmtNum = n => n.toFixed(2).replace('.', ',');

function decodeUint8Base64(value) {
  const raw = atob(value);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    bytes[i] = raw.charCodeAt(i);
  }
  return bytes;
}

function dateToISO(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function isoToDate(value) {
  const [year, month, day] = value.split('-').map(Number);
  return new Date(year, month - 1, day);
}

function replaceChildren(parent, children) {
  parent.textContent = '';
  parent.append(...children);
}

function predictionChannelNode(channel) {
  const wrapper = document.createElement('div');
  wrapper.className = 'pred-channel-item';

  const image = document.createElement('img');
  image.src = 'data:image/png;base64,' + channel.img;
  image.alt = channel.name;

  const label = document.createElement('div');
  label.className = 'pred-channel-label';
  label.textContent = channel.name;

  wrapper.append(image, label);
  return wrapper;
}

function predictionStatNode(labelText, valueText) {
  const wrapper = document.createElement('div');
  wrapper.className = 'pred-stat';

  const label = document.createElement('div');
  label.className = 'pred-stat-label';
  label.textContent = labelText;

  const value = document.createElement('div');
  value.className = 'pred-stat-value';
  value.textContent = valueText;

  wrapper.append(label, value);
  return wrapper;
}

// ===== Severity =====
function severityFromPeak(peak) {
  if (peak >= 0.8) return { label: 'Red',    cls: 'danger',  hint: 'critical — immediate response' };
  if (peak >= 0.5) return { label: 'Orange', cls: 'warning', hint: 'high — preparedness needed' };
  if (peak >= 0.2) return { label: 'Yellow', cls: '',        hint: 'moderate — keep monitoring' };
  return             { label: 'Green',  cls: '',        hint: 'low — normal conditions' };
}

function alertsHintText(n) {
  if (n === 0) return 'no critical zones detected';
  if (n === 1) return '1 zone requires attention';
  return n + ' zones require attention';
}

// ===== Period filter (giornaliero) =====
let currentPeriod = { year: null, month: null, day: null };
let picker = null;
let predPicker = null;

function appendPeriodParams(params) {
  if (currentPeriod.year)  params.set('year',  currentPeriod.year);
  if (currentPeriod.month) params.set('month', currentPeriod.month);
  if (currentPeriod.day)   params.set('day',   currentPeriod.day);
}

async function loadPeriods() {
  const res = await fetch(CONFIG.urls.historicalPeriods);
  const { days } = await res.json();

  const floodSet = new Set(
    days.map(({ year, month, day }) =>
      `${year}-${String(month).padStart(2,'0')}-${String(day).padStart(2,'0')}`)
  );

  picker = flatpickr('#periodPicker', {
    dateFormat: 'd/m/Y',
    allowInput: false,
    onDayCreate(dObj, dStr, fp, dayElem) {
      const d = dayElem.dateObj;
      const iso = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
      if (floodSet.has(iso)) dayElem.classList.add('flood-day');
    },
    onChange(selectedDates) {
      if (!selectedDates.length) {
        currentPeriod = { year: null, month: null, day: null };
      } else {
        const d = selectedDates[0];
        currentPeriod = { year: d.getFullYear(), month: d.getMonth() + 1, day: d.getDate() };
      }
      floodToken++;
      refresh();
    },
  });
}

document.getElementById('btnClearDate').addEventListener('click', () => {
  picker && picker.clear();
  currentPeriod = { year: null, month: null, day: null };
  floodToken++;
  refresh();
});

// ===== Routing state =====
let pointA = null, pointB = null;
let markerA = null, markerB = null;
let pathLayer = null;
let routeToken = 0;
let floodToken = 0;

function setHint(text) {
  document.getElementById('routeHint').textContent = text;
}

async function computeRoute() {
  if (!pointA || !pointB) return;
  const myToken = routeToken;
  const params = new URLSearchParams({
    a_lat: pointA.lat, a_lon: pointA.lng,
    b_lat: pointB.lat, b_lon: pointB.lng,
  });
  appendPeriodParams(params);

  try {
    const res = await fetch(CONFIG.urls.route + '?' + params);
    if (myToken !== routeToken) return;

    if (!res.ok) {
      setHint(res.status === 404 ? 'No path available' : 'Routing error');
      return;
    }
    const { path } = await res.json();
    if (myToken !== routeToken) return;

    if (pathLayer) map.removeLayer(pathLayer);
    pathLayer = L.polyline(path, {
      color: '#06b6d4',
      weight: 5,
      opacity: 0.9,
    }).addTo(map);
    setHint('Route auto-updates with flood changes');
  } catch (err) {
    console.error('route failed:', err);
    setHint('Routing error');
  }
}

function resetRoute() {
  routeToken++;
  if (markerA)   { map.removeLayer(markerA);   markerA = null; }
  if (markerB)   { map.removeLayer(markerB);   markerB = null; }
  if (pathLayer) { map.removeLayer(pathLayer); pathLayer = null; }
  pointA = null;
  pointB = null;
  setHint('Click two points to calculate a route');
}

map.on('click', (e) => {
  if (!pointA) {
    pointA = e.latlng;
    markerA = L.marker(e.latlng).addTo(map).bindTooltip('A', { permanent: true, direction: 'top' });
    setHint('Click destination point B');
  } else if (!pointB) {
    pointB = e.latlng;
    markerB = L.marker(e.latlng).addTo(map).bindTooltip('B', { permanent: true, direction: 'top' });
    computeRoute();
  } else {
    resetRoute();
    pointA = e.latlng;
    markerA = L.marker(e.latlng).addTo(map).bindTooltip('A', { permanent: true, direction: 'top' });
    setHint('Click destination point B');
  }
});

document.getElementById('btnReset').addEventListener('click', resetRoute);

// ===== Flood overlay loop =====
let floodLayer = null;

async function loadFloods() {
  const myToken = floodToken;

  const params = new URLSearchParams();
  appendPeriodParams(params);
  const url = CONFIG.urls.floodData +
              (params.toString() ? '?' + params : '');
  const res = await fetch(url);
  if (myToken !== floodToken) return;   // periodo cambiato nel frattempo

  const { bounds, size, data, stats } = await res.json();
  if (myToken !== floodToken) return;   // doppio check

  const values = decodeUint8Base64(data);

  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = colorByIntensity(values[y * size + x] / 255);
      const idx = (y * size + x) * 4;
      img.data[idx]     = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);

  if (floodLayer) map.removeLayer(floodLayer);
  floodLayer = L.imageOverlay(
    canvas.toDataURL(),
    [[bounds.south, bounds.west], [bounds.north, bounds.east]],
    { opacity: 0.7, interactive: false }
  ).addTo(map);

  // KPIs
  const affectedKm2 = stats.risk_cells * Math.pow(CONFIG.pixelM / 1000, 2);
  document.getElementById('kpiRisk').textContent = fmtNum(affectedKm2) + ' km²';
  document.getElementById('kpiRiskHint').textContent =
    fmtPct(stats.risk_cells / stats.total) + ' of monitored area';

  const sev = severityFromPeak(stats.peak);
  const peakEl = document.getElementById('kpiPeak');
  peakEl.textContent = sev.label;
  peakEl.className = 'kpi-value ' + sev.cls;
  document.getElementById('kpiPeakHint').textContent = sev.hint;

  const alertsEl = document.getElementById('kpiAlerts');
  alertsEl.textContent = fmtInt(stats.alerts);
  alertsEl.className = 'kpi-value ' +
    (stats.alerts >= 3 ? 'danger' : stats.alerts >= 1 ? 'warning' : '');
  document.getElementById('kpiAlertsHint').textContent = alertsHintText(stats.alerts);

  const now = new Date();
  document.getElementById('lastUpdate').textContent =
    String(now.getHours()).padStart(2, '0') + ':' +
    String(now.getMinutes()).padStart(2, '0') + ':' +
    String(now.getSeconds()).padStart(2, '0');

  if (pointA && pointB) computeRoute();
}

function refresh() {
  loadFloods().catch(err => console.error('flood load failed:', err));
}

const REFRESH_MS = CONFIG.refreshMs;
loadPeriods();
refresh();
setInterval(refresh, REFRESH_MS);

// ===== ResUNet prediction drawer =====
let modelLayer     = null;
let modelHeatmap   = null;
let predRunning    = false;

const predDrawer    = document.getElementById('predDrawer');
const predIdle      = document.getElementById('predIdle');
const predLoading   = document.getElementById('predLoading');
const predContent   = document.getElementById('predContent');
const btnRunPred    = document.getElementById('btnRunPred');
const btnToggle     = document.getElementById('btnToggleModelLayer');
const btnClosePred  = document.getElementById('btnClosePred');
const btnOpenPred   = document.getElementById('btnOpenPred');
const predDateInput = document.getElementById('predDateInput');

// Init date picker: default = today, max = today
const today = new Date();
predPicker = flatpickr(predDateInput, {
  dateFormat: 'd/m/Y',
  allowInput: false,
  defaultDate: today,
  maxDate: today,
});

btnOpenPred.addEventListener('click', () => predDrawer.classList.add('open'));
btnClosePred.addEventListener('click', () => predDrawer.classList.remove('open'));

function showPredState(state) {
  predIdle.hidden = state !== 'idle';
  predLoading.hidden = state !== 'loading';
  predContent.hidden = state !== 'done';
}

btnRunPred.addEventListener('click', async () => {
  if (predRunning) return;
  predRunning = true;
  btnRunPred.disabled = true;
  showPredState('loading');

  try {
    const selectedDate = predPicker.selectedDates[0]
      ? dateToISO(predPicker.selectedDates[0])
      : '';
    const url = CONFIG.urls.modelPrediction +
                (selectedDate ? '?date=' + selectedDate : '');
    const res  = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Server error ' + res.status);
    }

    // Update date picker to reflect what was actually used
    if (data.date) {
      predPicker.setDate(isoToDate(data.date), false);
    }

    // Input channels grid
    const grid = document.getElementById('predChannelGrid');
    replaceChildren(grid, data.channels.map(predictionChannelNode));

    // Stats
    const statsRow = document.getElementById('predStatsRow');
    const s = data.stats;
    replaceChildren(statsRow, [
      predictionStatNode('Max probability', `${(s.max_prob * 100).toFixed(1)} %`),
      predictionStatNode('Flooded area', `${s.flooded_pct.toFixed(1)} %`),
      predictionStatNode('Mean probability', `${(s.mean_prob * 100).toFixed(2)} %`),
      predictionStatNode('Threshold', String(data.threshold)),
    ]);

    // Output images
    document.getElementById('predRawImg').src    = 'data:image/png;base64,' + data.raw_img;
    document.getElementById('predThreshImg').src = 'data:image/png;base64,' + data.thresh_img;

    // Store heatmap for map overlay
    modelHeatmap = data.heatmap;
    btnToggle.disabled = false;

    showPredState('done');
  } catch (err) {
    console.error('prediction failed:', err);
    predIdle.textContent = 'Prediction failed — check server logs.';
    predIdle.classList.add('pred-error');
    predIdle.hidden = false;
    predLoading.hidden = true;
  } finally {
    predRunning = false;
    btnRunPred.disabled = false;
  }
});

// Toggle model heatmap overlay on Leaflet map
let modelLayerVisible = false;
btnToggle.addEventListener('click', () => {
  if (!modelHeatmap) return;

  if (modelLayerVisible) {
    if (modelLayer) { map.removeLayer(modelLayer); modelLayer = null; }
    modelLayerVisible = false;
    btnToggle.textContent = 'Toggle on map';
    return;
  }

  // Build canvas from model heatmap
  const size = modelHeatmap.length;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = colorByIntensity(modelHeatmap[y][x]);
      const idx = (y * size + x) * 4;
      img.data[idx]     = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);

  const bounds_data = CONFIG.bounds;
  modelLayer = L.imageOverlay(
    canvas.toDataURL(),
    [[bounds_data.south, bounds_data.west], [bounds_data.north, bounds_data.east]],
    { opacity: 0.75, interactive: false }
  ).addTo(map);

  modelLayerVisible = true;
  btnToggle.textContent = 'Remove from map';
});
