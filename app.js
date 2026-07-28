// ===== CONFIGURATION =====
const SUPABASE_URL = "https://lzbfhvjieikbdrtzwsqk.supabase.co";
const SUPABASE_KEY = "sb_publishable_G8QH-43Js3hCDE3IGRs2ww_pjT65efR";

const THRESH = {
  temperature: { warn: 50, crit: 75, unit: '°C', max: 100 },
  humidity:    { warn: 70, crit: 95, unit: '%',  max: 100 },
  vibration:   { warn: 400, crit: 600, unit: 'Hz', max: 1000 }
};
const METRIC_LABELS = { temperature: 'Temp.', humidity: 'Hum.', vibration: 'Vib.' };
const EQ_COLORS = ['#8BC34A', '#607D8B', '#E0A430', '#4FA3D1', '#C77DFF'];
const ZONE_COLOR = { normal: '#8BC34A', warning: '#E0A430', critical: '#D6524A' };

// ===== STATE =====
let state = { rows: [], byEquip: {}, order: [], ml: null };
let charts = {};
let selectedEquipmentId = null;
let currentRange = '24h';

// ===== ML RESULTS (produced by train_model.ipynb + predict_and_sync.py) =====
// La table ml_results est alimentée par un script Python côté serveur qui a
// réellement entraîné un Random Forest + Isolation Forest sur l'historique de
// ocp_sensor_data. Tant que cette table est vide (modèle pas encore synchronisé),
// on retombe sur des valeurs de démonstration clairement indiquées comme telles.
async function loadMLResults() {
  try {
    const url = `${SUPABASE_URL}/rest/v1/ml_results?select=*&id=eq.1&limit=1`;
    const res = await fetch(url, { headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const rows = await res.json();
    if (!Array.isArray(rows) || !rows.length) throw new Error('empty');
    state.ml = rows[0];
  } catch (err) {
    state.ml = null; // pas encore de modèle synchronisé -> renderML() utilisera le mode démo
  }
  renderML();
}

// ===== DOM HELPERS (null-safe, since each page only contains a subset of elements) =====
function $(id) { return document.getElementById(id); }
function setText(id, val) { const el = $(id); if (el) el.textContent = val; }
function setHTML(id, val) { const el = $(id); if (el) el.innerHTML = val; }
function setAttr(id, attr, val) { const el = $(id); if (el) el.setAttribute(attr, val); }

// ===== THEME =====
const THEME_KEY = 'ocp-theme';
function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(saved);
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  setText('themeToggle', theme === 'dark' ? '☀️' : '🌙');
  localStorage.setItem(THEME_KEY, theme);
}
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
  if (selectedEquipmentId) renderCharts(selectedEquipmentId, currentRange);
}
$('themeToggle')?.addEventListener('click', toggleTheme);
initTheme();

// ===== UTILITY FUNCTIONS =====
function timeAgo(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return 'il y a ' + Math.max(1, Math.round(diff)) + 's';
  if (diff < 3600) return 'il y a ' + Math.round(diff / 60) + 'min';
  if (diff < 86400) return 'il y a ' + Math.round(diff / 3600) + 'h';
  return d.toLocaleDateString('fr-FR');
}

function statusOf(row) {
  if (row.machine_failure === 1 || row.status === 'critical') return 'critical';
  if (row.status === 'warning') return 'warning';
  return row.status || 'normal';
}

function zoneOf(metric, value) {
  const t = THRESH[metric];
  if (value > t.crit) return 'critical';
  if (value > t.warn) return 'warning';
  return 'normal';
}

// Sévérité progressive par capteur (0 = normal, 60 = au seuil warn, 100 = au
// seuil crit, >100 au-delà) — aligné avec sensor_severity() du pipeline Python.
function sensorSeverity(metric, value) {
  const t = THRESH[metric];
  if (value <= t.warn) return 60 * value / Math.max(t.warn, 1e-6);
  if (value <= t.crit) {
    const span = Math.max(t.crit - t.warn, 1e-6);
    return 60 + 40 * (value - t.warn) / span;
  }
  const over = value - t.crit;
  return 100 + 40 * (over / Math.max(t.crit, 1e-6));
}

// Poids par capteur (alignés avec SENSOR_WEIGHTS dans ml_common.py) : la
// vibration et la température sont les indicateurs mécaniques les plus
// parlants, l'humidité a un rôle secondaire.
const SENSOR_WEIGHTS = { temperature: 0.40, vibration: 0.45, humidity: 0.15 };

// Renvoie le risque de panne (%) pour un équipement en donnant la priorité
// au vrai modèle ML (state.ml.predictions, alimenté par predict_and_sync.py)
// s'il est disponible pour cet équipement ; sinon utilise l'heuristique
// computeRisk() calculée localement à partir des dernières lectures.
function riskFor(equipId, rowsAsc) {
  const ml = state.ml;
  if (ml && Array.isArray(ml.predictions)) {
    const p = ml.predictions.find(x => String(x.equipment_id) === String(equipId));
    if (p && typeof p.risk_pct === 'number') return p.risk_pct;
  }
  return computeRisk(rowsAsc);
}

// Renvoie la prédiction ML complète (état, détail, cause) pour un
// équipement si le modèle réel est synchronisé, sinon null.
function mlPredictionFor(equipId) {
  const ml = state.ml;
  if (ml && Array.isArray(ml.predictions)) {
    return ml.predictions.find(x => String(x.equipment_id) === String(equipId)) || null;
  }
  return null;
}

function computeRisk(rowsAsc) {
  const last = rowsAsc[rowsAsc.length - 1];
  if (!last) return 0;
  let score = 0;
  Object.keys(SENSOR_WEIGHTS).forEach(m => {
    score += SENSOR_WEIGHTS[m] * sensorSeverity(m, last[m]);
  });
  const n = rowsAsc.length;
  if (n >= 6) {
    const recent = rowsAsc.slice(-5);
    const prior = rowsAsc.slice(-10, -5);
    const avg = (arr, k) => arr.reduce((s, r) => s + (r[k] || 0), 0) / arr.length;
    ['temperature', 'vibration', 'humidity'].forEach(m => {
      if (prior.length) {
        const delta = avg(recent, m) - avg(prior, m);
        if (delta > 0) score += Math.min(delta / THRESH[m].crit * 25, 8);
      }
    });
  }
  // Le score composite (0-100+) utilise les MÊMES points de bascule que
  // label_state() côté Python (45 = limite Healthy/Warning, 65 = limite
  // Warning/Critical). On le convertit ici en un risque affichable 0-100%
  // par une interpolation linéaire par morceaux sur ces mêmes bornes, pour
  // que "risk > 40" côté JS corresponde bien à un état Warning/Critical
  // réel plutôt qu'à un seuil arbitraire déconnecté du modèle.
  let riskPct;
  if (score < 45) {
    riskPct = (score / 45) * 25;              // 0..45   -> 0..25%  (Healthy)
  } else if (score < 65) {
    riskPct = 25 + ((score - 45) / 20) * 30;   // 45..65  -> 25..55% (Warning)
  } else {
    riskPct = 55 + Math.min((score - 65) / 35, 1) * 45; // 65+ -> 55..100% (Critical+)
  }
  return Math.max(0, Math.min(100, Math.round(riskPct)));
}

function polarToCartesian(cx, cy, r, angleDeg) {
  const a = (angleDeg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}
function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArcFlag = (endAngle - startAngle) > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

function gaugeSVG(metric, value) {
  const t = THRESH[metric];
  const min = 0, max = t.max;
  const cx = 50, cy = 54, r = 40;
  const toAngle = v => -90 + (Math.min(Math.max(v, min), max) - min) / (max - min) * 180;
  const warnA = toAngle(t.warn);
  const critA = toAngle(t.crit);
  const valA = toAngle(value);
  const zone = zoneOf(metric, value);
  let segs = '';
  segs += `<path d="${describeArc(cx,cy,r,-90,warnA)}" stroke="#8BC34A" stroke-width="8" fill="none" stroke-linecap="round"/>`;
  segs += `<path d="${describeArc(cx,cy,r,warnA,critA)}" stroke="#E0A430" stroke-width="8" fill="none" stroke-linecap="round"/>`;
  segs += `<path d="${describeArc(cx,cy,r,critA,90)}" stroke="#D6524A" stroke-width="8" fill="none" stroke-linecap="round"/>`;
  const needle = polarToCartesian(cx, cy, r - 11, valA);
  return `<svg viewBox="0 0 100 68">
    <path d="${describeArc(cx,cy,r,-90,90)}" stroke="rgba(230,232,227,0.08)" stroke-width="8" fill="none" stroke-linecap="round"/>
    ${segs}
    <line x1="${cx}" y1="${cy}" x2="${needle.x}" y2="${needle.y}" stroke="#E6E8E3" stroke-width="2" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="3.2" fill="${ZONE_COLOR[zone]}"/>
  </svg>`;
}

// ===== INGEST DATA =====
function ingest(rows) {
  const byEquip = {};
  rows.slice().reverse().forEach(r => {
    if (!byEquip[r.equipment_id]) byEquip[r.equipment_id] = [];
    byEquip[r.equipment_id].push(r);
  });
  const order = Object.keys(byEquip).sort((a, b) => a - b);
  state.rows = rows.slice().reverse();
  state.byEquip = byEquip;
  state.order = order;
}

// ===== DEMO DATA =====
function demoData() {
  const now = Date.now();
  const rows = [];
  for (let i = 60; i >= 0; i--) {
    const t = new Date(now - i * 15000);
    const drift = Math.max(0, 40 - i * 0.6);
    rows.push({
      equipment_id: 3,
      equipment_name: 'Pompe Centrifuge',
      location: "Centre OCP Laayoune",
      timestamp: t.toISOString(),
      temperature: 38 + drift * 0.9 + Math.random() * 3,
      humidity: 50 + drift * 0.5 + Math.random() * 4,
      vibration: 180 + drift * 9 + Math.random() * 20,
      light: 300 + Math.random() * 120,
      machine_failure: drift > 32 ? 1 : 0,
      status: drift > 32 ? 'critical' : (drift > 18 ? 'warning' : 'normal'),
      temp_failure: drift > 32 ? 1 : 0,
      humidity_failure: 0,
      vibration_failure: drift > 28 ? 1 : 0,
      light_failure: 0
    });
    rows.push({
      equipment_id: 5,
      equipment_name: 'Convoyeur Nord',
      location: "Centre OCP Laayoune",
      timestamp: t.toISOString(),
      temperature: 32 + Math.random() * 6,
      humidity: 55 + Math.random() * 8,
      vibration: 150 + Math.random() * 60,
      light: 400 + Math.random() * 200,
      machine_failure: 0,
      status: 'normal',
      temp_failure: 0,
      humidity_failure: 0,
      vibration_failure: 0,
      light_failure: 0
    });
    rows.push({
      equipment_id: 7,
      equipment_name: 'Broyeur 03',
      location: "Centre OCP Laayoune",
      timestamp: t.toISOString(),
      temperature: 45 + Math.random() * 10,
      humidity: 40 + Math.random() * 12,
      vibration: 200 + Math.random() * 80,
      light: 250 + Math.random() * 100,
      machine_failure: 0,
      status: 'normal',
      temp_failure: 0,
      humidity_failure: 0,
      vibration_failure: 0,
      light_failure: 0
    });
  }
  return rows;
}

// ===== LOAD DATA =====
async function loadData() {
  try {
    const url = `${SUPABASE_URL}/rest/v1/ocp_sensor_data?select=*&order=timestamp.desc&limit=400`;
    const res = await fetch(url, { headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const rows = await res.json();
    if (!Array.isArray(rows) || !rows.length) throw new Error('empty');
    ingest(rows);
    $('errBanner')?.classList.remove('show');
    $('liveDot')?.classList.remove('err');
    $('liveDot')?.classList.add('live');
    setText('syncLabel', 'Connecté — ' + new Date().toLocaleTimeString('fr-FR'));
    setText('footNote', 'Données en direct — Supabase / capteurs simulés ESP32');
    render();
  } catch (err) {
    $('errBanner')?.classList.add('show');
    $('liveDot')?.classList.remove('live');
    $('liveDot')?.classList.add('err');
    setText('syncLabel', 'Hors ligne — données de démo');
    if (!state.rows.length) {
      ingest(demoData());
      render();
    }
  }
}

// ===== RENDER FUNCTIONS =====
function render() {
  const groups = state.byEquip;
  const ids = state.order;
  const total = state.rows.length;
  const failCount = state.rows.filter(r => r.machine_failure === 1).length;
  const availability = total ? (100 - (failCount / total * 100)) : 100;
  let activeAlerts = 0;
  ids.forEach(id => {
    const latest = groups[id][groups[id].length - 1];
    if (statusOf(latest) !== 'normal') activeAlerts++;
  });

  // Overview KPIs
  setText('kpiAvail', availability.toFixed(1) + '%');
  setText('kpiEquip', ids.length);
  setText('kpiAlerts', activeAlerts);
  setText('kpiReadings', total.toLocaleString('fr-FR'));
  setText('lastUpdateNote', total ? ('Dernière lecture ' + timeAgo(state.rows[state.rows.length - 1]?.timestamp || state.rows[0]?.timestamp)) : '');

  const healthPct = Math.max(0, Math.min(100, Math.round(availability)));
  setText('heroPct', healthPct + '%');
  const circumference = 452;
  setAttr('heroArc', 'stroke-dashoffset', circumference - (circumference * healthPct / 100));

  // Overview stats
  const stats = [
    { label: 'Taux de disponibilité', value: availability.toFixed(1) + '%', sub: 'Exploitation continue' },
    { label: 'Équipements actifs', value: ids.length, sub: 'Mises à jour en temps réel' },
    { label: 'Capteurs actifs', value: Math.max(4, ids.length + 2), sub: 'Température, vibration, humidité' },
    { label: 'Messages MQTT', value: total.toLocaleString('fr-FR'), sub: 'Trafic de messages reçus' },
    { label: 'Prédictions aujourd\'hui', value: Math.max(18, Math.round(total / 4)), sub: 'Modèle IA / maintenance' },
    { label: 'Risque moyen', value: Math.round((activeAlerts / Math.max(1, ids.length)) * 100) + '%', sub: 'Criticité actuelle' }
  ];
  setHTML('overviewStats', stats.map(s =>
    `<div class="stat-card"><div class="top"><span>${s.label}</span><span>●</span></div><div class="value">${s.value}</div><div class="sub">${s.sub}</div></div>`
  ).join(''));

  renderMonitoring();
  renderEquipmentPage();
  renderMaintenance();
  renderAlerts();
  renderHistory();
  renderML();
  renderSettings();

  // Charts dropdown
  const sel = $('eqSelect');
  if (sel) {
    const prevVal = sel.value || selectedEquipmentId || ids[0];
    sel.innerHTML = ids.map(id =>
      `<option value="${id}">${groups[id][groups[id].length - 1].equipment_name} — ${groups[id][groups[id].length - 1].location}</option>`
    ).join('');
    if (ids.includes(prevVal)) sel.value = prevVal;
    selectedEquipmentId = sel.value || ids[0];
    renderCharts(selectedEquipmentId, currentRange);
  }
}

function renderMonitoring() {
  const grid = $('monitoringGrid');
  if (!grid) return;
  const groups = state.byEquip;
  const ids = state.order;
  if (!ids.length) {
    grid.innerHTML = '<div class="empty-state">Aucune donnée reçue pour le moment.</div>';
    return;
  }
  grid.innerHTML = ids.map((id, i) => {
    const rows = groups[id];
    const latest = rows[rows.length - 1];
    const st = statusOf(latest);
    const risk = riskFor(id, rows);
    const color = EQ_COLORS[i % EQ_COLORS.length];
    const gauges = ['temperature', 'humidity', 'vibration', 'light'].map(m =>
      `<div class="gauge-box">
        ${gaugeSVG(m, latest[m])}
        <div class="gauge-val">${Number(latest[m]).toFixed(1)}<span style="color:var(--text-faint); font-size:10px;"> ${THRESH[m].unit}</span></div>
        <div class="gauge-lbl">${METRIC_LABELS[m]}</div>
      </div>`
    ).join('');
    return `<div class="eq-card" style="--eq-color:${color};">
      <div class="eq-card-head">
        <div>
          <div class="eq-name">${latest.equipment_name} <span style="color:var(--text-faint); font-weight:500; font-size:13px;">#${id}</span></div>
          <div class="eq-loc">${latest.location}</div>
        </div>
        <span class="badge ${st}">${st === 'critical' ? 'Critique' : st === 'warning' ? 'Alerte' : 'Normal'}</span>
      </div>
      <div class="gauge-row">${gauges}</div>
      <div class="eq-foot">
        <span>Indice de risque</span>
        <div class="risk-track"><div class="risk-fill" style="width:${risk}%; background:${risk > 60 ? 'var(--red)' : risk > 30 ? 'var(--amber)' : 'var(--green-lime)'};"></div></div>
        <span>${timeAgo(latest.timestamp)}</span>
      </div>
    </div>`;
  }).join('');
}

function renderEquipmentPage() {
  const list = $('equipmentList');
  const detail = $('equipmentDetail');
  if (!list || !detail) return;
  const groups = state.byEquip;
  const ids = state.order;
  if (!ids.length) {
    list.innerHTML = '<div class="empty-state">Aucun équipement à afficher.</div>';
    detail.innerHTML = '<div class="empty-state">Aucune donnée disponible.</div>';
    return;
  }
  if (!selectedEquipmentId || !groups[selectedEquipmentId]) selectedEquipmentId = ids[0];
  list.innerHTML = ids.map(id => {
    const rows = groups[id];
    const latest = rows[rows.length - 1];
    const st = statusOf(latest);
    return `<button class="equipment-item ${selectedEquipmentId === id ? 'active' : ''}" data-eq="${id}">
      <div class="name">${latest.equipment_name}</div>
      <div class="meta">${latest.location} · ${st === 'critical' ? 'Critique' : st === 'warning' ? 'Alerte' : 'Normal'}</div>
    </button>`;
  }).join('');
  const rows = groups[selectedEquipmentId];
  const latest = rows[rows.length - 1];
  const st = statusOf(latest);
  const risk = riskFor(selectedEquipmentId, rows);
  const mlPred = mlPredictionFor(selectedEquipmentId);
  const healthScore = Math.max(15, 100 - risk);
  detail.innerHTML = `
    <div class="sec-head" style="margin-bottom:12px;">
      <div><span class="sec-tag">Détail</span><h2>${latest.equipment_name}</h2></div>
      <span class="badge ${st}">${st === 'critical' ? 'Critique' : st === 'warning' ? 'Alerte' : 'Normal'}</span>
    </div>
    <div class="stats-grid">
      <div class="stat-card"><div class="top"><span>Health score</span></div><div class="value">${healthScore}%</div><div class="sub">Calculé à partir des capteurs</div></div>
      <div class="stat-card"><div class="top"><span>Probabilité de panne</span></div><div class="value">${risk}%</div><div class="sub">${mlPred ? 'Modèle prédictif (Random Forest)' : 'Estimation locale'}</div></div>
      <div class="stat-card"><div class="top"><span>Dernière maintenance</span></div><div class="value">${Math.max(1, Math.round(risk / 10))} j</div><div class="sub">Intervention recommandée</div></div>
      <div class="stat-card"><div class="top"><span>Capteurs</span></div><div class="value">4</div><div class="sub">Température, humidité, vibration, lumière</div></div>
    </div>
    ${mlPred ? `<div class="panel" style="margin-top:16px;"><div class="sec-head" style="margin-bottom:8px;"><div><span class="sec-tag">Analyse IA</span><h2 style="font-size:16px;">${mlPred.state}</h2></div></div><p style="color:var(--text-faint); font-size:13px;">${mlPred.detail}</p></div>` : ''}
    <div class="panel" style="margin-top:16px;">
      <div class="sec-head" style="margin-bottom:8px;"><div><span class="sec-tag">État courant</span><h2 style="font-size:16px;">Dernières mesures</h2></div></div>
      <div class="alerts-list">
        <div class="alert-row"><span class="alert-time">Température</span><span>${latest.temperature.toFixed(1)} °C</span><span class="badge ${zoneOf('temperature', latest.temperature)}">${zoneOf('temperature', latest.temperature)}</span></div>
        <div class="alert-row"><span class="alert-time">Humidité</span><span>${latest.humidity.toFixed(1)} %</span><span class="badge ${zoneOf('humidity', latest.humidity)}">${zoneOf('humidity', latest.humidity)}</span></div>
        <div class="alert-row"><span class="alert-time">Vibration</span><span>${latest.vibration.toFixed(1)} Hz</span><span class="badge ${zoneOf('vibration', latest.vibration)}">${zoneOf('vibration', latest.vibration)}</span></div>
        <div class="alert-row"><span class="alert-time">Luminosité</span><span>${latest.light.toFixed(1)} lux</span><span class="badge ${zoneOf('light', latest.light)}">${zoneOf('light', latest.light)}</span></div>
      </div>
    </div>
  `;
  document.querySelectorAll('.equipment-item').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedEquipmentId = btn.dataset.eq;
      renderEquipmentPage();
      renderCharts(selectedEquipmentId, currentRange);
    });
  });
}

function renderMaintenance() {
  const cards = $('maintenanceCards');
  const list = $('maintList');
  if (!cards && !list) return;
  const groups = state.byEquip;
  const ids = state.order;
  const items = ids.slice(0, 4).map(id => {
    const rows = groups[id];
    const latest = rows[rows.length - 1];
    const risk = riskFor(id, rows);
    const action = risk > 70 ? 'Arrêt immédiat' : risk > 40 ? 'Inspection recommandée' : 'Aucune';
    const delay = risk > 70 ? '4 heures' : risk > 40 ? '2 jours' : '15 jours';
    return `<div class="maint-card">
      <h3 class="maint-title">${latest.equipment_name}</h3>
      <p class="maint-subtitle">Probabilité de panne estimée à ${risk}% · temps avant panne ${delay}</p>
      <div class="maint-list">
        <div class="maint-item"><span>Probabilité de panne</span><strong>${risk}%</strong></div>
        <div class="maint-item"><span>Temps estimé</span><strong>${delay}</strong></div>
        <div class="maint-item"><span>Action recommandée</span><strong>${action}</strong></div>
      </div>
    </div>`;
  }).join('');
  if (cards) cards.innerHTML = items;

  if (list) {
    list.innerHTML = ids.slice(0, 5).map(id => {
      const rows = groups[id];
      const latest = rows[rows.length - 1];
      const risk = riskFor(id, rows);
      const st = risk > 70 ? 'critical' : risk > 40 ? 'warning' : 'normal';
      const action = risk > 70 ? 'Arrêt immédiat conseillé' : risk > 40 ? 'Inspection recommandée' : 'Aucune action';
      return `<div class="alert-row">
        <span class="alert-time">${risk > 70 ? 'Immédiat' : risk > 40 ? '48-72h' : 'Cette semaine'}</span>
        <span><span class="alert-eq">${latest.equipment_name}</span> — <span class="alert-detail">Risque de panne ${risk}% · ${action}</span></span>
        <span class="badge ${st}">${st === 'critical' ? 'Critique' : st === 'warning' ? 'Alerte' : 'Normal'}</span>
      </div>`;
    }).join('');
  }

  const totalRisk = ids.reduce((sum, id) => sum + riskFor(id, groups[id]), 0);
  const avgRisk = ids.length ? Math.round(totalRisk / ids.length) : 0;
  const maintConfidence = Math.min(95, 100 - avgRisk / 2);
  setText('kpiMaintConf', Math.round(maintConfidence) + '%');
  setText('kpiMaintPlan', ids.length + 3);
  setText('kpiMaintCosts', `${(ids.length * 320 + 800).toLocaleString('fr-FR')}€`);
  setText('kpiMaintRec', ids.length);
  setText('maintPct', Math.round(maintConfidence) + '%');
  setAttr('maintArc', 'stroke-dashoffset', 452 - (452 * maintConfidence / 100));
}

function renderAlerts() {
  const list = $('alertsList');
  if (!list) return;
  const all = state.rows.filter(r => statusOf(r) !== 'normal').slice(-50).reverse();
  setText('alertCount', all.length + ' évènement(s) récent(s)');
  if (!all.length) {
    list.innerHTML = '<div class="empty-state">Aucune alerte récente — tous les équipements fonctionnent normalement.</div>';
    return;
  }
  list.innerHTML = all.map(r => {
    const st = statusOf(r);
    const triggers = [];
    if (r.temp_failure) triggers.push('température');
    if (r.humidity_failure) triggers.push('humidité');
    if (r.vibration_failure) triggers.push('vibration');
    if (r.light_failure) triggers.push('luminosité');
    const detail = triggers.length ? `Seuil dépassé : ${triggers.join(', ')}` : 'Paramètres hors plage normale';
    return `<div class="alert-row">
      <span class="alert-time">${timeAgo(r.timestamp)}</span>
      <span><span class="alert-eq">${r.equipment_name}</span> — <span class="alert-detail">${detail}</span></span>
      <span class="badge ${st}">${st === 'critical' ? 'Critique' : 'Alerte'}</span>
    </div>`;
  }).join('');
}

function renderHistory() {
  const timeline = $('historyTimeline');
  if (!timeline) return;
  const entries = state.rows.slice(-12).reverse();
  if (!entries.length) {
    timeline.innerHTML = '<div class="empty-state">Aucun historique disponible.</div>';
    return;
  }
  timeline.innerHTML = entries.map(r => {
    const st = statusOf(r);
    const label = st === 'critical' ? 'Anomalie critique' : st === 'warning' ? 'Anomalie modérée' : 'Évolution normale';
    return `<div class="timeline-item">
      <div class="time">${new Date(r.timestamp).toLocaleString('fr-FR')}</div>
      <div><strong>${r.equipment_name}</strong> · ${label}</div>
    </div>`;
  }).join('');
}

function renderML() {
  const rfMetrics = $('rfMetrics');
  const ifMetrics = $('ifMetrics');
  const predictionsEl = $('predictionList');
  const featuresEl = $('featureImportance');
  const ml = state.ml; // null tant que ml_results n'est pas encore alimentée

  setText('mlSourceNote', ml
    ? `Modèle réel · entraîné le ${new Date(ml.trained_at).toLocaleDateString('fr-FR')} à ${new Date(ml.trained_at).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })} sur ${ml.training_rows || '—'} lectures`
    : '⚠ Mode démonstration — modèle pas encore synchronisé (voir predict_and_sync.py)');

  // ----- Métriques Random Forest -----
  const rf = ml || { rf_accuracy: 94.2, rf_recall: 91, rf_f1: 0.927, rf_specificity: 96 };
  if (rfMetrics) {
    rfMetrics.innerHTML = [
      { label: 'Précision', value: (ml ? ml.rf_precision : 94.2) + '%', sub: 'Random Forest' },
      { label: 'Rappel', value: (ml ? ml.rf_recall : rf.rf_recall) + '%', sub: 'Détection des pannes' },
      { label: 'F1-Score', value: ml ? ml.rf_f1 : rf.rf_f1, sub: 'Équilibre performance' },
      { label: 'Spécificité', value: (ml ? ml.rf_specificity : rf.rf_specificity) + '%', sub: 'Vrais négatifs' }
    ].map(item =>
      `<div class="ml-metric"><div class="ml-metric-val">${item.value}</div><div class="ml-metric-lbl">${item.label}</div></div>`
    ).join('');
  }

  // ----- Métriques Isolation Forest -----
  if (ifMetrics) {
    const anomalyCount = ml ? (ml.predictions || []).filter(p => p.is_anomaly).length : 3;
    ifMetrics.innerHTML = [
      { label: 'Anomalies détectées', value: anomalyCount, sub: 'Isolation Forest' },
      { label: 'Spécificité', value: (ml ? ml.iso_specificity : 98.7) + '%', sub: 'Non-supervisé' },
      { label: 'Horizon prédiction', value: (ml ? ml.iso_horizon_hours : 72) + 'h', sub: 'Détection précoce' },
      { label: 'Contamination', value: (ml ? ml.iso_contamination * 100 : 5) + '%', sub: 'Seuil de détection' }
    ].map(item =>
      `<div class="ml-metric"><div class="ml-metric-val">${item.value}</div><div class="ml-metric-lbl">${item.label}</div></div>`
    ).join('');
  }

  // ----- Prédictions par équipement -----
  if (predictionsEl) {
    const predData = ml && ml.predictions
      ? ml.predictions.map(p => ({ name: p.equipment_name, detail: p.detail, risk: p.risk_pct, cls: p.risk_class }))
      : [
        { name: 'Pompe Centrifuge', detail: 'Donnée de démonstration', risk: 87, cls: 'high' },
        { name: 'Convoyeur Nord', detail: 'Donnée de démonstration', risk: 64, cls: 'medium' },
        { name: 'Broyeur 03', detail: 'Donnée de démonstration', risk: 28, cls: 'low' }
      ];
    predictionsEl.innerHTML = predData.map(p =>
      `<div class="prediction-row ${p.cls}">
        <div class="prediction-text">
          <strong>${p.name}</strong>
          <small>${p.detail}</small>
        </div>
        <div class="prediction-score" style="color:${p.risk > 70 ? 'var(--red)' : p.risk > 40 ? 'var(--amber)' : 'var(--green-lime)'};">${p.risk}% de risque</div>
      </div>`
    ).join('');
  }

  // ----- Feature importance -----
  if (featuresEl) {
    const fi = ml && ml.feature_importance
      ? Object.entries(ml.feature_importance).map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
      : [
        { name: 'Vibration', value: 93 },
        { name: 'Température', value: 84 },
        { name: 'Courant', value: 68 },
        { name: 'Humidité', value: 41 }
      ];
    featuresEl.innerHTML = fi.map(item =>
      `<div class="feature-item">
        <div style="display:flex;justify-content:space-between;align-items:center;"><strong>${item.name}</strong><span>${item.value}%</span></div>
        <div class="bar"><div class="fill" style="width:${item.value}%"></div></div>
      </div>`
    ).join('');
  }

  const groups = state.byEquip;
  const ids = state.order;
  const avgRisk = ml && ml.predictions && ml.predictions.length
    ? Math.round(ml.predictions.reduce((s, p) => s + p.risk_pct, 0) / ml.predictions.length)
    : (ids.length ? Math.round(ids.reduce((sum, id) => sum + riskFor(id, groups[id]), 0) / ids.length) : 0);
  const health = Math.max(15, 100 - avgRisk);

  setText('mlAccuracy', (ml ? ml.rf_accuracy : 94.2) + '%');
  setText('mlRecall', (ml ? ml.rf_recall : 91) + '%');
  setText('mlPredictions', ids.length);
  setText('mlAnomalies', ml && ml.predictions ? ml.predictions.filter(p => p.is_anomaly).length : ids.filter(id => riskFor(id, groups[id]) > 60).length);
  setText('mlHealthScore', health + '%');
  setAttr('mlArc', 'stroke-dashoffset', 452 - (452 * health / 100));
}

function renderSettings() {
  const fields = [
    ['tempWarn', THRESH.temperature.warn],
    ['tempCrit', THRESH.temperature.crit],
    ['humWarn', THRESH.humidity.warn],
    ['humCrit', THRESH.humidity.crit],
    ['vibWarn', THRESH.vibration.warn],
    ['vibCrit', THRESH.vibration.crit],
    ['lightWarn', THRESH.light.warn],
    ['lightCrit', THRESH.light.crit]
  ];
  fields.forEach(([id, value]) => {
    const el = $(id);
    if (el) el.value = value;
  });
}

// ===== CHARTS =====
// Durée réelle (en heures) associée à chaque bouton de plage.
const RANGE_HOURS = { '1h': 1, '24h': 24, '7d': 24 * 7, '30d': 24 * 30 };
// Nombre de points cible à afficher (pour ne pas surcharger le graphique
// sur les longues plages) — on sous-échantillonne si besoin après filtrage.
const RANGE_MAX_POINTS = { '1h': 60, '24h': 96, '7d': 168, '30d': 180 };

let chartsRequestToken = 0; // évite qu'une requête lente et périmée n'écrase un rendu plus récent

async function fetchChartRows(equipId, range) {
  const hours = RANGE_HOURS[range] || 24;
  const since = new Date(Date.now() - hours * 3600 * 1000).toISOString();
  // On charge large (jusqu'à 5000 lignes sur 30j) puis on sous-échantillonne
  // localement pour garder des graphiques lisibles et rapides à dessiner.
  const url = `${SUPABASE_URL}/rest/v1/ocp_sensor_data?select=*&equipment_id=eq.${equipId}&timestamp=gte.${since}&order=timestamp.asc&limit=5000`;
  const res = await fetch(url, { headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` } });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const rows = await res.json();
  return Array.isArray(rows) ? rows : [];
}

// Sous-échantillonne rows à ~maxPoints en gardant un pas régulier (au lieu
// de tronquer aux N dernières lignes, ce qui écraserait la couverture
// temporelle demandée par le bouton de plage).
function downsample(rows, maxPoints) {
  if (rows.length <= maxPoints) return rows;
  const step = rows.length / maxPoints;
  const out = [];
  for (let i = 0; i < maxPoints; i++) out.push(rows[Math.floor(i * step)]);
  return out;
}

function chartTimeLabel(ts, range) {
  const d = new Date(ts);
  if (range === '7d' || range === '30d') {
    return d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' }) + ' ' +
      d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  }
  return d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function renderCharts(equipId, range = currentRange) {
  if (!equipId || !state.byEquip[equipId]) return;
  const myToken = ++chartsRequestToken;

  let recentRows;
  try {
    const raw = await fetchChartRows(equipId, range);
    if (myToken !== chartsRequestToken) return; // une requête plus récente a pris le relais
    if (raw.length) {
      recentRows = downsample(raw, RANGE_MAX_POINTS[range] || 60);
    } else {
      // Repli : pas (encore) de données sur cette fenêtre temporelle réelle
      // (ex: capteurs actifs depuis moins d'1h/24h/etc.) -> on retombe sur
      // le cache local le plus proche disponible plutôt que d'afficher un
      // graphique vide.
      recentRows = state.byEquip[equipId].slice(-(RANGE_MAX_POINTS[range] || 60));
    }
  } catch (err) {
    // Supabase indisponible -> repli sur le cache local déjà chargé
    if (myToken !== chartsRequestToken) return;
    recentRows = state.byEquip[equipId].slice(-(RANGE_MAX_POINTS[range] || 60));
  }

  const labels = recentRows.map(r => chartTimeLabel(r.timestamp, range));

  const defs = [
    { key: 'temperature', canvas: 'chartTemp', color: '#8BC34A', thrEl: 'thrTemp' },
    { key: 'humidity', canvas: 'chartHum', color: '#607D8B', thrEl: 'thrHum' },
    { key: 'vibration', canvas: 'chartVib', color: '#E0A430', thrEl: 'thrVib' },
    { key: 'light', canvas: 'chartLight', color: '#4FA3D1', thrEl: 'thrLight' }
  ];

  defs.forEach(d => {
    const t = THRESH[d.key];
    const thrEl = $(d.thrEl);
    if (thrEl) thrEl.textContent = `Alerte > ${t.warn === t.crit ? '' : t.warn + ' ' + t.unit + ' · '}Critique > ${t.crit} ${t.unit}`;
    const canvas = $(d.canvas);
    if (!canvas) return;
    const data = recentRows.map(r => r[d.key]);
    const critLine = recentRows.map(() => t.crit);
    const ctx = canvas.getContext('2d');
    if (charts[d.canvas]) charts[d.canvas].destroy();
    charts[d.canvas] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: METRIC_LABELS[d.key], data, borderColor: d.color, backgroundColor: d.color + '22', borderWidth: 2, tension: 0.35, pointRadius: 0, fill: true },
          { label: 'Seuil critique', data: critLine, borderColor: '#D6524A', borderWidth: 1.3, borderDash: [5, 4], pointRadius: 0, fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#14261e', titleColor: '#E6E8E3', bodyColor: '#E6E8E3', borderColor: 'rgba(230,232,227,0.15)', borderWidth: 1 }
        },
        scales: {
          x: { ticks: { color: '#5E7269', maxTicksLimit: 6, font: { size: 10 } }, grid: { color: 'rgba(230,232,227,0.05)' } },
          y: { ticks: { color: '#5E7269', font: { size: 10 } }, grid: { color: 'rgba(230,232,227,0.05)' } }
        }
      }
    });
  });
}

// ===== SETTINGS HANDLERS =====
function saveSettings() {
  THRESH.temperature.warn = Number($('tempWarn').value) || THRESH.temperature.warn;
  THRESH.temperature.crit = Number($('tempCrit').value) || THRESH.temperature.crit;
  THRESH.humidity.warn = Number($('humWarn').value) || THRESH.humidity.warn;
  THRESH.humidity.crit = Number($('humCrit').value) || THRESH.humidity.crit;
  THRESH.vibration.warn = Number($('vibWarn').value) || THRESH.vibration.warn;
  THRESH.vibration.crit = Number($('vibCrit').value) || THRESH.vibration.crit;
  THRESH.light.warn = Number($('lightWarn').value) || THRESH.light.warn;
  THRESH.light.crit = Number($('lightCrit').value) || THRESH.light.crit;
  setText('settingsStatus', '✅ Seuils enregistrés avec succès.');
  if (selectedEquipmentId) renderCharts(selectedEquipmentId, currentRange);
}

function resetSettings() {
  THRESH.temperature = { warn: 50, crit: 75, unit: '°C', max: 100 };
  THRESH.humidity = { warn: 70, crit: 95, unit: '%', max: 100 };
  THRESH.vibration = { warn: 400, crit: 600, unit: 'Hz', max: 1000 };
  THRESH.light = { warn: 800, crit: 800, unit: 'lux', max: 1000 };
  renderSettings();
  setText('settingsStatus', '🔄 Seuils réinitialisés.');
  if (selectedEquipmentId) renderCharts(selectedEquipmentId, currentRange);
}

$('saveSettings')?.addEventListener('click', saveSettings);
$('resetSettings')?.addEventListener('click', resetSettings);

// ===== EVENT LISTENERS =====
$('eqSelect')?.addEventListener('change', e => {
  selectedEquipmentId = e.target.value;
  renderCharts(selectedEquipmentId, currentRange);
});

document.querySelectorAll('.range-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    currentRange = btn.dataset.range;
    document.querySelectorAll('.range-btn').forEach(b => b.classList.toggle('active', b === btn));
    if (selectedEquipmentId) renderCharts(selectedEquipmentId, currentRange);
  });
});

// ===== INIT =====
loadData();
loadMLResults();
setInterval(loadData, 20000);
setInterval(loadMLResults, 60000); // le modèle ne change pas toutes les 20s, 1 min suffit
