const state = {
  features: [], // [{name, type}]
  charts: { proba: null, radar: null },
};

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') n.className = v;
    else if (k === 'for') n.htmlFor = v;
    else n.setAttribute(k, v);
  });
  children.forEach((c) => n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
  return n;
}

async function loadSchema() {
  const loading = document.getElementById('schema-loading');
  const form = document.getElementById('predict-form');
  const inputs = document.getElementById('inputs');
  try {
    const res = await fetch('/schema');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Schema error');
    state.features = (data.features || []).map((f) => ({ name: f.name, type: f.type || 'unknown' }));
    if (!state.features.length) throw new Error('No features found in preprocessor.');

    inputs.innerHTML = '';
    state.features.forEach(({ name, type }, index) => {
      const id = `field-${name}`;
      const inputAttrs = { 
        id, 
        name, 
        placeholder: type === 'numeric' ? 'Enter numeric value' : 'Enter value',
        type: type === 'numeric' ? 'number' : 'text'
      };
      if (type === 'numeric') {
        inputAttrs.step = 'any';
      }
      const field = el('div', { 
        class: 'field', 
        style: `animation: fieldSlideIn 0.4s ease-out ${index * 0.03}s both` 
      }, [
        el('label', { for: id }, [`${name} ${type !== 'unknown' ? `(${type})` : ''}`.trim()]),
        el('input', inputAttrs),
      ]);
      inputs.appendChild(field);
    });

    loading.classList.add('hidden');
    form.classList.remove('hidden');
  } catch (err) {
    const loadingText = loading.querySelector('span');
    if (loadingText) {
      loadingText.textContent = `Failed to load schema: ${err.message}`;
    } else {
      loading.textContent = `Failed to load schema: ${err.message}`;
    }
  }
}

function buildPayload() {
  const payload = {};
  state.features.forEach(({ name }) => {
    const input = document.getElementById(`field-${name}`);
    if (!input) return;
    const val = input.value;
    payload[name] = val;
  });
  return payload;
}

function rnd(min, max, decimals = 4) {
  const v = Math.random() * (max - min) + min;
  return Number(v.toFixed(decimals));
}

function choice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomIsoDate() {
  const now = new Date();
  const daysBack = Math.floor(Math.random() * 365);
  const dt = new Date(now.getTime() - daysBack * 24 * 60 * 60 * 1000);
  return dt.toISOString().slice(0, 19);
}

function randomCategorical(name) {
  const n = name.toLowerCase();
  if (n.includes('date') || n.includes('epoch')) return randomIsoDate();
  if (n.includes('time_system')) return 'UTC';
  if (n.includes('ref_frame')) return 'TEME';
  if (n.includes('center_name')) return 'EARTH';
  if (n.includes('originator')) return '18 SPCS';
  if (n.includes('mean_element_theory')) return 'SGP4';
  if (n.includes('object_type')) return choice(['DEBRIS', 'PAYLOAD', 'ROCKET BODY']);
  if (n.includes('rcs_size')) return choice(['SMALL', 'MEDIUM', 'LARGE']);
  if (n.includes('country_code')) return choice(['US', 'PRC', 'CIS', 'IND', 'FR']);
  if (n.includes('classification_type')) return 'U';
  if (n.includes('ephemeris_type')) return '0';
  if (n.includes('site')) return choice(['FRGUI', 'PKMTR', 'SRI', 'TTMTR', 'XSC', 'AFETR']);
  if (n.includes('object_name')) return choice(['SL-8 DEB', 'CZ-4 DEB', 'BLOCK DM R/B', 'GSAT 1']);
  if (n.includes('object_id')) return choice(['2001-018A', '1999-057MB', '1965-108AS', '1979-028C']);
  if (n.startsWith('tle_line')) return '1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9990';
  if (n === 'tle_line0') return '0 ISS (ZARYA)';
  return 'UNKNOWN';
}

function randomNumeric(name) {
  const n = name.toLowerCase();
  if (n.includes('inclination')) return rnd(0, 180).toString();
  if (n.includes('ra_of_asc_node')) return rnd(0, 360).toString();
  if (n.includes('arg_of_pericenter')) return rnd(0, 360).toString();
  if (n.includes('mean_anomaly')) return rnd(0, 360).toString();
  if (n.includes('eccentricity')) return rnd(0, 0.99, 6).toString();
  if (n.includes('mean_motion_dot')) return rnd(-1e-3, 1e-3, 8).toString();
  if (n.includes('mean_motion_ddot')) return rnd(-1e-6, 1e-6, 10).toString();
  if (n === 'bstar' || n.includes('bstar')) return rnd(0, 0.05, 6).toString();
  if (n.includes('mean_motion')) return rnd(0.5, 16, 6).toString();
  if (n.includes('semimajor') || (n.includes('semi') && n.includes('axis'))) return rnd(6500, 45000).toString();
  if (n.includes('period')) return rnd(80, 1500).toString();
  if (n.includes('apoapsis')) return rnd(100, 40000).toString();
  if (n.includes('periapsis')) return rnd(100, 38000).toString();
  if (n.includes('id') || n.endsWith('_no') || n.includes('cat')) return Math.floor(rnd(1000, 60000, 0)).toString();
  return rnd(0, 1000).toString();
}

async function fillExample() {
  try {
    const res = await fetch('/example');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Failed to get example');
    const ex = data.example || {};
    state.features.forEach(({ name, type }) => {
      const input = document.getElementById(`field-${name}`);
      if (!input) return;
      const v = ex.hasOwnProperty(name) ? ex[name] : undefined;
      if (v === undefined || v === null || (typeof v === 'number' && !isFinite(v))) {
        input.value = type === 'numeric' ? randomNumeric(name) : randomCategorical(name);
      } else {
        input.value = String(v);
      }
    });
  } catch (err) {
    // Fallback to client-side generator
    state.features.forEach(({ name, type }) => {
      const input = document.getElementById(`field-${name}`);
      if (!input) return;
      input.value = type === 'numeric' ? randomNumeric(name) : randomCategorical(name);
    });
  }
}

function renderProbabilityChart(proba) {
  const ctx = document.getElementById('probaChart');
  if (!ctx) return;
  
  // Class labels mapping
  const classLabels = ['DEBRIS', 'PAYLOAD', 'ROCKET BODY', 'TBA'];
  const labels = Array.isArray(proba) && proba.length <= 4 
    ? proba.map((_, i) => classLabels[i] || `Class ${i}`)
    : (proba ? proba.map((_, i) => `Class ${i}`) : ['Class 0', 'Class 1']);
  
  const dataVals = Array.isArray(proba) && proba.length ? proba : [NaN, NaN];
  
  // Color scheme matching the design
  const colors = [
    'rgba(99, 102, 241, 0.8)',   // Primary
    'rgba(139, 92, 246, 0.8)',   // Secondary
    'rgba(6, 182, 212, 0.8)',     // Accent
    'rgba(16, 185, 129, 0.8)'     // Success
  ];
  
  const backgroundColor = dataVals.map((_, i) => colors[i % colors.length]);
  const borderColor = dataVals.map((_, i) => colors[i % colors.length].replace('0.8', '1'));
  
  const config = {
    type: 'bar',
    data: { 
      labels, 
      datasets: [{ 
        label: 'Probability', 
        data: dataVals, 
        backgroundColor,
        borderColor,
        borderWidth: 2,
        borderRadius: 8
      }] 
    },
    options: { 
      responsive: true,
      maintainAspectRatio: false,
      scales: { 
        y: { 
          beginAtZero: true, 
          max: 1,
          ticks: {
            color: 'rgba(203, 213, 225, 0.8)',
            font: { size: 11 }
          },
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          }
        },
        x: {
          ticks: {
            color: 'rgba(203, 213, 225, 0.8)',
            font: { size: 11 }
          },
          grid: {
            display: false
          }
        }
      }, 
      plugins: { 
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(30, 41, 59, 0.95)',
          titleColor: 'rgba(241, 245, 249, 1)',
          bodyColor: 'rgba(203, 213, 225, 1)',
          borderColor: 'rgba(99, 102, 241, 0.3)',
          borderWidth: 1,
          padding: 12,
          displayColors: true,
          callbacks: {
            label: function(context) {
              return `${context.label}: ${(context.parsed.y * 100).toFixed(1)}%`;
            }
          }
        }
      } 
    }
  };
  if (state.charts.proba) state.charts.proba.destroy();
  state.charts.proba = new Chart(ctx, config);
}

function renderInputRadar(payload) {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;
  // Take up to 8 numeric features for readability
  const numeric = state.features.filter(f => f.type === 'numeric').slice(0, 8);
  const labels = numeric.map(f => f.name);
  const vals = numeric.map(f => {
    const v = payload[f.name];
    const n = typeof v === 'string' ? Number(v) : v;
    return Number.isFinite(n) ? n : 0;
  });
  // Simple normalization to 0..1 by dividing by (abs) max among selected
  const maxAbs = Math.max(1, ...vals.map(v => Math.abs(v)));
  const norm = vals.map(v => (v / maxAbs));
  const config = {
    type: 'radar',
    data: { 
      labels, 
      datasets: [{ 
        label: 'Normalized inputs', 
        data: norm, 
        backgroundColor: 'rgba(99, 102, 241, 0.2)', 
        borderColor: 'rgba(99, 102, 241, 0.8)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(99, 102, 241, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
      }] 
    },
    options: { 
      responsive: true,
      maintainAspectRatio: false,
      scales: { 
        r: { 
          suggestedMin: -1, 
          suggestedMax: 1, 
          angleLines: { 
            color: 'rgba(148, 163, 184, 0.1)',
            lineWidth: 1
          }, 
          grid: { 
            color: 'rgba(148, 163, 184, 0.1)',
            lineWidth: 1
          },
          pointLabels: {
            color: 'rgba(203, 213, 225, 0.9)',
            font: { size: 11 }
          },
          ticks: {
            color: 'rgba(148, 163, 184, 0.6)',
            font: { size: 10 },
            backdropColor: 'transparent'
          }
        } 
      }, 
      plugins: { 
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(30, 41, 59, 0.95)',
          titleColor: 'rgba(241, 245, 249, 1)',
          bodyColor: 'rgba(203, 213, 225, 1)',
          borderColor: 'rgba(99, 102, 241, 0.3)',
          borderWidth: 1,
          padding: 12
        }
      } 
    }
  };
  if (state.charts.radar) state.charts.radar.destroy();
  state.charts.radar = new Chart(ctx, config);
}

async function submitPrediction(evt) {
  evt.preventDefault();
  const btn = document.getElementById('submit-btn');
  const btnText = btn.querySelector('.btn-text');
  const out = document.getElementById('prediction-output');
  const result = document.getElementById('result');
  const resultBadge = document.getElementById('result-badge');
  
  try {
    btn.disabled = true;
    btnText.textContent = 'Predicting…';
    out.textContent = 'Processing prediction…';
    result.classList.remove('hidden');
    result.style.animation = 'fadeInUp 0.5s ease-out';

    const payload = buildPayload();
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Prediction failed');

    // Show scalar class if single value
    const pred = data.prediction;
    const classLabels = ['DEBRIS', 'PAYLOAD', 'ROCKET BODY', 'TBA'];
    let predictionText = '';
    let predictionClass = '';
    
    if (typeof pred === 'number' && pred >= 0 && pred < classLabels.length) {
      predictionClass = classLabels[pred];
      predictionText = `Predicted Class: ${predictionClass} (${pred})`;
      resultBadge.textContent = predictionClass;
      resultBadge.classList.add('show');
    } else {
      predictionText = typeof pred === 'number' || typeof pred === 'string' 
        ? String(pred) 
        : JSON.stringify(pred, null, 2);
      resultBadge.classList.remove('show');
    }
    
    out.textContent = predictionText;

    // Render charts with slight delay for smooth animation
    setTimeout(() => {
      renderProbabilityChart(data.probabilities || null);
      renderInputRadar(payload);
    }, 100);
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
    resultBadge.classList.remove('show');
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Run Prediction';
  }
}

function clearForm() {
  state.features.forEach(({ name }) => {
    const input = document.getElementById(`field-${name}`);
    if (input) {
      input.value = '';
      input.style.animation = 'none';
      setTimeout(() => {
        input.style.animation = '';
      }, 10);
    }
  });
  const out = document.getElementById('prediction-output');
  const result = document.getElementById('result');
  const resultBadge = document.getElementById('result-badge');
  out.textContent = '—';
  resultBadge.classList.remove('show');
  
  // Clear charts
  if (state.charts.proba) {
    state.charts.proba.destroy();
    state.charts.proba = null;
  }
  if (state.charts.radar) {
    state.charts.radar.destroy();
    state.charts.radar = null;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadSchema();
  document.getElementById('predict-form').addEventListener('submit', submitPrediction);
  document.getElementById('fill-example').addEventListener('click', fillExample);
  document.getElementById('clear-btn').addEventListener('click', clearForm);
});
