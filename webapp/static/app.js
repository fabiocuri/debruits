'use strict';

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  templates: [],
  images:    [],
  config:    null,
  filter:    { series: '', format: '', lang: '' },
  tab:       'templates',
  gen:       { series: '', format: 'postcard', lang: 'pt', side: 'both' },
  busy:      false,
};

// ── API helpers ────────────────────────────────────────────────────────────────
async function GET(url)       { return (await fetch(url)).json(); }
async function PUT(url, body) { return (await fetch(url, { method:'PUT',  headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })).json(); }
async function POST(url,body) { return (await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })).json(); }

// ── Formatting ─────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b > 1048576) return (b/1048576).toFixed(1)+' MB';
  if (b > 1024)    return (b/1024).toFixed(0)+' KB';
  return b+' B';
}
function fmtDate(iso) {
  return new Date(iso).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'});
}

// ── Boot ───────────────────────────────────────────────────────────────────────
async function boot() {
  const [templates, images, config] = await Promise.all([
    GET('/api/templates'), GET('/api/images'), GET('/api/config'),
  ]);
  S.templates = templates;
  S.images    = images;
  S.config    = config;

  const seriesSet = new Set([
    ...templates.map(t => t.series),
    ...images.map(i => i.series),
  ].filter(Boolean));
  const allSeries = [...seriesSet].sort();

  buildSeriesPills(allSeries);
  buildGenSeries(allSeries);
  updateStats();
  renderTemplates();
  renderImages();
  renderParams();
}

// ── Stats line in header ───────────────────────────────────────────────────────
function updateStats() {
  const seriesSet = new Set(S.templates.map(t => t.series).filter(Boolean));
  const t = S.templates.length;
  const s = seriesSet.size;
  document.getElementById('header-stats').textContent =
    `${t} template${t!==1?'s':''} · ${s} series`;
}

// ── Filter pills ───────────────────────────────────────────────────────────────
function buildSeriesPills(series) {
  const c = document.getElementById('pills-series');
  c.innerHTML = `<button class="pill active" data-filter="series" data-val="">All</button>` +
    series.map(s => `<button class="pill" data-filter="series" data-val="${s}">${s}</button>`).join('');
  c.querySelectorAll('.pill').forEach(wireFilterPill);
}

function wireFilterPill(btn) {
  btn.addEventListener('click', () => {
    const key = btn.dataset.filter;
    document.querySelectorAll(`[data-filter="${key}"]`)
      .forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    S.filter[key] = btn.dataset.val;
    renderTemplates();
    renderImages();
  });
}

function wireStaticPills() {
  document.querySelectorAll('[data-filter]').forEach(wireFilterPill);
}

// ── Template grid ──────────────────────────────────────────────────────────────
function filteredTemplates() {
  return S.templates.filter(t => {
    if (S.filter.series && t.series !== S.filter.series) return false;
    if (S.filter.format && t.format !== S.filter.format) return false;
    if (S.filter.lang   && t.lang   !== S.filter.lang)   return false;
    return true;
  });
}

function renderTemplates() {
  const grid  = document.getElementById('grid-templates');
  const count = document.getElementById('count-templates');
  const items = filteredTemplates();

  count.textContent = `${items.length} template${items!==1?'s':''}`;

  if (!items.length) {
    grid.innerHTML = '<div class="empty">No templates yet — click + Generate</div>';
    return;
  }

  grid.innerHTML = items.map((t, i) => templateCard(t, i)).join('');

  grid.querySelectorAll('.card').forEach((card, i) => {
    const t = items[i];
    card.addEventListener('click', () => openLb(t));
    card.querySelector('.act-view')?.addEventListener('click', e => { e.stopPropagation(); openLb(t); });
    card.querySelector('.act-regen')?.addEventListener('click', e => { e.stopPropagation(); regen(t); });
  });
}

function templateCard(t) {
  const sideLabel = t.format==='fanzine' ? `Page ${t.page??'?'}` : (t.side??'').toUpperCase();
  const sideClass = t.side==='front'?'front': t.side==='back'?'back':'page';
  const langBadge = t.lang ? `<span class="badge ${t.lang}">${t.lang.toUpperCase()}</span>` : '';
  const regenBtn  = t.format!=='fanzine'
    ? `<button class="act-btn act-regen">↺ Regen</button>` : '';

  return `<div class="card" style="animation-delay:${Math.random()*.08}s">
    <img class="card-thumb portrait" src="/img/${t.path}" loading="lazy" alt="${t.series}">
    <div class="card-actions">
      <button class="act-btn act-view">⤢ View</button>
      ${regenBtn}
    </div>
    <div class="card-info">
      <div class="card-series">${t.series??'—'}</div>
      <div class="badges">
        <span class="badge ${t.format??''}">${t.format??''}</span>
        ${langBadge}
        <span class="badge ${sideClass}">${sideLabel}</span>
      </div>
      <div class="card-meta">${t.width}×${t.height} · ${fmtBytes(t.size_bytes)}<br>${fmtDate(t.modified)}</div>
    </div>
  </div>`;
}

// ── Images grid ────────────────────────────────────────────────────────────────
function renderImages() {
  const grid  = document.getElementById('grid-images');
  const count = document.getElementById('count-images');
  const items = S.images.filter(i =>
    !S.filter.series || i.series === S.filter.series
  );

  count.textContent = `${items.length} composite image${items.length!==1?'s':''}`;

  if (!items.length) {
    grid.innerHTML = '<div class="empty">No images in Final/</div>';
    return;
  }

  grid.innerHTML = items.map(img => `
    <div class="card" style="animation-delay:${Math.random()*.08}s">
      <img class="card-thumb square" src="/img/${img.path}" loading="lazy" alt="${img.series}">
      <div class="card-info">
        <div class="card-series">${img.series} <span style="color:var(--txt3);font-size:10px">#${img.index}</span></div>
        <div class="card-meta">${img.width}×${img.height} · ${fmtBytes(img.size_bytes)}<br>${fmtDate(img.modified)}</div>
      </div>
    </div>`).join('');

  grid.querySelectorAll('.card').forEach((card, i) => {
    card.addEventListener('click', () => openLb(items[i]));
  });
}

// ── Lightbox ───────────────────────────────────────────────────────────────────
function openLb(item) {
  document.getElementById('lb-img').src = `/img/${item.path}`;
  const parts = [
    item.series, item.format, item.lang?.toUpperCase(), item.side,
    item.page ? `page ${item.page}` : null,
    `${item.width}×${item.height}`, fmtBytes(item.size_bytes), fmtDate(item.modified),
  ].filter(Boolean);
  document.getElementById('lb-bar').textContent = parts.join('  ·  ');
  document.getElementById('lightbox').classList.remove('hidden');
}

function closeLb() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lb-img').src = '';
}

// ── Parameters ─────────────────────────────────────────────────────────────────
const PARAM_GROUPS = [
  {
    title: 'FRONT — Image Grid',
    params: [
      { key:'front_margin', label:'Outer margin',   min:.005, max:.10,  step:.005 },
      { key:'front_gap',    label:'Cell gap',       min:.002, max:.05,  step:.002 },
      { key:'stamp_right',  label:'Stamp → right',  min:.005, max:.10,  step:.005 },
      { key:'stamp_bottom', label:'Stamp → bottom', min:.002, max:.05,  step:.002 },
      { key:'stamp_size',   label:'Stamp size',     min:.02,  max:.15,  step:.005 },
    ],
  },
  {
    title: 'BACK — Divider Lines',
    params: [
      { key:'vline_x',  label:'Vert. line x',     min:.5,  max:1.0, step:.005 },
      { key:'vline_y0', label:'Vert. line start',  min:.0,  max:.5,  step:.01  },
      { key:'vline_y1', label:'Vert. line end',    min:.5,  max:1.0, step:.01  },
      { key:'hline_y',  label:'Horiz. line y',     min:.7,  max:1.0, step:.005 },
      { key:'hline_x0', label:'Horiz. start x',    min:.0,  max:.4,  step:.005 },
      { key:'hline_x1', label:'Horiz. end x',      min:.1,  max:.7,  step:.005 },
      { key:'line_width', label:'Line width (px)', min:1,   max:6,   step:1, integer:true },
    ],
  },
  {
    title: 'BACK — Title Text',
    params: [
      { key:'title_x',    label:'Title left x',  min:.0,  max:.4,  step:.005 },
      { key:'title_y',    label:'Title top y',   min:.7,  max:1.0, step:.005 },
      { key:'title_size', label:'Font size',     min:.008,max:.05, step:.002 },
    ],
  },
];

function renderParams() {
  if (!S.config) return;
  const body   = document.getElementById('params-body');
  const layout = S.config.layout;

  let html = '';

  // Layout sliders
  for (const g of PARAM_GROUPS) {
    html += `<div class="params-section">
      <div class="section-title">${g.title}</div>`;
    for (const p of g.params) {
      const v   = layout[p.key] ?? 0;
      const disp = p.integer ? v : v.toFixed(3);
      html += `<div class="param-row">
        <span class="param-label">${p.label}</span>
        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${v}"
          data-key="${p.key}" data-int="${!!p.integer}">
        <span class="param-val" id="pv-${p.key}">${disp}</span>
      </div>`;
    }
    html += '</div>';
  }

  // Canvas sizes (formats)
  html += `<div class="params-section">
    <div class="section-title">FORMAT CANVAS SIZE</div>
    <table class="fmt-table">
      <thead><tr><th>Format</th><th>Width px</th><th>Height px</th></tr></thead>
      <tbody>`;
  for (const [fmt, cfg] of Object.entries(S.config.formats)) {
    const [w, h] = cfg.size_px;
    html += `<tr>
      <td>${fmt}</td>
      <td><input class="num-input" type="number" data-fmt="${fmt}" data-dim="0" value="${w}" min="100" max="9999"></td>
      <td><input class="num-input" type="number" data-fmt="${fmt}" data-dim="1" value="${h}" min="100" max="9999"></td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Series titles
  html += `<div class="params-section">
    <div class="section-title">SERIES TITLES</div>
    <table class="titles-table">
      <thead><tr><th>Series</th><th>Português</th><th>English</th></tr></thead>
      <tbody>`;
  for (const [series, langs] of Object.entries(S.config.series_titles)) {
    html += `<tr>
      <td>${series}</td>
      <td><input class="txt-input" type="text" data-series="${series}" data-lang="pt" value="${langs.pt??''}"></td>
      <td><input class="txt-input" type="text" data-series="${series}" data-lang="en" value="${langs.en??''}"></td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  html += `<div class="params-actions">
    <button class="btn-primary" id="btn-save">Save Config</button>
    <button class="btn-secondary" id="btn-reset">Reset</button>
  </div>`;

  body.innerHTML = html;

  // Wire sliders
  body.querySelectorAll('input[type=range]').forEach(sl => {
    sl.addEventListener('input', () => {
      const k = sl.dataset.key;
      const v = sl.dataset.int==='true' ? parseInt(sl.value) : parseFloat(sl.value);
      document.getElementById(`pv-${k}`).textContent = sl.dataset.int==='true' ? v : v.toFixed(3);
      S.config.layout[k] = v;
    });
  });

  // Wire format size inputs
  body.querySelectorAll('[data-fmt]').forEach(inp => {
    inp.addEventListener('input', () => {
      const fmt = inp.dataset.fmt;
      const dim = parseInt(inp.dataset.dim);
      const v   = parseInt(inp.value) || 0;
      S.config.formats[fmt].size_px[dim] = v;
    });
  });

  // Wire titles
  body.querySelectorAll('[data-series]').forEach(inp => {
    inp.addEventListener('input', () => {
      const s = inp.dataset.series, l = inp.dataset.lang;
      if (!S.config.series_titles[s]) S.config.series_titles[s] = {};
      S.config.series_titles[s][l] = inp.value;
    });
  });

  document.getElementById('btn-save').addEventListener('click', saveConfig);
  document.getElementById('btn-reset').addEventListener('click', () => boot());
}

async function saveConfig() {
  const btn = document.getElementById('btn-save');
  btn.disabled = true; btn.textContent = 'Saving…';
  await PUT('/api/config', S.config);
  btn.disabled = false; btn.textContent = 'Saved ✓';
  setTimeout(() => { btn.textContent = 'Save Config'; }, 2200);
}

// ── Generate modal ─────────────────────────────────────────────────────────────
function buildGenSeries(series) {
  const c = document.getElementById('gen-pills-series');
  c.innerHTML = series.map((s, i) =>
    `<button class="pill${i===0?' active':''}" data-gen="series" data-val="${s}">${s}</button>`
  ).join('');
  if (series[0]) S.gen.series = series[0];
  c.querySelectorAll('.pill').forEach(wireGenPill);
}

function wireGenPill(btn) {
  btn.addEventListener('click', () => {
    const key = btn.dataset.gen;
    document.querySelectorAll(`[data-gen="${key}"]`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    S.gen[key] = btn.dataset.val;

    // hide side selector for fanzine
    if (key === 'format') {
      document.getElementById('gen-side-row').style.display =
        btn.dataset.val === 'fanzine' ? 'none' : '';
    }
  });
}

function openGenModal() {
  document.getElementById('gen-log').className = 'gen-log hidden';
  document.getElementById('gen-backdrop').classList.remove('hidden');

  // Pre-fill series from active filter
  if (S.filter.series) {
    document.querySelectorAll('[data-gen="series"]').forEach(b => {
      b.classList.toggle('active', b.dataset.val === S.filter.series);
      if (b.dataset.val === S.filter.series) S.gen.series = S.filter.series;
    });
  }
}

function closeGenModal() {
  document.getElementById('gen-backdrop').classList.add('hidden');
}

async function runGenerate() {
  if (S.busy) return;
  S.busy = true;
  const btn = document.getElementById('gen-run');
  const log = document.getElementById('gen-log');
  btn.disabled = true; btn.textContent = 'Generating…';
  log.className = 'gen-log'; log.textContent = 'Running…';

  try {
    const res = await POST('/api/generate', S.gen);
    log.textContent = (res.stdout + (res.stderr ? '\n' + res.stderr : '')).trim();
    log.className   = 'gen-log ' + (res.ok ? 'ok' : 'fail');

    if (res.ok) {
      S.templates = await GET('/api/templates');
      updateStats();
      renderTemplates();
    }
  } catch (e) {
    log.className = 'gen-log fail';
    log.textContent = 'Error: ' + e.message;
  }

  S.busy = false;
  btn.disabled = false; btn.textContent = 'Generate';
}

async function regen(t) {
  if (t.format === 'fanzine') return;
  const side = t.side === 'front' ? 'front' : t.side === 'back' ? 'back' : 'both';
  const res = await POST('/api/generate', { series:t.series, format:t.format, lang:t.lang, side });
  if (res.ok) {
    S.templates = await GET('/api/templates');
    renderTemplates();
  }
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.getElementById('tabs').querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
      S.tab = tab.dataset.tab;

      const hideFilter = ['params', 'info'].includes(tab.dataset.tab);
      document.getElementById('filter-bar').style.display = hideFilter ? 'none' : 'flex';
      document.querySelector('.main').classList.toggle('no-pad', tab.dataset.tab === 'info');

      if (tab.dataset.tab === 'info') {
        const iframe = document.getElementById('info-frame');
        const notice = document.getElementById('info-notice');
        if (!iframe.dataset.loaded) {
          GET('/api/config').then(cfg => {
            const url = cfg.info_dashboard_url || 'http://localhost:5001';
            document.getElementById('info-url').textContent = url;
            document.getElementById('info-open').href = url;
            document.getElementById('info-cmd').textContent =
              `cd ~/Desktop/interests-info-dashboard\nuvicorn app.main:app --port 5001`;
            iframe.dataset.loaded = '1';
            iframe.onload = () => {
              try {
                void iframe.contentWindow.location.href;
                notice.classList.add('hidden');
              } catch (_) {
                notice.classList.remove('hidden');
              }
            };
            iframe.onerror = () => notice.classList.remove('hidden');
            iframe.src = url;
          });
        }
      }
    });
  });
}

// ── Init ───────────────────────────────────────────────────────────────────────
function init() {
  setupTabs();
  wireStaticPills();

  document.getElementById('btn-generate').addEventListener('click', openGenModal);
  document.getElementById('gen-close').addEventListener('click', closeGenModal);
  document.getElementById('gen-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('gen-backdrop')) closeGenModal();
  });
  document.getElementById('gen-run').addEventListener('click', runGenerate);

  document.querySelectorAll('[data-gen]').forEach(wireGenPill);

  document.getElementById('lightbox').addEventListener('click', closeLb);
  document.getElementById('lb-close').addEventListener('click', e => { e.stopPropagation(); closeLb(); });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeLb(); closeGenModal(); }
  });

  boot();
}

document.addEventListener('DOMContentLoaded', init);
