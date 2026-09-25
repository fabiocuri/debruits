'use strict';

// Spread dimensions (2 pages side by side).
// mini-poster pages are A5 → spread = A4 landscape (297 × 210 mm)
// poster pages are A4      → spread = A3 landscape (420 × 297 mm)
const ZINE_FORMATS = {
  'mini-poster': { paperW: 297 / 25.4, paperH: 210 / 25.4 },  // A4 landscape
  'poster':      { paperW: 420 / 25.4, paperH: 297 / 25.4 },  // A3 landscape
};

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  prints:            [],
  images:            [],
  thumbnails:        {},
  config:            null,
  filter:            { series: '', format: '', lang: '', icon: '' },
  tab:               'prints',
  gen:               { series: '', format: 'postcard', lang: 'pt', count: null, countMode: 'all', icon: null, iconScale: 3, selectedImage: null, customTitle: null, bw: false },
  busy:              false,
  activeItemIdx:     null,
  imagesSeries:      null,
  // Zines state
  zinesImageSeries:  null,
  zinesImageSelected: new Set(),
  zines:             [],
  selectedZines:     new Set(),
  // Zine preview (step 2)
  zineSeriesFilter:  '',
  zineFormatFilter:  '',
  zineLangFilter:    '',
  zineIconFilter:    '',
  czineFormat:       'mini-poster',
  czineLang:         'pt',
  czineLayout:       [],
  czineTexts:        [],
  czineSize:         null,
  czineSpread:       0,
  czineActivePgIdx:  null,
  czineBw:           false,
  czineIcon:         null,
  czineCoverPath:    null,
  czineSeries:       '',
  czineEditName:     null,
  czineGlobalScale:  100,
  czineCoverScale:   100,
  printGlobalScale:  100,
  layout:            null,
  layoutSize:        null,
  stamp:             null,
  previewPath:       null,
  selected:          new Set(),
};

// ── API helpers ────────────────────────────────────────────────────────────────
async function GET(url)       { return (await fetch(url, { cache: 'no-store' })).json(); }
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
  const [prints, images, config] = await Promise.all([
    GET('/api/templates'), GET('/api/images'), GET('/api/config'),
  ]);
  S.prints    = prints;
  S.images    = images;
  S.config    = config;
  try { S.thumbnails = await GET('/api/thumbnails'); } catch (_) { S.thumbnails = {}; }

  const imageSeries = [...new Set(images.map(i => i.series).filter(Boolean))].sort();
  buildGenSeries(imageSeries);
  refreshPrintSeries();
  GET('/api/icons').then(buildIconPicker);
  renderPrints();
  renderText();
  renderImages();
}

// ── Shared toolbar builder ───────────────────────────────────────────────────────
// Both panels have the same 7-button toolbar; only the "+ Create" label/handler and
// the panel-scoped handlers differ. `opts` supplies those differences.
function buildToolbar(panel, opts) {
  const container = document.getElementById(`toolbar-${panel}`);
  if (!container) return;
  container.innerHTML = '';

  const mkBtn = (cls, id, label, hidden = false) => {
    const b = document.createElement('button');
    b.className = cls + (hidden ? ' hidden' : '');
    b.id = id;
    b.textContent = label;
    return b;
  };

  const createBtn     = mkBtn('btn-primary',   `btn-create-${panel}`,      opts.createLabel);
  const downloadBtn   = mkBtn('btn-secondary', `btn-download-${panel}`,    'Download All');
  const titlesBtn     = mkBtn('btn-secondary', `btn-titles-${panel}`,      'Theme Titles');
  const descBtn       = mkBtn('btn-secondary', `btn-desc-${panel}`,        'Back Description');
  const refreshBtn    = mkBtn('btn-secondary', `btn-refresh-${panel}`,     'Refresh');
  const selectAllBtn  = mkBtn('btn-secondary', `btn-select-all-${panel}`,  'Select All', true);
  const deleteBtn     = mkBtn('btn-danger',    `btn-delete-${panel}`,      'Delete Selected', true);

  [createBtn, downloadBtn, titlesBtn, descBtn, refreshBtn, selectAllBtn, deleteBtn]
    .forEach(b => container.appendChild(b));

  createBtn.addEventListener('click', opts.onCreate);
  downloadBtn.addEventListener('click', () => opts.onDownloadAll(downloadBtn));
  titlesBtn.addEventListener('click', openTitlesModal);
  descBtn.addEventListener('click', opts.onDesc);
  refreshBtn.addEventListener('click', () => opts.onRefresh(refreshBtn));
  selectAllBtn.addEventListener('click', opts.onSelectAll);
  deleteBtn.addEventListener('click', opts.onDelete);
}

// ── Shared filter sidebar builder ────────────────────────────────────────────────
// context is 'prints' (data-filter) or 'zines' (data-zfilter). Icon group is present
// for both but only populated/shown when there are icons.
function buildFilterSidebar(container, context) {
  if (!container) return;
  const attr = context === 'zines' ? 'data-zfilter' : 'data-filter';

  const formats = context === 'zines'
    ? [['', 'All'], ['mini-poster', 'Mini Poster'], ['poster', 'Poster']]
    : [['', 'All'], ['postcard', 'Postcard'], ['mini-poster', 'Mini Poster'], ['poster', 'Poster']];
  const langs = [['', 'All'], ['pt', 'PT'], ['en', 'EN'], ['es', 'ES'], ['fr', 'FR']];

  const seriesPillsId = context === 'zines' ? 'zine-pills-series' : 'pills-series';
  const iconPillsId   = context === 'zines' ? 'zine-pills-icon'   : 'pills-icon';
  const iconGroupId   = context === 'zines' ? 'zine-filter-group-icon' : 'filter-group-icon';
  const iconGroupHidden = context === 'zines' ? ' hidden' : '';

  const pillRow = pairs => pairs.map(([val, label], i) =>
    `<button class="pill${i === 0 ? ' active' : ''}" ${attr}="{K}" data-val="${val}">${label}</button>`
  ).join('');

  container.innerHTML = `
    <div class="filter-group">
      <span class="filter-label">Theme</span>
      <div class="pills" id="${seriesPillsId}"></div>
    </div>
    <div class="filter-group">
      <span class="filter-label">Format</span>
      <div class="pills">${pillRow(formats).replace(/\{K\}/g, 'format')}</div>
    </div>
    <div class="filter-group">
      <span class="filter-label">Lang</span>
      <div class="pills">${pillRow(langs).replace(/\{K\}/g, 'lang')}</div>
    </div>
    <div class="filter-group${iconGroupHidden}" id="${iconGroupId}">
      <span class="filter-label">Icon</span>
      <div class="pills" id="${iconPillsId}"></div>
    </div>`;
}

// ── Filter pills ───────────────────────────────────────────────────────────────
function refreshPrintSeries() {
  const series = [...new Set(S.prints.map(t => t.series).filter(Boolean))].sort();
  buildSeriesPills(series);
  buildIconPills();
}

function buildIconPills() {
  const icons = [...new Set(S.prints.filter(t => t.icon).map(t => t.icon))].sort();
  const c = document.getElementById('pills-icon');
  c.innerHTML =
    `<button class="pill active" data-filter="icon" data-val="">All</button>` +
    icons.map(ic =>
      `<button class="pill pill-icon" data-filter="icon" data-val="${ic}" title="${ic.replace(/\.[^.]+$/, '')}">
        <img src="/icon/${ic}" alt="${ic}">
      </button>`
    ).join('');
  c.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      c.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      S.filter.icon = btn.dataset.val;
      renderPrints();
    });
  });
}

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
    renderPrints();
  });
}

function wireStaticPills() {
  document.querySelectorAll('[data-filter]').forEach(wireFilterPill);
}

// ── Template grid ──────────────────────────────────────────────────────────────
function filteredPrints() {
  return S.prints.filter(t => {
    if (S.filter.series && t.series !== S.filter.series) return false;
    if (S.filter.format && t.format !== S.filter.format) return false;
    if (S.filter.lang   && t.lang   !== S.filter.lang)   return false;
    if (S.filter.icon   && t.icon   !== S.filter.icon)   return false;
    return true;
  });
}

function syncDeleteBtn() {
  document.getElementById('btn-delete-prints').classList.toggle('hidden', S.selected.size === 0);
}

function syncZineDeleteBtn() {
  document.getElementById('btn-delete-zines').classList.toggle('hidden', S.selectedZines.size === 0);
}

function renderPrints() {
  const grid  = document.getElementById('grid-prints');
  const count = document.getElementById('count-prints');
  const items = filteredPrints();

  count.textContent = `${items.length} ${items.length === 1 ? 'print' : 'prints'}`;
  document.getElementById('btn-select-all-prints').classList.toggle('hidden', !items.length);

  if (!items.length) {
    grid.innerHTML = '';
    return;
  }

  grid.innerHTML = items.map(t => printCard(t)).join('');

  grid.querySelectorAll('.card').forEach((card, i) => {
    const t = items[i];
    card.addEventListener('click', () => openLb(t));
    card.querySelector('.act-view')?.addEventListener('click', e => { e.stopPropagation(); openLb(t); });
    card.querySelector('.act-edit')?.addEventListener('click', e => { e.stopPropagation(); openEditModal(t); });
    card.querySelector('.act-dl')?.addEventListener('click', e => e.stopPropagation());

    const cb = card.querySelector('.card-check input');
    cb.addEventListener('change', () => {
      if (cb.checked) S.selected.add(t.path);
      else            S.selected.delete(t.path);
      card.classList.toggle('selected', cb.checked);
      syncDeleteBtn();
    });
    cb.addEventListener('click', e => e.stopPropagation());
  });
}

function printCard(t) {
  const thumbSrc = t.thumb ? `/img/${t.thumb}` : `/img/${t.path}`;
  const checked  = S.selected.has(t.path) ? 'checked' : '';
  const langBadge = t.lang ? `<span class="badge ${t.lang}">${t.lang.toUpperCase()}</span>` : '';

  return `<div class="card${S.selected.has(t.path) ? ' selected' : ''}" style="animation-delay:${Math.random()*.08}s">
    <label class="card-check" title="Select"><input type="checkbox" ${checked}></label>
    <img class="card-thumb portrait" src="${thumbSrc}" loading="lazy" alt="${t.series}">
    <div class="card-actions">
      <button class="act-btn act-edit">Edit</button>
      <button class="act-btn act-view">View</button>
      <a class="act-btn act-dl" href="/api/download/${t.path}">PDF</a>
    </div>
    <div class="card-info">
      <div class="card-series">${t.series??'—'}</div>
      <div class="badges">
        <span class="badge ${t.format??''}">${t.format??''}</span>
        ${langBadge}
      </div>
      <div class="card-meta">${fmtBytes(t.size_bytes)}<br>${fmtDate(t.modified)}</div>
    </div>
  </div>`;
}

// ── Images grid ────────────────────────────────────────────────────────────────
function renderImages() {
  const grid  = document.getElementById('grid-images');
  const count = document.getElementById('count-images');

  if (S.imagesSeries === null) {
    // Series overview
    const byS = {};
    S.images.forEach(img => {
      if (!byS[img.series]) byS[img.series] = [];
      byS[img.series].push(img);
    });
    const series = Object.keys(byS).sort();

    if (count) { count.textContent = ''; count.classList.add('hidden'); }

    if (!series.length) {
      grid.innerHTML = '<div class="empty">No images in Final/</div>';
      return;
    }

    grid.innerHTML = series.map(s => {
      const imgs = byS[s];
      const thumbPath = S.thumbnails[s] || imgs[0].path;
      return `<div class="img-tile" data-series="${s}">
        <img src="/img/${thumbPath}" loading="lazy" alt="${s}">
        <div class="img-tile-overlay">
          <div class="img-tile-name">${s}</div>
        </div>
      </div>`;
    }).join('');

    grid.querySelectorAll('.img-tile').forEach(tile => {
      tile.addEventListener('click', () => {
        S.imagesSeries = tile.dataset.series;
        renderImages();
      });
    });

  } else {
    // Image detail view — images for one series
    const items = S.images.filter(i => i.series === S.imagesSeries);
    if (count) {
      count.innerHTML = `<button class="back-btn" id="images-back">Back</button> <span>${S.imagesSeries} — ${items.length} ${items.length === 1 ? 'image' : 'images'}</span>`;
      count.classList.remove('hidden');
      document.getElementById('images-back').addEventListener('click', () => {
        S.imagesSeries = null;
        renderImages();
      });
    }

    const coverPath = S.thumbnails[S.imagesSeries] || items[0]?.path;

    grid.innerHTML = items.map(img => {
      const isCover = img.path === coverPath;
      return `<div class="img-tile img-card">
        <img src="/img/${img.path}" loading="lazy" alt="${img.series}">
        <button class="set-cover-btn${isCover ? ' is-cover' : ''}" data-path="${img.path}" title="Set as theme cover">◉</button>
        <div class="img-tile-overlay">
          <div class="img-tile-meta">${img.width}x${img.height}</div>
          <div class="img-tile-meta">${fmtBytes(img.size_bytes)}</div>
        </div>
      </div>`;
    }).join('');

    grid.querySelectorAll('.img-card').forEach((tile, i) => {
      tile.addEventListener('click', e => {
        if (e.target.closest('.set-cover-btn')) return;
        openLb(items[i]);
      });
      tile.querySelector('.set-cover-btn').addEventListener('click', async e => {
        e.stopPropagation();
        const path = e.currentTarget.dataset.path;
        S.thumbnails[S.imagesSeries] = path;
        await PUT('/api/thumbnails', { series: S.imagesSeries, path });
        renderImages();
      });
    });
  }
}

// ── Lightbox ───────────────────────────────────────────────────────────────────
function openLb(item) {
  const lbV = item.modified ? `${item.modified}`.replace(/\D/g,'') : Date.now();
  const lbPath = item.thumb ? `/img/${item.thumb}` : `/img/${item.path}`;
  document.getElementById('lb-img').src = `${lbPath}?v=${lbV}`;
  const parts = [
    item.series, item.format, item.lang?.toUpperCase(),
    item.page ? `page ${item.page}` : null,
    item.width ? `${item.width}x${item.height}` : null,
    fmtBytes(item.size_bytes), fmtDate(item.modified),
  ].filter(Boolean);
  document.getElementById('lb-bar').textContent = parts.join('  ·  ');
  document.getElementById('lightbox').classList.remove('hidden');
}

function closeLb() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lb-img').src = '';
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

function buildIconPicker(icons) {
  [
    { containerId: 'gen-icon-picker',   onSelect: name => { S.gen.icon = name || null; } },
    { containerId: 'czine-icon-picker', onSelect: name => { S.czineIcon = name || null; renderCzineSpread(S.czineSpread); } },
  ].forEach(({ containerId, onSelect }) => {
    const c = document.getElementById(containerId);
    if (!c) return;
    const noneEl = document.createElement('div');
    noneEl.className = 'icon-opt active';
    noneEl.dataset.icon = '';
    noneEl.textContent = 'None';
    noneEl.addEventListener('click', () => selectIcon('', c, onSelect));
    c.appendChild(noneEl);
    icons.forEach(ic => {
      const el = document.createElement('img');
      el.className = 'icon-opt';
      el.src = `/${ic.url}`;
      el.title = ic.name.replace(/\.png$/i, '');
      el.dataset.icon = ic.name;
      el.addEventListener('click', () => selectIcon(ic.name, c, onSelect));
      c.appendChild(el);
    });
  });
}

function selectIcon(name, container, onSelect) {
  container.querySelectorAll('.icon-opt').forEach(el => el.classList.remove('active'));
  container.querySelector(`[data-icon="${name}"]`).classList.add('active');
  onSelect(name);
}

function selectCzineIcon(name) {
  const c = document.getElementById('czine-icon-picker');
  if (!c) return;
  c.querySelectorAll('.icon-opt').forEach(el => el.classList.remove('active'));
  const el = c.querySelector(`[data-icon="${name}"]`);
  if (el) el.classList.add('active');
  S.czineIcon = name || null;
}

function wireGenPill(btn) {
  btn.addEventListener('click', () => {
    const key = btn.dataset.gen;
    document.querySelectorAll(`[data-gen="${key}"]`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    S.gen[key] = btn.dataset.val;

    if (key === 'count-mode') {
      S.gen.countMode = btn.dataset.val;
      setImagePickerVisibility(S.gen.series);
    }

    if (key === 'lang') buildBackTextSelect(S.gen.series, btn.dataset.val);
    if (key === 'series') {
      updateCountRow(btn.dataset.val);
      buildBackTextSelect(btn.dataset.val, S.gen.lang);
    }
  });
}

function buildBackTextSelect(series, lang) {
  const sel = document.getElementById('gen-back-text-select');
  const customInput = document.getElementById('gen-back-custom-input');

  const raw     = ((S.config?.series_titles || {})[series] || {})[lang];
  const options = Array.isArray(raw) ? raw.filter(Boolean) : (raw ? [raw] : []);

  sel.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join('')
    + `<option value="__custom__">Custom…</option>`;

  if (options.length) {
    sel.value = options[0];
    customInput.classList.add('hidden');
    customInput.value = '';
    S.gen.customTitle = options[0];
  } else {
    sel.value = '__custom__';
    customInput.classList.remove('hidden');
    S.gen.customTitle = null;
  }
}

function updateCountRow(series) {
  const n = S.images.filter(i => i.series === series).length;
  const row = document.getElementById('gen-count-row');
  if (n < 1) { row.classList.add('hidden'); return; }
  row.classList.remove('hidden');
  S.gen.count = n;
  setImagePickerVisibility(series);
}

function setImagePickerVisibility(series) {
  const row = document.getElementById('gen-image-picker-row');
  const n = S.images.filter(i => i.series === series).length;
  if (S.gen.countMode === 'one') {
    S.gen.count = 1;
    row.classList.remove('hidden');
    buildImagePicker(series);
  } else {
    S.gen.count = n;
    row.classList.add('hidden');
    S.gen.selectedImage = null;
  }
}

function buildImagePicker(series) {
  const container = document.getElementById('gen-image-picker');
  const imgs = S.images
    .filter(i => i.series === series)
    .sort((a, b) => a.index - b.index);
  container.innerHTML = '';
  // Auto-select first so there's always a valid selection before Preview is clicked
  S.gen.selectedImage = imgs[0]?.path || null;

  imgs.forEach((img, i) => {
    const el = document.createElement('img');
    el.src       = `/img/${img.path}`;
    el.title     = img.filename;
    el.className = 'image-picker-opt' + (i === 0 ? ' active' : '');
    el.addEventListener('click', () => {
      container.querySelectorAll('.image-picker-opt').forEach(x => x.classList.remove('active'));
      el.classList.add('active');
      S.gen.selectedImage = img.path;
    });
    container.appendChild(el);
  });
}

function openGenModal() {
  document.getElementById('gen-modal-title').textContent = 'Create';
  showGenStep(1);
  document.getElementById('gen-log').className = 'gen-log hidden';
  document.getElementById('gen-backdrop').classList.remove('hidden');

  if (S.filter.series) {
    document.querySelectorAll('[data-gen="series"]').forEach(b => {
      b.classList.toggle('active', b.dataset.val === S.filter.series);
      if (b.dataset.val === S.filter.series) S.gen.series = S.filter.series;
    });
  }
  if (S.gen.series) {
    updateCountRow(S.gen.series);
    buildBackTextSelect(S.gen.series, S.gen.lang);
  }
}

function closeGenModal() {
  document.getElementById('gen-backdrop').classList.add('hidden');
  document.getElementById('gen-modal').classList.remove('modal-wide');
  document.getElementById('gen-modal-title').textContent = 'Create';
  S.gen.selectedImage = null;
  document.getElementById('gen-image-picker-row').classList.add('hidden');
  S.gen.customTitle = null;
}

async function openEditModal(t) {
  S.gen.series        = t.series   || '';
  S.gen.format        = t.format   || 'postcard';
  S.gen.lang          = t.lang     || 'pt';
  S.gen.icon          = null;
  S.gen.iconScale     = 3;
  S.gen.selectedImage = null;

  ['series', 'format', 'lang'].forEach(key => {
    document.querySelectorAll(`[data-gen="${key}"]`).forEach(b =>
      b.classList.toggle('active', b.dataset.val === S.gen[key]));
  });

  updateCountRow(S.gen.series);
  buildBackTextSelect(S.gen.series, S.gen.lang);

  const picker = document.getElementById('gen-icon-picker');
  if (picker) selectIcon('', picker, name => { S.gen.icon = name || null; });
  document.getElementById('gen-log').className = 'gen-log hidden';

  document.getElementById('gen-modal-title').textContent = 'Edit';
  document.getElementById('gen-preview-title').textContent =
    `${t.series} · ${t.format}` + (t.lang ? ` · ${t.lang.toUpperCase()}` : '');

  // Always show flat image first as placeholder
  document.getElementById('gen-canvas-wrap').classList.add('hidden');
  const prevImg = document.getElementById('gen-preview-img');
  prevImg.classList.remove('hidden');
  prevImg.src = `/img/${t.path}?t=${Date.now()}`;

  showGenStep(2);
  document.getElementById('gen-backdrop').classList.remove('hidden');

  // Try to restore the saved layout (no new generation, no disk write)
  try {
    const data = await GET(`/api/layout-data/${t.path}`);
    if (data && data.layout && data.layout.length) {
      S.layout     = data.layout;
      S.layoutSize = data.size_px;
      S.stamp      = data.stamp || null;
      if (data.icon) { S.gen.icon = data.icon; S.gen.iconScale = 3; }
      resetPrintScale();
      prevImg.classList.add('hidden');
      document.getElementById('gen-canvas-wrap').classList.remove('hidden');
      renderDraggablePreview();
      return;
    }
  } catch (_) {}
  // No saved layout — generate a preview-only draft (nothing written to disk)
  runPreview(true);
}

function resetPrintScale() {
  S.printGlobalScale = 100;
  document.getElementById('print-scale-slider').value = 100;
  document.getElementById('print-scale-val').textContent = '100%';
}

function showGenStep(n) {
  document.getElementById('gen-step-1').classList.toggle('hidden', n !== 1);
  document.getElementById('gen-step-2').classList.toggle('hidden', n !== 2);
  document.getElementById('gen-modal').classList.toggle('modal-wide', n === 2);
}

async function runPreview(previewOnly = false) {
  if (S.busy) return;
  if (!S.gen.series) {
    document.getElementById('gen-log').className = 'gen-log fail';
    document.getElementById('gen-log').textContent = 'Select a series first.';
    return;
  }
  S.busy = true;
  const btn = document.getElementById('gen-preview-btn');
  const log = document.getElementById('gen-log');
  btn.disabled = true; btn.textContent = 'Generating…';
  log.className = 'gen-log'; log.textContent = 'Running…';

  try {
    // Snap back-text selection from DOM so stale state never reaches the backend
    const sel = document.getElementById('gen-back-text-select');
    const customInput = document.getElementById('gen-back-custom-input');
    if (sel) {
      S.gen.customTitle = sel.value === '__custom__'
        ? (customInput.value.trim() || null)
        : (sel.value || null);
    }
    const res = await POST('/api/generate', { ...S.gen, previewOnly });
    const hasContent = res.ok && (res.previewOnly || (res.paths && res.paths.length));
    if (hasContent) {
      document.getElementById('gen-preview-title').textContent =
        `${S.gen.series} · ${S.gen.format} · ${S.gen.lang.toUpperCase()}`;
      log.className = 'gen-log hidden';

      if (res.layout && res.layout.length) {
        S.layout      = res.layout;
        S.layoutSize  = res.size_px;
        S.stamp       = res.stamp || null;
        S.previewPath = res.paths[0] || null;
        resetPrintScale();
        document.getElementById('gen-preview-img').classList.add('hidden');
        document.getElementById('gen-canvas-wrap').classList.remove('hidden');
        renderDraggablePreview();
      } else {
        S.layout = null;
        document.getElementById('gen-canvas-wrap').classList.add('hidden');
        const img = document.getElementById('gen-preview-img');
        img.classList.remove('hidden');
        img.src = `/img/${res.paths[0]}?t=${Date.now()}`;
      }

      showGenStep(2);
      // Only reload template list when we actually saved something
      if (!previewOnly) {
        S.prints = await GET('/api/templates');
        if (S.gen.customTitle) S.config = await GET('/api/config');
        refreshPrintSeries();
        renderPrints();
        renderText();
      }
    } else {
      log.textContent = (res.stdout + '\n' + (res.stderr || '')).trim() || 'Generation failed.';
      log.className   = 'gen-log fail';
    }
  } catch (e) {
    log.className = 'gen-log fail';
    log.textContent = 'Error: ' + e.message;
  }

  S.busy = false;
  btn.disabled = false; btn.textContent = 'Preview >';
}

function renderDraggablePreview() {
  const wrap = document.getElementById('gen-canvas-wrap');
  wrap.innerHTML = '';

  const [FW, FH] = S.layoutSize;
  const maxW  = Math.min(660, window.innerWidth  - 80);
  const maxH  = Math.floor(window.innerHeight * 0.62);
  const scale = Math.min(maxW / FW, maxH / FH);

  wrap.style.width  = Math.round(FW * scale) + 'px';
  wrap.style.height = Math.round(FH * scale) + 'px';

  S.activeItemIdx = null;
  document.getElementById('item-adj-bar').classList.add('hidden');

  S.layout.forEach((item, i) => {
    if (!('rot'        in item)) item.rot        = 0;
    if (!('brightness' in item)) item.brightness = 100;
    if (!('contrast'   in item)) item.contrast   = 100;

    const box = document.createElement('div');
    box.className  = 'drag-item';
    box.dataset.idx = i;
    box.style.left      = Math.round(item.x * scale) + 'px';
    box.style.top       = Math.round(item.y * scale) + 'px';
    box.style.width     = Math.round(item.w * scale) + 'px';
    box.style.height    = Math.round(item.h * scale) + 'px';
    box.style.transform = `rotate(${item.rot}deg)`;

    const img = document.createElement('img');
    img.src = `/img/${item.url}`;
    img.draggable = false;
    applyItemFilter(img, item.brightness, item.contrast);
    box.appendChild(img);

    const handle = document.createElement('div');
    handle.className = 'resize-handle';
    box.appendChild(handle);

    const rotHandle = document.createElement('div');
    rotHandle.className = 'rotate-handle';
    rotHandle.title = 'Rotate';
    box.appendChild(rotHandle);

    makeDraggable(box, handle, rotHandle, scale, () => selectDragItem(i));
    wrap.appendChild(box);
  });

  // Icon overlay — always on top (z-index 20)
  if (S.gen.icon && S.stamp) {
    const st   = S.stamp;
    const sz   = Math.round(st.size * scale);
    const left = Math.round(st.x * scale);
    const top  = Math.round(st.y * scale);

    const icon = document.createElement('img');
    icon.src       = `/icon/${S.gen.icon}`;
    icon.className  = 'preview-stamp';
    icon.draggable  = false;
    icon.style.left   = left + 'px';
    icon.style.top    = top  + 'px';
    icon.style.width  = sz   + 'px';
    icon.style.height = sz   + 'px';
    wrap.appendChild(icon);

    const fontSize  = Math.round(sz * 0.28);
    const lineH     = Math.round(fontSize * 1.0);
    const totalH    = lineH * 2 + 2;
    const label = document.createElement('div');
    label.className = 'preview-stamp-label';
    label.innerHTML = 'D B U T<br>E R I S';
    label.style.left      = left + 'px';
    label.style.top       = (top - totalH) + 'px';
    label.style.width     = sz + 'px';
    label.style.fontSize  = fontSize + 'px';
    wrap.appendChild(label);
  }
}

function applyItemFilter(img, brightness, contrast) {
  const bw = S.gen.bw ? ' grayscale(1)' : '';
  img.style.filter = `brightness(${brightness}%) contrast(${contrast}%)${bw}`;
}

function selectDragItem(idx) {
  S.activeItemIdx = idx;
  document.querySelectorAll('.drag-item').forEach((b, i) =>
    b.classList.toggle('drag-selected', i === idx));
  const item = S.layout[idx];
  const bar  = document.getElementById('item-adj-bar');
  bar.classList.remove('hidden');
  const bEl = document.getElementById('adj-brightness');
  const cEl = document.getElementById('adj-contrast');
  bEl.value = item.brightness ?? 100;
  cEl.value = item.contrast   ?? 100;
  document.getElementById('adj-brightness-val').textContent = bEl.value;
  document.getElementById('adj-contrast-val').textContent   = cEl.value;
}

function refreshAllItemFilters() {
  document.querySelectorAll('#gen-canvas-wrap .drag-item').forEach((box, i) => {
    const img  = box.querySelector('img');
    const item = S.layout[i];
    if (img && item) applyItemFilter(img, item.brightness ?? 100, item.contrast ?? 100);
  });
}

function makeDraggable(box, handle, rotHandle, scale, onClickFn, layoutArr) {
  const L   = layoutArr != null ? layoutArr : S.layout;
  const idx = () => parseInt(box.dataset.idx);

  // Drag to move; click (no movement) → onClickFn
  box.addEventListener('mousedown', e => {
    if (e.target === handle || e.target === rotHandle) return;
    e.preventDefault();
    const startX   = e.clientX;
    const startY   = e.clientY;
    const origLeft = parseInt(box.style.left);
    const origTop  = parseInt(box.style.top);
    let   moved    = false;
    box.style.zIndex = 10;

    const onMove = e => {
      if (Math.abs(e.clientX - startX) > 3 || Math.abs(e.clientY - startY) > 3) moved = true;
      box.style.left = (origLeft + e.clientX - startX) + 'px';
      box.style.top  = (origTop  + e.clientY - startY) + 'px';
    };
    const onUp = () => {
      box.style.zIndex = '';
      if (!moved && onClickFn) { onClickFn(); }
      L[idx()].x = Math.round(parseInt(box.style.left) / scale);
      L[idx()].y = Math.round(parseInt(box.style.top)  / scale);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });

  // Drag handle to resize (proportional)
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    e.stopPropagation();
    const startX  = e.clientX;
    const startY  = e.clientY;
    const origW   = parseInt(box.style.width);
    const origH   = parseInt(box.style.height);
    const aspect  = origW / origH;
    box.style.zIndex = 10;

    const onMove = e => {
      const dx   = e.clientX - startX;
      const dy   = e.clientY - startY;
      const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
      const newW  = Math.max(24, origW + delta);
      const newH  = Math.max(24, Math.round(newW / aspect));
      box.style.width  = newW + 'px';
      box.style.height = newH + 'px';
    };
    const onUp = () => {
      box.style.zIndex = '';
      L[idx()].w = Math.round(parseInt(box.style.width)  / scale);
      L[idx()].h = Math.round(parseInt(box.style.height) / scale);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });

  // Rotate
  rotHandle.addEventListener('mousedown', e => {
    e.preventDefault();
    e.stopPropagation();
    const i    = idx();
    const rect = box.getBoundingClientRect();
    const cx   = rect.left + rect.width  / 2;
    const cy   = rect.top  + rect.height / 2;
    const startAngle = Math.atan2(e.clientY - cy, e.clientX - cx) * 180 / Math.PI;
    const origRot    = L[i].rot || 0;

    const onMove = e => {
      const angle  = Math.atan2(e.clientY - cy, e.clientX - cx) * 180 / Math.PI;
      const newRot = origRot + (angle - startAngle);
      L[i].rot      = newRot;
      box.style.transform  = `rotate(${newRot}deg)`;
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });
}

// ── Text tab ───────────────────────────────────────────────────────────────────
function normalizeTitlesClient() {
  if (!S.config.series_titles) S.config.series_titles = {};
  for (const s of Object.keys(S.config.series_titles)) {
    for (const l of ['pt', 'en']) {
      const v = S.config.series_titles[s][l];
      if (!Array.isArray(v)) S.config.series_titles[s][l] = v ? [v] : [];
    }
  }
}

function optsHtml(series, lang) {
  const opts = (S.config.series_titles[series] || {})[lang] || [];
  return opts.map((v, i) => `
    <div class="opt-row">
      <input class="txt-input opt-input" type="text"
             data-series="${series}" data-lang="${lang}" data-idx="${i}"
             value="${v.replace(/"/g, '&quot;')}" placeholder="—">
      <button class="opt-remove" data-series="${series}" data-lang="${lang}" data-idx="${i}">-</button>
    </div>`).join('') +
    `<button class="opt-add" data-series="${series}" data-lang="${lang}">+</button>`;
}

function renderText() {
  if (S.config) normalizeTitlesClient();
}

function openTitlesModal() {
  if (!S.config) return;
  normalizeTitlesClient();

  const all = new Set([
    ...S.images.map(i => i.series).filter(Boolean),
    ...S.prints.map(t => t.series).filter(Boolean),
  ]);
  const series = [...all].sort();
  series.forEach(s => {
    if (!S.config.series_titles[s]) S.config.series_titles[s] = { pt: [], en: [] };
  });

  const LANGS = [['pt','Português'],['en','English'],['es','Español'],['fr','Français']];
  const body = document.getElementById('titles-modal-body');
  let html = `<div class="text-editor"><div class="titles-cards">`;
  for (const s of series) {
    html += `<div class="title-card">
      <div class="title-card-hd">${s}</div>
      <div class="title-card-langs">` +
      LANGS.map(([l, label]) => `
        <div class="title-card-lang">
          <span class="title-lang-label">${label}</span>
          ${optsHtml(s, l)}
        </div>`).join('') +
      `</div></div>`;
  }
  html += `</div></div>`;
  body.innerHTML = html;

  body.querySelectorAll('.opt-input').forEach(inp => {
    inp.addEventListener('input', () => {
      const { series: s, lang: l, idx } = inp.dataset;
      S.config.series_titles[s][l][+idx] = inp.value;
    });
  });
  body.querySelectorAll('.opt-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      const { series: s, lang: l, idx } = btn.dataset;
      S.config.series_titles[s][l].splice(+idx, 1);
      openTitlesModal();
    });
  });
  body.querySelectorAll('.opt-add').forEach(btn => {
    btn.addEventListener('click', () => {
      const { series: s, lang: l } = btn.dataset;
      if (!Array.isArray(S.config.series_titles[s][l])) S.config.series_titles[s][l] = [];
      S.config.series_titles[s][l].push('');
      openTitlesModal();
    });
  });

  document.getElementById('titles-backdrop').classList.remove('hidden');
}

// ── Back Description modal (shared: prints edit back_footer, zines edit back_description) ──
function openDescModal(configKey) {
  if (!S.config) return;
  const data = S.config[configKey] || {};
  const body = document.getElementById('desc-modal-body');
  body.innerHTML = `
    <div class="back-footer-editor">
      <div class="footer-row"><label>Português</label><textarea id="desc-pt" rows="5">${data.pt || ''}</textarea></div>
      <div class="footer-row"><label>English</label><textarea id="desc-en" rows="5">${data.en || ''}</textarea></div>
      <div class="footer-row"><label>Español</label><textarea id="desc-es" rows="5">${data.es || ''}</textarea></div>
      <div class="footer-row"><label>Français</label><textarea id="desc-fr" rows="5">${data.fr || ''}</textarea></div>
    </div>`;
  ['pt', 'en', 'es', 'fr'].forEach(l => {
    document.getElementById(`desc-${l}`).addEventListener('input', e => {
      if (!S.config[configKey]) S.config[configKey] = {};
      S.config[configKey][l] = e.target.value;
    });
  });
  document.getElementById('desc-backdrop').classList.remove('hidden');
}

function closeDescModal() {
  document.getElementById('desc-backdrop').classList.add('hidden');
}

async function saveConfig(btnEl) {
  btnEl.disabled = true; btnEl.textContent = 'Saving…';
  await PUT('/api/config', S.config);
  btnEl.disabled = false; btnEl.textContent = 'Saved ✓';
  setTimeout(() => { btnEl.textContent = 'Save'; }, 2000);
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function activateTab(tabId) {
  document.querySelectorAll('.home-tab').forEach(t => t.classList.toggle('active', t.dataset.homeTab === tabId));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`panel-${tabId}`).classList.add('active');
  S.tab = tabId;
  S.imagesSeries = null;
  S.zinesImageSeries = null;
  S.zinesImageSelected.clear();
  if (tabId === 'zines') renderZines();
}

function setupTabs() {
  document.querySelectorAll('[data-home-tab]').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.homeTab));
  });
}

// ── Paper calculator ───────────────────────────────────────────────────────────
function buildPaperTable() {
  const formats = [
    { id: 'postcard',    label: 'Postcard',    w: 10.5, h: 16 },
    { id: 'mini-poster', label: 'Mini Poster', w: 16,   h: 26 },
    { id: 'poster',      label: 'Poster',      w: 21,   h: 30 },
  ];
  for (const f of formats) {
    const tbl = document.getElementById(`paper-table-${f.id}`);
    let html = `<thead><tr><th></th><th>${f.label}<br><span class="paper-size">${f.w}x${f.h} cm</span></th></tr></thead><tbody>`;
    for (let g = 1; g <= 20; g++) {
      const gsm = Math.round(g / ((f.w * f.h) / 10000));
      if (gsm > 300) continue;
      html += `<tr><td class="paper-g">${g}g</td><td>${gsm} g/m<sup>2</sup></td></tr>`;
    }
    html += '</tbody>';
    tbl.innerHTML = html;
  }
}

// ── Zines panel ────────────────────────────────────────────────────────────────

function renderZineSeriesFilter() {
  const container = document.getElementById('zine-pills-series');
  if (!container) return;
  const series = [...new Set(S.zines.map(z => z.series).filter(Boolean))].sort();
  container.innerHTML =
    `<button class="pill${!S.zineSeriesFilter ? ' active' : ''}" data-zsf="">All</button>` +
    series.map(s => `<button class="pill${S.zineSeriesFilter === s ? ' active' : ''}" data-zsf="${s}">${s}</button>`).join('');
  container.querySelectorAll('[data-zsf]').forEach(btn => {
    btn.addEventListener('click', () => {
      S.zineSeriesFilter = btn.dataset.zsf;
      renderZineSeriesFilter();
      renderZinesGrid();
    });
  });
}

function renderZineIconFilter() {
  const container = document.getElementById('zine-pills-icon');
  if (!container) return;
  const icons = [...new Set(S.zines.filter(z => z.icon).map(z => z.icon))].sort();
  const group = document.getElementById('zine-filter-group-icon');
  if (!icons.length) {
    group?.classList.add('hidden');
    return;
  }
  group?.classList.remove('hidden');
  container.innerHTML =
    `<button class="pill${!S.zineIconFilter ? ' active' : ''}" data-zif="">All</button>` +
    icons.map(ic =>
      `<button class="pill pill-icon${S.zineIconFilter === ic ? ' active' : ''}" data-zif="${ic}" title="${ic.replace(/\.[^.]+$/, '')}">
        <img src="/icon/${ic}" alt="${ic}">
      </button>`
    ).join('');
  container.querySelectorAll('[data-zif]').forEach(btn => {
    btn.addEventListener('click', () => {
      S.zineIconFilter = btn.dataset.zif;
      renderZineIconFilter();
      renderZinesGrid();
    });
  });
}

function renderZinesGrid() {
  const grid  = document.getElementById('grid-zines');
  const count = document.getElementById('count-zines-grid');
  const items = S.zines
    .filter(z => !S.zineSeriesFilter  || z.series === S.zineSeriesFilter)
    .filter(z => !S.zineFormatFilter  || z.format === S.zineFormatFilter)
    .filter(z => !S.zineLangFilter    || z.lang   === S.zineLangFilter)
    .filter(z => !S.zineIconFilter    || z.icon   === S.zineIconFilter);

  count.textContent = `${items.length} ${items.length === 1 ? 'zine' : 'zines'}`;
  document.getElementById('btn-select-all-zines').classList.toggle('hidden', !items.length);

  if (!items.length) { grid.innerHTML = ''; return; }

  grid.innerHTML = items.map(z => {
    const checked  = S.selectedZines.has(z.name);
    const cv = z.modified ? `${z.modified}`.replace(/\D/g,'') : Date.now();
    const thumb    = z.cover ? `<img class="card-thumb portrait" src="/img/${z.cover}?v=${cv}" loading="lazy" alt="${z.series}">` : `<div class="card-thumb portrait card-thumb-empty"></div>`;
    const editBtn  = z.layout?.length ? `<button class="act-btn act-edit">Edit</button>` : '';
    const viewBtn  = z.cover ? `<button class="act-btn act-view">View</button>` : '';
    return `<div class="card${checked ? ' selected' : ''}">
      <label class="card-check" title="Select"><input type="checkbox"${checked ? ' checked' : ''}></label>
      ${thumb}
      <div class="card-actions">
        ${editBtn}
        ${viewBtn}
        <a class="act-btn act-dl" href="/api/zines/download/${z.name}?v=${cv}" download="${z.name}.pdf">PDF</a>
      </div>
      <div class="card-info">
        <div class="card-series">${z.series || '—'}</div>
        <div class="badges">
          <span class="badge ${z.format || ''}">${z.format || ''}</span>
          ${z.lang ? `<span class="badge ${z.lang}">${z.lang.toUpperCase()}</span>` : ''}
        </div>
        <div class="card-meta">${fmtBytes(z.size_bytes)}<br>${fmtDate(z.modified)}</div>
      </div>
    </div>`;
  }).join('');

  grid.querySelectorAll('.card').forEach((card, i) => {
    const z = items[i];
    card.addEventListener('click', e => {
      if (e.target.closest('.card-check') || e.target.closest('.card-actions')) return;
      if (z.cover) openLb({ path: z.cover, series: z.series, width: 0, height: 0, size_bytes: z.size_bytes, modified: z.modified });
    });
    card.querySelector('.act-view')?.addEventListener('click', e => {
      e.stopPropagation();
      openLb({ path: z.cover, series: z.series, width: 0, height: 0, size_bytes: z.size_bytes, modified: z.modified });
    });
    card.querySelector('.act-edit')?.addEventListener('click', e => { e.stopPropagation(); openZineEditModal(z); });
    card.querySelector('.act-dl')?.addEventListener('click', e => e.stopPropagation());
    const cb = card.querySelector('.card-check input');
    cb.addEventListener('change', () => {
      if (cb.checked) S.selectedZines.add(z.name); else S.selectedZines.delete(z.name);
      card.classList.toggle('selected', cb.checked);
      syncZineDeleteBtn();
    });
    cb.addEventListener('click', e => e.stopPropagation());
  });
}

async function renderZines() {
  S.zines = await GET('/api/zines');
  renderZineSeriesFilter();
  renderZineIconFilter();
  renderZinesGrid();
  syncZineDeleteBtn();
}

function openZineEditModal(z) {
  if (!z.layout || !z.layout.length) return;
  S.czineEditName    = z.name;
  S.czineSeries      = z.series;
  S.czineLayout      = z.layout.map(item => ({ ...item }));
  S.czineSize        = { canvasW: z.canvasW || 680, canvasH: z.canvasH || 380 };
  S.czineBw          = !!z.bw;
  S.czineFormat      = z.format || 'mini-poster';
  S.czineLang        = z.lang   || 'pt';
  S.czineSpread      = 0;
  S.czineActivePgIdx = null;
  S.czineGlobalScale = z.globalScale || 100;
  const numSpreads   = Math.ceil(S.czineLayout.length / 2);
  const HALF_W = (z.canvasW || 680) / 2;
  const CANVAS_H = z.canvasH || 380;
  S.czineTexts = (z.texts && z.texts.length)
    ? z.texts.map((t, i) => ({
        left:  { ...t.left,  text: i === 0 ? autoSplitTitle(t.left.text  || '') : (t.left.text  || '') },
        right: { ...t.right, text: t.right.text || '' },
      }))
    : Array.from({ length: numSpreads }, () => ({
        left:  { text: '', x: 0,      y: Math.round(CANVAS_H / 2) - 12 },
        right: { text: '', x: HALF_W, y: Math.round(CANVAS_H / 2) - 12 },
      }));

  selectCzineIcon(z.icon || '');

  // Detect cover image from layout[0] (non-white = cover photo is set)
  const firstItem = S.czineLayout[0];
  S.czineCoverPath = (firstItem && firstItem.url && !firstItem.white) ? firstItem.url : null;
  if (S.czineCoverPath) {
    const coverScale = Math.round((firstItem.w / ((z.canvasW || 680) / 2)) * 100);
    S.czineCoverScale = coverScale;
    document.getElementById('czine-cover-scale-slider').value = coverScale;
    document.getElementById('czine-cover-scale-val').textContent = coverScale + '%';
  } else {
    S.czineCoverScale = 100;
  }
  document.getElementById('czine-cover-scale-ctrl').classList.toggle('hidden', !S.czineCoverPath);

  document.getElementById('czine-scale-slider').value   = S.czineGlobalScale;
  document.getElementById('czine-scale-val').textContent = S.czineGlobalScale + '%';
  document.getElementById('czine-pg-bw-toggle').classList.toggle('active', S.czineBw);
  document.getElementById('czine-pg-adj-bar').classList.add('hidden');
  document.getElementById('czine-save-status').textContent = '';
  const saveBtn = document.getElementById('czine-save-btn');
  saveBtn.disabled = false; saveBtn.textContent = 'Save';
  showCzineStep(2);
  document.getElementById('czine-backdrop').classList.remove('hidden');
  renderCzineSpread(0);
}

function openCzineModal() {
  document.getElementById('czine-status').textContent = '';
  document.getElementById('czine-preview-btn').disabled = true;
  document.getElementById('czine-images-section').classList.add('hidden');
  document.getElementById('czine-image-grid').innerHTML = '';
  document.getElementById('czine-adj-bar').classList.add('hidden');
  S.czineEditName = null;
  S.czineBw = false;
  selectCzineIcon('');
  showCzineStep(1);

  // Build theme pills from S.images series
  const seriesSet = [...new Set(S.images.map(i => i.series).filter(Boolean))].sort();
  const pillsEl = document.getElementById('czine-pills-series');
  pillsEl.innerHTML = seriesSet.map(s =>
    `<button class="pill" data-czs="${s}">${s}</button>`
  ).join('');

  // Pre-select current browsed theme if available
  const preselect = S.zinesImageSeries;
  pillsEl.querySelectorAll('[data-czs]').forEach(btn => {
    if (btn.dataset.czs === preselect) {
      btn.classList.add('active');
      loadCzineImages(preselect);
    }
    btn.addEventListener('click', () => {
      pillsEl.querySelectorAll('[data-czs]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadCzineImages(btn.dataset.czs);
    });
  });

  document.getElementById('czine-backdrop').classList.remove('hidden');
}

function applyCzineFilter(img, tile) {
  const b   = parseInt(tile.dataset.brightness || '100');
  const c   = parseInt(tile.dataset.contrast   || '100');
  const bwS = S.czineBw ? ' grayscale(1)' : '';
  img.style.filter = `brightness(${b}%) contrast(${c}%)${bwS}`;
}

function selectCzineTile(tile) {
  document.querySelectorAll('#czine-image-grid .czine-active').forEach(t => t.classList.remove('czine-active'));
  tile.classList.add('czine-active');
  const bar  = document.getElementById('czine-adj-bar');
  bar.classList.remove('hidden');
  const bEl  = document.getElementById('czine-adj-brightness');
  const cEl  = document.getElementById('czine-adj-contrast');
  bEl.value  = tile.dataset.brightness || '100';
  cEl.value  = tile.dataset.contrast   || '100';
  document.getElementById('czine-adj-brightness-val').textContent = bEl.value;
  document.getElementById('czine-adj-contrast-val').textContent   = cEl.value;
}

function loadCzineImages(series) {
  const items = S.images.filter(i => i.series === series)
                        .slice().sort((a, b) => a.index - b.index);
  const section = document.getElementById('czine-images-section');
  const gridEl  = document.getElementById('czine-image-grid');
  document.getElementById('czine-adj-bar').classList.add('hidden');

  // Preserve cover selection across series switches
  const prevCover = S.czineCoverPath;

  gridEl.innerHTML = items.map((img, i) => `
    <div class="czine-img-tile" data-brightness="100" data-contrast="100" data-path="${img.path}">
      <label class="czine-img-check">
        <input type="checkbox" checked data-path="${img.path}">
      </label>
      <button class="czine-cover-btn" title="Set as fanzine cover">COVER</button>
      <img src="/img/${img.path}" loading="lazy" alt="${img.series}">
      <div class="czine-img-num">${i + 1}</div>
    </div>`).join('');

  // Restore cover highlight if the same image is in this series
  if (prevCover) {
    const prevTile = gridEl.querySelector(`.czine-img-tile[data-path="${CSS.escape(prevCover)}"]`);
    if (prevTile) prevTile.classList.add('czine-cover-active');
  }

  gridEl.querySelectorAll('.czine-img-tile').forEach(tile => {
    const cb  = tile.querySelector('input');
    const img = tile.querySelector('img');
    // Left-click: select for adjustments; checkbox handles include/exclude
    tile.addEventListener('click', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.target.tagName === 'LABEL') return;
      if (e.target.classList.contains('czine-cover-btn')) return;
      selectCzineTile(tile);
    });
    cb.addEventListener('change', () => {
      tile.classList.toggle('czine-excluded', !cb.checked);
    });
    cb.addEventListener('click', e => e.stopPropagation());
    tile.querySelector('.czine-cover-btn').addEventListener('click', e => {
      e.stopPropagation();
      const isActive = tile.classList.contains('czine-cover-active');
      gridEl.querySelectorAll('.czine-cover-active').forEach(t => t.classList.remove('czine-cover-active'));
      if (!isActive) {
        tile.classList.add('czine-cover-active');
        S.czineCoverPath = tile.dataset.path;
      } else {
        S.czineCoverPath = null;
      }
    });
  });

  section.classList.remove('hidden');
  document.getElementById('czine-preview-btn').disabled = false;
}

// ── Czine step helpers ────────────────────────────────────────────────────────

function showCzineStep(n) {
  document.getElementById('czine-step-1').classList.toggle('hidden', n !== 1);
  document.getElementById('czine-step-2').classList.toggle('hidden', n !== 2);
}

function closeCzineModal() {
  document.getElementById('czine-backdrop').classList.add('hidden');
  showCzineStep(1);
  S.czineLayout       = [];
  S.czineTexts        = [];
  S.czineCoverPath    = null;
  S.czineCoverScale   = 100;
  document.getElementById('czine-cover-scale-ctrl').classList.add('hidden');
  S.czineActivePgIdx  = null;
  S.czineGlobalScale  = 100;
  const saveBtn = document.getElementById('czine-save-btn');
  saveBtn.disabled = false; saveBtn.textContent = 'Save';
  document.getElementById('czine-save-status').textContent = '';
}

function setCzineBw(bw) {
  S.czineBw = bw;
  document.getElementById('czine-pg-bw-toggle').classList.toggle('active', bw);
}

function autoSplitTitle(t) {
  // If >1 word precedes the last geographic preposition, split before it.
  const re = /\b(DE|DAS|DOS|FROM|VON|VAN|DI|DES|DEL|DEN|DEGLI|DU)\b/gi;
  let last = null, m;
  while ((m = re.exec(t)) !== null) last = m;
  if (!last) return t;
  const before = t.slice(0, last.index).trim();
  if (before.split(/\s+/).length > 1) {
    return before + '\n' + t.slice(last.index);
  }
  return t;
}

function buildCzineLayout(imageEntries) {
  const CANVAS_W = 680;
  const HALF_W   = 340;
  const MARGIN   = 30;
  const fmt      = ZINE_FORMATS[S.czineFormat] || ZINE_FORMATS['postcard'];
  const CANVAS_H = Math.round(CANVAS_W * fmt.paperH / fmt.paperW);
  S.czineSize = { canvasW: CANVAS_W, canvasH: CANVAS_H };

  // odd M  → 3 trailing blanks (N-2, N-1, N)
  // even M → 4 trailing blanks (N-3, N-2, N-1, N)
  // Round to next multiple of 4 so PDF imposition adds no hidden extra blanks.
  const M        = imageEntries.length;
  const baseLen  = (M % 2 === 0) ? (M + 6) : (M + 5);
  const totalLen = Math.ceil(baseLen / 4) * 4;
  const nTrailing = totalLen - M - 2;

  const mkWhite = absIdx => ({
    url: null, path: null,
    x: (absIdx % 2) * HALF_W, y: 0, w: HALF_W, h: CANVAS_H,
    rot: 0, brightness: 100, contrast: 100, white: true,
  });

  const layout = [mkWhite(0), mkWhite(1)];  // cover (physical pages 1, 2)

  // Place selected cover image at position 0 (the front cover page)
  if (S.czineCoverPath) {
    layout[0] = {
      url: S.czineCoverPath, path: S.czineCoverPath,
      x: 0, y: 0, w: HALF_W, h: CANVAS_H,
      rot: 0, brightness: 100, contrast: 100,
    };
  }

  imageEntries.forEach((entry, i) => {
    const absIdx = layout.length;
    const slotX  = (absIdx % 2) * HALF_W;
    const meta   = S.images.find(img => img.path === entry.path);
    const srcW   = meta?.width  || 1000;
    const srcH   = meta?.height || 1000;
    const availW = HALF_W - 2 * MARGIN;
    const availH = CANVAS_H - 2 * MARGIN;
    const fs     = Math.min(availW / srcW, availH / srcH);
    const w      = Math.round(srcW * fs);
    const h      = Math.round(srcH * fs);
    layout.push({
      url: entry.path, path: entry.path,
      x: slotX + Math.round((HALF_W - w) / 2),
      y: Math.round((CANVAS_H - h) / 2),
      w, h, rot: 0,
      brightness: entry.brightness || 100,
      contrast:   entry.contrast   || 100,
    });
  });

  for (let i = 0; i < nTrailing; i++) layout.push(mkWhite(layout.length));

  S.czineLayout = layout;

  const numSpreads = Math.ceil(layout.length / 2);
  S.czineTexts = Array.from({ length: numSpreads }, () => ({
    left:  { text: '', x: 0,      y: Math.round(CANVAS_H / 2) - 12 },
    right: { text: '', x: HALF_W, y: Math.round(CANVAS_H / 2) - 12 },
  }));

  // Auto-populate cover: theme title on front (left, centered) and back description on inside (right, top)
  const seriesTitles = (S.config?.series_titles || {})[S.czineSeries] || {};
  const _rawTitle = (seriesTitles[S.czineLang]?.[0] || seriesTitles.pt?.[0] || seriesTitles.en?.[0] || seriesTitles.es?.[0] || seriesTitles.fr?.[0] || '').trim().toUpperCase();
  const titleText = autoSplitTitle(_rawTitle);
  if (titleText) {
    S.czineTexts[0].left.text = titleText;
    S.czineTexts[0].left.x   = 0;
    S.czineTexts[0].left.y   = Math.round(CANVAS_H / 2) - 12;
  }

  const bd = S.config?.back_description || {};
  const bdText = (bd[S.czineLang] || bd.pt || bd.en || bd.es || bd.fr || '').trim().toUpperCase();
  const CANVAS_MARGIN = 28; // matches Python CANVAS_MARGIN; baked into position so preview == PDF
  if (bdText) {
    S.czineTexts[0].right.text = bdText;
    S.czineTexts[0].right.x   = HALF_W + CANVAS_MARGIN;
    S.czineTexts[0].right.y   = 24;
  }
}

function openCzinePreview() {
  const series = document.querySelector('#czine-pills-series .pill.active')?.dataset.czs || '';
  if (!series) return;
  S.czineFormat = document.querySelector('[data-czf].active')?.dataset.czf || 'mini-poster';
  S.czineLang   = document.querySelector('#czine-pills-lang .pill.active')?.dataset.czl || 'pt';

  const imageEntries =[...document.querySelectorAll('#czine-image-grid input[type=checkbox]:checked')]
    .map(cb => {
      const tile = cb.closest('.czine-img-tile');
      return {
        path:       cb.dataset.path,
        brightness: parseInt(tile.dataset.brightness || '100'),
        contrast:   parseInt(tile.dataset.contrast   || '100'),
      };
    });

  if (!imageEntries.length) {
    document.getElementById('czine-status').textContent = 'No images selected.';
    return;
  }

  S.czineSeries      = series;
  S.czineSpread      = 0;
  S.czineActivePgIdx = null;
  S.czineGlobalScale = 100;
  const coverTile = document.querySelector('#czine-image-grid .czine-cover-active');
  S.czineCoverPath = coverTile ? coverTile.dataset.path : null;
  buildCzineLayout(imageEntries);

  document.getElementById('czine-scale-slider').value = 100;
  document.getElementById('czine-scale-val').textContent = '100%';
  S.czineCoverScale = 100;
  document.getElementById('czine-cover-scale-slider').value = 100;
  document.getElementById('czine-cover-scale-val').textContent = '100%';
  document.getElementById('czine-cover-scale-ctrl').classList.toggle('hidden', !S.czineCoverPath);
  document.getElementById('czine-pg-bw-toggle').classList.toggle('active', S.czineBw);
  document.getElementById('czine-pg-adj-bar').classList.add('hidden');
  document.getElementById('czine-save-status').textContent = '';
  showCzineStep(2);
  renderCzineSpread(0);
}

function applyCzineItemFilter(img, brightness, contrast) {
  const bwS = S.czineBw ? ' grayscale(1)' : '';
  img.style.filter = `brightness(${brightness}%) contrast(${contrast}%)${bwS}`;
}

function refreshAllCzineItemFilters() {
  document.querySelectorAll('#czine-spread-canvas .drag-item').forEach(box => {
    const img  = box.querySelector('img');
    const item = S.czineLayout[parseInt(box.dataset.idx)];
    if (img && item) applyCzineItemFilter(img, item.brightness ?? 100, item.contrast ?? 100);
  });
}

function selectCzineItem(absIdx) {
  S.czineActivePgIdx = absIdx;
  document.querySelectorAll('#czine-spread-canvas .drag-item').forEach(b =>
    b.classList.toggle('drag-selected', parseInt(b.dataset.idx) === absIdx));
  const item = S.czineLayout[absIdx];
  const bar  = document.getElementById('czine-pg-adj-bar');
  bar.classList.remove('hidden');
  const bEl  = document.getElementById('czine-pg-brightness');
  const cEl  = document.getElementById('czine-pg-contrast');
  bEl.value  = item.brightness ?? 100;
  cEl.value  = item.contrast   ?? 100;
  document.getElementById('czine-pg-brightness-val').textContent = bEl.value;
  document.getElementById('czine-pg-contrast-val').textContent   = cEl.value;
}

function renderCzineSpread(spreadIdx) {
  const canvas = document.getElementById('czine-spread-canvas');
  canvas.innerHTML = '';

  const { canvasW, canvasH } = S.czineSize;
  canvas.style.width  = canvasW + 'px';
  canvas.style.height = canvasH + 'px';

  const div = document.createElement('div');
  div.className = 'czine-spread-divider';
  canvas.appendChild(div);

  const startIdx = spreadIdx * 2;
  [0, 1].forEach(slot => {
    const absIdx = startIdx + slot;
    if (absIdx >= S.czineLayout.length) return;
    const item = S.czineLayout[absIdx];

    if (item.white || !item.url) return;  // mandatory white page — canvas background shows through

    const box = document.createElement('div');
    box.className = 'drag-item';
    box.dataset.idx = absIdx;
    box.style.left      = item.x + 'px';
    box.style.top       = item.y + 'px';
    box.style.width     = item.w + 'px';
    box.style.height    = item.h + 'px';
    box.style.transform = `rotate(${item.rot}deg)`;

    const img = document.createElement('img');
    img.src       = `/img/${item.url}`;
    img.draggable = false;
    applyCzineItemFilter(img, item.brightness, item.contrast);
    box.appendChild(img);

    const handle = document.createElement('div');
    handle.className = 'resize-handle';
    box.appendChild(handle);

    const rotHandle = document.createElement('div');
    rotHandle.className = 'rotate-handle';
    box.appendChild(rotHandle);

    makeDraggable(box, handle, rotHandle, 1, () => selectCzineItem(absIdx), S.czineLayout);
    canvas.appendChild(box);
  });

  // Page numbers — skip cover (first) and back cover (last) spread
  // Visible numbering starts at 1 from the first content page (N-2 offset)
  const totalSpreads = Math.ceil(S.czineLayout.length / 2);
  const isEdgeSpread = spreadIdx === 0 || spreadIdx === totalSpreads - 1;
  if (!isEdgeSpread) {
    const pgL = spreadIdx * 2 - 1;
    const pgR = spreadIdx * 2;
    const itemL = S.czineLayout[startIdx];
    const itemR = S.czineLayout[startIdx + 1];
    if (itemL && itemL.url && !itemL.white) {
      const numLeft  = document.createElement('div');
      numLeft.className  = 'czine-pg-num left';
      numLeft.textContent = pgL;
      canvas.appendChild(numLeft);
    }
    if (itemR && itemR.url && !itemR.white) {
      const numRight = document.createElement('div');
      numRight.className  = 'czine-pg-num right';
      numRight.textContent = pgR;
      canvas.appendChild(numRight);
    }
  }

  // Text labels
  const td = S.czineTexts[spreadIdx];
  if (td) {
    ['left', 'right'].forEach((side, slot) => {
      const ti = td[side];
      if (!ti.text) return;
      if (slot === 1 && startIdx + 1 >= S.czineLayout.length) return;
      const box = document.createElement('div');
      box.className = 'czine-text-item';
      if (spreadIdx === 0 && side === 'right') box.classList.add('text-justify');
      if (spreadIdx === 0 && side === 'left')  box.classList.add('text-bold');
      box.dataset.side = side;
      box.dataset.spread = spreadIdx;
      const CANVAS_MARGIN = 28;
      const isBackDesc = (spreadIdx === 0 && side === 'right');
      box.style.left  = ti.x + 'px';
      box.style.top   = ti.y + 'px';
      box.style.width = (isBackDesc ? canvasW / 2 - 2 * CANVAS_MARGIN : canvasW / 2) + 'px';
      box.textContent = ti.text;
      canvas.appendChild(box);
      makeCzineTextDraggable(box, spreadIdx, side);
    });
  }

  // Sync text inputs
  const leftInput  = document.getElementById('czine-text-left');
  const rightInput = document.getElementById('czine-text-right');
  if (leftInput)  leftInput.value  = td?.left.text  || '';
  if (rightInput) rightInput.value = td?.right.text || '';

  // Sidebar labels: cover/back show "Cover"/"Back", content shows page number (N-2)
  const labelL = isEdgeSpread
    ? (spreadIdx === 0 ? 'Cover' : 'Back')
    : `Page ${spreadIdx * 2 - 1}`;
  const labelR = isEdgeSpread
    ? (spreadIdx === 0 ? 'Cover' : 'Back')
    : `Page ${spreadIdx * 2}`;
  document.getElementById('czine-text-label-left').textContent  = labelL;
  document.getElementById('czine-text-label-right').textContent = labelR;

  // Icon stamp on front cover (left slot of spread 0), bottom-right — same as prints
  if (spreadIdx === 0 && S.czineIcon) {
    const HALF_W          = canvasW / 2;
    const STAMP_SIZE_FRAC  = 0.048;
    const STAMP_RIGHT_FRAC = 0.020;
    const STAMP_BOTTOM_FRAC = 0.012;
    const ICON_SCALE       = 2;
    const base = Math.round(STAMP_SIZE_FRAC * HALF_W);
    const sz   = Math.round(base * ICON_SCALE);
    const x0   = Math.round(HALF_W - STAMP_RIGHT_FRAC * HALF_W) - sz;
    const y0   = Math.round(canvasH - STAMP_BOTTOM_FRAC * canvasH) - sz;

    const stamp = document.createElement('img');
    stamp.src       = `/icon/${S.czineIcon}`;
    stamp.className = 'preview-stamp';
    stamp.draggable = false;
    stamp.style.left   = x0 + 'px';
    stamp.style.top    = y0 + 'px';
    stamp.style.width  = sz + 'px';
    stamp.style.height = sz + 'px';
    canvas.appendChild(stamp);

    const fontSize = Math.round(sz * 0.28);
    const lineH    = Math.round(fontSize * 1.0);
    const label    = document.createElement('div');
    label.className    = 'preview-stamp-label';
    label.innerHTML    = 'D B U T<br>E R I S';
    label.style.left   = x0 + 'px';
    label.style.top    = (y0 - lineH * 2 - 2) + 'px';
    label.style.width  = sz + 'px';
    label.style.fontSize = fontSize + 'px';
    canvas.appendChild(label);
  }

  document.getElementById('czine-spread-label').textContent = `Spread ${spreadIdx + 1} / ${totalSpreads}`;
  document.getElementById('czine-prev-spread').disabled = (spreadIdx === 0);
  document.getElementById('czine-next-spread').disabled = (spreadIdx >= totalSpreads - 1);

  S.czineActivePgIdx = null;
  document.getElementById('czine-pg-adj-bar').classList.add('hidden');
}

function makeCzineTextDraggable(box, spreadIdx, side) {
  let startX, startY, origX, origY;
  box.addEventListener('mousedown', e => {
    e.stopPropagation();
    startX = e.clientX; startY = e.clientY;
    const ti = S.czineTexts[spreadIdx][side];
    origX = ti.x; origY = ti.y;
    document.querySelectorAll('.czine-text-item').forEach(b => b.classList.remove('text-selected'));
    box.classList.add('text-selected');
    const onMove = mv => {
      ti.x = origX + (mv.clientX - startX);
      ti.y = origY + (mv.clientY - startY);
      box.style.left = ti.x + 'px';
      box.style.top  = ti.y + 'px';
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

function updateCzineTextItem(spreadIdx, side) {
  const td  = S.czineTexts[spreadIdx];
  if (!td) return;
  const ti  = td[side];
  const canvas = document.getElementById('czine-spread-canvas');
  // Remove existing text box for this side
  canvas.querySelectorAll(`.czine-text-item[data-side="${side}"]`).forEach(b => b.remove());
  if (!ti.text) return;
  const box = document.createElement('div');
  box.className = 'czine-text-item';
  box.dataset.side   = side;
  box.dataset.spread = spreadIdx;
  box.style.left  = ti.x + 'px';
  box.style.top   = ti.y + 'px';
  box.style.width = (S.czineSize.canvasW / 2) + 'px';
  box.textContent = ti.text;
  canvas.appendChild(box);
  makeCzineTextDraggable(box, spreadIdx, side);
}

async function saveCzine() {
  const btn    = document.getElementById('czine-save-btn');
  const status = document.getElementById('czine-save-status');
  btn.disabled = true; btn.textContent = 'Saving…'; status.textContent = '';
  try {
    const res = await POST('/api/zine/create', {
      name:        S.czineEditName || null,
      series:      S.czineSeries,
      format:      S.czineFormat,
      lang:        S.czineLang,
      layout:      S.czineLayout,
      canvasW:     S.czineSize.canvasW,
      canvasH:     S.czineSize.canvasH,
      bw:          S.czineBw,
      texts:       S.czineTexts,
      icon:        S.czineIcon || null,
      globalScale: S.czineGlobalScale,
      coverScale:  S.czineCoverScale,
    });
    if (res.ok) {
      S.zines = await GET('/api/zines');
      renderZineSeriesFilter();
      renderZineIconFilter();
      renderZinesGrid();
      closeCzineModal();
    } else {
      status.textContent = 'Error: ' + (res.error || 'Unknown');
      btn.disabled = false; btn.textContent = 'Save';
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
    btn.disabled = false; btn.textContent = 'Save';
  }
}

async function deleteSelectedZines() {
  if (!S.selectedZines.size) return;
  const names = [...S.selectedZines];
  const res = await fetch('/api/zines/delete', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  }).then(r => r.json());
  if (res.ok) {
    S.selectedZines.clear();
    syncZineDeleteBtn();
    S.zines = await GET('/api/zines');
    renderZineSeriesFilter();
    renderZineIconFilter();
    renderZinesGrid();
  }
}

async function deleteSelectedPrints() {
  if (!S.selected.size) return;
  const paths = [...S.selected];
  const res = await fetch('/api/templates', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  }).then(r => r.json());
  if (res.ok) {
    S.selected.clear();
    syncDeleteBtn();
    S.prints = await GET('/api/templates');
    refreshPrintSeries();
    renderPrints();
  }
}

async function downloadAllPrints(btn) {
  const saved = S.prints.filter(t => t.path);
  if (!saved.length) { alert('No saved prints to download.'); return; }
  btn.disabled = true; btn.textContent = 'Packing…';
  try {
    const res = await fetch('/api/download-all');
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || 'Download failed');
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'DE_BRUITS_HQ.zip';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (_) { alert('Download failed'); }
  finally { btn.disabled = false; btn.textContent = 'Download All'; }
}

async function downloadAllZines(btn) {
  btn.disabled = true; btn.textContent = 'Packing…';
  try {
    const res = await fetch('/api/zines/download-all');
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || 'Download failed');
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'DE_BRUITS_ZINES.zip';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (_) { alert('Download failed'); }
  finally { btn.disabled = false; btn.textContent = 'Download All'; }
}

async function refreshPrints(btn) {
  btn.disabled = true; btn.textContent = 'Refreshing…';
  try {
    const res = await POST('/api/refresh-prints', {});
    if (res.ok) {
      S.prints = await GET('/api/templates');
      refreshPrintSeries();
      renderPrints();
      btn.textContent = `Done (${res.count})`;
    } else {
      btn.textContent = 'Error';
    }
  } catch { btn.textContent = 'Error'; }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Refresh'; }, 2500);
}

async function refreshZines(btn) {
  btn.disabled = true; btn.textContent = 'Refreshing…';
  try {
    const res = await POST('/api/refresh-zines', {});
    if (res.ok) {
      S.zines = await GET('/api/zines');
      renderZinesGrid();
      btn.textContent = `Done (${res.count})`;
    } else {
      btn.textContent = 'Error';
    }
  } catch { btn.textContent = 'Error'; }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Refresh'; }, 2500);
}

function selectAllPrints() {
  filteredPrints().forEach(t => S.selected.add(t.path));
  renderPrints();
}

function selectAllZines() {
  const visible = S.zineSeriesFilter
    ? S.zines.filter(z => z.series === S.zineSeriesFilter)
    : S.zines;
  visible.forEach(z => S.selectedZines.add(z.name));
  renderZinesGrid();
  syncZineDeleteBtn();
}

function showHome() {
  document.querySelectorAll('.home-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-home').classList.add('active');
  S.imagesSeries = null;
  renderImages();
}

// ── Init ───────────────────────────────────────────────────────────────────────
function init() {
  setupTabs();

  // Build both panel toolbars from the shared template.
  buildToolbar('prints', {
    createLabel:   '+ Create Print',
    onCreate:      openGenModal,
    onDownloadAll: downloadAllPrints,
    onDesc:        () => openDescModal('back_footer'),
    onRefresh:     refreshPrints,
    onSelectAll:   selectAllPrints,
    onDelete:      deleteSelectedPrints,
  });
  buildToolbar('zines', {
    createLabel:   '+ Create Zine',
    onCreate:      openCzineModal,
    onDownloadAll: downloadAllZines,
    onDesc:        () => openDescModal('back_description'),
    onRefresh:     refreshZines,
    onSelectAll:   selectAllZines,
    onDelete:      deleteSelectedZines,
  });

  // Build both filter sidebars from the shared template.
  buildFilterSidebar(document.getElementById('filters-prints'), 'prints');
  buildFilterSidebar(document.getElementById('filters-zines'), 'zines');

  wireStaticPills();

  document.getElementById('logo-btn').addEventListener('click', showHome);

  document.getElementById('czine-close').addEventListener('click', closeCzineModal);
  document.getElementById('czine-close-2').addEventListener('click', closeCzineModal);
  document.getElementById('czine-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('czine-backdrop')) closeCzineModal();
  });
  document.getElementById('czine-preview-btn').addEventListener('click', openCzinePreview);

  // Format pills (czine step 1)
  document.querySelectorAll('[data-czf]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-czf]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      S.czineFormat = btn.dataset.czf;
    });
  });

  // Step 1 language pills
  document.querySelectorAll('#czine-pills-lang [data-czl]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#czine-pills-lang [data-czl]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      S.czineLang = btn.dataset.czl;
    });
  });

  // Zines panel format + lang filter pills
  document.querySelectorAll('[data-zfilter]').forEach(btn => {
    btn.addEventListener('click', () => {
      const filterKey = btn.dataset.zfilter;
      btn.closest('.pills').querySelectorAll('[data-zfilter]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (filterKey === 'format') S.zineFormatFilter = btn.dataset.val;
      if (filterKey === 'lang')   S.zineLangFilter   = btn.dataset.val;
      renderZinesGrid();
    });
  });

  // Step 2 navigation
  document.getElementById('czine-prev-spread').addEventListener('click', () => {
    if (S.czineSpread > 0) { S.czineSpread--; renderCzineSpread(S.czineSpread); }
  });
  document.getElementById('czine-next-spread').addEventListener('click', () => {
    const total = Math.ceil(S.czineLayout.length / 2);
    if (S.czineSpread < total - 1) { S.czineSpread++; renderCzineSpread(S.czineSpread); }
  });
  document.getElementById('czine-back-btn').addEventListener('click', () => {
    showCzineStep(1);
    // Rebuild series pills (may be empty when coming from openZineEditModal)
    const seriesSet = [...new Set(S.images.map(i => i.series).filter(Boolean))].sort();
    const pillsEl = document.getElementById('czine-pills-series');
    pillsEl.innerHTML = seriesSet.map(s =>
      `<button class="pill${s === S.czineSeries ? ' active' : ''}" data-czs="${s}">${s}</button>`
    ).join('');
    pillsEl.querySelectorAll('[data-czs]').forEach(btn => {
      btn.addEventListener('click', () => {
        pillsEl.querySelectorAll('[data-czs]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadCzineImages(btn.dataset.czs);
      });
    });
    if (S.czineSeries) loadCzineImages(S.czineSeries);
  });
  document.getElementById('czine-save-btn').addEventListener('click', saveCzine);

  // Global scale slider (step 2)
  document.getElementById('czine-scale-slider').addEventListener('input', e => {
    const newScale = parseInt(e.target.value);
    const factor   = newScale / S.czineGlobalScale;
    document.getElementById('czine-scale-val').textContent = newScale + '%';
    S.czineLayout.forEach((item, idx) => {
      if (idx === 0 && S.czineCoverPath) return;  // cover has its own slider
      const cx = item.x + item.w / 2;
      const cy = item.y + item.h / 2;
      item.w = Math.max(10, Math.round(item.w * factor));
      item.h = Math.max(10, Math.round(item.h * factor));
      item.x = Math.round(cx - item.w / 2);
      item.y = Math.round(cy - item.h / 2);
    });
    S.czineGlobalScale = newScale;
    renderCzineSpread(S.czineSpread);
  });

  document.getElementById('czine-cover-scale-slider').addEventListener('input', e => {
    if (!S.czineCoverPath) return;
    const newScale = parseInt(e.target.value);
    const factor   = newScale / S.czineCoverScale;
    document.getElementById('czine-cover-scale-val').textContent = newScale + '%';
    const item = S.czineLayout[0];
    const cx = item.x + item.w / 2;
    const cy = item.y + item.h / 2;
    item.w = Math.max(10, Math.round(item.w * factor));
    item.h = Math.max(10, Math.round(item.h * factor));
    item.x = Math.round(cx - item.w / 2);
    item.y = Math.round(cy - item.h / 2);
    S.czineCoverScale = newScale;
    renderCzineSpread(S.czineSpread);
  });

  // B&W toggle step 2
  document.getElementById('czine-pg-bw-toggle').addEventListener('click', () => {
    setCzineBw(!S.czineBw);
    refreshAllCzineItemFilters();
  });

  // Adj sliders step 2
  document.getElementById('czine-pg-brightness').addEventListener('input', e => {
    if (S.czineActivePgIdx === null) return;
    S.czineLayout[S.czineActivePgIdx].brightness = parseInt(e.target.value);
    document.getElementById('czine-pg-brightness-val').textContent = e.target.value;
    const box = document.querySelector(`#czine-spread-canvas [data-idx="${S.czineActivePgIdx}"]`);
    const img = box?.querySelector('img');
    if (img) applyCzineItemFilter(img, S.czineLayout[S.czineActivePgIdx].brightness, S.czineLayout[S.czineActivePgIdx].contrast);
  });
  document.getElementById('czine-pg-contrast').addEventListener('input', e => {
    if (S.czineActivePgIdx === null) return;
    S.czineLayout[S.czineActivePgIdx].contrast = parseInt(e.target.value);
    document.getElementById('czine-pg-contrast-val').textContent = e.target.value;
    const box = document.querySelector(`#czine-spread-canvas [data-idx="${S.czineActivePgIdx}"]`);
    const img = box?.querySelector('img');
    if (img) applyCzineItemFilter(img, S.czineLayout[S.czineActivePgIdx].brightness, S.czineLayout[S.czineActivePgIdx].contrast);
  });

  // Text label inputs (czine step 2)
  document.getElementById('czine-center-left').addEventListener('click', () => {
    const td = S.czineTexts[S.czineSpread];
    if (!td) return;
    td.left.x = 0;
    updateCzineTextItem(S.czineSpread, 'left');
  });
  document.getElementById('czine-center-right').addEventListener('click', () => {
    const td = S.czineTexts[S.czineSpread];
    if (!td) return;
    td.right.x = S.czineSize.canvasW / 2;
    updateCzineTextItem(S.czineSpread, 'right');
  });

  document.getElementById('czine-text-left').addEventListener('input', e => {
    const td = S.czineTexts[S.czineSpread];
    if (!td) return;
    const upper = e.target.value.toUpperCase();
    td.left.text = upper;
    e.target.value = upper;
    updateCzineTextItem(S.czineSpread, 'left');
  });
  document.getElementById('czine-text-right').addEventListener('input', e => {
    const td = S.czineTexts[S.czineSpread];
    if (!td) return;
    const upper = e.target.value.toUpperCase();
    td.right.text = upper;
    e.target.value = upper;
    updateCzineTextItem(S.czineSpread, 'right');
  });


  // Per-image brightness/contrast (Czine modal)
  document.getElementById('czine-adj-brightness').addEventListener('input', e => {
    const tile = document.querySelector('#czine-image-grid .czine-active');
    if (!tile) return;
    tile.dataset.brightness = e.target.value;
    document.getElementById('czine-adj-brightness-val').textContent = e.target.value;
    applyCzineFilter(tile.querySelector('img'), tile);
  });
  document.getElementById('czine-adj-contrast').addEventListener('input', e => {
    const tile = document.querySelector('#czine-image-grid .czine-active');
    if (!tile) return;
    tile.dataset.contrast = e.target.value;
    document.getElementById('czine-adj-contrast-val').textContent = e.target.value;
    applyCzineFilter(tile.querySelector('img'), tile);
  });

  document.getElementById('czine-select-all').addEventListener('click', () => {
    document.querySelectorAll('#czine-image-grid input[type=checkbox]').forEach(cb => {
      cb.checked = true;
      cb.closest('.czine-img-tile').classList.remove('czine-excluded');
    });
  });
  document.getElementById('czine-select-none').addEventListener('click', () => {
    document.querySelectorAll('#czine-image-grid input[type=checkbox]').forEach(cb => {
      cb.checked = false;
      cb.closest('.czine-img-tile').classList.add('czine-excluded');
    });
  });

  // Shared Back Description modal wiring (opened by either panel's Back Description button)
  document.getElementById('desc-close').addEventListener('click', closeDescModal);
  document.getElementById('desc-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('desc-backdrop')) closeDescModal();
  });
  document.getElementById('btn-save-desc').addEventListener('click', () =>
    saveConfig(document.getElementById('btn-save-desc')));

  document.getElementById('titles-close').addEventListener('click', () =>
    document.getElementById('titles-backdrop').classList.add('hidden'));
  document.getElementById('titles-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('titles-backdrop'))
      document.getElementById('titles-backdrop').classList.add('hidden');
  });
  document.getElementById('btn-save-titles').addEventListener('click', e =>
    saveConfig(document.getElementById('btn-save-titles')));

  document.getElementById('gen-close').addEventListener('click', closeGenModal);
  document.getElementById('gen-backdrop').addEventListener('click', e => {
    if (e.target === document.getElementById('gen-backdrop')) closeGenModal();
  });
  document.getElementById('gen-preview-btn').addEventListener('click', () => runPreview(false));
  document.getElementById('gen-back-btn').addEventListener('click', () => {
    S.layout = null; S.gen.bw = false;
    resetPrintScale();
    showGenStep(1);
  });

  // Global scale slider (Print preview)
  document.getElementById('print-scale-slider').addEventListener('input', e => {
    if (!S.layout) return;
    const newScale = parseInt(e.target.value);
    const factor   = newScale / S.printGlobalScale;
    document.getElementById('print-scale-val').textContent = newScale + '%';
    S.layout.forEach(item => {
      const cx = item.x + item.w / 2;
      const cy = item.y + item.h / 2;
      item.w = Math.max(10, Math.round(item.w * factor));
      item.h = Math.max(10, Math.round(item.h * factor));
      item.x = Math.round(cx - item.w / 2);
      item.y = Math.round(cy - item.h / 2);
    });
    S.printGlobalScale = newScale;
    renderDraggablePreview();
  });

  // B&W toggle (Print preview)
  document.getElementById('btn-bw-toggle').addEventListener('click', () => {
    S.gen.bw = !S.gen.bw;
    document.getElementById('btn-bw-toggle').classList.toggle('active', S.gen.bw);
    refreshAllItemFilters();
  });

  // Per-element brightness/contrast sliders (Print preview)
  document.getElementById('adj-brightness').addEventListener('input', e => {
    if (S.activeItemIdx === null) return;
    S.layout[S.activeItemIdx].brightness = parseInt(e.target.value);
    document.getElementById('adj-brightness-val').textContent = e.target.value;
    const img = document.querySelector(`#gen-canvas-wrap [data-idx="${S.activeItemIdx}"] img`);
    if (img) applyItemFilter(img, S.layout[S.activeItemIdx].brightness, S.layout[S.activeItemIdx].contrast);
  });
  document.getElementById('adj-contrast').addEventListener('input', e => {
    if (S.activeItemIdx === null) return;
    S.layout[S.activeItemIdx].contrast = parseInt(e.target.value);
    document.getElementById('adj-contrast-val').textContent = e.target.value;
    const img = document.querySelector(`#gen-canvas-wrap [data-idx="${S.activeItemIdx}"] img`);
    if (img) applyItemFilter(img, S.layout[S.activeItemIdx].brightness, S.layout[S.activeItemIdx].contrast);
  });
  document.getElementById('gen-save-btn').addEventListener('click', async () => {
    if (S.layout) {
      try {
        const res = await POST('/api/compose', { layout: S.layout, ...S.gen });
        if (res.ok) {
          S.prints = await GET('/api/templates');
          refreshPrintSeries();
          renderPrints();
          renderText();
        }
      } catch (_) {}
    }
    closeGenModal();
  });
  document.getElementById('gen-close-2').addEventListener('click', closeGenModal);

  document.querySelectorAll('[data-gen]').forEach(wireGenPill);

  // Wire back-text select (options from config + Custom…)
  document.getElementById('gen-back-text-select').addEventListener('change', () => {
    const sel = document.getElementById('gen-back-text-select');
    const customInput = document.getElementById('gen-back-custom-input');
    if (sel.value === '__custom__') {
      customInput.classList.remove('hidden');
      customInput.focus();
      S.gen.customTitle = customInput.value.trim() || null;
    } else {
      customInput.classList.add('hidden');
      customInput.value = '';
      S.gen.customTitle = sel.value || null;
    }
  });
  document.getElementById('gen-back-custom-input').addEventListener('input', e => {
    S.gen.customTitle = e.target.value.trim() || null;
  });

  document.getElementById('lightbox').addEventListener('click', closeLb);
  document.getElementById('lb-close').addEventListener('click', e => { e.stopPropagation(); closeLb(); });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeLb(); closeGenModal(); }
  });

  buildPaperTable();

  boot();
}

document.addEventListener('DOMContentLoaded', init);
