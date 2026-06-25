(() => {
  // ── State ──
  let currentFile   = null;
  let lastChartJson = null;

  // ── DOM refs ──
  const uploadArea      = document.getElementById('uploadArea');
  const fileInput       = document.getElementById('fileInput');
  const fileNameEl      = document.getElementById('fileName');
  const colSelectArea   = document.getElementById('colSelectArea');
  const vColEl          = document.getElementById('vCol');
  const cColEl          = document.getElementById('cCol');
  const analyzeBtn      = document.getElementById('analyzeBtn');
  const loadingArea     = document.getElementById('loadingArea');
  const errorArea       = document.getElementById('errorArea');
  const resultsSection  = document.getElementById('resultsSection');
  const themeLightBtn   = document.getElementById('themeLight');
  const themeDarkBtn    = document.getElementById('themeDark');

  // ── Dark mode ──
  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function applyTheme(dark) {
    document.documentElement.classList.toggle('dark', dark);
    themeLightBtn.classList.toggle('active', !dark);
    themeDarkBtn.classList.toggle('active', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
    if (lastChartJson) rerenderChart();
  }

  // Restore saved theme on load
  if (localStorage.getItem('theme') === 'dark') applyTheme(true);

  themeLightBtn.addEventListener('click', () => applyTheme(false));
  themeDarkBtn.addEventListener('click',  () => applyTheme(true));

  // ── Step indicator ──
  function setStep(n) {
    [1, 2, 3].forEach(i => {
      const el = document.getElementById(`step${i}`);
      if (!el) return;
      el.classList.remove('active', 'done');
      if (i < n)      el.classList.add('done');
      else if (i === n) el.classList.add('active');
    });
  }

  // ── Upload interactions ──
  uploadArea.addEventListener('click', () => fileInput.click());

  uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
  });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  });
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFileSelected(e.target.files[0]);
  });

  async function handleFileSelected(file) {
    currentFile = file;
    fileNameEl.textContent = file.name;
    fileNameEl.classList.remove('hidden');
    hideError();
    resultsSection.classList.add('hidden');
    colSelectArea.classList.add('hidden');
    lastChartJson = null;

    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch('/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) { showError(data.error); return; }
      populateSelects(data.columns);
      colSelectArea.classList.remove('hidden');
      setStep(2);
    } catch {
      showError('파일 업로드 중 오류가 발생했습니다.');
    }
  }

  function populateSelects(cols) {
    [vColEl, cColEl].forEach((sel, idx) => {
      sel.innerHTML = '';
      cols.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
      if (idx === 1 && cols.length > 1) sel.selectedIndex = 1;
    });
  }

  // ── Analyze ──
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    hideError();
    resultsSection.classList.add('hidden');
    loadingArea.classList.remove('hidden');
    analyzeBtn.disabled = true;

    const fd = new FormData();
    fd.append('file', currentFile);
    fd.append('v_col', vColEl.value);
    fd.append('c_col', cColEl.value);

    try {
      const res  = await fetch('/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      loadingArea.classList.add('hidden');
      analyzeBtn.disabled = false;
      if (data.error) { showError(data.error); return; }
      renderResults(data);
      setStep(3);
    } catch {
      loadingArea.classList.add('hidden');
      analyzeBtn.disabled = false;
      showError('분석 중 오류가 발생했습니다.');
    }
  });

  // ── Render pipeline ──
  function renderResults(data) {
    lastChartJson = data.chart;
    renderChart(data.chart);
    renderFeatureTable(data.features);
    renderReport(data.top2);
    resultsSection.classList.remove('hidden');
  }

  // Patch Plotly layout/trace colors for current theme
  function applyChartTheme(chartObj) {
    if (isDark()) {
      // Warm dark palette — stone/amber
      chartObj.layout.paper_bgcolor = '#1c1917';
      chartObj.layout.plot_bgcolor  = '#292524';
      if (chartObj.layout.xaxis) { chartObj.layout.xaxis.gridcolor = '#3a3531'; chartObj.layout.xaxis.color = '#78716c'; }
      if (chartObj.layout.yaxis) { chartObj.layout.yaxis.gridcolor = '#3a3531'; chartObj.layout.yaxis.color = '#78716c'; }
      chartObj.layout.font = { ...(chartObj.layout.font || {}), color: '#a8a29e', family: 'DM Sans, sans-serif', size: 12 };
      if (chartObj.data?.[0]?.marker) chartObj.data[0].marker.color = '#524b45';
      if (chartObj.data?.[1]?.line)   chartObj.data[1].line.color   = '#e0b050';  // amber accent
    } else {
      chartObj.layout.paper_bgcolor = '#f5f4f0';
      chartObj.layout.plot_bgcolor  = '#ffffff';
      if (chartObj.layout.xaxis) { chartObj.layout.xaxis.gridcolor = '#e3e2dd'; chartObj.layout.xaxis.color = '#a8a29e'; }
      if (chartObj.layout.yaxis) { chartObj.layout.yaxis.gridcolor = '#e3e2dd'; chartObj.layout.yaxis.color = '#a8a29e'; }
      chartObj.layout.font = { ...(chartObj.layout.font || {}), color: '#57534e', family: 'DM Sans, sans-serif', size: 12 };
      if (chartObj.data?.[0]?.marker) chartObj.data[0].marker.color = '#d4d1cb';
      if (chartObj.data?.[1]?.line)   chartObj.data[1].line.color   = '#1e3a5f';
    }
    return chartObj;
  }

  function renderChart(chartJson) {
    const chartObj = applyChartTheme(JSON.parse(chartJson));
    Plotly.newPlot('plotlyChart', chartObj.data, chartObj.layout, {
      responsive: true,
      displayModeBar: 'hover',
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines'],
      displaylogo: false
    });
  }

  function rerenderChart() {
    if (!lastChartJson) return;
    const chartObj = applyChartTheme(JSON.parse(lastChartJson));
    Plotly.react('plotlyChart', chartObj.data, chartObj.layout);
  }

  function renderFeatureTable(features) {
    const tbody = document.getElementById('featureBody');
    tbody.innerHTML = '';
    features.forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>
          <span class="tip-wrap">
            <strong style="font-weight:500;">${escHtml(f.name)}</strong>
            <span class="tip-badge">?</span>
            <span class="tip-text">${escHtml(f.tip)}</span>
          </span>
        </td>
        <td class="mono">${f.value}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderReport(top2) {
    const grid = document.getElementById('reportGrid');
    grid.innerHTML = '';
    const labels = ['Primary Finding', 'Secondary Hypothesis'];

    top2.forEach((item, i) => {
      const card = document.createElement('div');
      card.className = i === 0 ? 'report-card-primary' : 'report-card-secondary';

      let refHtml = '';
      if (item.papers?.length) {
        const extLinkIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`;

        const refItems = item.papers.map((p, j) => `
          <div class="ref-card">
            <div class="ref-card-header">
              <div class="ref-card-num">${j + 1}</div>
              <div class="ref-card-content">
                <div class="ref-card-title">${escHtml(p.title)}</div>
                <div class="ref-card-meta">${escHtml(p.journal)}</div>
              </div>
            </div>
            <div class="ref-card-insight">
              <span class="ref-card-insight-label">AI Insight</span>
              <p class="ref-card-insight-text">${escHtml(p.summary)}</p>
            </div>
            <div class="ref-card-footer">
              <a href="${escHtml(p.link)}" target="_blank" rel="noopener noreferrer" class="ref-card-link">
                ${extLinkIcon} View full paper
              </a>
            </div>
          </div>
        `).join('');

        refHtml = `
          <div class="ref-section">
            <button class="ref-toggle-btn" onclick="toggleAccordion(this)">
              <span class="ref-header-label">References</span>
              <span class="ref-header-rule"></span>
              <span class="ref-chevron">▾</span>
            </button>
            <div class="ref-entries hidden">
              ${refItems}
              <div class="disclaimer">※ 본 논문 정보는 AI가 매칭한 것으로 정확하지 않을 수 있습니다.</div>
            </div>
          </div>
        `;
      }

      card.innerHTML = `
        <div class="finding-label">${labels[i]}</div>
        <div class="${i === 0 ? 'defect-name-primary' : 'defect-name-secondary'}">${escHtml(item.category)}</div>
        <span class="conf-stat">${item.prob}% confidence</span>
        <div class="defect-desc">${escHtml(item.desc)}</div>
        ${refHtml}
      `;
      grid.appendChild(card);
    });
  }

  // ── Global accordion toggle ──
  window.toggleAccordion = btn => {
    const content = btn.nextElementSibling;
    const opening = content.classList.contains('hidden');
    content.classList.toggle('hidden', !opening);
    btn.classList.toggle('open', opening);
  };

  // ── Helpers ──
  function showError(msg) {
    errorArea.textContent = msg;
    errorArea.classList.remove('hidden');
  }
  function hideError() {
    errorArea.classList.add('hidden');
    errorArea.textContent = '';
  }
  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
