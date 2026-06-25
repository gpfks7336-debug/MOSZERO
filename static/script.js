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

  if (localStorage.getItem('theme') === 'dark') applyTheme(true);

  themeLightBtn.addEventListener('click', () => applyTheme(false));
  themeDarkBtn.addEventListener('click',  () => applyTheme(true));

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

  function applyChartTheme(chartObj) {
    if (isDark()) {
      chartObj.layout.paper_bgcolor = '#292524';
      chartObj.layout.plot_bgcolor  = '#1c1917';
      if (chartObj.layout.xaxis) { chartObj.layout.xaxis.gridcolor = '#3a3531'; chartObj.layout.xaxis.color = '#a8a29e'; }
      if (chartObj.layout.yaxis) { chartObj.layout.yaxis.gridcolor = '#3a3531'; chartObj.layout.yaxis.color = '#a8a29e'; }
      chartObj.layout.font = { ...(chartObj.layout.font || {}), color: '#a8a29e', family: 'Pretendard, sans-serif', size: 12 };
      if (chartObj.data?.[0]?.marker) chartObj.data[0].marker.color = '#524b45';
      if (chartObj.data?.[1]?.line)   chartObj.data[1].line.color   = '#e2e1da';
    } else {
      chartObj.layout.paper_bgcolor = '#ffffff';
      chartObj.layout.plot_bgcolor  = '#ffffff';
      if (chartObj.layout.xaxis) { chartObj.layout.xaxis.gridcolor = '#e3e2dd'; chartObj.layout.xaxis.color = '#78716c'; }
      if (chartObj.layout.yaxis) { chartObj.layout.yaxis.gridcolor = '#e3e2dd'; chartObj.layout.yaxis.color = '#78716c'; }
      chartObj.layout.font = { ...(chartObj.layout.font || {}), color: '#292524', family: 'Pretendard, sans-serif', size: 12 };
      if (chartObj.data?.[0]?.marker) chartObj.data[0].marker.color = '#cbd5e1';
      if (chartObj.data?.[1]?.line)   chartObj.data[1].line.color   = '#2563eb';
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

  // 💡 [HTML 구조 매칭 완료]: tip-text가 tip-badge 내부에 정렬 타겟으로 종속됨
  function renderFeatureTable(features) {
    const tbody = document.getElementById('featureBody');
    tbody.innerHTML = '';
    features.forEach(f => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>
          <span class="tip-wrap">
            <strong style="font-weight:600; color:inherit;">${escHtml(f.name)}</strong>
            <span class="tip-badge">
              ?
              <span class="tip-text">${escHtml(f.tip)}</span>
            </span>
          </span>
        </td>
        <td class="mono">${f.value}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // 🌟 심사위원 맞춤형 "불량 유형 매칭 확률" 게이지 바 연동
  function renderReport(top2) {
    const grid = document.getElementById('reportGrid');
    grid.innerHTML = '';
    const labels = ['Primary Finding', 'Secondary Hypothesis'];

    top2.forEach((item, i) => {
      const card = document.createElement('div');
      card.className = 'diagnostic-card';

      let refHtml = '';
      if (item.papers?.length) {
        const refItems = item.papers.map((p) => `
          <div class="academic-card">
            <h4 class="academic-title">${escHtml(p.title)}</h4>
            <div class="academic-meta-badge">${escHtml(p.journal)}</div>
            <div class="ai-summary-box">
              <span class="ai-summary-tag">✦ AI 핵심 분석 및 연구 요약</span>
              <p class="ai-summary-text">${escHtml(p.summary)}</p>
            </div>
            <a href="${escHtml(p.link)}" target="_blank" rel="noopener noreferrer" class="view-paper-btn">
              원문 보기 ➔
            </a>
          </div>
        `).join('');

        refHtml = `
          <div class="ref-section">
            <button class="ref-toggle-btn" onclick="toggleAccordion(this)">
              <span>References (${item.papers.length})</span>
              <span class="ref-chevron">▾</span>
            </button>
            <div class="ref-entries hidden">
              <div class="papers-container">
                ${refItems}
              </div>
            </div>
          </div>
        `;
      }

      card.innerHTML = `
        <div style="font-size:0.75rem; font-weight:700; color:#78716c; text-transform:uppercase; letter-spacing:0.05em;">${labels[i]}</div>
        <div style="font-size:1.35rem; font-weight:700; letter-spacing:-0.02em;">${escHtml(item.category)}</div>
        
        <div class="confidence-container">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:600; color:#78716c; margin-bottom:4px;">
            <span>불량 유형 매칭 확률</span>
            <span>${item.prob}%</span>
          </div>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" id="gauge-${i}"></div>
          </div>
        </div>

        <div style="font-size:0.92rem; line-height:1.6; color:inherit; margin-top:4px; word-break:keep-all;">${escHtml(item.desc)}</div>
        ${refHtml}
      `;
      grid.appendChild(card);

      setTimeout(() => {
        const fillBar = document.getElementById(`gauge-${i}`);
        if (fillBar) fillBar.style.width = `${item.prob}%`;
      }, 50);
    });
  }

  window.toggleAccordion = btn => {
    const content = btn.nextElementSibling;
    const opening = content.classList.contains('hidden');
    content.classList.toggle('hidden', !opening);
    btn.classList.toggle('open', opening);
  };

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