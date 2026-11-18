(function () {
  const input = document.getElementById('image');
  if (input) {
    const dz = document.getElementById('dropzone'), prev = document.getElementById('preview'),
          err = document.getElementById('clientError'), form = document.getElementById('uploadForm'),
          MAX = 16 * 1024 * 1024, OK = ['image/jpeg', 'image/png', 'image/webp'];
    function show(file) {
      err.textContent = '';
      if (!file) { prev.classList.add('d-none'); return false; }
      if (!OK.includes(file.type)) { err.textContent = 'Unsupported file type. Use JPG, PNG or WEBP.'; input.value = ''; return false; }
      if (file.size > MAX) { err.textContent = 'File is larger than 16 MB.'; input.value = ''; return false; }
      prev.src = URL.createObjectURL(file); prev.classList.remove('d-none'); return true;
    }
    input.addEventListener('change', () => show(input.files[0]));
    ['dragenter', 'dragover'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(input.files[0]); } });
    dz.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); input.click(); } });
    document.getElementById('clearBtn').addEventListener('click', () => { input.value = ''; show(null); });
    form.addEventListener('submit', ev => {
      if (!input.files.length) { ev.preventDefault(); err.textContent = 'Please choose an image first.'; return; }
      document.getElementById('spin').classList.remove('d-none'); document.getElementById('submitBtn').disabled = true;
    });
  }
  const data = document.getElementById('stats-data');
  if (data && window.Chart) {
    const s = JSON.parse(data.textContent), labels = Object.keys(s.per_class);
    new Chart(document.getElementById('classChart'), { type: 'doughnut',
      data: { labels: labels.map(l => l.replace('_', ' ')), datasets: [{ data: Object.values(s.per_class), backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#6366f1'] }] },
      options: { maintainAspectRatio: false } });
    new Chart(document.getElementById('trendChart'), { type: 'line',
      data: { labels: s.trend.map(t => t.date), datasets: [{ label: 'Predictions', data: s.trend.map(t => t.count), borderColor: '#14532d', backgroundColor: 'rgba(16,185,129,.2)', fill: true, tension: .3 }] },
      options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
  }
})();
