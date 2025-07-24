// Dashboard module for client-side rendering
const Dashboard = (function() {
  'use strict';

  // DOM element factory
  function el(tag, attrs = {}, children = []) {
    const elem = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') elem.className = v;
      else if (k === 'innerHTML') elem.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') {
        Object.assign(elem.style, v);
      } else elem.setAttribute(k, v);
    });
    children.forEach(child => {
      if (typeof child === 'string') elem.appendChild(document.createTextNode(child));
      else if (child) elem.appendChild(child);
    });
    return elem;
  }

  // Format numbers with appropriate units
  function fmt(num, decimals = 1) {
    const value = Number(num);
    return value > 10 ? value.toFixed(0) : value.toFixed(decimals);
  }

  function fmtMinutes(min) {
    const value = Number(min);
    if (value >= 1440) {
      return (value / 1440).toFixed(1) + 'd';
    }
    if (value >= 60) {
      return Math.round(value / 60) + 'h';
    }
    return Math.round(value) + 'm';
  }

  // Create a stat card
  function statCard(title, value, subtitle, color) {
    return el('div', {className: 'stat-card'}, [
      el('h3', {}, [title]),
      el('p', {className: 'stat-value', style: color ? {color} : {}}, [String(value)]),
      el('p', {className: 'stat-subtitle'}, [subtitle])
    ]);
  }

  // Create a progress card
  function progressCard(title, processed, repairable) {
    const total = processed + repairable;
    const pct = total > 0 ? Math.round((processed / total) * 100) : 100;
    return el('div', {className: 'progress-card'}, [
      el('h3', {}, [title]),
      el('div', {className: 'progress-bar'}, [
        el('div', {
          className: 'progress-fill',
          style: {width: `${pct}%`}
        }, [`${pct}%`])
      ]),
      el('div', {className: 'progress-stats'}, [
        el('span', {}, [`${processed} processed`]),
        el('span', {}, [`${repairable} pending`])
      ])
    ]);
  }

  // Build activity chart
  function buildChart(container, data, config = {}) {
    const {valueKey = 'value', unit = '', color = null, maxBars = 30} = config;
    
    if (!data.length) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    const chart = el('div', {className: 'bar-chart'});
    const maxVal = Math.max(...data.map(d => d[valueKey])) || 1;
    const skip = Math.ceil(data.length / maxBars);

    data.forEach((d, i) => {
      if (i % skip !== 0) return;
      
      const height = (d[valueKey] / maxVal) * 100;
      const bar = el('div', {
        className: 'bar',
        style: color ? {height: `${height}%`, background: color} : {height: `${height}%`}
      });
      
      bar.appendChild(el('div', {className: 'bar-label'}, [d.day || d.label]));
      
      if (d[valueKey] > 0) {
        bar.appendChild(
          el('div', {className: 'bar-value'}, [`${d[valueKey]}${unit}`])
        );
      }
      
      chart.appendChild(bar);
    });

    container.appendChild(chart);
  }

  // Build heatmap
  function buildHeatmap(container, data) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const maxVal = Math.max(...data.flat()) || 1;
    
    const heatmap = el('div', {className: 'heatmap'});
    
    // Empty top-left corner
    heatmap.appendChild(el('div'));
    
    // Hour headers
    const header = el('div', {className: 'heatmap-header'});
    for (let h = 0; h < 24; h++) {
      header.appendChild(el('div', {className: 'heatmap-hour'}, [String(h)]));
    }
    heatmap.appendChild(header);
    
    // Days with cells
    for (let d = 0; d < 7; d++) {
      heatmap.appendChild(el('div', {className: 'heatmap-label'}, [days[d]]));
      
      for (let h = 0; h < 24; h++) {
        const intensity = data[d][h] / maxVal;
        const cell = el('div', {
          className: 'heatmap-cell',
          style: {background: `rgba(102,126,234,${intensity})`},
          title: `${days[d]} ${h}:00 - ${Math.round(data[d][h])} min`
        });
        heatmap.appendChild(cell);
      }
    }
    
    container.appendChild(heatmap);
  }

  // Build topics grid
  function buildTopics(container, counts, minutes, meta = {}) {
    const names = Object.keys(counts);
    if (!names.length) {
      container.style.display = 'none';
      return;
    }
    
    container.style.display = 'block';
    const sorted = names.sort((a, b) => counts[b] - counts[a]);
    const grid = el('div', {className: 'topics-grid'});
    
    sorted.forEach(name => {
      const info = meta[name] || {};
      const title = info.title || name;
      const color = info.color || null;
      const card = el('div', {className: 'topic-card'}, [
        el('div', {className: 'topic-name', style: color ? {color} : {}}, [title]),
        el('div', {className: 'topic-stats'}, [
          el('span', {}, [String(counts[name])]),
          el('span', {}, [fmtMinutes(minutes[name] || 0)])
        ])
      ]);
      grid.appendChild(card);
    });
    
    container.appendChild(el('h2', {}, ['Topics']));
    container.appendChild(grid);
  }

  // Main render function
  function render(data) {
    if (!data) return;
    
    const stats = data.stats || {};
    const summary = data.summary_html || '';
    
    // Clear loading state and notices
    document.getElementById('loading').style.display = 'none';
    document.getElementById('notice').innerHTML = '';
    
    // Show main content
    const main = document.getElementById('mainContent');
    main.style.display = 'block';
    
    // Handle empty data
    if (!stats.days || Object.keys(stats.days).length === 0) {
      document.getElementById('notice').appendChild(
        el('div', {className: 'alert alert-warning'}, [
          el('strong', {}, ['No data available. ']),
          'Run journal_stats.py to generate statistics.'
        ])
      );
      return;
    }
    
    // Calculate derived values
    const days = Object.keys(stats.days).sort();
    const totals = stats.totals || {};
    const totalDays = days.length;
    const totalAudioHours = fmt((stats.total_audio_seconds || 0) / 3600);
    const totalStorageBytes = (stats.total_audio_bytes || 0) + (stats.total_image_bytes || 0);
    const totalStorageMB = totalStorageBytes / (1024 * 1024);
    const totalStorageValue = totalStorageMB >= 1000
      ? fmt(totalStorageMB / 1024)
      : fmt(totalStorageMB);
    const totalStorageUnit = totalStorageMB >= 1000 ? 'GB total' : 'MB total';
    const completion = totals.audio_flac > 0 ?
      Math.round((totals.audio_json / totals.audio_flac) * 100) : 100;
    
    // Render stats cards
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = ''; // Clear existing content
    statsGrid.appendChild(statCard('Total Days', totalDays, 'days recorded'));
    statsGrid.appendChild(statCard('Audio Duration', totalAudioHours, 'hours recorded'));
    statsGrid.appendChild(statCard('Storage Used', totalStorageValue, totalStorageUnit));
    statsGrid.appendChild(statCard('Processing Status', `${completion}%`, 'complete'));
    
    // Render progress cards
    const progressSection = document.getElementById('progressSection');
    progressSection.innerHTML = ''; // Clear existing content
    progressSection.appendChild(
      progressCard('Audio Transcription', totals.audio_json || 0, totals.repair_hear || 0)
    );
    progressSection.appendChild(
      progressCard('Screenshot Analysis', totals.desc_json || 0, totals.repair_see || 0)
    );
    
    // Prepare chart data
    const recent = days.slice(-30);
    const activityData = recent.map(day => ({
      day: day.slice(4, 6) + '/' + day.slice(6, 8),
      value: stats.days[day].activity || 0
    }));
    
    const audioData = recent.map(day => ({
      day: day.slice(4, 6) + '/' + day.slice(6, 8),
      hours: parseFloat(fmt((stats.days[day].audio_seconds || 0) / 3600))
    }));
    
    // Render charts
    buildChart(document.getElementById('activityChart'), activityData);
    buildChart(document.getElementById('audioChart'), audioData, {
      valueKey: 'hours',
      unit: 'h',
      color: 'linear-gradient(to top, #f093fb, #f5576c)'
    });
    
    // Render heatmap
    if (stats.heatmap) {
      buildHeatmap(document.getElementById('heatmap'), stats.heatmap);
    }
    
    // Render topics
    if (stats.topic_counts && Object.keys(stats.topic_counts).length > 0) {
      buildTopics(
        document.getElementById('topicsSection'),
        stats.topic_counts,
        stats.topic_minutes,
        data.topics || {}
      );
    }
    
    // Render repairs if needed
    const repairs = ['repair_hear', 'repair_see', 'repair_reduce', 'repair_entity', 'repair_ponder'];
    const hasRepairs = repairs.some(key => (totals[key] || 0) > 0);
    
    if (hasRepairs) {
      const repairSection = document.getElementById('repairSection');
      const alert = el('div', {className: 'chart-section alert-repair'}, [
        el('h2', {}, ['Items Needing Repair']),
        el('div', {className: 'stats-grid', id: 'repairGrid'})
      ]);
      
      const repairGrid = alert.querySelector('#repairGrid');
      const repairLabels = {
        repair_hear: 'Audio',
        repair_see: 'Screenshots',
        repair_reduce: 'Summaries',
        repair_entity: 'Entities',
        repair_ponder: 'Ponder'
      };
      
      repairs.forEach(key => {
        const count = totals[key] || 0;
        if (count > 0) {
          repairGrid.appendChild(
            statCard(repairLabels[key], count, '', '#f5576c')
          );
        }
      });
      
      repairSection.appendChild(alert);
    }
    
    // Render summary if available
    if (summary) {
      document.getElementById('summarySection').innerHTML = summary;
    }
  }

  // Public API
  return {
    load: function(url) {
      fetch(url, {
        credentials: 'same-origin'  // Include cookies for authentication
      })
        .then(response => {
          if (!response.ok) {
            if (response.status === 401 || response.redirected) {
              // Redirected to login, reload the page
              window.location.reload();
              return;
            }
            throw new Error('Failed to load data');
          }
          return response.json();
        })
        .then(data => {
          if (data) render(data);
        })
        .catch(error => {
          document.getElementById('loading').style.display = 'none';
          document.getElementById('notice').appendChild(
            el('div', {className: 'alert alert-error'}, [
              'Failed to load dashboard data: ' + error.message
            ])
          );
        });
    }
  };
})();

// Export for use in templates
window.Dashboard = Dashboard;