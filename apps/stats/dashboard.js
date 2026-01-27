// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

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

  // Format byte counts with GB/MB/KB suffixes
  function fmtBytes(num) {
    const value = Number(num);
    if (value >= 1e12) return (value / 1e12).toFixed(1) + ' TB';
    if (value >= 1e9) return (value / 1e9).toFixed(1) + ' GB';
    if (value >= 1e6) return (value / 1e6).toFixed(1) + ' MB';
    if (value >= 1e3) return (value / 1e3).toFixed(1) + ' KB';
    return String(Math.round(value)) + ' B';
  }

  // Format token counts with Bil/Mil suffixes
  function fmtTokens(num) {
    const value = Number(num);
    if (value >= 1e9) {
      return (value / 1e9).toFixed(1) + 'B';
    }
    if (value >= 1e6) {
      return (value / 1e6).toFixed(1) + 'M';
    }
    if (value >= 1e3) {
      return (value / 1e3).toFixed(1) + 'K';
    }
    return String(Math.round(value));
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

  // Build stacked token chart
  function buildTokenChart(container, tokenData, model) {
    container.innerHTML = ''; // Clear existing content
    
    if (!tokenData || !model) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['Select a model to view token usage'])
      );
      return;
    }

    // Get last 30 days of data
    const days = Object.keys(tokenData).sort().slice(-30);
    
    if (!days.length) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No token data available for this model'])
      );
      return;
    }

    // Calculate max total for scaling
    let maxTotal = 0;
    const chartData = days.map(day => {
      let input = 0, reasoning = 0, output = 0;

      if (model === 'total') {
        // Sum all models for this day
        const dayModels = tokenData[day] || {};
        Object.values(dayModels).forEach(modelData => {
          input += modelData.input_tokens || 0;
          reasoning += modelData.reasoning_tokens || 0;
          output += modelData.output_tokens || 0;
        });
      } else {
        // Single model data
        const dayData = tokenData[day][model] || {};
        input = dayData.input_tokens || 0;
        reasoning = dayData.reasoning_tokens || 0;
        output = dayData.output_tokens || 0;
      }

      const total = input + reasoning + output;
      maxTotal = Math.max(maxTotal, total);
      return { day, input, reasoning, output, total };
    });

    if (maxTotal === 0) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No token usage for this model in the last 30 days'])
      );
      return;
    }

    const chart = el('div', {className: 'bar-chart'});
    
    chartData.forEach(d => {
      const height = (d.total / maxTotal) * 100;
      const bar = el('div', {
        className: 'bar',
        style: {height: `${height}%`, background: 'transparent', overflow: 'visible'}
      });
      
      // Create stacked segments
      const stack = el('div', {className: 'bar-stack', style: {height: '100%'}});
      
      // Calculate segment heights as percentages of the bar
      if (d.total > 0) {
        const inputPct = (d.input / d.total) * 100;
        const reasoningPct = (d.reasoning / d.total) * 100;
        const outputPct = (d.output / d.total) * 100;

        if (d.output > 0) {
          stack.appendChild(el('div', {
            className: 'stack-segment output',
            style: {height: `${outputPct}%`}
          }));
        }
        if (d.reasoning > 0) {
          stack.appendChild(el('div', {
            className: 'stack-segment reasoning',
            style: {height: `${reasoningPct}%`}
          }));
        }
        if (d.input > 0) {
          stack.appendChild(el('div', {
            className: 'stack-segment input',
            style: {height: `${inputPct}%`}
          }));
        }
      }
      
      bar.appendChild(stack);

      if (d.total > 0) {
        let formatted;
        if (d.total >= 1000000) {
          formatted = `${Math.round(d.total/1000000)}m`;
        } else if (d.total >= 1000) {
          formatted = `${Math.round(d.total/1000)}k`;
        } else {
          formatted = String(d.total);
        }
        bar.appendChild(el('div', {className: 'bar-value'}, [formatted]));
        bar.setAttribute('title', `${d.day.slice(4, 6)}/${d.day.slice(6, 8)} - Input: ${d.input}, Reasoning: ${d.reasoning}, Output: ${d.output}`);
      }
      
      chart.appendChild(bar);
    });

    container.appendChild(chart);
    
    // Add legend
    const legend = el('div', {className: 'token-legend'}, [
      el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: '#667eea'}}),
        'Input'
      ]),
      el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: '#9b59b6'}}),
        'Reasoning'
      ]),
      el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: '#e91e63'}}),
        'Output'
      ])
    ]);
    container.appendChild(legend);
  }

  // Build stacked hours chart (audio + screen)
  function buildStackedHoursChart(container, data) {
    container.innerHTML = ''; // Clear existing content

    if (!data || !data.length) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    // Calculate max total for scaling
    const maxTotal = Math.max(...data.map(d => d.audio + d.screen)) || 1;

    const chart = el('div', {className: 'bar-chart'});

    data.forEach(d => {
      const total = d.audio + d.screen;
      const height = (total / maxTotal) * 100;
      const bar = el('div', {
        className: 'bar',
        style: {height: `${height}%`, background: 'transparent', overflow: 'visible'}
      });

      // Create stacked segments
      const stack = el('div', {className: 'bar-stack', style: {height: '100%'}});

      // Calculate segment heights as percentages of the bar
      if (total > 0) {
        const audioPct = (d.audio / total) * 100;
        const screenPct = (d.screen / total) * 100;

        // Screen on top (pink/purple)
        if (d.screen > 0) {
          stack.appendChild(el('div', {
            className: 'stack-segment',
            style: {
              height: `${screenPct}%`,
              background: '#e91e63'
            }
          }));
        }
        // Audio on bottom (blue/purple)
        if (d.audio > 0) {
          stack.appendChild(el('div', {
            className: 'stack-segment',
            style: {
              height: `${audioPct}%`,
              background: '#667eea'
            }
          }));
        }
      }

      bar.appendChild(stack);

      if (total > 0) {
        const formatted = total > 10 ? Math.round(total) : total.toFixed(1);
        bar.appendChild(el('div', {className: 'bar-value'}, [`${formatted}h`]));
        const titleParts = [`${d.day} - Audio: ${d.audio.toFixed(1)}h, Screen: ${d.screen.toFixed(1)}h`];
        if (d.bytes) titleParts.push(`Disk: ${fmtBytes(d.bytes)}`);
        bar.setAttribute('title', titleParts.join(', '));
      }

      chart.appendChild(bar);
    });

    container.appendChild(chart);

    // Add legend
    const legend = el('div', {className: 'token-legend'}, [
      el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: '#667eea'}}),
        'Audio'
      ]),
      el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: '#e91e63'}}),
        'Screen'
      ])
    ]);
    container.appendChild(legend);
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

  // Generate consistent colors for categories
  function getCategoryColor(index, total) {
    // Use a palette of distinct colors
    const palette = [
      '#667eea', '#764ba2', '#e91e63', '#00bcd4', '#4caf50',
      '#ff9800', '#9c27b0', '#3f51b5', '#009688', '#ff5722',
      '#795548', '#607d8b', '#8bc34a', '#ffc107', '#673ab7'
    ];
    return palette[index % palette.length];
  }

  // Build stacked category chart (for Events or Facets)
  function buildStackedCategoryChart(container, countsByDay, meta = {}) {
    container.innerHTML = '';

    if (!countsByDay || Object.keys(countsByDay).length === 0) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    // Get last 30 days sorted
    const days = Object.keys(countsByDay).sort().slice(-30);
    if (!days.length) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    // Collect all unique categories across all days
    const allCategories = new Set();
    days.forEach(day => {
      Object.keys(countsByDay[day] || {}).forEach(cat => allCategories.add(cat));
    });
    const categories = Array.from(allCategories).sort();

    if (!categories.length) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    // Assign colors to categories
    const categoryColors = {};
    categories.forEach((cat, i) => {
      const info = meta[cat] || {};
      categoryColors[cat] = info.color || getCategoryColor(i, categories.length);
    });

    // Calculate max total for scaling
    let maxTotal = 0;
    const chartData = days.map(day => {
      const dayCounts = countsByDay[day] || {};
      const total = Object.values(dayCounts).reduce((sum, c) => sum + c, 0);
      maxTotal = Math.max(maxTotal, total);
      return { day, counts: dayCounts, total };
    });

    if (maxTotal === 0) {
      container.appendChild(
        el('div', {className: 'empty-chart'}, ['No data available'])
      );
      return;
    }

    const chart = el('div', {className: 'bar-chart'});

    chartData.forEach(d => {
      const height = (d.total / maxTotal) * 100;
      const bar = el('div', {
        className: 'bar',
        style: {height: `${height}%`, background: 'transparent', overflow: 'visible'}
      });

      // Create stacked segments
      const stack = el('div', {className: 'bar-stack', style: {height: '100%'}});

      // Build tooltip showing breakdown
      const tooltipParts = [`${d.day.slice(4, 6)}/${d.day.slice(6, 8)}`];

      if (d.total > 0) {
        // Stack segments from top to bottom (reverse order for visual stacking)
        categories.slice().reverse().forEach(cat => {
          const count = d.counts[cat] || 0;
          if (count > 0) {
            const pct = (count / d.total) * 100;
            const info = meta[cat] || {};
            const title = info.title || cat;
            stack.appendChild(el('div', {
              className: 'stack-segment',
              style: {
                height: `${pct}%`,
                background: categoryColors[cat]
              }
            }));
            tooltipParts.push(`${title}: ${count}`);
          }
        });
      }

      bar.appendChild(stack);

      if (d.total > 0) {
        bar.appendChild(el('div', {className: 'bar-value'}, [String(d.total)]));
        bar.setAttribute('title', tooltipParts.join('\n'));
      }

      chart.appendChild(bar);
    });

    container.appendChild(chart);

    // Add legend
    const legend = el('div', {className: 'token-legend'});
    categories.forEach(cat => {
      const info = meta[cat] || {};
      const title = info.title || cat;
      legend.appendChild(el('div', {className: 'legend-item'}, [
        el('div', {className: 'legend-color', style: {background: categoryColors[cat]}}),
        title
      ]));
    });
    container.appendChild(legend);
  }

  // Main render function
  function render(data) {
    if (!data) return;

    const stats = data.stats || {};

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
          'Run think-journal-stats to generate statistics.'
        ])
      );
      return;
    }
    
    // Calculate derived values
    const days = Object.keys(stats.days).sort();
    const totals = stats.totals || {};
    const totalDays = days.length;
    const totalAudioHours = Math.round((stats.total_audio_duration || 0) / 3600);
    const totalScreenHours = Math.round((stats.total_screen_duration || 0) / 3600);

    // Calculate total tokens across all models
    const tokenTotals = stats.token_totals_by_model || {};
    const totalTokens = Object.values(tokenTotals).reduce((sum, model) => {
      return sum + (model.total_tokens || 0);
    }, 0);

    // Render stats cards
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = ''; // Clear existing content
    statsGrid.appendChild(statCard('Total Days', totalDays, 'days recorded'));
    statsGrid.appendChild(statCard('Audio Hours', totalAudioHours, 'hours'));
    statsGrid.appendChild(statCard('Screen Hours', totalScreenHours, 'hours'));
    statsGrid.appendChild(statCard('Total Tokens', fmtTokens(totalTokens), 'tokens'));
    statsGrid.appendChild(statCard('Disk Usage', fmtBytes(totals.day_bytes || 0), 'journal days'));
    
    // Render progress cards
    const progressSection = document.getElementById('progressSection');
    progressSection.innerHTML = ''; // Clear existing content
    progressSection.appendChild(
      progressCard('Audio Processing', totals.audio_sessions || 0, totals.pending_segments || 0)
    );
    progressSection.appendChild(
      progressCard('Agent Outputs', totals.outputs_processed || 0, totals.outputs_pending || 0)
    );
    
    // Token usage setup
    const tokenUsage = stats.token_usage_by_day || {};
    const models = Object.keys(tokenTotals).sort();
    
    // Populate model selector
    const modelSelector = document.getElementById('modelSelector');
    if (models.length > 0) {
      modelSelector.innerHTML = '';
      
      // Add "Total" option first
      const totalOption = el('option', {value: 'total'}, ['Total']);
      modelSelector.appendChild(totalOption);
      
      // Add individual models
      models.forEach(model => {
        const option = el('option', {value: model}, [model]);
        modelSelector.appendChild(option);
      });
      
      // Set total as default
      modelSelector.value = 'total';
      
      // Initial render
      buildTokenChart(document.getElementById('tokenChart'), tokenUsage, 'total');
      
      // Handle model selection changes
      modelSelector.addEventListener('change', function() {
        buildTokenChart(document.getElementById('tokenChart'), tokenUsage, this.value);
      });
    } else {
      // No token data available
      buildTokenChart(document.getElementById('tokenChart'), null, null);
    }
    
    // Combined audio + screen chart data
    const recent = days.slice(-30);
    const hoursData = recent.map(day => {
      const dayData = stats.days[day];
      const audioHours = (dayData.audio_duration || 0) / 3600;
      const screenHours = (dayData.screen_duration || 0) / 3600;
      return {
        day: day.slice(4, 6) + '/' + day.slice(6, 8),
        audio: audioHours,
        screen: screenHours,
        bytes: dayData.day_bytes || 0
      };
    });

    // Render stacked hours chart
    buildStackedHoursChart(document.getElementById('audioChart'), hoursData);
    
    // Render heatmap
    if (stats.heatmap) {
      buildHeatmap(document.getElementById('heatmap'), stats.heatmap);
    }

    // Render Facets stacked bar chart
    buildStackedCategoryChart(
      document.getElementById('facetsChart'),
      stats.facet_counts_by_day || {},
      {}  // No special metadata for facets
    );

    // Render Events stacked bar chart
    buildStackedCategoryChart(
      document.getElementById('eventsChart'),
      stats.topic_counts_by_day || {},
      data.insights || {}  // Use insight metadata for titles/colors
    );
    
    // Render repairs if needed
    const repairs = ['pending_segments', 'outputs_pending'];
    const hasRepairs = repairs.some(key => (totals[key] || 0) > 0);

    if (hasRepairs) {
      const repairSection = document.getElementById('repairSection');
      const alert = el('div', {className: 'chart-section alert-repair'}, [
        el('h2', {}, ['Items Needing Processing']),
        el('div', {className: 'stats-grid', id: 'repairGrid'})
      ]);

      const repairGrid = alert.querySelector('#repairGrid');
      const repairLabels = {
        pending_segments: 'Pending Segments',
        outputs_pending: 'Agent Outputs'
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