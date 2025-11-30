/**
 * Month Picker Widget
 * Dropdown calendar for date navigation with per-app heat map support.
 */
window.MonthPicker = (function() {
  // State
  let container = null;
  let currentMonth = null;      // YYYYMM being displayed
  let selectedDay = null;       // YYYYMMDD from date-nav
  let dayLabel = null;          // Original day label text
  let app = null;               // Current app name
  let availableDays = new Set();
  let isVisible = false;

  // External elements (set during init)
  let labelEl = null;
  let prevBtn = null;
  let nextBtn = null;

  // Cache: {YYYYMM: {data, facet}}
  const cache = {};
  const providers = {};

  // Constants
  const TODAY = getToday();
  const CURRENT_MONTH = TODAY.slice(0, 6);
  const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

  function getToday() {
    const now = new Date();
    return now.getFullYear() +
      String(now.getMonth() + 1).padStart(2, '0') +
      String(now.getDate()).padStart(2, '0');
  }

  function parseYM(ym) {
    return {
      year: parseInt(ym.slice(0, 4)),
      month: parseInt(ym.slice(4)) - 1
    };
  }

  function formatYM(year, month) {
    const d = new Date(year, month, 1);
    return d.getFullYear() + String(d.getMonth() + 1).padStart(2, '0');
  }

  function adjacentMonth(ym, delta) {
    const { year, month } = parseYM(ym);
    return formatYM(year, month + delta);
  }

  function getMonthLabel(ym) {
    const { year, month } = parseYM(ym);
    const d = new Date(year, month, 1);
    const monthName = d.toLocaleString('default', { month: 'short' });
    const yearShort = String(year).slice(2);
    return `${monthName} '${yearShort}`;
  }

  function getDaysInMonth(ym) {
    const { year, month } = parseYM(ym);
    return new Date(year, month + 1, 0).getDate();
  }

  function getStartDayOfWeek(ym) {
    const { year, month } = parseYM(ym);
    return new Date(year, month, 1).getDay();
  }

  // Data fetching
  async function fetchAvailableDays() {
    try {
      const resp = await fetch('/app/calendar/api/days');
      if (resp.ok) {
        const days = await resp.json();
        availableDays = new Set(days || []);
      }
    } catch (e) {
      console.warn('[MonthPicker] Failed to fetch available days:', e);
    }
  }

  async function fetchMonthData(ym) {
    const provider = providers[app];
    if (!provider) return null;

    try {
      const facet = window.selectedFacet || null;
      return await provider(ym, facet);
    } catch (e) {
      console.warn(`[MonthPicker] Provider error for ${ym}:`, e);
      return null;
    }
  }

  async function getMonthData(ym) {
    const facet = window.selectedFacet || null;
    const cacheKey = ym;

    if (cache[cacheKey]?.facet === facet) {
      return cache[cacheKey].data;
    }

    const data = await fetchMonthData(ym);
    cache[cacheKey] = { data, facet };
    return data;
  }

  function preloadAdjacentMonths(ym) {
    const prev = adjacentMonth(ym, -1);
    const next = adjacentMonth(ym, 1);
    getMonthData(prev);
    getMonthData(next);
  }

  // Rendering
  function render() {
    if (!container) return;

    const data = cache[currentMonth]?.data || {};
    const daysInMonth = getDaysInMonth(currentMonth);
    const startDay = getStartDayOfWeek(currentMonth);

    // Calculate max for heat map scaling
    let maxCount = 0;
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = currentMonth + String(d).padStart(2, '0');
      maxCount = Math.max(maxCount, data[dateStr] || 0);
    }

    // Build HTML (no header - nav is in date-nav bar)
    let html = `
      <div class="mp-weekdays">
        ${WEEKDAYS.map(d => `<span>${d}</span>`).join('')}
      </div>
      <div class="mp-grid">
    `;

    // Leading empty cells
    for (let i = 0; i < startDay; i++) {
      html += `<div class="mp-day mp-other"></div>`;
    }

    // Days of month
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = currentMonth + String(d).padStart(2, '0');
      const count = data[dateStr] || 0;
      const exists = availableDays.has(dateStr);

      const classes = ['mp-day'];
      if (dateStr === TODAY) classes.push('mp-today');
      if (dateStr === selectedDay) classes.push('mp-selected');
      if (count === 0 || !exists) classes.push('mp-empty');

      const rawIntensity = maxCount > 0 ? count / maxCount : 0;
      const intensity = count > 0 ? 0.2 + (rawIntensity * 0.8) : 0;

      html += `<div class="${classes.join(' ')}" data-day="${dateStr}" style="--intensity: ${intensity}">${d}</div>`;
    }

    // Trailing empty cells
    const totalCells = startDay + daysInMonth;
    const remainder = totalCells % 7;
    if (remainder > 0) {
      for (let i = 0; i < 7 - remainder; i++) {
        html += `<div class="mp-day mp-other"></div>`;
      }
    }

    html += '</div>';
    container.innerHTML = html;

    // Update the date-nav label to show month
    if (labelEl) {
      labelEl.textContent = getMonthLabel(currentMonth);
    }
  }

  async function showMonth(ym) {
    currentMonth = ym;
    await getMonthData(ym);
    render();
    preloadAdjacentMonths(ym);
  }

  function navigateMonth(delta) {
    if (!currentMonth) return;
    showMonth(adjacentMonth(currentMonth, delta));
  }

  // Event handlers
  function handleClick(e) {
    const day = e.target.closest('.mp-day');
    if (day && !day.classList.contains('mp-other') && !day.classList.contains('mp-empty')) {
      const dateStr = day.dataset.day;
      if (dateStr) {
        window.location.href = `/app/${app}/${dateStr}`;
      }
    }
  }

  function handleKeydown(e) {
    if (!isVisible) return;

    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    }
  }

  function handleClickOutside(e) {
    if (!isVisible) return;
    const dateNav = document.querySelector('.date-nav');
    const facetBar = document.querySelector('.facet-bar');
    // Only close if clicking outside both date-nav and facet-bar
    if (dateNav && !dateNav.contains(e.target) && (!facetBar || !facetBar.contains(e.target))) {
      hide();
    }
  }

  function handleFacetSwitch() {
    if (isVisible && currentMonth) {
      showMonth(currentMonth);
    }
  }

  // Public API
  function init(options) {
    app = options.app;
    selectedDay = options.currentDay;
    currentMonth = selectedDay ? selectedDay.slice(0, 6) : CURRENT_MONTH;

    container = document.querySelector(options.container || '.month-picker');
    labelEl = document.getElementById('date-nav-label');
    prevBtn = document.getElementById('date-nav-prev');
    nextBtn = document.getElementById('date-nav-next');

    if (labelEl) {
      dayLabel = labelEl.textContent;
    }

    if (!container) {
      console.warn('[MonthPicker] Container not found');
      return;
    }

    container.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('click', handleClickOutside);
    window.addEventListener('facet.switch', handleFacetSwitch);

    // Prefetch data
    fetchAvailableDays().then(() => {
      getMonthData(currentMonth);
    });
  }

  function show() {
    if (!container) return;
    isVisible = true;
    container.classList.add('open');
    document.querySelector('.date-nav')?.classList.add('picker-open');
    showMonth(currentMonth);
  }

  function hide() {
    if (!container) return;
    isVisible = false;
    container.classList.remove('open');
    document.querySelector('.date-nav')?.classList.remove('picker-open');

    // Restore day label
    if (labelEl && dayLabel) {
      labelEl.textContent = dayLabel;
    }
  }

  function toggle() {
    isVisible ? hide() : show();
  }

  function registerDataProvider(appName, fn) {
    providers[appName] = fn;
  }

  return {
    init,
    show,
    hide,
    toggle,
    isOpen: () => isVisible,
    navigateMonth,
    registerDataProvider
  };
})();
