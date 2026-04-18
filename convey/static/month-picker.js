// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 sol pbc

/**
 * Month Picker Widget
 * Dropdown calendar for date navigation with per-app heat map support.
 *
 * Day clickability is derived from stats data - if a day has data in the
 * stats response (count > 0), it's clickable. Future dates can be clickable
 * via allowFutureDates option even without data.
 */
window.MonthPicker = (function() {
  // State
  let container = null;
  let currentMonth = null;      // YYYYMM being displayed
  let selectedDay = null;       // YYYYMMDD from date-nav
  let dayLabel = null;          // Original day label text
  let app = null;               // Current app name
  let isVisible = false;
  let allowFutureDates = false; // Whether future dates without data are clickable
  let pendingFocusDay = null;   // Set before render to auto-focus: YYYYMMDD, 'first', or 'last'

  // External elements (set during init)
  let labelEl = null;

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

  function isFutureDay(dateStr) {
    return dateStr > TODAY;
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

  function getFullMonthLabel(ym) {
    const { year, month } = parseYM(ym);
    const d = new Date(year, month, 1);
    return d.toLocaleString('default', { month: 'long' }) + ' ' + year;
  }

  function formatDayLabel(dateStr) {
    const year = parseInt(dateStr.slice(0, 4), 10);
    const month = parseInt(dateStr.slice(4, 6), 10) - 1;
    const day = parseInt(dateStr.slice(6, 8), 10);
    return new Date(year, month, day).toLocaleDateString('default', { weekday: 'long', month: 'long', day: 'numeric' });
  }

  function isFocusable(cell) {
    return !cell.classList.contains('mp-other') && !cell.classList.contains('mp-empty');
  }

  function getActiveDay(focusableCells) {
    return focusableCells.find(el => el.dataset.day === selectedDay) ||
           focusableCells.find(el => el.dataset.day === TODAY) ||
           focusableCells[0] || null;
  }

  function focusCell(cell) {
    const grid = container.querySelector('.mp-grid');
    if (!grid) return;
    grid.querySelectorAll('.mp-day[tabindex]').forEach(el => {
      el.setAttribute('tabindex', '-1');
    });
    cell.setAttribute('tabindex', '0');
    cell.focus();
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

    // Build cells array
    const cells = [];

    // Leading empty cells
    for (let i = 0; i < startDay; i++) {
      cells.push('<div class="mp-day mp-other" role="gridcell"></div>');
    }

    // Days of month
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = currentMonth + String(d).padStart(2, '0');
      const count = data[dateStr] || 0;
      const hasData = dateStr in data && count > 0;
      const isFuture = isFutureDay(dateStr);

      const classes = ['mp-day'];
      if (dateStr === TODAY) classes.push('mp-today');
      if (dateStr === selectedDay) classes.push('mp-selected');

      if (isFuture && !hasData && allowFutureDates) {
        classes.push('mp-future');
      } else if (!hasData) {
        classes.push('mp-empty');
      }

      const rawIntensity = maxCount > 0 ? count / maxCount : 0;
      const intensity = count > 0 ? 0.2 + (rawIntensity * 0.8) : 0;
      const dateLabel = formatDayLabel(dateStr);
      const countText = count === 1 ? '1 item' : count > 0 ? `${count} items` : 'no items';

      let attrs = `class="${classes.join(' ')}" role="gridcell" data-day="${dateStr}" style="--intensity: ${intensity}" aria-label="${dateLabel}, ${countText}"`;
      if (dateStr === TODAY) attrs += ' aria-current="date"';
      if (dateStr === selectedDay) attrs += ' aria-selected="true"';
      if (count > 0) attrs += ` title="${countText}"`;

      cells.push(`<div ${attrs}>${d}</div>`);
    }

    // Trailing empty cells
    const remainder = cells.length % 7;
    if (remainder > 0) {
      for (let i = 0; i < 7 - remainder; i++) {
        cells.push('<div class="mp-day mp-other" role="gridcell"></div>');
      }
    }

    // Group cells into rows of 7
    let gridHtml = '';
    for (let i = 0; i < cells.length; i += 7) {
      gridHtml += `<div role="row">${cells.slice(i, i + 7).join('')}</div>`;
    }

    // Build full HTML — outer div owns the ARIA grid role so the
    // columnheader row and the data rows share the same grid context.
    const html = `
      <div role="grid" aria-label="${getFullMonthLabel(currentMonth)}">
        <div class="mp-weekdays" role="row">
          ${WEEKDAYS.map(d => `<span role="columnheader">${d}</span>`).join('')}
        </div>
        <div class="mp-grid" role="rowgroup">
          ${gridHtml}
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Apply roving tabindex
    const focusableCells = Array.from(container.querySelectorAll('.mp-day')).filter(isFocusable);
    const activeCell = getActiveDay(focusableCells);
    focusableCells.forEach(el => {
      el.setAttribute('tabindex', el === activeCell ? '0' : '-1');
    });

    // Handle pending focus
    if (pendingFocusDay) {
      let target;
      if (pendingFocusDay === 'first') {
        target = focusableCells[0];
      } else if (pendingFocusDay === 'last') {
        target = focusableCells[focusableCells.length - 1];
      } else {
        target = container.querySelector(`.mp-day[data-day="${pendingFocusDay}"]`);
        if (!target || !isFocusable(target)) {
          target = activeCell;
        }
      }
      pendingFocusDay = null;
      if (target) focusCell(target);
    }

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

  function handleGridKeydown(e) {
    const cell = e.target.closest('.mp-day');
    if (!cell || !isFocusable(cell)) return;

    const grid = container.querySelector('.mp-grid');
    if (!grid || !grid.contains(cell)) return;

    const allCells = Array.from(grid.querySelectorAll('.mp-day'));
    const focusableCells = allCells.filter(isFocusable);
    const cellIndex = allCells.indexOf(cell);
    const focusIndex = focusableCells.indexOf(cell);

    let targetCell = null;

    switch (e.key) {
      case 'ArrowRight': {
        e.preventDefault();
        e.stopPropagation();
        if (focusIndex < focusableCells.length - 1) {
          targetCell = focusableCells[focusIndex + 1];
        } else {
          pendingFocusDay = 'first';
          navigateMonth(1);
          return;
        }
        break;
      }
      case 'ArrowLeft': {
        e.preventDefault();
        e.stopPropagation();
        if (focusIndex > 0) {
          targetCell = focusableCells[focusIndex - 1];
        } else {
          pendingFocusDay = 'last';
          navigateMonth(-1);
          return;
        }
        break;
      }
      case 'ArrowDown': {
        e.preventDefault();
        e.stopPropagation();
        const downIndex = cellIndex + 7;
        if (downIndex < allCells.length && isFocusable(allCells[downIndex])) {
          targetCell = allCells[downIndex];
        } else {
          pendingFocusDay = 'first';
          navigateMonth(1);
          return;
        }
        break;
      }
      case 'ArrowUp': {
        e.preventDefault();
        e.stopPropagation();
        const upIndex = cellIndex - 7;
        if (upIndex >= 0 && isFocusable(allCells[upIndex])) {
          targetCell = allCells[upIndex];
        } else {
          pendingFocusDay = 'last';
          navigateMonth(-1);
          return;
        }
        break;
      }
      case 'Home': {
        e.preventDefault();
        e.stopPropagation();
        const rowStart = cellIndex - (cellIndex % 7);
        for (let i = rowStart; i < rowStart + 7 && i < allCells.length; i++) {
          if (isFocusable(allCells[i])) {
            targetCell = allCells[i];
            break;
          }
        }
        break;
      }
      case 'End': {
        e.preventDefault();
        e.stopPropagation();
        const rowStart = cellIndex - (cellIndex % 7);
        for (let i = Math.min(rowStart + 6, allCells.length - 1); i >= rowStart; i--) {
          if (isFocusable(allCells[i])) {
            targetCell = allCells[i];
            break;
          }
        }
        break;
      }
      case 'Enter':
      case ' ': {
        e.preventDefault();
        const dateStr = cell.dataset.day;
        if (dateStr) {
          window.location.href = `/app/${app}/${dateStr}`;
        }
        return;
      }
      default:
        return;
    }

    if (targetCell) focusCell(targetCell);
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
    allowFutureDates = options.allowFutureDates || false;

    container = document.querySelector(options.container || '.month-picker');
    labelEl = document.getElementById('date-nav-label');

    if (labelEl) {
      dayLabel = labelEl.textContent;
      labelEl.setAttribute('tabindex', '-1');
    }

    if (!container) {
      console.warn('[MonthPicker] Container not found');
      return;
    }

    container.addEventListener('click', handleClick);
    container.addEventListener('keydown', handleGridKeydown);
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('click', handleClickOutside);
    window.addEventListener('facet.switch', handleFacetSwitch);

    // Prefetch current month data
    getMonthData(currentMonth);
  }

  function show() {
    if (!container) return;
    isVisible = true;
    container.classList.add('open');
    document.querySelector('.date-nav')?.classList.add('picker-open');
    pendingFocusDay = selectedDay || TODAY;
    showMonth(currentMonth);
  }

  function hide() {
    if (!container) return;
    isVisible = false;
    container.classList.remove('open');
    document.querySelector('.date-nav')?.classList.remove('picker-open');

    pendingFocusDay = null;
    if (labelEl && dayLabel) {
      labelEl.textContent = dayLabel;
    }
    if (labelEl) labelEl.focus();
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
    registerDataProvider,
    getToday
  };
})();
