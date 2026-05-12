// Mock data is the baseline used when the API server is unreachable.
// loadIndex() (below) replaces months[] entirely with the 12 most-recent
// real months from the server, and loadDay()/loadSegment() lazy-fetch
// per-day rollups + per-segment audio/screen jsonl on demand.
let months = window.timelineData.months;
let realHourPlan = {};
let realDayPlan = {};
const segmentAvail = {};        // "monthIdx:day:hour" → buckets (12 entries)
const dayCache = new Map();      // "YYYYMMDD" → /app/timeline/api/day response
const segCache = new Map();      // "<day>/<stream>/<seg>" → /app/timeline/api/segment response
const monthCache = {};

const ACCENT_ROTATION = ["blue", "teal", "amber", "coral"];
const MONTH_FULL_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"];

function isoDay(monthIndex, day) {
  const m = months[monthIndex];
  if (!m || !m.ym) return null;
  return m.ym + String(day).padStart(2, "0");
}
function isoToMonthIdx(yyyymm) {
  return months.findIndex((m) => m.ym === yyyymm);
}
function dayFromOrigin(origin) {
  const day8 = (origin || "").slice(0, 8);
  return /^\d{8}$/.test(day8) ? parseInt(day8.slice(6, 8), 10) : null;
}
function minuteFromOrigin(origin) {
  const seg = (origin || "").split("/").pop() || "";
  if (seg.length < 6 || !/^\d{6}/.test(seg)) return null;
  return parseInt(seg.slice(2, 4), 10);
}

// Compose a wall-clock label from a "seconds-from-segment-start" offset
// anchored to a per-call meta {startSec}. Used by the river view.
function segmentTimeLabel(meta, secondsFromStart) {
  const total = meta.startSec + Math.floor(secondsFromStart);
  const hh = Math.floor(total / 3600) % 24;
  const mm = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  return String(hh).padStart(2, "0") + ":" +
         String(mm).padStart(2, "0") + ":" +
         String(ss).padStart(2, "0");
}

// ── Data loaders (lazy, cached) ──────────────────────────────────────

async function loadIndex() {
  try {
    const res = await fetch("/app/timeline/api/index", { cache: "no-store" });
    if (!res.ok) {
      console.info(`/app/timeline/api/index failed (${res.status}); falling back to mock months`);
      return;
    }
    const idx = await res.json();
    rebuildMonthsFromIndex(idx);
    console.info(`loaded /app/timeline/api/index (${idx.months.length} months, year_top=${idx.year_top.length})`);
  } catch (e) {
    console.warn("/app/timeline/api/index fetch failed; falling back to mock months", e);
  }
}

function rebuildMonthsFromIndex(idx) {
  const newMonths = idx.months.map((m, i) => {
    const fullName = MONTH_FULL_NAMES[m.month_num - 1];
    const head = (m.month_top || [])[0] || null;
    const yearEvent = head
      ? { title: head.title, text: head.description, origin: head.origin || "" }
      : { title: `${fullName} ${m.year}`,
          text: m.day_count ? `${m.day_count} day${m.day_count === 1 ? "" : "s"} with observations` : "no observations yet",
          _empty: true };
    return {
      name: fullName,
      short: fullName.slice(0, 3).toUpperCase(),
      year: m.year,
      month_num: m.month_num,
      ym: m.ym,
      accent: ACCENT_ROTATION[i % 4],
      days: m.days_in_month,
      first_weekday: m.first_weekday,
      side: i % 2 === 0 ? "top" : "bottom",
      yearEvent,
      dayEvents: {},
      day_count: m.day_count,
      days_with_data: new Set(m.days_with_data || []),
      daysWithData: new Set(m.days_with_data || []),
    };
  });
  months = newMonths;
  // Stash on the global so console-debugging is easier.
  window.timelineData.months = months;
}

async function loadMonth(ym) {
  if (Object.prototype.hasOwnProperty.call(monthCache, ym)) return monthCache[ym];
  try {
    const res = await fetch(`/app/timeline/api/month/${ym}`, { cache: "no-store" });
    if (!res.ok) {
      console.info(`/app/timeline/api/month/${ym} failed (${res.status}); using empty month`);
      monthCache[ym] = null;
      return null;
    }
    const payload = await res.json();
    monthCache[ym] = payload;
    const monthIndex = months.findIndex((m) => m.ym === ym);
    if (monthIndex >= 0) {
      const month = months[monthIndex];
      month.dayEvents = {};
      let toggle = true;
      for (const [day, info] of Object.entries(payload.days || {}).sort()) {
        const pickArr = info.day_top || [];
        const pick = pickArr[0] || null;
        if (!pick) continue;
        const dayNum = parseInt(day.slice(6, 8), 10);
        month.dayEvents[day] = {
          day: dayNum,
          side: toggle ? "top" : "bottom",
          title: pick.title,
          text: pick.description,
          origin: pick.origin || "",
        };
        toggle = !toggle;
      }
      const daysWithData = new Set(payload.days_with_data || []);
      month.days_with_data = daysWithData;
      month.daysWithData = daysWithData;
    }
    return payload;
  } catch (e) {
    console.warn(`/app/timeline/api/month/${ym} fetch failed`, e);
    monthCache[ym] = null;
    return null;
  }
}

async function loadDay(yyyymmdd) {
  if (dayCache.has(yyyymmdd)) return dayCache.get(yyyymmdd);
  let data = null;
  try {
    const res = await fetch(`/app/timeline/api/day/${yyyymmdd}`, { cache: "no-store" });
    if (res.ok) data = await res.json();
  } catch (e) { console.warn("loadDay failed", yyyymmdd, e); }
  if (!data) data = { day: yyyymmdd, day_top: [], hours: {}, hours_avail: {} };
  dayCache.set(yyyymmdd, data);
  // Populate the prototype's per-render lookups.
  const monthIdx = isoToMonthIdx(yyyymmdd.slice(0, 6));
  if (monthIdx >= 0) populateDayLookups(monthIdx, yyyymmdd, data);
  return data;
}

function populateDayLookups(monthIdx, yyyymmdd, data) {
  const dayInt = parseInt(yyyymmdd.slice(6, 8), 10);
  // Day-view hour events: first pick of each hour with picks, alternating sides.
  const dayPlan = [];
  let toggle = true;
  for (const hh of Object.keys(data.hours || {}).sort()) {
    const picks = data.hours[hh].picks || [];
    if (!picks.length) continue;
    const p = picks[0];
    dayPlan.push({
      hour: parseInt(hh, 10),
      side: toggle ? "top" : "bottom",
      kind: "work",
      title: p.title, text: p.description, origin: p.origin || "",
    });
    toggle = !toggle;
  }
  if (dayPlan.length) realDayPlan[`${monthIdx}:${dayInt}`] = dayPlan;
  // Hour view minute events.
  for (const [hh, hd] of Object.entries(data.hours || {})) {
    const picks = hd.picks || [];
    if (!picks.length) continue;
    realHourPlan[`${monthIdx}:${dayInt}:${parseInt(hh, 10)}`] = pickListToMinutePlan(picks);
  }
  // Per-cell availability: drives hour-view tinting + click gating.
  for (const [hh, ha] of Object.entries(data.hours_avail || {})) {
    segmentAvail[`${monthIdx}:${dayInt}:${parseInt(hh, 10)}`] = ha.buckets;
  }
}

function pickListToMinutePlan(picks) {
  const used = new Set();
  const fallbackSlots = [5, 20, 35, 50];
  const out = [];
  picks.slice(0, 4).forEach((p, i) => {
    let slot;
    const m = minuteFromOrigin(p.origin);
    if (m == null) slot = fallbackSlots[i];
    else slot = Math.max(0, Math.min(55, Math.floor(m / 5) * 5));
    const orig = slot;
    while (used.has(slot) && slot < 55) slot += 5;
    if (used.has(slot)) {
      slot = orig;
      while (used.has(slot) && slot > 0) slot -= 5;
    }
    used.add(slot);
    out.push({
      minute: slot,
      side: i % 2 === 0 ? "top" : "bottom",
      title: p.title, text: p.description, origin: p.origin || "",
    });
  });
  out.sort((a, b) => a.minute - b.minute);
  return out;
}

async function loadSegment(origin) {
  if (segCache.has(origin)) return segCache.get(origin);
  try {
    const res = await fetch(`/app/timeline/api/segment/${origin}`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = await res.json();
    segCache.set(origin, data);
    return data;
  } catch (e) { console.warn("loadSegment failed", origin, e); return null; }
}

// Frame category → CSS color variable (shared with the prototype palette).
const SCREEN_CATEGORY_COLOR = {
  terminal:    "var(--ink)",
  code:        "var(--ink)",
  coding:      "var(--ink)",
  browsing:    "var(--teal)",
  productivity:"var(--amber)",
  reading:     "var(--muted)",
  messaging:   "var(--coral)",
  meeting:     "var(--teal)",
  media:       "var(--coral)",
  other:       "var(--muted)",
};
function categoryColor(primary) {
  return SCREEN_CATEGORY_COLOR[(primary || "").toLowerCase()] || "var(--muted)";
}

// Featured = frames with extracted text content (the meaningful ones to
// surface as visible serif marginalia). Non-featured render as ticks only.
function isFeatured(frame) {
  return !!(frame.content && Object.keys(frame.content).length);
}

// Excerpt the most important content from a frame for the inline detail
// panel — visual_description first, then any text content.
function frameDetailText(frame) {
  const a = frame.analysis || {};
  const c = frame.content || {};
  const parts = [];
  if (a.visual_description) parts.push(a.visual_description);
  for (const [k, v] of Object.entries(c)) {
    if (typeof v === "string") parts.push(`[${k}]\n${v}`);
  }
  return parts.join("\n\n");
}

function clearActiveMarks() {
  for (const el of document.querySelectorAll(".river-tick.is-active, .river-audio-dot.is-active")) {
    el.classList.remove("is-active");
  }
}

// The river renderer stashes the rendered segment's data here so the
// click-driven detail handlers can find frames + transcript lines
// without re-fetching.
let _activeSegment = null;
let _activeMeta = null;

function showSegmentDetail(frameId) {
  const detail = document.getElementById("segment-detail");
  if (!detail || !_activeSegment || !_activeSegment.screen) return;
  const frame = _activeSegment.screen.frames.find((f) => f.frame_id === frameId);
  if (!frame) return;
  const a = frame.analysis || {};
  const tLabel = segmentTimeLabel(_activeMeta, frame.timestamp || 0);
  const featured = isFeatured(frame);
  detail.innerHTML = `
    <div class="seg-detail-meta">
      <span class="seg-detail-time">${tLabel}</span>
      <span class="seg-detail-cat" style="--cat:${categoryColor(a.primary)}">${escapeHtml(a.primary || "?")}</span>
      <span class="seg-detail-frame">frame #${frame.frame_id}</span>
    </div>
    <div class="seg-detail-desc">${escapeHtml(a.visual_description || "")}</div>
    ${featured ? Object.entries(frame.content).map(([k, v]) =>
      `<div class="seg-detail-content">
         <div class="seg-detail-content-tag">${escapeHtml(k)}</div>
         <pre class="seg-detail-content-body">${escapeHtml(typeof v === "string" ? v : JSON.stringify(v, null, 2))}</pre>
       </div>`).join("") : ""}
  `;
  clearActiveMarks();
  const active = document.querySelector(`.river-tick[data-frame-id="${frameId}"]`);
  if (active) active.classList.add("is-active");
}

function showSegmentAudioDetail(audioIndex) {
  const detail = document.getElementById("segment-detail");
  if (!detail || !_activeSegment || !_activeSegment.audio) return;
  const lines = _activeSegment.audio.lines;
  const line = lines[audioIndex];
  if (!line) return;
  const sp = line.speaker || 1;
  const speakerColor = ["var(--blue)","var(--teal)","var(--coral)","var(--amber)"][sp - 1] || "var(--muted)";
  // Stitch together a small context window: 1 line before + this + 1 after
  const before = lines[audioIndex - 1];
  const after = lines[audioIndex + 1];
  const renderLine = (l, isFocus) =>
    l ? `<div class="seg-detail-line ${isFocus ? "is-focus" : ""}">
           <span class="seg-detail-line-time">${escapeHtml(l.start || "")}</span>
           <span class="seg-detail-line-sp">s${l.speaker || "?"}</span>
           <span class="seg-detail-line-text">${escapeHtml(l.corrected || l.text || "")}</span>
           ${l.emotion ? `<span class="seg-detail-line-emotion">${escapeHtml(l.emotion)}</span>` : ""}
         </div>` : "";
  detail.innerHTML = `
    <div class="seg-detail-meta">
      <span class="seg-detail-time">${escapeHtml(line.start || "")}</span>
      <span class="seg-detail-cat" style="--cat:${speakerColor}">speaker ${sp}</span>
      <span class="seg-detail-frame">audio #${audioIndex + 1} of ${lines.length}</span>
    </div>
    <div class="seg-detail-lines">
      ${renderLine(before, false)}
      ${renderLine(line, true)}
      ${renderLine(after, false)}
    </div>
  `;
  clearActiveMarks();
  const active = document.querySelector(`.river-audio-dot[data-audio-index="${audioIndex}"]`);
  if (active) active.classList.add("is-active");
}

function clearSegmentDetail() {
  const detail = document.getElementById("segment-detail");
  if (detail) detail.innerHTML = `<div class="seg-detail-empty">click a tick or audio dot on the river to see what sol observed at that moment</div>`;
  clearActiveMarks();
}

const timeline = document.querySelector("#timeline-root");
let selectedMonth = null;
let selectedDay = null;
let selectedHour = null;
let selectedMinute = null;
let appScreen = "cover";
let transitionToken = 0;

const holidays = new Map(
  [
    [0, 1, "New Year's Day"],
    [0, 20, "Martin Luther King Jr. Day"],
    [1, 14, "Valentine's Day"],
    [1, 17, "Presidents' Day"],
    [2, 17, "St. Patrick's Day"],
    [3, 20, "Easter"],
    [4, 11, "Mother's Day"],
    [4, 26, "Memorial Day"],
    [5, 15, "Father's Day"],
    [5, 19, "Juneteenth"],
    [6, 4, "Independence Day"],
    [8, 1, "Labor Day"],
    [9, 13, "Indigenous Peoples' Day"],
    [9, 31, "Halloween"],
    [10, 11, "Veterans Day"],
    [10, 27, "Thanksgiving"],
    [11, 24, "Christmas Eve"],
    [11, 25, "Christmas Day"],
    [11, 31, "New Year's Eve"],
  ].map(([monthIndex, day, name]) => [`${monthIndex}-${day}`, name]),
);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function solLogoSvg(className = "sol-logo") {
  return `
    <svg class="${className}" xmlns="http://www.w3.org/2000/svg" viewBox="2.5 2.5 27 27" role="img" aria-label="sol logo">
      <title>sol</title>
      <path fill="#F5C740" d="M16.0 2.5 L18.6 7.3 A9.1 9.1 0 0 0 13.4 7.3 Z M23.9 5.1 L23.2 10.5 A9.1 9.1 0 0 0 19.0 7.4 Z M28.8 11.8 L25.1 15.8 A9.1 9.1 0 0 0 23.5 10.9 Z M28.8 20.2 L23.5 21.1 A9.1 9.1 0 0 0 25.1 16.2 Z M23.9 26.9 L19.0 24.6 A9.1 9.1 0 0 0 23.2 21.5 Z M16.0 29.5 L13.4 24.7 A9.1 9.1 0 0 0 18.6 24.7 Z M8.1 26.9 L8.8 21.5 A9.1 9.1 0 0 0 13.0 24.6 Z M3.2 20.2 L6.9 16.2 A9.1 9.1 0 0 0 8.5 21.1 Z M3.2 11.8 L8.5 10.9 A9.1 9.1 0 0 0 6.9 15.8 Z M8.1 5.1 L13.0 7.4 A9.1 9.1 0 0 0 8.8 10.5 Z"/>
      <circle cx="16" cy="16" r="8.0" fill="none" stroke="#E8923A" stroke-width="1.2"/>
      <path fill="#E8923A" fill-rule="evenodd" d="M12.079 18.795C13.489 18.795 14.229 18.065 14.229 17.155C14.229 16.365 13.729 15.835 12.229 15.535C11.149 15.315 10.939 15.095 10.939 14.725C10.939 14.345 11.399 14.135 11.989 14.135C12.499 14.135 12.859 14.235 13.199 14.555C13.399 14.745 13.729 14.815 13.949 14.665C14.159 14.505 14.169 14.255 13.989 14.035C13.589 13.545 12.889 13.245 12.009 13.245C10.989 13.245 9.959 13.735 9.959 14.755C9.959 15.525 10.529 16.075 11.879 16.335C12.919 16.525 13.249 16.815 13.239 17.215C13.229 17.615 12.809 17.895 12.039 17.895C11.429 17.895 10.889 17.625 10.659 17.375C10.469 17.175 10.189 17.125 9.929 17.335C9.699 17.515 9.659 17.825 9.859 18.035C10.299 18.475 11.149 18.795 12.079 18.795Z M16.999 18.795C18.609 18.795 19.749 17.645 19.749 16.025C19.739 14.395 18.599 13.245 16.999 13.245C15.379 13.245 14.239 14.395 14.239 16.025C14.239 17.645 15.379 18.795 16.999 18.795ZM16.999 17.895C15.959 17.895 15.219 17.125 15.219 16.025C15.219 14.925 15.959 14.145 16.999 14.145C18.039 14.145 18.769 14.925 18.769 16.025C18.769 17.125 18.039 17.895 16.999 17.895Z M21.569 18.755H21.589C21.989 18.755 22.269 18.545 22.269 18.255C22.269 17.965 22.079 17.755 21.819 17.755H21.569C21.279 17.755 21.069 17.405 21.069 16.905V11.445C21.069 11.155 20.859 10.945 20.569 10.945C20.279 10.945 20.069 11.155 20.069 11.445V16.905C20.069 17.985 20.689 18.755 21.569 18.755Z"/>
    </svg>
  `;
}

function getTransitionFrames(mode, phase) {
  const enter = phase === "enter";
  const frames = {
    "zoom-in": enter
      ? [{ opacity: 0, transform: "scale(0.985) translateY(8px)" }, { opacity: 1, transform: "scale(1) translateY(0)" }]
      : [{ opacity: 1, transform: "scale(1) translateY(0)" }, { opacity: 0, transform: "scale(1.015) translateY(-8px)" }],
    "zoom-out": enter
      ? [{ opacity: 0, transform: "scale(1.015) translateY(-8px)" }, { opacity: 1, transform: "scale(1) translateY(0)" }]
      : [{ opacity: 1, transform: "scale(1) translateY(0)" }, { opacity: 0, transform: "scale(0.985) translateY(8px)" }],
    lateral: enter
      ? [{ opacity: 0, transform: "translateX(18px)" }, { opacity: 1, transform: "translateX(0)" }]
      : [{ opacity: 1, transform: "translateX(0)" }, { opacity: 0, transform: "translateX(-18px)" }],
  };

  return frames[mode] || frames["zoom-in"];
}

async function setTimeline(markup, mode = "zoom-in") {
  const token = ++transitionToken;
  const shouldReduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const currentView = timeline.firstElementChild;

  if (!currentView || mode === "none" || shouldReduceMotion) {
    timeline.innerHTML = markup;
    return;
  }

  await currentView
    .animate(getTransitionFrames(mode, "leave"), {
      duration: 95,
      easing: "cubic-bezier(.2, 0, .2, 1)",
      fill: "forwards",
    })
    .finished.catch(() => {});

  if (token !== transitionToken) return;

  timeline.innerHTML = markup;
  const nextView = timeline.firstElementChild;
  if (!nextView) return;

  nextView.animate(getTransitionFrames(mode, "enter"), {
    duration: 145,
    easing: "cubic-bezier(.2, 0, .2, 1)",
    fill: "both",
  });
}

function renderCover(mode = "zoom-out") {
  appScreen = "cover";
  selectedMonth = null;
  selectedDay = null;
  selectedHour = null;
  selectedMinute = null;

  setTimeline(`
    <div class="cover-screen">
      <button class="cover-button" type="button" data-start-demo onclick="event.stopPropagation(); showTimeline('zoom-in');" aria-label="Start timeline demo">
        ${solLogoSvg()}
      </button>
    </div>
  `, mode);
}

function renderClosing(mode = "zoom-in") {
  appScreen = "closing";

  setTimeline(`
    <div class="closing-screen">
      <div class="closing-content">
        <div class="closing-logo" aria-hidden="true">${solLogoSvg()}</div>
        <p class="install-url">solstone.app/install</p>
      </div>
    </div>
  `, mode);
}

function showTimeline(mode = "zoom-in") {
  selectedMonth = null;
  selectedDay = null;
  selectedHour = null;
  selectedMinute = null;
  renderYear(mode);
}

function goNextScreen() {
  if (appScreen === "cover") {
    showTimeline("zoom-in");
    return;
  }

  if (appScreen === "timeline") {
    renderClosing("zoom-in");
  }
}

function goPreviousScreen() {
  if (appScreen === "closing") {
    showTimeline("zoom-out");
    return;
  }

  if (appScreen === "timeline") {
    renderCover("zoom-out");
  }
}

function eventColumn(day, span, days) {
  const start = Math.max(1, Math.min(day - 2, days - span + 1));
  return `${start} / span ${span}`;
}

function hourColumn(hour, span = 4) {
  const start = Math.max(1, Math.min(hour + 1, 25 - span));
  return `${start} / span ${span}`;
}

function segmentColumn(minute, span = 3) {
  const index = Math.floor(minute / 5) + 1;
  const start = Math.max(1, Math.min(index, 13 - span));
  return `${start} / span ${span}`;
}

function getDayMeta(monthIndex, day) {
  // Use the real year from the dynamically-built months[] entry so
  // weekend computation matches the actual calendar (e.g., Jun 14
  // 2025 is a Saturday, but Jun 14 2026 is a Sunday).
  const m = months[monthIndex] || {};
  const year = m.year || 2025;
  const monthNum = (m.month_num != null ? m.month_num - 1 : monthIndex);
  const date = new Date(year, monthNum, day);
  const weekday = date.getDay();
  const holiday = holidays.get(`${monthIndex}-${day}`);
  return {
    dayType: weekday === 0 || weekday === 6 ? "weekend" : "weekday",
    holiday,
  };
}

function formatHour(hour) {
  if (hour === 0) return "12a";
  if (hour < 12) return `${hour}a`;
  if (hour === 12) return "12p";
  return `${hour - 12}p`;
}

function formatTime(hour, minute = 0) {
  const suffix = hour < 12 ? "a" : "p";
  const normalizedHour = hour % 12 === 0 ? 12 : hour % 12;
  return `${normalizedHour}:${String(minute).padStart(2, "0")}${suffix}`;
}

function getDayEvent(monthIndex, day) {
  const yyyymmdd = isoDay(monthIndex, day);
  if (!yyyymmdd || !months[monthIndex].dayEvents) return null;
  return months[monthIndex].dayEvents[yyyymmdd] || null;
}

function getDayPlan(monthIndex, day) {
  // Real per-day rollup data overrides the synthetic plan when populated.
  const realKey = `${monthIndex}:${day}`;
  if (realDayPlan[realKey]) {
    return realDayPlan[realKey];
  }

  const { dayType, holiday } = getDayMeta(monthIndex, day);
  const keyEvent = getDayEvent(monthIndex, day);
  const isWeekend = dayType === "weekend";
  const base = isWeekend
    ? [
        { hour: 7, side: "top", kind: "personal", title: "Slow Start", text: "Breakfast, a walk, and a lighter read on the week." },
        { hour: 10, side: "bottom", kind: "personal", title: "Personal Block", text: "Errands, family time, or a longer reset away from the desk." },
        { hour: 19, side: "bottom", kind: "personal", title: "Evening Offline", text: "Dinner, downtime, and a clear stop before the next week." },
      ]
    : [
        { hour: 6, side: "top", kind: "personal", title: "Morning Reset", text: "Walk, breakfast, and a quick scan of the day's shape." },
        { hour: 8, side: "bottom", kind: "work", title: "Deep Work", text: "Protected build time before calls and inbox pressure arrive." },
        { hour: 13, side: "bottom", kind: "personal", title: "Lunch Walk", text: "A short reset between maker work and afternoon coordination." },
        { hour: 16, side: "top", kind: "work", title: "Calls & Follow-up", text: "Customer notes, investor replies, or team coordination." },
        { hour: 20, side: "bottom", kind: "personal", title: "Home Block", text: "Dinner, cleanup, and a cleaner boundary around the evening." },
      ];

  if (keyEvent) {
    base.splice(isWeekend ? 2 : 2, 0, {
      hour: isWeekend ? 12 : 11,
      side: "top",
      kind: "work",
      title: keyEvent.title,
      text: keyEvent.text,
      featured: true,
    });
  }

  if (holiday) {
    base.splice(1, 0, {
      hour: 9,
      side: "bottom",
      kind: "personal",
      title: holiday,
      text: "Keeps the day lighter and protects space around the holiday.",
      featured: true,
    });
  }

  return base.sort((a, b) => a.hour - b.hour);
}

function getHourEvent(monthIndex, day, hour) {
  return getDayPlan(monthIndex, day).find((event) => event.hour === hour);
}

function getMinutePlan(monthIndex, day, hour) {
  // Real per-segment data overrides the synthetic plan when populated.
  const realKey = `${monthIndex}:${day}:${hour}`;
  if (realHourPlan[realKey]) {
    return realHourPlan[realKey];
  }

  const hourEvent = getHourEvent(monthIndex, day, hour);
  const isWorkHour = hour >= 8 && hour <= 17;
  const base = hourEvent
    ? [
        { minute: 0, side: "top", title: "Set Context", text: "Opens notes, names the goal, and removes small distractions." },
        { minute: 20, side: "bottom", title: hourEvent.title, text: hourEvent.text, featured: true },
        { minute: 45, side: "top", title: "Name Next Step", text: "Writes the decision, owner, and next concrete action before moving on." },
      ]
    : isWorkHour
      ? [
          { minute: 5, side: "top", title: "Focus Setup", text: "Chooses one concrete task and closes unrelated tabs." },
          { minute: 20, side: "bottom", title: "Build / Decide", text: "Uses the middle of the hour for the hardest work or decision." },
          { minute: 50, side: "top", title: "Commit Notes", text: "Leaves the workspace in a state that is easy to resume." },
        ]
      : [
          { minute: 5, side: "top", title: "Transition", text: "Moves out of the previous context and checks what the body needs." },
          { minute: 25, side: "bottom", title: "Personal Reset", text: "Walks, eats, reads, or handles home logistics without rushing." },
          { minute: 50, side: "top", title: "Light Plan", text: "Sets up the next hour without turning the break into work." },
        ];

  return base;
}

function getSegmentEvent(monthIndex, day, hour, minute) {
  return getMinutePlan(monthIndex, day, hour).find((event) => event.minute === minute);
}

function getFiveMinutePlan(monthIndex, day, hour, minute) {
  const segmentEvent = getSegmentEvent(monthIndex, day, hour, minute);
  const isWorkHour = hour >= 8 && hour <= 17;
  const base = segmentEvent
    ? [
        { offset: 0, side: "top", title: "Orient", text: "Reads the immediate cue and decides what this tiny window is for." },
        { offset: 2, side: "bottom", title: segmentEvent.title, text: segmentEvent.text, featured: true },
        { offset: 4, side: "top", title: "Leave Trace", text: "Notes the result so the next block starts with context." },
      ]
    : isWorkHour
      ? [
          { offset: 0, side: "top", title: "Open Thread", text: "Pulls up the exact file, note, or message needed now." },
          { offset: 2, side: "bottom", title: "Make Progress", text: "Uses the middle minutes for one concrete edit or decision." },
          { offset: 4, side: "top", title: "Close Loop", text: "Saves the small result and names the next action." },
        ]
      : [
          { offset: 0, side: "top", title: "Pause", text: "Steps out of the previous context and slows the pace." },
          { offset: 2, side: "bottom", title: "Reset", text: "Takes care of the small personal need that this break is for." },
          { offset: 4, side: "top", title: "Return Cue", text: "Sets up the next transition without over-planning." },
        ];

  return base;
}

function renderYear(mode = "zoom-out") {
  appScreen = "timeline";
  setTimeline(`
    <div class="year-view">
      ${months
        .map(
          (month, index) => `
            <article class="milestone timeline-${month.side} accent-${month.accent}" style="grid-column: ${index + 1}">
              <div class="timeline-card">
                <div class="timeline-date">${month.name} ${month.year || ""}</div>
                <h2>${escapeHtml(month.yearEvent.title)}</h2>
                <p>${escapeHtml(month.yearEvent.text)}</p>
              </div>
              <button class="timeline-node" type="button" data-month="${index}" aria-label="Open ${month.name} ${month.year || ""}">
                ${month.short}
              </button>
            </article>
          `,
        )
        .join("")}
    </div>
  `, mode);
}

async function renderMonth(index, mode = "zoom-in") {
  appScreen = "timeline";
  const month = months[index];
  const previous = index > 0 ? months[index - 1] : null;
  const next = index < months.length - 1 ? months[index + 1] : null;
  const monthEvents = Object.values(month.dayEvents || {}).filter(Boolean);
  const topEvents = monthEvents.filter((event) => event.side === "top");
  const bottomEvents = monthEvents.filter((event) => event.side === "bottom");
  const eventDays = new Map(monthEvents.map((event) => [event.day, event.side]));

  await setTimeline(`
    <div class="month-view accent-${month.accent}" style="--days: ${month.days}">
      ${previous ? renderEdgeMonth(previous, index - 1, "prev") : ""}
      ${next ? renderEdgeMonth(next, index + 1, "next") : ""}

      <section class="timeline-focus-panel" aria-label="${month.name} ${month.year || ""} daily timeline">
        <svg class="month-connectors" aria-hidden="true"></svg>

        <div class="timeline-focus-heading">
          <button class="timeline-focus-node" type="button" data-month="${index}" aria-label="Return to year view">
            ${month.short}
          </button>
        </div>

        <div class="events-lane timeline-top" aria-label="${month.name} highlighted events above the daily timeline">
          ${topEvents.map((event) => renderDayEvent(event, month.days, "top")).join("")}
        </div>

        <div class="day-grid" aria-label="${month.name} ${month.year || ""} days">
          ${Array.from({ length: month.days }, (_, dayIndex) => {
            const day = dayIndex + 1;
            const side = eventDays.get(day);
            const { dayType, holiday } = getDayMeta(index, day);
            const classes = ["day-cell", dayType, holiday ? "holiday" : "", side ? `has-event timeline-${side}` : ""]
              .filter(Boolean)
              .join(" ");
            const label = `${month.name} ${day}, ${month.year || ""}${holiday ? `, ${holiday}` : ""}`;
            return `
              <button class="${classes}" type="button" data-month="${index}" data-day="${day}" title="${escapeHtml(label)}" aria-label="Open ${escapeHtml(label)}">
                ${day}
                ${holiday ? '<span class="holiday-mark" aria-hidden="true"></span>' : ""}
              </button>
            `;
          }).join("")}
        </div>

        <div class="events-lane timeline-bottom" aria-label="${month.name} highlighted events below the daily timeline">
          ${bottomEvents.map((event) => renderDayEvent(event, month.days, "bottom")).join("")}
        </div>
      </section>
    </div>
  `, mode);
  layoutMonth();
  const view = document.querySelector(".month-view");
  if (view && view.getAnimations) {
    Promise.all(view.getAnimations().map((a) => a.finished.catch(() => {})))
      .then(() => layoutMonth());
  }
}

async function renderDay(monthIndex, day, mode = "zoom-in") {
  appScreen = "timeline";
  const month = months[monthIndex];
  const previous = day > 1 ? day - 1 : null;
  const next = day < month.days ? day + 1 : null;
  // Lazy-fetch the day's rollup so realDayPlan/realHourPlan/segmentAvail
  // are populated before the day-view renders.
  const yyyymmdd = isoDay(monthIndex, day);
  if (yyyymmdd) await loadDay(yyyymmdd);
  const plan = getDayPlan(monthIndex, day);
  const topEvents = plan.filter((event) => event.side === "top");
  const bottomEvents = plan.filter((event) => event.side === "bottom");
  const eventHours = new Map(plan.map((event) => [event.hour, event]));
  const { dayType, holiday } = getDayMeta(monthIndex, day);
  const dayLabel = `${month.short} ${day}`;

  await setTimeline(`
    <div class="day-view accent-${month.accent}">
      ${previous ? renderEdgeDay(monthIndex, previous, "prev") : ""}
      ${next ? renderEdgeDay(monthIndex, next, "next") : ""}

      <section class="hour-panel" aria-label="${month.name} ${day}, ${month.year || ""} hourly timeline">
        <svg class="day-connectors" aria-hidden="true"></svg>

        <div class="timeline-focus-heading">
          <button class="day-focus-node" type="button" data-month="${monthIndex}" data-return-month="true" aria-label="Return to ${month.name} ${month.year || ""}">
            ${dayLabel}
          </button>
        </div>

        <div class="hour-lane timeline-top" aria-label="${month.name} ${day} highlighted events above the hourly timeline">
          ${topEvents.map(renderHourEvent).join("")}
        </div>

        <div class="hour-grid" aria-label="${month.name} ${day}, ${month.year || ""} hours">
          ${Array.from({ length: 24 }, (_, hour) => {
            const event = eventHours.get(hour);
            const hourKind = hour >= 8 && hour <= 17 ? "work" : "personal";
            const classes = ["hour-cell", hourKind, event ? `has-hour-event timeline-${event.side}` : ""]
              .filter(Boolean)
              .join(" ");
            const label = `${formatHour(hour)} on ${month.name} ${day}, ${month.year || ""}${event ? `, ${event.title}` : ""}`;
            return `
              <button class="${classes}" type="button" data-month="${monthIndex}" data-day="${day}" data-hour="${hour}" title="${escapeHtml(label)}" aria-label="Open ${escapeHtml(label)}">
                ${formatHour(hour)}
              </button>
            `;
          }).join("")}
        </div>

        <div class="hour-lane timeline-bottom" aria-label="${month.name} ${day} highlighted events below the hourly timeline">
          ${bottomEvents.map(renderHourEvent).join("")}
        </div>
      </section>
    </div>
  `, mode);
  layoutDay();
  // Settle pass once enter animation completes.
  const view = document.querySelector(".day-view");
  if (view && view.getAnimations) {
    Promise.all(view.getAnimations().map((a) => a.finished.catch(() => {})))
      .then(() => layoutDay());
  }
}

// Generic layout primitive used by every "axis with events above and
// below" view (hour view, day view, eventually month + year). For each
// side: each card's ideal left = its anchor cell's center − cardWidth/2;
// sort by anchor key; forward-pass to push apart any overlap; then draw
// SVG dotted connectors from card edge to anchor cell edge so slants
// appear when cards had to slide off their cells.
//
// opts: {
//   viewSelector,    // e.g. ".minute-view"
//   panelSelector,   // e.g. ".minute-panel"   (layout origin for SVG)
//   gridSelector,    // e.g. ".minute-grid"
  //   laneSelectors,   // e.g. [".minute-lane.timeline-top", ".minute-lane.timeline-bottom"]
//   eventSelector,   // e.g. ".minute-event"
//   cellSelector,    // e.g. ".segment-cell[data-minute='${k}']"  template
//   anchorAttr,      // e.g. "data-anchor-minute"
//   svgSelector,     // e.g. ".minute-connectors"
//   cardWidth,       // e.g. 170
//   cardGap,         // e.g. 14
// }
function layoutScale(opts) {
  const view = document.querySelector(opts.viewSelector);
  if (!view) return;
  const panel = view.querySelector(opts.panelSelector);
  const grid = view.querySelector(opts.gridSelector);
  const svg = view.querySelector(opts.svgSelector);
  if (!panel || !grid || !svg) return;

  // Mobile responsive layouts use a stacked block flow; skip the
  // absolute-positioned overlay entirely so it doesn't fight CSS.
  const isMobile = window.matchMedia("(max-width: 800px)").matches;
  if (isMobile) {
    svg.innerHTML = "";
    for (const c of view.querySelectorAll(opts.eventSelector)) c.style.left = "";
    return;
  }

  const panelRect = panel.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${panelRect.width} ${panelRect.height}`);
  svg.style.width = panelRect.width + "px";
  svg.style.height = panelRect.height + "px";
  svg.innerHTML = "";

  const ns = "http://www.w3.org/2000/svg";
  const accent = getComputedStyle(view).getPropertyValue("--accent").trim() || "#0f4c81";

  for (const sideName of ["top", "bottom"]) {
    const lane = view.querySelector(opts.laneSelectorFor(sideName));
    if (!lane) continue;
    const cards = Array.from(lane.querySelectorAll(opts.eventSelector));
    if (!cards.length) continue;

    const laneRect = lane.getBoundingClientRect();
    const items = cards.map((card) => {
      const anchor = parseInt(card.getAttribute(opts.anchorAttr), 10);
      const cell = grid.querySelector(opts.cellSelectorFor(anchor));
      const cellRect = cell ? cell.getBoundingClientRect() : null;
      const cellCenterInLane = cellRect
        ? cellRect.left + cellRect.width / 2 - laneRect.left
        : 0;
      return {
        card,
        cell,
        anchor,
        idealLeft: cellCenterInLane - opts.cardWidth / 2,
      };
    }).sort((a, b) => a.anchor - b.anchor);

    // Forward pass: never let a card overlap its left neighbor.
    let prevRight = -Infinity;
    for (const it of items) {
      it.left = Math.max(it.idealLeft, prevRight + opts.cardGap);
      prevRight = it.left + opts.cardWidth;
    }
    for (const it of items) {
      it.card.style.left = it.left + "px";
    }

    // Connectors — drawn in panel coords so the SVG layer can overlap
    // both lanes and the central grid.
    for (const it of items) {
      if (!it.cell) continue;
      const cardRect = it.card.getBoundingClientRect();
      const cellRect = it.cell.getBoundingClientRect();

      const cardEdgeY = sideName === "top" ? cardRect.bottom : cardRect.top;
      const cellEdgeY = sideName === "top" ? cellRect.top : cellRect.bottom;

      const x1 = cardRect.left + cardRect.width / 2 - panelRect.left;
      const y1 = cardEdgeY - panelRect.top;
      const x2 = cellRect.left + cellRect.width / 2 - panelRect.left;
      const y2 = cellEdgeY - panelRect.top;

      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      line.setAttribute("stroke", accent);
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("stroke-dasharray", "2 4");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("opacity", "0.55");
      svg.appendChild(line);

      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", x2);
      dot.setAttribute("cy", y2);
      dot.setAttribute("r", "4");
      dot.setAttribute("fill", accent);
      svg.appendChild(dot);
    }
  }
}

// Per-scale wrappers — fixed selectors and card sizing.
const LAYOUT_MINUTE = {
  viewSelector: ".minute-view",
  panelSelector: ".minute-panel",
  gridSelector: ".minute-grid",
  svgSelector: ".minute-connectors",
  eventSelector: ".minute-event",
  anchorAttr: "data-anchor-minute",
  laneSelectorFor: (s) => `.minute-lane.timeline-${s}`,
  cellSelectorFor: (k) => `.segment-cell[data-minute="${k}"]`,
  cardWidth: 170,
  cardGap: 14,
};
const LAYOUT_DAY = {
  viewSelector: ".day-view",
  panelSelector: ".hour-panel",
  gridSelector: ".hour-grid",
  svgSelector: ".day-connectors",
  eventSelector: ".hour-event",
  anchorAttr: "data-anchor-hour",
  laneSelectorFor: (s) => `.hour-lane.timeline-${s}`,
  cellSelectorFor: (k) => `.hour-cell[data-hour="${k}"]`,
  cardWidth: 170,
  cardGap: 12,
};
const LAYOUT_MONTH = {
  viewSelector: ".month-view",
  panelSelector: ".timeline-focus-panel",
  gridSelector: ".day-grid",
  svgSelector: ".month-connectors",
  eventSelector: ".day-event",
  anchorAttr: "data-anchor-day",
  laneSelectorFor: (s) => `.events-lane.timeline-${s}`,
  cellSelectorFor: (k) => `.day-cell[data-day="${k}"]`,
  cardWidth: 170,
  cardGap: 12,
};

function layoutMinute() { layoutScale(LAYOUT_MINUTE); }
function layoutDay()    { layoutScale(LAYOUT_DAY); }
function layoutMonth()  { layoutScale(LAYOUT_MONTH); }

// Re-layout the active scale on resize.
window.addEventListener("resize", () => {
  if (document.querySelector(".minute-view")) layoutMinute();
  if (document.querySelector(".day-view")) layoutDay();
  if (document.querySelector(".month-view")) layoutMonth();
});

async function renderMinute(monthIndex, day, hour, mode = "zoom-in") {
  appScreen = "timeline";
  const month = months[monthIndex];
  const previous = hour > 0 ? hour - 1 : null;
  const next = hour < 23 ? hour + 1 : null;
  // Make sure the day's data (rollup picks + per-cell availability) is
  // loaded before we compute the plan + grid.
  const yyyymmdd = isoDay(monthIndex, day);
  if (yyyymmdd) await loadDay(yyyymmdd);
  const plan = getMinutePlan(monthIndex, day, hour);
  const topEvents = plan.filter((event) => event.side === "top");
  const bottomEvents = plan.filter((event) => event.side === "bottom");
  const eventMinutes = new Map(plan.map((event) => [event.minute, event]));
  const focusLabel = `${month.short} ${day} ${formatHour(hour)}`;
  const buckets = segmentAvail[`${monthIndex}:${day}:${hour}`] || [];

  await setTimeline(`
    <div class="minute-view accent-${month.accent}">
      ${previous !== null ? renderEdgeHour(monthIndex, day, previous, "prev") : ""}
      ${next !== null ? renderEdgeHour(monthIndex, day, next, "next") : ""}

      <section class="minute-panel" aria-label="${month.name} ${day}, ${month.year || ""} ${formatHour(hour)} five-minute timeline">
        <svg class="minute-connectors" aria-hidden="true"></svg>

        <div class="timeline-focus-heading">
          <button class="minute-focus-node" type="button" data-month="${monthIndex}" data-day="${day}" data-return-day="true" aria-label="Return to ${month.name} ${day}, ${month.year || ""}">
            ${focusLabel}
          </button>
        </div>

        <div class="minute-lane timeline-top" aria-label="${formatHour(hour)} segment events above the timeline">
          ${topEvents.map(renderMinuteEvent).join("")}
        </div>

        <div class="minute-grid" aria-label="${formatHour(hour)} five-minute segments">
          ${Array.from({ length: 12 }, (_, segmentIndex) => {
            const minute = segmentIndex * 5;
            const event = eventMinutes.get(minute);
            const bucket = buckets[segmentIndex] || null;
            const hasData = !!(bucket && bucket.best_origin);
            // Availability tint: both = accent, screen-only = teal,
            // audio-only = coral, none = grey/disabled.
            let availClass = "avail-none";
            if (hasData && bucket.has_audio && bucket.has_screen) availClass = "avail-both";
            else if (hasData && bucket.has_screen) availClass = "avail-screen";
            else if (hasData && bucket.has_audio) availClass = "avail-audio";
            const classes = ["segment-cell", event ? `timeline-focus timeline-${event.side}` : "", availClass].filter(Boolean).join(" ");
            const availLabel = hasData
              ? (bucket.has_audio && bucket.has_screen ? "audio + screen"
                 : bucket.has_screen ? "screen only"
                 : bucket.has_audio ? "audio only" : "metadata only")
              : "no observation";
            const label = `${formatTime(hour, minute)} · ${availLabel}${event ? `, ${event.title}` : ""}`;
            const disabled = hasData ? "" : "disabled aria-disabled=\"true\"";
            return `
              <button class="${classes}" type="button" ${disabled} data-month="${monthIndex}" data-day="${day}" data-hour="${hour}" data-minute="${minute}" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}">
                ${String(minute).padStart(2, "0")}
              </button>
            `;
          }).join("")}
        </div>

        <div class="minute-lane timeline-bottom" aria-label="${formatHour(hour)} segment events below the timeline">
          ${bottomEvents.map(renderMinuteEvent).join("")}
        </div>
      </section>
    </div>
  `, mode);
  layoutMinute();
  // Re-layout once the enter animation settles — getBoundingClientRect
  // reads animated transforms, so an early-pass call lands the line
  // endpoints a few pixels off. Settling is ~145ms.
  const view = document.querySelector(".minute-view");
  if (view && view.getAnimations) {
    Promise.all(view.getAnimations().map((a) => a.finished.catch(() => {})))
      .then(() => layoutMinute());
  }
}

// Empty-state river when a 5-min cell has no underlying segment data.
// The hour view should disable empty cells, so this is a defensive render.
async function renderEmptySegment(monthIndex, day, hour, minute, mode, focusLabel) {
  const month = months[monthIndex];
  const previous = minute > 0 ? minute - 5 : null;
  const next = minute < 55 ? minute + 5 : null;
  await setTimeline(`
    <div class="segment-view accent-${month.accent}">
      ${previous !== null ? renderEdgeSegment(monthIndex, day, hour, previous, "prev") : ""}
      ${next !== null ? renderEdgeSegment(monthIndex, day, hour, next, "next") : ""}
      <section class="segment-panel">
        <div class="timeline-focus-heading">
          <button class="five-focus-node" type="button"
                  data-month="${monthIndex}" data-day="${day}" data-hour="${hour}"
                  data-return-hour="true">${focusLabel}</button>
        </div>
        <div class="segment-empty">no observation in this slice</div>
      </section>
    </div>
  `, mode);
}

async function renderFiveMinute(monthIndex, day, hour, minute, mode = "zoom-in") {
  // The 5-min view breaks from the event-cards-around-an-axis pattern
  // of higher levels. Here we visualize what sol *actually observed* in
  // that 5-minute window — screen frames as ticks above the time axis,
  // transcript lines as dots below. No cards, no slants. Data loads
  // dynamically per the cell's best_origin from the day endpoint.
  appScreen = "timeline";
  const month = months[monthIndex];
  const previous = minute > 0 ? minute - 5 : null;
  const next = minute < 55 ? minute + 5 : null;
  const focusLabel = `${formatTime(hour, minute)}`;

  // Look up which segment to load from the cached day data.
  const yyyymmdd = isoDay(monthIndex, day);
  if (yyyymmdd) await loadDay(yyyymmdd);
  const buckets = segmentAvail[`${monthIndex}:${day}:${hour}`] || [];
  const bucketIdx = Math.floor(minute / 5);
  const bucket = buckets[bucketIdx] || null;
  const origin = bucket && bucket.best_origin ? bucket.best_origin : null;

  // No data → render an empty-state river. (Cell shouldn't have been
  // clickable in the first place; this is a defensive fallback.)
  if (!origin) {
    return renderEmptySegment(monthIndex, day, hour, minute, mode, focusLabel);
  }

  // Derive the segment's wall-clock start from its segment name's HHMMSS.
  // origin format: "YYYYMMDD/<stream>/<HHMMSS_LEN>" or "YYYYMMDD/<HHMMSS_LEN>"
  const parts = origin.split("/");
  const segName = parts[parts.length - 1];
  const segMatch = /^(\d{2})(\d{2})(\d{2})_(\d{1,6})$/.exec(segName);
  const startSec = segMatch ? parseInt(segMatch[1],10)*3600 + parseInt(segMatch[2],10)*60 + parseInt(segMatch[3],10) : (hour*3600 + minute*60);
  const dur = segMatch ? parseInt(segMatch[4], 10) : 300;
  const stream = parts.length === 3 ? parts[1] : "";
  const dayStr = `${parts[0].slice(0,4)}-${parts[0].slice(4,6)}-${parts[0].slice(6,8)}`;
  const meta = { day: dayStr, startSec, durationSec: dur, stream };

  const sample = await loadSegment(origin);
  if (!sample) {
    return renderEmptySegment(monthIndex, day, hour, minute, mode, focusLabel);
  }

  // Stash for the click-driven detail handlers.
  _activeSegment = sample;
  _activeMeta = meta;

  const audioHeader = sample.audio?.header || {};
  const screenHeader = sample.screen?.header || {};
  const audioLines = sample.audio?.lines || [];
  const screenFrames = sample.screen?.frames || [];

  const setting = audioHeader.setting || screenHeader.setting || "—";
  const rawTopics = audioHeader.topics ?? "";
  const topics = Array.isArray(rawTopics)
    ? rawTopics.map((s) => String(s).trim()).filter(Boolean)
    : String(rawTopics).split(",").map((s) => s.trim()).filter(Boolean);
  const fmtPct = (sec) => `${(sec / dur * 100).toFixed(2)}%`;

  // Pre-render screen ticks (one per frame) and audio dots/lines.
  const featuredCount = screenFrames.filter(isFeatured).length;
  const screenMarks = screenFrames.map((f) => {
    const a = f.analysis || {};
    const featured = isFeatured(f);
    const left = fmtPct(f.timestamp || 0);
    const tipText = `${segmentTimeLabel(meta, f.timestamp || 0)} · ${a.primary || "?"}\n${(a.visual_description || "").slice(0, 200)}`;
    // No always-visible labels — too crowded with 19 featured frames.
    // Tick height = featured signal; full content surfaces via title
    // hover and the click-to-detail panel.
    return `<button class="river-tick screen ${featured ? "is-featured" : ""}"
      data-frame-id="${f.frame_id}"
      style="left:${left}; --cat:${categoryColor(a.primary)};"
      title="${escapeHtml(tipText)}"
      type="button">
      <span class="river-tick-bar"></span>
      ${featured ? `<span class="river-tick-pip"></span>` : ""}
    </button>`;
  }).join("");

  const audioMarks = audioLines.length
    ? audioLines.map((line, i) => {
        // Convert "HH:MM:SS" → seconds offset from segment start.
        const [hh, mm, ss] = (line.start || "00:00:00").split(":").map(Number);
        const lineSec = hh * 3600 + mm * 60 + ss;
        const offset = Math.max(0, Math.min(dur, lineSec - meta.startSec));
        const sp = line.speaker || 1;
        const speakerColor = ["var(--blue)","var(--teal)","var(--coral)","var(--amber)"][sp - 1] || "var(--muted)";
        const tipText = `${line.start} · speaker ${sp}\n${(line.text || "").slice(0, 200)}`;
        return `<button class="river-audio-dot"
          data-audio-index="${i}"
          style="left:${fmtPct(offset)}; --cat:${speakerColor};"
          title="${escapeHtml(tipText)}"
          aria-label="${escapeHtml(tipText)}"
          type="button"></button>`;
      }).join("")
    : `<div class="river-empty">no microphone input in this slice</div>`;

  // Minute markers along the axis: 0, 60, 120, 180, 240, (the right edge is the segment end)
  const axisMarks = [0, 60, 120, 180, 240].map((s) =>
    `<div class="axis-mark" style="left:${fmtPct(s)};"><span>${segmentTimeLabel(meta, s).slice(0, 5)}</span></div>`
  ).join("");
  const startHHMM = segmentTimeLabel(meta, 0).slice(0, 5);
  const endHHMM = segmentTimeLabel(meta, dur).slice(0, 5);
  const minutesStr = `${Math.floor(dur / 60)} min${dur % 60 ? ` ${dur % 60}s` : ""}`;

  await setTimeline(`
    <div class="segment-view accent-${month.accent}">
      ${previous !== null ? renderEdgeSegment(monthIndex, day, hour, previous, "prev") : ""}
      ${next !== null ? renderEdgeSegment(monthIndex, day, hour, next, "next") : ""}

      <section class="segment-panel" aria-label="${month.name} ${day}, ${month.year || ""} ${focusLabel} segment observations">
        <div class="timeline-focus-heading">
          <button class="five-focus-node" type="button"
                  data-month="${monthIndex}" data-day="${day}" data-hour="${hour}"
                  data-return-hour="true"
                  aria-label="Return to ${formatHour(hour)} on ${month.name} ${day}, ${month.year || ""}">
            ${focusLabel}
          </button>
        </div>

        <header class="segment-header">
          <div class="seg-header-row">
            <span class="seg-header-time">${meta.day} · ${startHHMM} → ${endHHMM} · ${minutesStr}</span>
            <span class="seg-header-mid">${escapeHtml(meta.stream || "—")} observer</span>
            <span class="seg-header-end">${escapeHtml(setting)} setting</span>
          </div>
          ${topics.length ? `<div class="seg-topics">${topics.map((t) => `<span class="topic-chip">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
        </header>

        <div class="segment-river">
          <div class="river-screen" aria-label="screen frames sol observed">
            ${screenMarks}
          </div>
          <div class="river-axis">
            ${axisMarks}
          </div>
          <div class="river-audio" aria-label="microphone input">
            ${audioMarks}
          </div>
        </div>

        <div class="segment-detail" id="segment-detail">
          <div class="seg-detail-empty">click a tick or audio dot on the river to see what sol observed at that moment</div>
        </div>

        <footer class="segment-footer">
          ${screenFrames.length} frames analyzed
          · ${audioLines.length} transcript line${audioLines.length === 1 ? "" : "s"}
          · ${featuredCount} frames with extracted text
        </footer>
      </section>
    </div>
  `, mode);

  // Wire click handlers for tick + audio-dot selection.
  for (const tick of document.querySelectorAll(".river-tick[data-frame-id]")) {
    tick.addEventListener("click", (e) => {
      e.stopPropagation();
      const fid = parseInt(tick.getAttribute("data-frame-id"), 10);
      if (tick.classList.contains("is-active")) clearSegmentDetail();
      else showSegmentDetail(fid);
    });
  }
  for (const dot of document.querySelectorAll(".river-audio-dot[data-audio-index]")) {
    dot.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(dot.getAttribute("data-audio-index"), 10);
      if (dot.classList.contains("is-active")) clearSegmentDetail();
      else showSegmentAudioDetail(idx);
    });
  }
}

function renderEdgeDay(monthIndex, day, position) {
  const month = months[monthIndex];
  return `
    <button class="edge-day timeline-${position}" type="button" data-month="${monthIndex}" data-day="${day}" aria-label="Open ${month.name} ${day}, ${month.year || ""}">
      ${day}
    </button>
  `;
}

function renderEdgeHour(monthIndex, day, hour, position) {
  const month = months[monthIndex];
  return `
    <button class="edge-hour timeline-${position}" type="button" data-month="${monthIndex}" data-day="${day}" data-hour="${hour}" aria-label="Open ${formatHour(hour)} on ${month.name} ${day}, ${month.year || ""}">
      ${formatHour(hour)}
    </button>
  `;
}

function renderEdgeSegment(monthIndex, day, hour, minute, position) {
  return `
    <button class="edge-segment timeline-${position}" type="button" data-month="${monthIndex}" data-day="${day}" data-hour="${hour}" data-minute="${minute}" aria-label="Open ${formatTime(hour, minute)}">
      ${formatTime(hour, minute)}
    </button>
  `;
}

function renderEdgeMonth(month, index, position) {
  return `
    <button class="edge-node timeline-${position} accent-${month.accent}" type="button" data-month="${index}" aria-label="Open ${month.name} ${month.year || ""}">
      ${month.short}
    </button>
  `;
}

function renderDayEvent(event, days, side) {
  // Origin is internal-only: kept on the title attr for hover/inspection
  // and on the data attribute for layout, never rendered visibly.
  const originAttr = event.origin ? ` title="${escapeHtml(event.origin)}"` : "";
  return `
    <article class="day-event timeline-${side}" data-anchor-day="${event.day}" data-side="${side}"${originAttr}>
      <div class="day-date">Day ${event.day}</div>
      <h3>${escapeHtml(event.title)}</h3>
      <p>${escapeHtml(event.text)}</p>
    </article>
  `;
}

function renderHourEvent(event) {
  const originAttr = event.origin ? ` title="${escapeHtml(event.origin)}"` : "";
  return `
    <article class="hour-event timeline-${event.side}" data-anchor-hour="${event.hour}" data-side="${event.side}"${originAttr}>
      <div class="hour-time">${formatHour(event.hour)}</div>
      <h3>${escapeHtml(event.title)}</h3>
      <p>${escapeHtml(event.text)}</p>
    </article>
  `;
}

function renderMinuteEvent(event) {
  const originAttr = event.origin ? ` title="${escapeHtml(event.origin)}"` : "";
  return `
    <article class="minute-event timeline-${event.side}" data-anchor-minute="${event.minute}" data-side="${event.side}"${originAttr}>
      <div class="minute-time">${String(event.minute).padStart(2, "0")}</div>
      <h3>${escapeHtml(event.title)}</h3>
      <p>${escapeHtml(event.text)}</p>
    </article>
  `;
}

function renderMicroEvent(event, hour, minute) {
  return `
    <article class="micro-event timeline-${event.side}" style="grid-column: ${event.offset + 1} / span 1">
      <div class="micro-time">${formatTime(hour, minute + event.offset)}</div>
      <h3>${escapeHtml(event.title)}</h3>
      <p>${escapeHtml(event.text)}</p>
    </article>
  `;
}

timeline.addEventListener("click", async (event) => {
  const startButton = event.target.closest("[data-start-demo]");
  if (startButton) {
    showTimeline("zoom-in");
    return;
  }

  const closeButton = event.target.closest("[data-close-demo]");
  if (closeButton) {
    renderClosing("zoom-in");
    return;
  }

  const returnHourButton = event.target.closest("[data-return-hour]");
  if (returnHourButton) {
    const monthIndex = Number(returnHourButton.dataset.month);
    const day = Number(returnHourButton.dataset.day);
    const hour = Number(returnHourButton.dataset.hour);
    if (Number.isInteger(monthIndex) && Number.isInteger(day) && Number.isInteger(hour)) {
      selectedMonth = monthIndex;
      selectedDay = day;
      selectedHour = hour;
      selectedMinute = null;
      renderMinute(monthIndex, day, hour, "zoom-out");
    }
    return;
  }

  const returnDayButton = event.target.closest("[data-return-day]");
  if (returnDayButton) {
    const monthIndex = Number(returnDayButton.dataset.month);
    const day = Number(returnDayButton.dataset.day);
    if (Number.isInteger(monthIndex) && Number.isInteger(day)) {
      selectedMonth = monthIndex;
      selectedDay = day;
      selectedHour = null;
      selectedMinute = null;
      renderDay(monthIndex, day, "zoom-out");
    }
    return;
  }

  const returnMonthButton = event.target.closest("[data-return-month]");
  if (returnMonthButton) {
    const monthIndex = Number(returnMonthButton.dataset.month);
    if (Number.isInteger(monthIndex)) {
      selectedMonth = monthIndex;
      selectedDay = null;
      selectedHour = null;
      selectedMinute = null;
      if (months[monthIndex]?.ym) await loadMonth(months[monthIndex].ym);
      renderMonth(monthIndex, "zoom-out");
    }
    return;
  }

  const minuteButton = event.target.closest("[data-minute]");
  if (minuteButton) {
    const monthIndex = Number(minuteButton.dataset.month);
    const day = Number(minuteButton.dataset.day);
    const hour = Number(minuteButton.dataset.hour);
    const minute = Number(minuteButton.dataset.minute);
    if (
      Number.isInteger(monthIndex) &&
      Number.isInteger(day) &&
      Number.isInteger(hour) &&
      Number.isInteger(minute)
    ) {
      selectedMonth = monthIndex;
      selectedDay = day;
      selectedHour = hour;
      selectedMinute = minute;
      renderFiveMinute(monthIndex, day, hour, minute, minuteButton.classList.contains("edge-segment") ? "lateral" : "zoom-in");
    }
    return;
  }

  const hourButton = event.target.closest("[data-hour]");
  if (hourButton) {
    const monthIndex = Number(hourButton.dataset.month);
    const day = Number(hourButton.dataset.day);
    const hour = Number(hourButton.dataset.hour);
    if (Number.isInteger(monthIndex) && Number.isInteger(day) && Number.isInteger(hour)) {
      selectedMonth = monthIndex;
      selectedDay = day;
      selectedHour = hour;
      selectedMinute = null;
      renderMinute(monthIndex, day, hour, hourButton.classList.contains("edge-hour") ? "lateral" : "zoom-in");
    }
    return;
  }

  const dayButton = event.target.closest("[data-day]");
  if (dayButton) {
    const monthIndex = Number(dayButton.dataset.month);
    const day = Number(dayButton.dataset.day);
    if (Number.isInteger(monthIndex) && Number.isInteger(day)) {
      selectedMonth = monthIndex;
      selectedDay = day;
      selectedHour = null;
      selectedMinute = null;
      renderDay(monthIndex, day, dayButton.classList.contains("edge-day") ? "lateral" : "zoom-in");
    }
    return;
  }

  const button = event.target.closest("[data-month]");
  if (!button) return;

  const index = Number(button.dataset.month);
  if (!Number.isInteger(index)) return;

  if (selectedMonth === index && button.classList.contains("timeline-focus-node")) {
    selectedMonth = null;
    selectedDay = null;
    selectedHour = null;
    selectedMinute = null;
    renderYear("zoom-out");
    return;
  }

  selectedMonth = index;
  selectedDay = null;
  selectedHour = null;
  selectedMinute = null;
  if (months[index]?.ym) await loadMonth(months[index].ym);
  renderMonth(index, button.classList.contains("edge-node") ? "lateral" : "zoom-in");
});

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    event.preventDefault();
    goNextScreen();
    return;
  }

  if (event.key === "ArrowLeft") {
    event.preventDefault();
    goPreviousScreen();
  }
});

// Block initial render on the index fetch so the first paint already
// reflects real-month names + headlines. Cover screen is content-free,
// so a small delay is invisible to the user.
loadIndex().finally(() => renderCover("none"));
