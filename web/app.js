/* CurbIQ dashboard — build-free, reads precomputed artifacts from /api/*. */
const $ = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "–" : n.toLocaleString());
const fmt1 = (n) => (n == null ? "–" : (+n).toFixed(1));

const COLOR_STOPS = [
  [0.0, [43, 108, 176]], [0.25, [56, 189, 248]], [0.5, [250, 204, 21]],
  [0.75, [251, 146, 60]], [1.0, [255, 59, 48]],
];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    const [p1, c1] = COLOR_STOPS[i - 1], [p2, c2] = COLOR_STOPS[i];
    if (t <= p2) {
      const f = (t - p1) / (p2 - p1 || 1);
      const c = c1.map((v, k) => Math.round(v + f * (c2[k] - v)));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(255,59,48)";
}

Chart.defaults.color = "#8a97a8";
Chart.defaults.borderColor = "#232c3b";
Chart.defaults.font.family = "system-ui, sans-serif";

const DATA = {};
const ACCENT = "#22d3ee", ACCENT2 = "#a78bfa", HOT = "#ff3b30", WARN = "#ffb020", GOOD = "#34d399";
const UNIT_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#ffb020", "#fb923c", "#ff3b30", "#60a5fa", "#f472b6"];
let map, hexLayer, heatLayer, junctionLayer, zoneLayer, blindLayer, novelLayer, patrolLayer, weeklyLayer;
let weekTimer = null, deckInstance = null;
const charts = {};
const builtTabs = new Set();

async function getJSON(name) {
  const r = await fetch(`/api/${name}`);
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  return r.json();
}

async function init() {
  try {
    const [kpis, cells, priority, junctions, zones, forecast, fairness, calibration, geo, patrol, timeseries, weekly, model, emerging, manifest] =
      await Promise.all(["kpis", "cells", "priority", "junctions", "zones", "forecast",
        "fairness", "calibration", "geo-validation", "patrol", "timeseries", "weekly", "model-metrics", "emerging", "manifest"].map(getJSON));
    Object.assign(DATA, { kpis, cells: cells.cells, kanon: cells.k_anon, priority,
      junctions, zones, forecast: forecast.cells, fairness, calibration, geo, patrol, timeseries, weekly, model, emerging, manifest });
  } catch (e) {
    $("loading").textContent = "Failed to load artifacts — run `python build_all.py`. " + e.message;
    return;
  }
  $("loading").style.display = "none";
  renderKPIs();
  renderMetaPills();
  buildMap();
  populateOffences();
  renderHexes();
  renderPriorityTable();
  renderJunctionTable();
  renderGeoStats();
  renderPatrolStats();
  ensureTab("overview");
  wireControls();
  setupTimeline();
  setup3D();
  $("footer").innerHTML = `Artifacts v${DATA.manifest.version} · ${fmt(DATA.manifest.dataset.records)} raw records · `
    + `${DATA.kanon.frac_suppressed ? (DATA.kanon.frac_suppressed * 100).toFixed(0) : 0}% cells k-anon suppressed · `
    + `License ${DATA.manifest.license}`;
}

function renderKPIs() {
  const k = DATA.kpis;
  const items = [
    ["Violations", fmt(k.total_violations), false],
    ["Hotspots", fmt(k.n_hotspots), false],
    ["Blind spots", fmt(k.n_blind_spots), true],
    ["Eve. enforce", (k.evening_peak_enforcement_share * 100).toFixed(1) + "%", true],
    ["Forecast PAI@5", fmt1(k.forecast_pai_at_5) + "×", false],
    ["Moran z", fmt1(k.global_moran_z), false],
  ];
  $("kpis").innerHTML = items.map(([l, v, a]) =>
    `<div class="kpi"><div class="v ${a ? "alert" : ""}">${v}</div><div class="l">${l}</div></div>`).join("");
}

function renderMetaPills() {
  const k = DATA.kpis;
  $("meta-pills").innerHTML = [
    `${fmt(k.n_police_stations)} stations`, `${fmt(k.n_junctions)} junctions`,
    `${fmt(k.n_h3_cells)} H3 cells`, k.date_range.join(" → "),
  ].map((t) => `<span class="pill">${t}</span>`).join("");
}

function buildMap() {
  map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([12.9716, 77.5946], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "© OpenStreetMap, © CARTO", subdomains: "abcd", maxZoom: 19,
  }).addTo(map);
  hexLayer = L.layerGroup().addTo(map);
  junctionLayer = L.layerGroup();
  zoneLayer = L.layerGroup();
  blindLayer = L.layerGroup();
  novelLayer = L.layerGroup();
  patrolLayer = L.layerGroup();
  weeklyLayer = L.layerGroup();
}

function activeCells() {
  const metric = $("metric").value;
  const minpri = +$("minpri").value;
  const off = $("offence").value;
  const hotonly = $("t-hotonly").checked;
  return DATA.cells.filter((c) =>
    c[metric] != null && c.priority_score >= minpri
    && (!off || c.top_offence === off) && (!hotonly || c.is_hotspot));
}

function renderHexes() {
  hexLayer.clearLayers();
  if (!$("t-hex").checked) return;
  const metric = $("metric").value;
  const cells = activeCells();
  const vals = cells.map((c) => +c[metric]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (metric === "gi_z") lo = Math.max(lo, -3);
  $("legmin").textContent = isFinite(lo) ? fmt1(lo) : "low";
  $("legmax").textContent = isFinite(hi) ? fmt1(hi) : "high";
  const span = hi - lo || 1;
  for (const c of cells) {
    const t = (+c[metric] - lo) / span;
    const ring = h3.cellToBoundary(c.h3);          // [[lat,lng],...]
    const poly = L.polygon(ring, {
      fillColor: ramp(t), color: c.is_hotspot ? "#ffffff" : ramp(t),
      weight: c.is_hotspot ? 1.1 : 0.3, fillOpacity: 0.62, opacity: c.is_hotspot ? 0.9 : 0.4,
    });
    poly.bindPopup(cellPopup(c));
    hexLayer.addLayer(poly);
  }
}

function cellPopup(c) {
  const row = (l, v) => `<div class="popup-row"><span>${l}</span><span>${v}</span></div>`;
  return `<b>Cell ${c.h3.slice(0, 8)}…</b>`
    + (c.zone_id ? ` <span class="badge">${c.zone_id}</span>` : "")
    + (c.is_hotspot ? ` <span class="badge" style="color:#ff3b30;border-color:#ff3b30">HOTSPOT ${c.gi_band || ""}</span>` : "")
    + `<div style="margin-top:6px">`
    + row("Violations", fmt(c.count))
    + row("Priority", fmt1(c.priority_score) + " (#" + c.priority_rank + ")")
    + row("Gi* z-score", fmt1(c.gi_z))
    + row("Congestion CIS", fmt1(c.cis_score))
    + row("Modeled extra delay", fmt1(c.extra_delay_pct) + "%")
    + row("Forecast (next day)", fmt1(c.forecast_area))
    + row("Top offence", c.top_offence || "–")
    + (c.is_blind_spot ? `<div style="color:#ffb020;margin-top:4px">⚠ Under-enforcement blind spot</div>` : "")
    + `</div>`;
}

function toggleHeat() {
  if ($("t-heat").checked) {
    const pts = DATA.cells.map((c) => [c.lat, c.lon, Math.min(1, c.count / 2000)]);
    heatLayer = L.heatLayer(pts, { radius: 18, blur: 22, maxZoom: 15,
      gradient: { 0.2: "#2b6cb0", 0.4: "#38bdf8", 0.6: "#facc15", 0.8: "#fb923c", 1: "#ff3b30" } });
    heatLayer.addTo(map);
  } else if (heatLayer) { map.removeLayer(heatLayer); }
}

function toggleJunctions() {
  junctionLayer.clearLayers();
  if (!$("t-junctions").checked) { map.removeLayer(junctionLayer); return; }
  for (const j of DATA.junctions) {
    const r = 4 + Math.sqrt(j.count) / 12;
    L.circleMarker([j.lat, j.lon], { radius: r, color: ACCENT2, weight: 1.5,
      fillColor: ACCENT2, fillOpacity: 0.4 })
      .bindPopup(`<b>${j.junction_id}</b><div class="popup-row"><span>Violations</span><span>${fmt(j.count)}</span></div>`
        + `<div class="popup-row"><span>Peak share</span><span>${(j.peak_share * 100).toFixed(0)}%</span></div>`
        + `<div class="popup-row"><span>Top offence</span><span>${j.top_offence}</span></div>`)
      .addTo(junctionLayer);
  }
  junctionLayer.addTo(map);
}

function toggleZones() {
  zoneLayer.clearLayers();
  if (!$("t-zones").checked) { map.removeLayer(zoneLayer); return; }
  for (const z of DATA.zones) {
    L.marker([z.lat, z.lon]).bindPopup(
      `<b>${z.zone_id}</b> — ${fmt(z.count)} violations<br>${z.n_cells} cells · peak Gi* z ${fmt1(z.peak_gi_z)}<br>${z.top_offence}`)
      .addTo(zoneLayer);
  }
  zoneLayer.addTo(map);
}

function toggleBlind() {
  blindLayer.clearLayers();
  if (!$("t-blind").checked) { map.removeLayer(blindLayer); return; }
  for (const b of DATA.priority.blind_spots) {
    L.circleMarker([b.lat, b.lon], { radius: 7, color: WARN, weight: 2, fillOpacity: 0.15 })
      .bindPopup(`<b>Blind spot</b><div class="popup-row"><span>Gap (pctile)</span><span>${fmt1(b.under_enforcement_gap)}</span></div>`
        + `<div class="popup-row"><span>Recorded</span><span>${fmt(b.count)}</span></div>`
        + `<div class="popup-row"><span>Congestion CIS</span><span>${fmt1(b.cis_score)}</span></div>`)
      .addTo(blindLayer);
  }
  blindLayer.addTo(map);
}

function toggleNovel() {
  novelLayer.clearLayers();
  if (!$("t-novel").checked) { map.removeLayer(novelLayer); return; }
  for (const n of DATA.geo.novel_hotspots) {
    L.circleMarker([n.lat, n.lon], { radius: 8, color: ACCENT2, weight: 2, fillColor: ACCENT2, fillOpacity: 0.2 })
      .bindPopup(`<b>Novel hotspot</b> (off-junction)`
        + `<div class="popup-row"><span>Violations</span><span>${fmt(n.count)}</span></div>`
        + `<div class="popup-row"><span>Nearest junction</span><span>${fmt(Math.round(n.nearest_ref_m))} m</span></div>`
        + `<div class="popup-row"><span>Top offence</span><span>${n.top_offence || "–"}</span></div>`)
      .addTo(novelLayer);
  }
  novelLayer.addTo(map);
}

function renderGeoStats() {
  const g = DATA.geo;
  $("geo-badge").textContent = g.is_official_list ? "official list" : `${g.n_reference_points} BTP junctions`;
  $("geo-stats").innerHTML = [
    ["precision@50", (g.precision_at_n.top50["300m"] * 100).toFixed(0) + "%"],
    ["precision@154", (g.precision_at_n.top154["300m"] * 100).toFixed(0) + "%"],
    ["novel hotspots", g.n_novel_hotspots],
    ["median dist", Math.round(g.median_nearest_m_top50) + "m"],
  ].map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function populateOffences() {
  const sel = $("offence");
  // only offences that are actually a cell's dominant offence (else filter blanks map)
  const present = [...new Set(DATA.cells.map((c) => c.top_offence).filter(Boolean))].sort();
  present.forEach((o) => {
    const opt = document.createElement("option"); opt.value = o; opt.textContent = o; sel.appendChild(opt);
  });
}

function renderPriorityTable() {
  const rows = DATA.priority.top.slice(0, 10).map((c, i) =>
    `<tr><td>${i + 1}</td><td>${c.top_offence || "–"}</td>`
    + `<td class="num">${fmt(c.count)}</td><td class="num">${fmt1(c.priority_score)}</td>`
    + `<td class="num">${fmt1(c.cis_score)}</td></tr>`).join("");
  $("tbl-priority").innerHTML = `<tr><th>#</th><th>Offence</th><th class="num">Viol.</th>`
    + `<th class="num">Priority</th><th class="num">CIS</th></tr>` + rows;
}

function renderJunctionTable() {
  const rows = DATA.junctions.slice(0, 8).map((j) =>
    `<tr><td>${j.junction_id.replace(/^BTP\d+\s*-\s*/, "")}</td>`
    + `<td class="num">${fmt(j.count)}</td><td class="num">${(j.peak_share * 100).toFixed(0)}%</td></tr>`).join("");
  $("tbl-junctions").innerHTML = `<tr><th>Junction</th><th class="num">Viol.</th><th class="num">Peak</th></tr>` + rows;
}

/* ---- charts (lazy per tab) ---- */
function ensureTab(tab) {
  if (builtTabs.has(tab)) return;
  builtTabs.add(tab);
  ({ overview: buildOverviewCharts, temporal: buildTemporalCharts,
     model: buildModelCharts, equity: buildEquityCharts }[tab] || (() => {}))();
}

function buildOverviewCharts() {
  const cc = DATA.priority.coverage_curve;
  charts.coverage = new Chart($("ch-coverage"), {
    type: "line",
    data: { labels: cc.frac_locations.map((f) => (f * 100).toFixed(0) + "%"),
      datasets: [{ label: "Violations captured", data: cc.frac_violations_captured.map((v) => v * 100),
        borderColor: ACCENT, backgroundColor: "rgba(34,211,238,.15)", fill: true, pointRadius: 0, tension: .3 }] },
    options: { plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "% of locations enforced" } },
        y: { title: { display: true, text: "% violations captured" }, max: 100 } } },
  });
}

function buildTemporalCharts() {
  const t = DATA.timeseries, f = DATA.fairness.temporal;
  charts.hourly = new Chart($("ch-hourly"), {
    type: "bar",
    data: { labels: t.hourly_ist.hour,
      datasets: [
        { label: "Enforcement", data: t.hourly_ist.count, backgroundColor: ACCENT, order: 2 },
        { label: "Congestion risk (scaled)", type: "line", order: 1, borderColor: HOT, pointRadius: 0, tension: .3,
          data: f.risk_share.map((r) => r * Math.max(...t.hourly_ist.count) * 5) },
      ] },
    options: { scales: { x: { title: { display: true, text: "Hour (IST)" } } } },
  });
  charts.daily = new Chart($("ch-daily"), {
    type: "line",
    data: { labels: t.daily.date,
      datasets: [{ data: t.daily.count, borderColor: ACCENT2, pointRadius: 0, tension: .25, fill: false }] },
    options: { plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 8 } } } },
  });
  const vc = t.vehicle_category;
  charts.vehicle = new Chart($("ch-vehicle"), {
    type: "doughnut",
    data: { labels: Object.keys(vc), datasets: [{ data: Object.values(vc),
      backgroundColor: ["#22d3ee", "#a78bfa", "#34d399", "#ffb020", "#fb923c", "#ff3b30", "#64748b"] }] },
    options: { plugins: { legend: { position: "right", labels: { boxWidth: 10, font: { size: 10 } } } } },
  });
  const off = t.top_offences;
  charts.offence = new Chart($("ch-offence"), {
    type: "bar",
    data: { labels: Object.keys(off).map((o) => o.length > 22 ? o.slice(0, 22) + "…" : o),
      datasets: [{ data: Object.values(off), backgroundColor: ACCENT }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } } },
  });
}

function buildModelCharts() {
  const m = DATA.model.forecast, h = m.holdout.metrics;
  $("model-stats").innerHTML = [
    ["PAI@5%", fmt1(h["pai@5"]) + "×"], ["PAI@20%", fmt1(h["pai@20"]) + "×"],
    ["ROC-AUC", fmt1(h.roc_auc * 100) + "%"], ["R²", fmt1(h.r2)],
    ["MAE", fmt1(h.mae)], ["PEI@5%", fmt1(h["pei@5"])],
  ].map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  const b = m.baselines;
  const names = ["model", ...Object.keys(b)];
  const pai = [h["pai@5"], ...Object.values(b).map((x) => x["pai@5"])];
  charts.baselines = new Chart($("ch-baselines"), {
    type: "bar",
    data: { labels: names, datasets: [{ data: pai,
      backgroundColor: names.map((n) => n === "model" ? GOOD : "#475569") }] },
    options: { plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "PAI@5% (×)" } } } },
  });
  const fi = m.feature_importances;
  const top = Object.entries(fi).slice(0, 12);
  charts.features = new Chart($("ch-features"), {
    type: "bar",
    data: { labels: top.map((x) => x[0]), datasets: [{ data: top.map((x) => x[1]), backgroundColor: ACCENT2 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } } },
  });

  const cal = DATA.calibration;
  $("calib-badge").textContent = cal.synthetic ? "synthetic probe" : "real probe";
  $("calib-stats").innerHTML = [
    ["ρ default", fmt1(cal.spearman_default)],
    ["ρ calibrated", fmt1(cal.spearman_calibrated)],
    ["isotonic R²", fmt1(cal.isotonic_r2)],
    ["MAE", fmt1(cal.isotonic_mae_pct) + "%"],
  ].map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  charts.calib = new Chart($("ch-calib"), {
    type: "scatter",
    data: { datasets: [
      { label: "cells", data: cal.scatter.map((p) => ({ x: p.cis, y: p.observed_pct })),
        backgroundColor: "rgba(34,211,238,.5)", pointRadius: 2 },
      { label: "isotonic fit", type: "line", showLine: true, tension: 0, pointRadius: 0,
        borderColor: HOT, data: cal.scatter.map((p) => ({ x: p.cis, y: p.fit_pct })) },
    ] },
    options: { plugins: { legend: { labels: { boxWidth: 10, font: { size: 9 } } } },
      scales: { x: { title: { display: true, text: "calibrated CIS composite" } },
        y: { title: { display: true, text: "observed congestion %" } } } },
  });
}

function buildEquityCharts() {
  const f = DATA.fairness;
  const g = f.temporal;
  charts.gap = new Chart($("ch-gap"), {
    type: "bar",
    data: { labels: g.hour,
      datasets: [{ data: g.under_enforcement_gap.map((v) => v * 100),
        backgroundColor: g.under_enforcement_gap.map((v) => v > 0 ? WARN : "#475569") }] },
    options: { plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "Hour (IST)" } },
        y: { title: { display: true, text: "gap (× risk - enforce, %)" } } } },
  });
  const se = f.spatial_equity;
  $("equity-stats").innerHTML = [
    ["Disparate impact", fmt1(se.disparate_impact_ratio), se.disparate_impact_flag],
    ["Parity diff", fmt1(se.statistical_parity_diff * 100) + "%", se.statistical_parity_flag],
    ["Stations", se.n_stations, false],
  ].map(([l, v, a]) => `<div class="stat"><div class="v" style="${a ? "color:#ff3b30" : ""}">${v}</div><div class="l">${l}</div></div>`).join("");
  const ue = se.most_under_enforced;
  $("tbl-equity").innerHTML = `<tr><th>Under-enforced station</th><th class="num">Coverage</th></tr>`
    + ue.map((s) => `<tr><td>${s.police_station}</td><td class="num">${fmt1(s.coverage_ratio)}</td></tr>`).join("");

  const em = DATA.emerging.by_category;
  charts.emerging = new Chart($("ch-emerging"), {
    type: "polarArea",
    data: { labels: Object.keys(em), datasets: [{ data: Object.values(em),
      backgroundColor: ["#ff3b30", "#fb923c", "#ffb020", "#22d3ee", "#a78bfa", "#34d399", "#64748b"] }] },
    options: { plugins: { legend: { position: "right", labels: { boxWidth: 10, font: { size: 10 } } } } },
  });
}

function togglePatrol() {
  patrolLayer.clearLayers();
  if (!$("t-patrol").checked) { map.removeLayer(patrolLayer); return; }
  const dep = DATA.patrol.depot;
  L.circleMarker([dep.lat, dep.lon], { radius: 7, color: "#fff", weight: 2, fillColor: "#0b0e14", fillOpacity: 0.95 })
    .bindPopup("<b>Patrol depot</b>").addTo(patrolLayer);
  DATA.patrol.routes.forEach((rt, i) => {
    if (!rt.stops.length) return;
    const col = UNIT_COLORS[i % UNIT_COLORS.length];
    const pts = [[dep.lat, dep.lon], ...rt.stops.map((s) => [s.lat, s.lon]), [dep.lat, dep.lon]];
    L.polyline(pts, { color: col, weight: 3, opacity: 0.85, dashArray: "4 6" }).addTo(patrolLayer);
    rt.stops.forEach((s) => {
      L.circleMarker([s.lat, s.lon], { radius: 5, color: col, weight: 2, fillColor: col, fillOpacity: 0.7 })
        .bindPopup(`<b>${rt.unit}</b> · stop ${s.seq}`
          + `<div class="popup-row"><span>ETA</span><span>${s.eta}</span></div>`
          + `<div class="popup-row"><span>Priority</span><span>${fmt1(s.priority)}</span></div>`
          + `<div class="popup-row"><span>Offence</span><span>${s.top_offence || "–"}</span></div>`)
        .addTo(patrolLayer);
    });
  });
  patrolLayer.addTo(map);
}

function renderPatrolStats() {
  const p = DATA.patrol;
  $("patrol-badge").textContent = p.solver === "ortools" ? "OR-Tools VRP" : "greedy VRP";
  const active = p.routes.filter((r) => r.n_stops > 0).length;
  $("patrol-stats").innerHTML = [
    ["units used", `${active}/${p.n_units}`],
    ["coverage", p.coverage_pct + "%"],
    ["total km", p.total_distance_km],
    ["shift", p.shift_start],
  ].map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

/* ---- time-slider: weekly hotspot evolution ---- */
function setupTimeline() {
  const wk = DATA.weekly;
  if (!wk || !wk.weeks.length) return;
  const slider = $("week-slider");
  slider.max = wk.weeks.length - 1;
  slider.value = wk.weeks.length - 1;
  slider.addEventListener("input", () => renderWeek(+slider.value));
  $("week-all").addEventListener("click", showAllTime);
  $("week-play").addEventListener("click", togglePlay);
}

function renderWeek(idx) {
  const wk = DATA.weekly;
  if (map.hasLayer(hexLayer)) map.removeLayer(hexLayer);
  weeklyLayer.clearLayers();
  const maxc = wk.max_count || 1;
  for (const c of wk.cells) {
    const v = c.counts[idx];
    if (!v) continue;
    const t = Math.sqrt(v / maxc);
    const col = ramp(t);
    L.polygon(h3.cellToBoundary(c.h3), { fillColor: col, color: col, weight: 0.3, fillOpacity: 0.7 })
      .bindPopup(`<b>${wk.weeks[idx]}</b><div class="popup-row"><span>Violations</span><span>${fmt(v)}</span></div>`)
      .addTo(weeklyLayer);
  }
  weeklyLayer.addTo(map);
  $("week-label").textContent = wk.weeks[idx];
}

function showAllTime() {
  stopPlay();
  if (map.hasLayer(weeklyLayer)) map.removeLayer(weeklyLayer);
  hexLayer.addTo(map);
  renderHexes();
  $("week-label").textContent = "All-time";
}

function togglePlay() {
  if (weekTimer) { stopPlay(); return; }
  $("week-play").textContent = "⏸";
  const slider = $("week-slider");
  weekTimer = setInterval(() => {
    const v = (+slider.value + 1) % DATA.weekly.weeks.length;
    slider.value = v;
    renderWeek(v);
  }, 750);
}

function stopPlay() {
  if (weekTimer) { clearInterval(weekTimer); weekTimer = null; $("week-play").textContent = "▶"; }
}

/* ---- optional deck.gl 3D extruded hexes (self-contained overlay) ----
   @deck.gl/leaflet ships no browser UMD, so instead of syncing to Leaflet we
   render a standalone deck.gl Deck (its own controller) over the map div. */
function deckAvailable() {
  return !!(window.deck && deck.Deck && deck.H3HexagonLayer);
}
function rampRGBA(t) {
  const m = ramp(t).match(/\d+/g).map(Number);
  return [m[0], m[1], m[2], 210];
}
function setup3D() {
  if (!deckAvailable()) return;            // leave the toggle hidden if deck.gl absent
  $("t-3d-wrap").style.display = "flex";
  $("t-3d").addEventListener("change", toggle3D);
}
function destroyDeck() {
  if (deckInstance) { try { deckInstance.finalize(); } catch (e) {} deckInstance = null; }
  const c = document.getElementById("deck-canvas");
  if (c) c.remove();
}
function toggle3D() {
  destroyDeck();
  if (!$("t-3d").checked) return;
  const metric = $("metric").value;
  const cells = activeCells();
  const vals = cells.map((c) => +c[metric]);
  const lo = Math.min(...vals), hi = Math.max(...vals) || 1, span = hi - lo || 1;
  const data = cells.map((c) => ({ h3: c.h3, t: (+c[metric] - lo) / span, count: c.count, off: c.top_offence }));
  const canvas = document.createElement("canvas");
  canvas.id = "deck-canvas";
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:500;background:#0b0e14;";
  document.getElementById("map").appendChild(canvas);
  const c0 = map.getCenter();
  deckInstance = new deck.Deck({
    canvas,
    initialViewState: { longitude: c0.lng, latitude: c0.lat, zoom: map.getZoom() - 1, pitch: 50, bearing: 15 },
    controller: true,
    layers: [new deck.H3HexagonLayer({
      id: "h3-3d", data, extruded: true, getHexagon: (d) => d.h3,
      getElevation: (d) => d.count, elevationScale: 1.5,
      getFillColor: (d) => rampRGBA(d.t), opacity: 0.92, pickable: true,
    })],
    getTooltip: ({ object }) => object && { text: `${object.off || ""}\n${object.count} violations` },
  });
}

function wireControls() {
  $("metric").addEventListener("change", () => { renderHexes(); if (deckInstance) toggle3D(); });
  $("t-hex").addEventListener("change", renderHexes);
  $("t-hotonly").addEventListener("change", renderHexes);
  $("offence").addEventListener("change", renderHexes);
  $("minpri").addEventListener("input", (e) => { $("minpri-v").textContent = e.target.value; renderHexes(); });
  $("t-heat").addEventListener("change", toggleHeat);
  $("t-junctions").addEventListener("change", toggleJunctions);
  $("t-zones").addEventListener("change", toggleZones);
  $("t-blind").addEventListener("change", toggleBlind);
  $("t-novel").addEventListener("change", toggleNovel);
  $("t-patrol").addEventListener("change", togglePatrol);
  document.querySelectorAll(".tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      $("tab-" + tab).classList.add("active");
      ensureTab(tab);
    });
  });
}

init();

// installable PWA + offline shell
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
