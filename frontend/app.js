/* ResaleIQ frontend — vanilla JS, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
const NS = "http://www.w3.org/2000/svg";

let META = null, METRICS = null;
let lastResult = null;

/* ---------------- nav + reveal ---------------- */
const nav = $("#nav");
addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 10), { passive: true });

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    if (e.target.dataset.count !== undefined) countUp(e.target);
    io.unobserve(e.target);
  }
}, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });
$$(".reveal").forEach((el, i) => { el.style.transitionDelay = `${(i % 4) * 70}ms`; io.observe(el); });
$$("[data-count]").forEach((el) => io.observe(el));

function countUp(el, to = +el.dataset.count, fmt = (v) => Math.round(v).toLocaleString("en-IN"), dur = 1400) {
  const from = 0, t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4);
    el.textContent = fmt(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------- segmented controls ---------------- */
function seg(root, onChange) {
  const pill = $(".seg-pill", root), btns = $$(".seg-btn", root);
  const place = () => {
    const a = btns.find((b) => b.classList.contains("active"));
    if (!a) return;
    pill.style.width = a.offsetWidth + "px";
    pill.style.transform = `translateX(${a.offsetLeft - 3}px)`;
  };
  btns.forEach((b) => b.addEventListener("click", () => {
    if (b.classList.contains("active")) return;
    btns.forEach((x) => x.classList.toggle("active", x === b));
    place(); onChange && onChange(b.dataset.v);
  }));
  addEventListener("resize", place);
  new ResizeObserver(place).observe(root);
  requestAnimationFrame(place);
  return { set(v) { btns.forEach((x) => x.classList.toggle("active", x.dataset.v === v)); place(); } };
}

/* ---------------- charts ---------------- */
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  parent && parent.appendChild(n);
  return n;
}

function lineChart(svg, series, { xLabel = (x) => x, yLabel = (y) => y, yMin, yMax, marker } = {}) {
  svg.innerHTML = ""; svg.classList.remove("in");
  const W = 600, H = 300, L = 56, R = 16, T = 14, B = 34;
  const xs = series[0].points.map((p) => p.x);
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  const y0 = yMin ?? Math.min(...allY) * 0.92, y1 = yMax ?? Math.max(...allY) * 1.05;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
  const g = el("g", { class: "grid" }, svg);
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4;
    el("line", { x1: L, x2: W - R, y1: Y(yv), y2: Y(yv) }, g);
    el("text", { x: L - 8, y: Y(yv) + 4, "text-anchor": "end" }, svg).textContent = yLabel(yv);
  }
  const step = Math.max(1, Math.ceil(xs.length / 9));
  xs.forEach((x, i) => { if (i % step === 0) el("text", { x: X(x), y: H - 10, "text-anchor": "middle" }, svg).textContent = xLabel(x); });
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    const path = el("path", { d, class: "line draw", stroke: s.color, ...(s.dash ? { "stroke-dasharray": "6 6" } : {}) }, svg);
    const len = path.getTotalLength(); path.style.setProperty("--len", len);
    if (s.dash) path.classList.remove("draw");
    const last = s.points[s.points.length - 1];
    el("circle", { cx: X(last.x), cy: Y(last.y), r: 4, fill: s.color, class: "dot" }, svg);
  }
  if (marker) {
    el("line", { x1: X(marker.x), x2: X(marker.x), y1: T, y2: H - B, stroke: "currentColor", "stroke-opacity": .25, "stroke-dasharray": "3 4", class: "dot" }, svg);
    el("circle", { cx: X(marker.x), cy: Y(marker.y), r: 7, fill: "var(--accent)", stroke: "var(--card)", "stroke-width": 3, class: "dot" }, svg);
    const t = el("text", { x: X(marker.x), y: Y(marker.y) - 14, "text-anchor": "middle", class: "dot", style: "fill:var(--text);font-weight:600;font-size:12px" }, svg);
    t.textContent = marker.label;
  }
  requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.add("in")));
}

/* Findings chart: Module 6 same-variant curves */
function drawFindings() {
  const mk = (start, loss) => Array.from({ length: 8 }, (_, a) => ({ x: a, y: start * Math.pow(1 - loss, a) * 100 }));
  lineChart($("#findChart"), [
    { points: mk(0.90, 0.061), color: "var(--petrol)" },
    { points: mk(0.90, 0.076), color: "var(--ola)" },
    { points: mk(0.88, 0.12), color: "var(--ev)" },
  ], { xLabel: (x) => `${x}y`, yLabel: (y) => `${Math.round(y)}%`, yMin: 30, yMax: 100 });
}
const fc = $("#findChart");
new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { drawFindings(); o.disconnect(); } }, { threshold: .3 }).observe(fc);

/* Road-tax side note bars (Classic 350, 2–5 yrs old) */
const TAX = [["Tamil Nadu", 8, 115000], ["Delhi", 8, 118000], ["Haryana", 6, 120000], ["Rajasthan", 8, 121000], ["Gujarat", 6, 121050],
  ["Maharashtra", 12, 125000], ["Uttar Pradesh", 10, 125000], ["Punjab", 9, 132500], ["Telangana", 12, 133300], ["Karnataka", 18, 135000]];
(function taxBars() {
  const box = $("#taxBars"), min = 100000, max = 137000;
  box.innerHTML = TAX.map(([s, r, p]) => `<div class="bar-row"><span>${s}<span class="rate">${r}%</span></span>
    <div class="bar-track"><div class="bar-fill" data-w="${((p - min) / (max - min)) * 100}"></div></div><span class="v">${inr(p)}</span></div>`).join("");
  new IntersectionObserver((e, o) => {
    if (!e[0].isIntersecting) return;
    $$(".bar-fill", box).forEach((b, i) => setTimeout(() => (b.style.width = b.dataset.w + "%"), i * 60));
    o.disconnect();
  }, { threshold: .3 }).observe(box);
})();

/* ---------------- metrics table ---------------- */
const SEG_NAMES = { overall: "All test ads", baseline_rule: "Simple rule: new price × typical % kept", site_valuation: "The listing site's own valuation",
  ours_same_ads: "Ours, on the same ads", "Petrol motorcycle": "Petrol motorcycles", "Petrol scooter": "Petrol scooters",
  "Electric scooter": "Electric scooters", "Electric motorcycle": "Electric motorcycles", "CNG motorcycle": "CNG motorcycles" };
function renderMetrics(kind) {
  if (!METRICS || !METRICS[kind]) return;
  const pct = (x) => (x * 100).toFixed(1) + "%";
  if (kind.endsWith("_market")) {
    const segs = METRICS[kind].segments;
    $("#metricHead").innerHTML = "<tr><th>Ads tested</th><th>n</th><th>Typical miss</th><th>Within 20%</th><th class='barcol'></th></tr>";
    $("#metricTable tbody").innerHTML = Object.entries(segs).map(([k, v]) => "<tr class='" + (k === "overall" ? "best" : "") + "'><td>" +
      esc(SEG_NAMES[k] || k) + "</td><td>" + v.n.toLocaleString("en-IN") + "</td><td>" + pct(v.median_abs_pct_err) + "</td><td>" +
      pct(v.within_20pct) + "</td><td><div class='r2bar'><i data-w='" + (v.within_20pct * 100).toFixed(1) + "'></i></div></td></tr>").join("");
  } else {
    const m = METRICS[kind].models;
    const best = Object.entries(m).sort((a, b) => a[1].MAE - b[1].MAE)[0][0];
    $("#metricHead").innerHTML = "<tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th class='barcol'></th></tr>";
    $("#metricTable tbody").innerHTML = Object.entries(m).map(([n, v]) => "<tr class='" + (n === best ? "best" : "") + "'><td>" + n +
      "</td><td>" + inr(v.MAE) + "</td><td>" + inr(v.RMSE) + "</td><td>" + v.R2.toFixed(3) +
      "</td><td><div class='r2bar'><i data-w='" + (Math.max(0, v.R2) * 100).toFixed(1) + "'></i></div></td></tr>").join("");
  }
  requestAnimationFrame(() => $$("#metricTable .r2bar i").forEach((i) => (i.style.width = i.dataset.w + "%")));
}
seg($("#metricSeg"), renderMetrics);

/* ---------------- form ---------------- */
const F = { brand: $("#brand"), model: $("#model"), variant: $("#variant"), year: $("#year"), km: $("#km"), owner: $("#owner"),
  seller: $("#seller"), state: $("#state"), city: $("#city"), trans: $("#trans"), fuel: $("#fuel"), cc: $("#cc"),
  power: $("#power"), seats: $("#seats"), exs: $("#exs"), kwh: $("#kwh") };
let KIND = "2w", FUELF = "petrol"; // 2w | car ; petrol | electric
let VARIANTS = [];
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function modelsFor(brand) {
  const list = META[KIND].brands[brand] || [];
  if (KIND === "car") return list;
  return list.filter((m) => (FUELF === "electric") === (m.fuel === "electric"));
}
function brandsFor() { return Object.keys(META[KIND].brands).filter((b) => modelsFor(b).length); }
function current() { return modelsFor(F.brand.value).find((m) => m.model === F.model.value); }
function currentVariant() { return VARIANTS.find((v) => v.id === F.variant.value); }

function fillBrands(keep) {
  const bs = brandsFor(), prev = keep ?? F.brand.value;
  F.brand.innerHTML = bs.map((b) => `<option>${esc(b)}</option>`).join("");
  if (bs.includes(prev)) F.brand.value = prev;
  fillModels();
}
const TYPE_LABEL = { motorcycle: "Motorcycles", scooter: "Scooters", moped: "Mopeds" };
function fillModels(keep) {
  const list = modelsFor(F.brand.value), prev = keep ?? F.model.value;
  const groups = new Map();
  for (const m of list) {
    const g = !m.on_sale ? "No longer sold new"
      : KIND === "car" ? "On sale" : `${TYPE_LABEL[m.type] || "Two-wheelers"} on sale`;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  const order = [...groups.keys()].sort((a, b) => a.startsWith("No longer") - b.startsWith("No longer"));
  F.model.innerHTML = order.map((g) => `<optgroup label="${esc(g)}">${groups.get(g).map((m) =>
    `<option value="${esc(m.model)}">${esc(m.model)}${m.fuel === "cng" ? " (CNG)" : ""}</option>`).join("")}</optgroup>`).join("");
  if (list.some((m) => m.model === prev)) F.model.value = prev;
  onModel();
}
let variantReq = 0;
async function onModel() {
  const m = current();
  if (!m) return;
  setYears(m);
  if (m.cc) F.cc.value = Math.round(m.cc);
  if (m.new_price) F.exs.value = Math.round(m.new_price);
  F.kwh.value = m.kwh_est || "";
  if (KIND === "car" && m.engine === "ml") {
    if (m.power) F.power.value = Math.round(m.power);
    if (m.seats) F.seats.value = m.seats;
    if (m.kaggle_fuel) F.fuel.value = m.kaggle_fuel;
    if (m.transmission) F.trans.value = m.transmission;
  }
  if (KIND === "car" && m.engine === "market") {
    const f = { petrol: "Petrol", diesel: "Diesel", cng: "CNG", electric: "Electric", hybrid: "Hybrid" }[m.fuel];
    if (f) F.fuel.value = f;
    if (m.transmission_mode && m.transmission_mode !== "NA") F.trans.value = m.transmission_mode;
  }
  VARIANTS = [];
  F.variant.innerHTML = `<option value="">Not sure</option>`;
  updateFields();
  if (m.engine !== "market") return;
  const my = ++variantReq;
  const q = new URLSearchParams({ type: KIND, brand: F.brand.value, model: F.model.value });
  try {
    const vs = await fetch("/api/variants?" + q).then((r) => r.json());
    if (my !== variantReq) return;
    VARIANTS = vs;
    F.variant.innerHTML = `<option value="">Not sure</option>` + vs.map((v) =>
      `<option value="${esc(v.id)}">${esc(v.name)} (${v.years[0] === v.years[1] ? v.years[0] : v.years[0] + "–" + v.years[1]})${v.new_price ? " · " + inr(v.new_price) + " new" : ""}</option>`).join("");
  } catch { /* variants are optional */ }
  updateFields();
}
function updateFields() {
  const m = current() || {};
  const v = currentVariant();
  const flags = new Set([KIND, m.engine, m.fuel === "electric" ? "ev" : "ice"]);
  if (KIND === "car" && (m.engine !== "market" || !v)) flags.add("carfree");
  $$("[data-for]").forEach((f) => f.classList.toggle("hide", !f.dataset.for.split(" ").every((t) => flags.has(t))));
  $("#advEmpty").hidden = $$("#adv .field").some((f) => !f.classList.contains("hide"));
  const yrs = m.years ? ` · ads from ${m.years[0]}–${m.years[1]}` : "";
  $("#engineHint").innerHTML = !m.engine ? "" : m.engine === "market"
    ? `Priced live by <b>XGBoost</b> · learned from ${m.n.toLocaleString("en-IN")} real 2026 ads of this model${yrs}`
    : m.engine === "ml" ? `Priced by <b>XGBoost</b> on older 2020-21 listings · no 2026 ads of this model`
    : `Few ads exist · <b>estimated</b> from the official price ${inr(m.new_price)}`;
}

function setYears(m) {
  const now = META.now, cur = +F.year.value || now - 4;
  let [y0, y1] = (m && m.years) || [now - 15, now];
  y0 = Math.max(1995, Math.min(y0, now)); y1 = Math.min(now, Math.max(y1, y0));
  F.year.innerHTML = Array.from({ length: y1 - y0 + 1 }, (_, i) => `<option>${y1 - i}</option>`).join("");
  F.year.value = String(Math.max(y0, Math.min(y1, cur)));
}
function fillCities() {
  const cs = META.cities[F.state.value] || [], prev = F.city.value;
  const sorted = [...cs].sort((a, b) => a.localeCompare(b));
  F.city.innerHTML = sorted.map((c) => `<option>${esc(c)}</option>`).join("") + `<option value="">Other town</option>`;
  F.city.value = cs.includes(prev) ? prev : (cs[0] || "");
}

seg($("#typeSeg"), (v) => {
  KIND = v === "2w" ? "2w" : "car";
  $("#fuelSeg").style.display = KIND === "2w" ? "" : "none";
  if (KIND === "car" && +F.km.value < 20000) F.km.value = 40000;
  if (KIND === "2w" && +F.km.value > 60000) F.km.value = 15000;
  fillBrands(KIND === "car" ? "Maruti Suzuki" : undefined);
  if (lastResult) schedule();
});
seg($("#fuelSeg"), (v) => { FUELF = v; fillBrands(v === "electric" ? "Ather" : undefined); if (lastResult) schedule(); });

F.brand.addEventListener("change", () => { fillModels(); if (lastResult) schedule(); });
F.model.addEventListener("change", () => { onModel(); if (lastResult) schedule(); });
F.variant.addEventListener("change", () => { updateFields(); if (lastResult) schedule(); });
F.state.addEventListener("change", () => { fillCities(); if (lastResult) schedule(); });
["year", "km", "owner", "seller", "city", "trans", "fuel", "cc", "power", "seats", "exs", "kwh"].forEach((k) =>
  F[k].addEventListener("input", () => lastResult && schedule()));

let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(() => run(false), 220); }
$("#form").addEventListener("submit", (e) => { e.preventDefault(); run(true); });

function readKm() {
  const k = Math.round(+F.km.value);
  if (!Number.isFinite(k) || k < 0) return 0;
  return Math.min(k, 999999);
}

let inflight = null, lastRtt = 0;
async function run(scrollTo) {
  const m = current();
  if (!m) return;
  const v = currentVariant();
  const body = { type: KIND, brand: F.brand.value, model: F.model.value, year: +F.year.value, km: readKm(),
    owners: +F.owner.value, seller: F.seller.value, state: F.state.value, city: F.city.value || null };
  if (v) body.variant = v.id;
  if (KIND === "car" && (m.engine !== "market" || !v)) Object.assign(body, { fuel: F.fuel.value, transmission: F.trans.value });
  if (m.engine === "ml") body.cc = +F.cc.value || null;
  if (KIND === "car" && m.engine === "ml") Object.assign(body, { power: +F.power.value || null, seats: +F.seats.value || null });
  if (m.engine === "dep" && +F.exs.value >= 10000) body.new_price = +F.exs.value;
  if (m.fuel === "electric" && +F.kwh.value > 0) body.kwh = Math.min(50, +F.kwh.value);

  const btn = $("#go"); btn.classList.add("loading");
  inflight?.abort(); inflight = new AbortController();
  const t0 = performance.now();
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: inflight.signal });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(typeof e.detail === "string" ? e.detail : "please check the details");
    }
    const data = await r.json();
    lastRtt = performance.now() - t0;
    show(data, !lastResult, m);
    lastResult = data;
    $("#formNote").textContent = "Prices in 2026 rupees. Asking prices on listing sites, not final sale prices.";
  } catch (err) {
    if (err.name !== "AbortError") $("#formNote").textContent = "Couldn't get a price: " + err.message;
  } finally { btn.classList.remove("loading"); }
  if (scrollTo && innerWidth < 920) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------------- result ---------------- */
let shownPrice = 0;
function show(d, first, m) {
  $("#empty").hidden = true; const res = $("#res"); res.hidden = false;
  if (first) { res.style.animation = "none"; res.offsetHeight; res.style.animation = ""; }
  const isMl = d.engine.kind === "ml", isEv = m.fuel === "electric";
  const evMl = isMl && isEv;
  const badge = $("#engineBadge");
  badge.className = "engine-badge" + (isMl ? "" : " dep");
  badge.innerHTML = `<i></i>${isMl ? `Predicted live by ${esc(d.engine.name)}` : `Estimate from official price`}`;
  badge.title = `Server compute ${d.engine.ms} ms · round trip ${Math.round(lastRtt)} ms`;
  $("#resTitle").textContent = d.title;
  $("#warn").hidden = !d.warning; $("#warn").textContent = d.warning || "";
  const pe = $("#price"), from = shownPrice, to = d.price, t0 = performance.now();
  const step = (t) => { const p = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - p, 4); pe.textContent = inr(from + (to - from) * e); if (p < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step); shownPrice = to;
  $("#range").textContent = `Fair range ${inr(d.low)} – ${inr(d.high)}`;

  const minis = [];
  if (isMl) {
    minis.push(["Age", d.age + " yr"]);
    minis.push(d.similar ? [`${d.similar.scope === "variant" ? "Same variant" : "All variants"}, all-India 2026 ads (${d.similar.count})`, inr(d.similar.median)] : ["Typical listing", inr(d.base_price)]);
    minis.push(evMl && d.battery_health ? ["Battery health", "~" + Math.round(d.battery_health) + "%"]
      : d.value_kept ? ["Value kept", d.value_kept + "%"] : d.new_price ? ["New today", inr(d.new_price)] : ["Typical listing", inr(d.base_price)]);
  } else {
    minis.push(["Value kept", d.value_kept + "%"]);
    minis.push(d.similar ? [`All-India 2026 ads (${d.similar.count})`, inr(d.similar.median)]
      : isEv ? ["Battery health", "~" + Math.round(d.battery_health) + "%"] : ["Age", d.age + " yr"]);
    minis.push([KIND === "car" ? "New (ex-showroom)" : "New on-road", inr(d.new_price)]);
  }
  $("#miniStats").innerHTML = minis.map(([k, v]) => `<div class="mini"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#summary").textContent = d.summary + (d.brand_note ? " " + d.brand_note : "");

  $("#whyIntro").innerHTML = isMl
    ? `We start from a <b>typical ad</b> (${inr(d.base_price)}) and show how much each detail you entered pushes the price up (green) or down (red). These come straight from the model's own calculation (Tree SHAP). They compare with an average ${KIND === "car" ? "car" : "two-wheeler"}, so a newer-than-average vehicle shows a positive age effect.`
    : `We start from the price when new, then apply each factor in turn.`;
  const maxAbs = Math.max(10, ...d.factors.map((f) => Math.abs(f.effect || 0)));
  $("#factors").innerHTML = d.factors.map((f, i) => {
    const e = f.effect, has = e !== null && e !== undefined, cls = !has ? "" : e >= 0 ? "up" : "down";
    const w = has ? Math.min(50, (Math.abs(e) / maxAbs) * 50) : 0;
    return `<div class="factor" style="animation-delay:${i * 55}ms">
      <div class="name">${esc(f.feature)}<small>${esc(f.value)}</small></div>
      <div class="eff ${cls}">${has ? (e > 0 ? "+" : "") + e + "%" : ""}</div>
      ${has ? `<div class="track"><i class="${cls}" data-w="${w}"></i></div>` : ""}
      <div class="note">${esc(f.note)}</div></div>`;
  }).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#factors .track i").forEach((i) => (i.style.width = i.dataset.w + "%"))));
  $("#engineBox").innerHTML = `<b>How this was priced.</b> ${esc(d.engine.detail)}`;

  const tn = $("#taxNote");
  if (d.road_tax) {
    const t = d.road_tax, pct = (x) => (x * 100).toFixed(1).replace(".0", "") + "%";
    const gap = t.rate - t.cheapest_rate;
    tn.hidden = false;
    tn.innerHTML = `<b>Road tax side note.</b> A new one (~${inr(t.new_price_est)}) pays <b>${pct(t.rate)}</b> road tax in ${esc(t.state)}` +
      (gap > 0 ? ` vs ${pct(t.cheapest_rate)} in ${esc(t.cheapest_state)}. Our Classic 350 check found ~₹1,430 more on the used price per extra 1% of tax, so roughly ${inr(gap * t.new_price_est)} of this price may reflect the higher tax.` : `, the lowest of the states we tracked.`) +
      ` <span class="fine">Source: ${esc(t.source)}.</span>`;
  } else tn.hidden = true;
  $("#method").textContent = d.method + " " + (d.range_note || "");

  const money = (y) => y >= 1e5 ? `₹${(y / 1e5).toFixed(1)}L` : `₹${Math.round(y / 1e3)}k`;
  if (isEv && !isMl) {
    const col = "var(--ev)";
    $("#curveIntro").textContent = "Share of the new on-road price kept, by age — this EV vs a petrol two-wheeler of the same type.";
    lineChart($("#curveChart"), [
      { points: d.curve.map((c) => ({ x: c.age, y: c.petrol })), color: "var(--petrol)", dash: true },
      { points: d.curve.map((c) => ({ x: c.age, y: c.kept })), color: col },
    ], { xLabel: (x) => `${x}y`, yLabel: (y) => `${Math.round(y)}%`, yMin: 20, yMax: 100,
      marker: { x: Math.min(10, d.age), y: d.curve[Math.min(10, d.age)].kept, label: d.curve[Math.min(10, d.age)].kept + "%" } });
    $("#curveLegend").innerHTML = `<span class="lg" style="background:${col}"></span> ${esc(F.brand.value)} <span class="lg lg-p"></span> Petrol (same type)`;
  } else {
    $("#curveIntro").textContent = "Predicted listing price of this exact vehicle at every age, holding km and owners fixed.";
    const pts = d.curve.map((c) => ({ x: c.age, y: c.price }));
    const ai = Math.min(pts.length - 1, d.age);
    lineChart($("#curveChart"), [{ points: pts, color: "var(--accent)" }],
      { xLabel: (x) => `${x}y`, yLabel: money, marker: { x: ai, y: pts[ai].y, label: inr(d.price) } });
    $("#curveLegend").innerHTML = `<span class="lg lg-p"></span> Predicted price · dot = your vehicle`;
  }
}

/* result tabs */
(function tabs() {
  const root = $("#resTabs"), pill = $(".tab-pill", root), btns = $$(".tab", root);
  btns.forEach((b, i) => b.addEventListener("click", () => {
    btns.forEach((x) => x.classList.toggle("active", x === b));
    pill.style.transform = `translateX(${i * 100}%)`;
    $$(".pane").forEach((p) => p.classList.toggle("active", p.dataset.p === b.dataset.t));
    if (b.dataset.t === "curve" && lastResult) { const s = $("#curveChart"); s.classList.remove("in"); requestAnimationFrame(() => requestAnimationFrame(() => s.classList.add("in"))); }
  }));
})();

/* ---------------- boot ---------------- */
(async function boot() {
  const [m, x] = await Promise.all([fetch("/api/meta").then((r) => r.json()), fetch("/api/metrics").then((r) => r.json())]);
  META = m; METRICS = x;
  F.state.innerHTML = META.states.map((s) => `<option>${esc(s)}</option>`).join("");
  F.state.value = "Karnataka"; fillCities();
  const sm = $("#statModels"); sm.dataset.count = META.counts["2w"] + META.counts.car;
  if (sm.classList.contains("in")) countUp(sm);
  renderMetrics(METRICS.bike_market ? "bike_market" : "bike");
  F.year.value = String(META.now - 4);
  fillBrands("Royal Enfield"); F.model.value = "Classic 350"; onModel();
})();
