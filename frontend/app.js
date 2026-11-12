/* ResaleIQ frontend — vanilla JS, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
const NS = "http://www.w3.org/2000/svg";

let META = null, METRICS = null, MODE = "bike"; // bike | ev | car
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
    { points: mk(0.90, 0.069), color: "var(--petrol)" },
    { points: mk(0.91, 0.073), color: "var(--ev)" },
    { points: mk(0.83, 0.12), color: "var(--ola)" },
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
function renderMetrics(kind) {
  if (!METRICS) return;
  const m = METRICS[kind].models;
  const best = Object.entries(m).sort((a, b) => b[1].R2 - a[1].R2)[0][0];
  $("#metricTable tbody").innerHTML = Object.entries(m).map(([n, v]) => `<tr class="${n === best ? "best" : ""}">
    <td>${n}</td><td>${inr(v.MAE)}</td><td>${inr(v.RMSE)}</td><td>${v.R2.toFixed(3)}</td>
    <td><div class="r2bar"><i data-w="${(v.R2 * 100).toFixed(1)}"></i></div></td></tr>`).join("");
  requestAnimationFrame(() => $$("#metricTable .r2bar i").forEach((i) => (i.style.width = i.dataset.w + "%")));
}
seg($("#metricSeg"), renderMetrics);

/* ---------------- form ---------------- */
const F = { brand: $("#brand"), model: $("#model"), year: $("#year"), km: $("#km"), owner: $("#owner"), state: $("#state"),
  trans: $("#trans"), fuel: $("#fuel"), cc: $("#cc"), power: $("#power"), seats: $("#seats"), exs: $("#exs"), kwh: $("#kwh") };

const opts = (arr) => arr.map((o) => (typeof o === "string" ? `<option>${o}</option>` : `<option value="${o.v}">${o.t}</option>`)).join("");

function brandsFor() {
  if (MODE === "ev") return META.ev.brands;
  return META[MODE].brands;
}

function fillBrands() {
  const b = Object.keys(brandsFor());
  F.brand.innerHTML = opts(b);
  fillModels();
}
function fillModels() {
  const list = brandsFor()[F.brand.value] || [];
  F.model.innerHTML = MODE === "ev"
    ? opts(list.map((m) => ({ v: m.model, t: m.model })))
    : opts(list.map((m) => ({ v: m.key, t: m.model })));
  fillSpecs();
}
function fillSpecs() {
  const list = brandsFor()[F.brand.value] || [];
  const m = list.find((x) => (MODE === "ev" ? x.model : x.key) === F.model.value);
  if (!m) return;
  if (MODE === "ev") { F.exs.value = m.price; F.kwh.value = m.kwh; }
  else F.cc.value = Math.round(m.cc);
  if (MODE === "car") {
    F.power.value = Math.round(m.power); F.seats.value = m.seats;
    if (m.fuel) F.fuel.value = m.fuel;
    if (m.transmission) F.trans.value = m.transmission;
  }
}

function setMode(mode) {
  MODE = mode;
  $$("[data-for]").forEach((f) => f.classList.toggle("hide", !f.dataset.for.split(" ").includes(MODE)));
  const now = META.now, span = MODE === "ev" ? 8 : 20;
  const cur = +F.year.value || now - 3;
  F.year.innerHTML = opts(Array.from({ length: span + 1 }, (_, i) => String(now - i)));
  F.year.value = String(Math.max(now - span, Math.min(now, cur)));
  if (MODE === "car" && +F.km.value < 20000) F.km.value = 40000;
  fillBrands();
  if (lastResult) schedule();
}

const typeSeg = seg($("#typeSeg"), (v) => {
  $("#fuelSeg").style.display = v === "2w" ? "" : "none";
  setMode(v === "2w" ? ($("#fuelSeg .active").dataset.v) : "car");
});
seg($("#fuelSeg"), (v) => setMode(v));

F.brand.addEventListener("change", () => { fillModels(); if (lastResult) schedule(); });
F.model.addEventListener("change", () => { fillSpecs(); if (lastResult) schedule(); });
["year", "km", "owner", "state", "trans", "fuel", "cc", "power", "seats", "exs", "kwh"].forEach((k) =>
  F[k].addEventListener("input", () => lastResult && schedule()));

let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(() => run(false), 220); }

$("#form").addEventListener("submit", (e) => { e.preventDefault(); run(true); });

let inflight = null;
async function run(scrollTo) {
  const body = {
    type: MODE, brand: F.brand.value, model: F.model.value, year: +F.year.value,
    km: Math.max(0, +F.km.value || 0), owners: +F.owner.value,
  };
  if (MODE !== "car") body.state = F.state.value;
  if (MODE === "bike") body.cc = +F.cc.value || null;
  if (MODE === "car") Object.assign(body, { fuel: F.fuel.value, transmission: F.trans.value, cc: +F.cc.value || null, power: +F.power.value || null, seats: +F.seats.value || null });
  if (MODE === "ev") Object.assign(body, { ex_showroom: +F.exs.value || null, kwh: +F.kwh.value || null });

  const btn = $("#go"); btn.classList.add("loading");
  inflight?.abort(); inflight = new AbortController();
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: inflight.signal });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    show(data, !lastResult);
    lastResult = data;
    if (scrollTo && innerWidth < 920) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    if (err.name !== "AbortError") $("#formNote").textContent = "Couldn't get a price: " + err.message;
  } finally { btn.classList.remove("loading"); }
}

/* ---------------- result ---------------- */
let shownPrice = 0;
function show(d, first) {
  $("#empty").hidden = true; const res = $("#res"); res.hidden = false;
  if (first) { res.style.animation = "none"; res.offsetHeight; res.style.animation = ""; }
  $("#resTitle").textContent = d.title;
  const pe = $("#price"), from = shownPrice, to = d.price, t0 = performance.now();
  const step = (t) => { const p = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - p, 4); pe.textContent = inr(from + (to - from) * e); if (p < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step); shownPrice = to;
  $("#range").textContent = `Fair range ${inr(d.low)} – ${inr(d.high)}`;

  const minis = [];
  if (MODE === "ev") {
    minis.push(["Value kept", d.value_kept + "%"], ["Battery health", "~" + Math.round(d.battery_health) + "%"], ["New on-road", inr(d.new_price)]);
  } else {
    minis.push(["Age", d.age + " yr"]);
    minis.push(d.similar ? [`Similar ads (${d.similar.count})`, inr(d.similar.median)] : ["Similar ads", "—"]);
    minis.push(["Typical listing", inr(d.base_price)]);
  }
  $("#miniStats").innerHTML = minis.map(([k, v]) => `<div class="mini"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#summary").textContent = d.summary + (d.brand_note ? " " + d.brand_note : "");

  // why
  $("#whyIntro").innerHTML = MODE === "ev"
    ? `Start from the new on-road price, then apply each factor in turn. EV prices come from our same-variant depreciation study, not the Kaggle model (it has almost no EVs).`
    : `We start from the <b>typical listing</b> in the data (${inr(d.base_price)}) and add the push from each feature, measured with Tree SHAP. Green raises the price, red lowers it. Effects are relative to an average ${MODE === "car" ? "car" : "bike"}, so a newer-than-average vehicle shows a positive age effect.`;
  const maxAbs = Math.max(10, ...d.factors.map((f) => Math.abs(f.effect || 0)));
  $("#factors").innerHTML = d.factors.map((f, i) => {
    const e = f.effect, has = e !== null && e !== undefined, cls = !has ? "" : e >= 0 ? "up" : "down";
    const w = has ? Math.min(50, (Math.abs(e) / maxAbs) * 50) : 0;
    return `<div class="factor" style="animation-delay:${i * 55}ms">
      <div class="name">${f.feature}<small>${f.value}</small></div>
      <div class="eff ${cls}">${has ? (e > 0 ? "+" : "") + e + "%" : ""}</div>
      ${has ? `<div class="track"><i class="${cls}" data-w="${w}"></i></div>` : ""}
      <div class="note">${f.note}</div></div>`;
  }).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#factors .track i").forEach((i) => (i.style.width = i.dataset.w + "%"))));

  const tn = $("#taxNote");
  if (d.road_tax) {
    const t = d.road_tax, pct = (x) => (x * 100).toFixed(1).replace(".0", "") + "%";
    const gap = t.rate - t.cheapest_rate;
    tn.hidden = false;
    tn.innerHTML = `<b>Road tax side note.</b> A new one (~${inr(t.new_price_est)}) pays <b>${pct(t.rate)}</b> road tax in ${t.state}` +
      (gap > 0 ? ` vs ${pct(t.cheapest_rate)} in ${t.cheapest_state}. Our Classic 350 check found ~₹1,430 more on the used price per extra 1% of tax, so roughly ${inr(gap * t.new_price_est)} of this price may reflect the higher tax.` : `, the lowest of the states we tracked.`) +
      ` <span class="fine">Source: ${t.source}.</span>`;
  } else tn.hidden = true;
  $("#method").textContent = d.method + " " + (d.range_note || "");

  // curve
  if (MODE === "ev") {
    $("#curveIntro").textContent = "Share of the new on-road price kept, by age — this EV's brand curve vs a petrol two-wheeler of the same new price.";
    lineChart($("#curveChart"), [
      { points: d.curve.map((c) => ({ x: c.age, y: c.petrol })), color: "var(--petrol)", dash: true },
      { points: d.curve.map((c) => ({ x: c.age, y: c.ev })), color: F.brand.value === "Ola" ? "var(--ola)" : "var(--ev)" },
    ], { xLabel: (x) => `${x}y`, yLabel: (y) => `${Math.round(y)}%`, yMin: 30, yMax: 100,
      marker: { x: Math.min(8, d.age), y: d.value_kept, label: d.value_kept + "%" } });
    $("#curveLegend").innerHTML = `<span class="lg" style="background:${F.brand.value === "Ola" ? "var(--ola)" : "var(--ev)"}"></span> ${F.brand.value} <span class="lg lg-p"></span> Petrol (same new price)`;
  } else {
    $("#curveIntro").textContent = "Predicted listing price of this exact vehicle at every age, holding km and owners fixed.";
    const pts = d.curve.map((c) => ({ x: c.age, y: c.price }));
    lineChart($("#curveChart"), [{ points: pts, color: "var(--accent)" }],
      { xLabel: (x) => `${x}y`, yLabel: (y) => y >= 1e5 ? `₹${(y / 1e5).toFixed(1)}L` : `₹${Math.round(y / 1e3)}k`,
        marker: { x: Math.min(15, d.age), y: d.price, label: inr(d.price) } });
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
  F.state.innerHTML = opts(META.bike.states);
  F.state.value = "Karnataka";
  const sc = $("#statCars"); sc.dataset.count = METRICS.car.cleaning.rows_loaded;
  if (sc.classList.contains("in")) countUp(sc);
  renderMetrics("bike");
  setMode("bike");
  // sensible default
  F.brand.value = "Royal Enfield"; fillModels();
  const cls = [...F.model.options].find((o) => o.textContent === "Classic"); if (cls) { F.model.value = cls.value; fillSpecs(); }
  F.year.value = String(META.now - 4);
})();
