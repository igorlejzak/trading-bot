// Prosty schodkowy wykres liniowy w SVG z crosshairem i tooltipem. Wspólny dla portfela i backtestu.
// series: [{name, color, pts: [[ms, value, extra?], ...]}], opts: {base, fmt, title}
function lineChart(el, series, opts = {}) {
  const fmt = opts.fmt || (v => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  series = series.filter(s => s.pts.length);
  if (!series.length) { el.innerHTML = '<div class="fallback">No data yet.</div>'; return; }
  el.innerHTML = '<svg></svg><div class="tip"></div>';
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip");
  const W = el.clientWidth, H = el.clientHeight, multi = series.length > 1;
  const m = { l: 60, r: multi ? 132 : 20, t: 14, b: 26 };
  const all = series.flatMap(s => s.pts);
  const x0 = Math.min(...all.map(p => p[0])), x1 = Math.max(...all.map(p => p[0]));
  let y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  if (opts.base != null) { y0 = Math.min(y0, opts.base); y1 = Math.max(y1, opts.base); }
  const pad = (y1 - y0) * 0.08 || 5; y0 -= pad; y1 += pad;
  const X = t => m.l + (t - x0) / ((x1 - x0) || 1) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const nice = raw => { const p = 10 ** Math.floor(Math.log10(raw)), n = raw / p; return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p; };
  const step = nice((y1 - y0) / 5);
  let grid = "";
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step)
    grid += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" ${opts.base != null && Math.abs(v - opts.base) < 1e-9 ? "" : 'stroke-dasharray="2 4"'}/>
             <text class="axis" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${Number(v).toLocaleString("en-US", { maximumFractionDigits: step < 1 ? 2 : 0 })}</text>`;
  // oś czasu: dni przy krótkim zakresie, miesiące albo kwartały przy dłuższym
  const ticks = [], spanD = (x1 - x0) / 864e5, d = new Date(x0);
  d.setUTCHours(0, 0, 0, 0);
  if (spanD < 45) {
    const ev = spanD > 20 ? 7 : spanD > 9 ? 2 : 1;
    for (d.setUTCDate(d.getUTCDate() + 1); d.getTime() <= x1; d.setUTCDate(d.getUTCDate() + ev))
      ticks.push([d.getTime(), d.toLocaleString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })]);
  } else {
    const ev = spanD > 500 ? 6 : spanD > 200 ? 2 : 1;
    d.setUTCDate(1); d.setUTCMonth(d.getUTCMonth() + 1);
    if (ev > 1) while (d.getUTCMonth() % ev) d.setUTCMonth(d.getUTCMonth() + 1);
    for (; d.getTime() <= x1; d.setUTCMonth(d.getUTCMonth() + ev))
      ticks.push([d.getTime(), d.toLocaleString("en-US", { month: "short", timeZone: "UTC" }) + (d.getUTCMonth() === 0 || ev > 1 ? " " + String(d.getUTCFullYear()).slice(2) : "")]);
  }
  const xt = ticks.map(([t, l]) => `<text class="axis" x="${X(t)}" y="${H - 7}" text-anchor="middle">${l}</text>`).join("");
  // schodki: wartość zmienia się tylko w chwili zamknięcia transakcji
  const path = s => s.pts.map((p, i) => i ? `H${X(p[0]).toFixed(1)}V${Y(p[1]).toFixed(1)}` : `M${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const last = s => s.pts[s.pts.length - 1][1];
  const ends = series.map(s => ({ s, y: Y(last(s)) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = grid + xt +
    series.map(s => `<path d="${path(s)}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`).join("") +
    (multi ? ends.map(({ s, y }) => `<text class="lbl" x="${W - m.r + 8}" y="${y + 4}">${s.name.length > 17 ? s.name.slice(0, 16) + "…" : s.name}</text>`).join("") : "") +
    `<line class="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" visibility="hidden"/><g class="dots"></g>
     <rect class="hit" x="${m.l}" y="${m.t}" width="${Math.max(0, W - m.l - m.r)}" height="${Math.max(0, H - m.t - m.b)}" fill="transparent"/>`;
  const hit = svg.querySelector(".hit"), xh = svg.querySelector(".xh"), dots = svg.querySelector(".dots");
  const valAt = (s, t) => { let v = null; for (const p of s.pts) { if (p[0] > t) break; v = p[1]; } return v; };
  hit.onmousemove = e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    const t = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    xh.setAttribute("x1", px); xh.setAttribute("x2", px); xh.setAttribute("visibility", "visible");
    const vals = series.map(s => ({ s, v: valAt(s, t) }));
    dots.innerHTML = vals.filter(o => o.v != null).map(o => `<circle cx="${px}" cy="${Y(o.v)}" r="4" fill="${o.s.color}" stroke="var(--panel)" stroke-width="2"/>`).join("");
    tip.innerHTML = `<div class="t">${new Date(t).toISOString().slice(0, 16).replace("T", " ")} UTC</div>` +
      vals.map(o => `<div class="r"><span><span class="swatch" style="background:${o.s.color}"></span>${o.s.name}</span><b>${o.v == null ? "not started" : fmt(o.v)}</b></div>`).join("");
    tip.style.display = "block";
    const tx = px * r.width / W;
    tip.style.left = (tx > r.width / 2 ? tx - tip.offsetWidth - 12 : tx + 12) + "px";
    tip.style.top = "10px";
  };
  hit.onmouseleave = () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); dots.innerHTML = ""; };
}
