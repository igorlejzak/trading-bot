// Backtest w przeglądarce. Ta sama logika co research/ (livemode.prepare_state, grid.build_signals,
// grid_avg.simulate_avg), ale na świeczkach 15m zamiast 1m:
// - wskaźniki ze ZAMKNIĘTYCH świeczek + aktualna cena (jak bot live), cena wejścia = pierwsza cena w świeczce,
//   przy której warunek jest spełniony (bisekcja między open a high/low),
// - kolejne wejścia to zlecenia limit co X% od pierwszego wejścia, TP od średniej, SL dopiero po ostatnim wejściu,
// - gdy w jednej świeczce dotknięte są TP i SL, liczymy SL (ostrożnie). Na 1m research rozstrzyga to dokładniej.
(function (root) {
  const BB_WINDOW = 20, BB_DEV = 2, RSI_WINDOW = 14, SMOOTH = 3, ST_HI = 80, ST_LO = 20;

  function rollingSum(a, w) {
    const out = new Float64Array(a.length).fill(NaN);
    let s = 0, nn = 0;
    for (let i = 0; i < a.length; i++) {
      if (!isNaN(a[i])) { s += a[i]; nn++; }
      if (i >= w) { if (!isNaN(a[i - w])) { s -= a[i - w]; nn--; } }
      if (i >= w - 1 && nn === w) out[i] = s;
    }
    return out;
  }
  function rollingMinMax(a, w, isMax) {
    const out = new Float64Array(a.length).fill(NaN);
    for (let i = w - 1; i < a.length; i++) {
      let v = isMax ? -Infinity : Infinity, ok = true;
      for (let j = i - w + 1; j <= i; j++) { if (isNaN(a[j])) { ok = false; break; } v = isMax ? Math.max(v, a[j]) : Math.min(v, a[j]); }
      if (ok) out[i] = v;
    }
    return out;
  }

  // Stan z zamkniętych świeczek: wartości dla świeczki t pochodzą ze świeczki t-1 (jak prepare_state).
  function prepareState(c) {
    const n = c.length, a = 1 / RSI_WINDOW;
    const ag = new Float64Array(n).fill(NaN), al = new Float64Array(n).fill(NaN), rsi = new Float64Array(n).fill(NaN);
    let g = NaN, l = NaN, cnt = 0;
    for (let i = 1; i < n; i++) {
      const d = c[i] - c[i - 1], up = Math.max(d, 0), dn = Math.max(-d, 0);
      if (cnt === 0) { g = up; l = dn; } else { g = (1 - a) * g + a * up; l = (1 - a) * l + a * dn; }
      cnt++;
      if (cnt >= RSI_WINDOW) { ag[i] = g; al[i] = l; rsi[i] = 100 - 100 / (1 + g / l); }
    }
    const rmin = rollingMinMax(rsi, RSI_WINDOW, false), rmax = rollingMinMax(rsi, RSI_WINDOW, true);
    const srsi = new Float64Array(n);
    for (let i = 0; i < n; i++) srsi[i] = (rsi[i] - rmin[i]) / (rmax[i] - rmin[i]);
    const ksum = rollingSum(srsi, SMOOTH), k = ksum.map(v => v / SMOOTH);
    const sq = c.map(v => v * v);
    const s_sum = rollingSum(c, BB_WINDOW - 1), s_sq = rollingSum(sq, BB_WINDOW - 1);
    const rmin13 = rollingMinMax(rsi, RSI_WINDOW - 1, false), rmax13 = rollingMinMax(rsi, RSI_WINDOW - 1, true);
    const S = { s_sum: [], s_sq: [], last: [], ag: [], al: [], rmin: [], rmax: [], srsi1: [], srsi2: [], k1: [], k2: [] };
    for (const key in S) S[key] = new Float64Array(n).fill(NaN);
    for (let t = 1; t < n; t++) {
      S.s_sum[t] = s_sum[t - 1]; S.s_sq[t] = s_sq[t - 1]; S.last[t] = c[t - 1];
      S.ag[t] = ag[t - 1]; S.al[t] = al[t - 1]; S.rmin[t] = rmin13[t - 1]; S.rmax[t] = rmax13[t - 1];
      S.srsi1[t] = srsi[t - 1]; S.k1[t] = k[t - 1];
      if (t >= 2) { S.srsi2[t] = srsi[t - 2]; S.k2[t] = k[t - 2]; }
    }
    return S;
  }

  function ind(S, t, p) {
    const mean = (S.s_sum[t] + p) / BB_WINDOW;
    const std = Math.sqrt(Math.max((S.s_sq[t] + p * p) / BB_WINDOW - mean * mean, 0));
    const a = 1 / RSI_WINDOW, d = p - S.last[t];
    const ag = (1 - a) * S.ag[t] + a * Math.max(d, 0), al = (1 - a) * S.al[t] + a * Math.max(-d, 0);
    const rsi = 100 - 100 / (1 + ag / al);
    const lo = Math.min(S.rmin[t], rsi), hi = Math.max(S.rmax[t], rsi);
    const srsi = (rsi - lo) / (hi - lo);
    const k = (S.srsi2[t] + S.srsi1[t] + srsi) / SMOOTH, dd = (S.k2[t] + S.k1[t] + k) / SMOOTH;
    return [mean + BB_DEV * std, mean - BB_DEV * std, k * 100, dd * 100];
  }
  // Math.min/max z NaN dają NaN, a porównania z NaN są fałszywe - tak samo jak w numpy.
  function condShort(S, t, p, bb, stoch) {
    const [h, , k, d] = ind(S, t, p);
    return p > h * (1 + bb) && (!stoch || (k > ST_HI && d > ST_HI));
  }
  function condLong(S, t, p, bb, stoch) {
    const [, l, k, d] = ind(S, t, p);
    return p < l * (1 - bb) && (!stoch || (k < ST_LO && d < ST_LO));
  }
  function firstPrice(cond, start, end) {
    if (cond(start)) return start;
    for (let i = 0; i < 30; i++) { const mid = (start + end) / 2; if (cond(mid)) end = mid; else start = mid; }
    return end;
  }

  function signals(D, S, bb, stoch) {
    const idx = [], side = [], price = [];
    for (let t = 1; t < D.c.length; t++) {
      if (condShort(S, t, D.h[t], bb, stoch)) {
        idx.push(t); side.push(-1); price.push(firstPrice(p => condShort(S, t, p, bb, stoch), D.o[t], D.h[t]));
      } else if (condLong(S, t, D.l[t], bb, stoch)) {
        idx.push(t); side.push(1); price.push(firstPrice(p => condLong(S, t, p, bb, stoch), D.o[t], D.l[t]));
      }
    }
    return { idx, side, price };
  }

  function firstHit(hi, lo, start, up, down) {
    for (let j = start; j < hi.length; j++) if (hi[j] >= up || lo[j] <= down) return j;
    return -1;
  }

  // simulate_avg: zwraca transakcje {i, j, side, gross, filled, fills, win}
  function simulate(hi, lo, sig, tp, sl, weights, dists) {
    const out = [], nW = weights.length;
    let s = 0;
    while (s < sig.idx.length) {
      const i = sig.idx[s], side = sig.side[s], p1 = sig.price[s];
      let filled = weights[0], qty = weights[0] / p1, lastP = p1, k = 1, j = i, allowTp = false, exitP = null, win = false;
      for (;;) {
        const avg = filled / qty;
        if (side === 1) {
          const tpL = avg * (1 + tp);
          if (k < nW) {
            const nxt = p1 * (1 - dists[k - 1]);
            if (lo[j] <= nxt) { filled += weights[k]; qty += weights[k] / nxt; lastP = nxt; k++; allowTp = false; continue; }
            if (allowTp && hi[j] >= tpL) { exitP = tpL; win = true; break; }
            j = firstHit(hi, lo, j + 1, tpL, nxt);
          } else {
            const slL = lastP * (1 - sl);
            if (lo[j] <= slL) { exitP = slL; win = false; break; }
            if (allowTp && hi[j] >= tpL) { exitP = tpL; win = true; break; }
            j = firstHit(hi, lo, j + 1, tpL, slL);
          }
        } else {
          const tpL = avg * (1 - tp);
          if (k < nW) {
            const nxt = p1 * (1 + dists[k - 1]);
            if (hi[j] >= nxt) { filled += weights[k]; qty += weights[k] / nxt; lastP = nxt; k++; allowTp = false; continue; }
            if (allowTp && lo[j] <= tpL) { exitP = tpL; win = true; break; }
            j = firstHit(hi, lo, j + 1, nxt, tpL);
          } else {
            const slL = lastP * (1 + sl);
            if (hi[j] >= slL) { exitP = slL; win = false; break; }
            if (allowTp && lo[j] <= tpL) { exitP = tpL; win = true; break; }
            j = firstHit(hi, lo, j + 1, slL, tpL);
          }
        }
        if (j === -1) break;            // pozycja otwarta do końca danych - pomijamy
        allowTp = true;
      }
      if (exitP === null) break;
      const gross = side === 1 ? qty * exitP - filled : filled - qty * exitP;
      out.push({ i, j, side, entry: p1, avg: filled / qty, exit: exitP, gross, filled, fills: k, win });
      while (s < sig.idx.length && sig.idx[s] <= j) s++;   // searchsorted(..., j, side="right")
    }
    return out;
  }

  function summarize(trades, D, fee, capital) {
    const net = trades.map(t => (t.gross - 2 * fee * t.filled) * capital);
    let eq = 0, peak = 0, maxDd = 0;
    const curve = [[D.t0 + (trades.length ? trades[0].i : 0) * D.step, 0]];
    trades.forEach((t, n) => {
      eq += net[n]; peak = Math.max(peak, eq); maxDd = Math.min(maxDd, eq - peak);
      curve.push([D.t0 + t.j * D.step, eq]);
    });
    const wins = net.filter(v => v > 0), losses = net.filter(v => v <= 0);
    const years = {};
    trades.forEach((t, n) => {
      const y = new Date(D.t0 + t.j * D.step).getUTCFullYear();
      (years[y] ||= { pnl: 0, trades: 0 }); years[y].pnl += net[n]; years[y].trades++;
    });
    return {
      trades: trades.length, wins: wins.length, net: eq, maxDd,
      avgWin: wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : null,
      avgLoss: losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : null,
      avgFills: trades.length ? trades.reduce((a, t) => a + t.fills, 0) / trades.length : 0,
      curve, years, netList: net,
    };
  }

  // Losowe wejścia z TYMI SAMYMI zasadami wyjścia: punkt odniesienia "czy sygnał w ogóle coś daje".
  function randomBenchmark(D, tp, sl, weights, dists, fee, capital, from, to, seeds = 3) {
    const res = [];
    for (let seed = 1; seed <= seeds; seed++) {
      let x = seed * 2654435761 % 4294967296;
      const rnd = () => (x = (x * 1664525 + 1013904223) % 4294967296) / 4294967296;
      const idx = [], side = [], price = [];
      for (let t = from; t < to; t++) if (rnd() < 1 / 3) { idx.push(t); side.push(rnd() < .5 ? 1 : -1); price.push(D.o[t]); }
      const tr = simulate(D.h, D.l, { idx, side, price }, tp, sl, weights, dists);
      res.push(summarize(tr, D, fee, capital));
    }
    const avg = f => res.reduce((a, r) => a + f(r), 0) / res.length;
    return { perTrade: avg(r => r.trades ? r.net / r.trades : 0), winRate: avg(r => r.trades ? r.wins / r.trades : 0), trades: avg(r => r.trades) };
  }

  const stateCache = new WeakMap();
  // params: {bb, stoch, tp, sl, weights:[...], dists:[...], fee, capital, from, to (ms)}
  function backtest(D, P) {
    if (!stateCache.has(D)) stateCache.set(D, prepareState(D.c));
    const S = stateCache.get(D);
    const from = Math.max(1, Math.ceil(((P.from ?? D.t0) - D.t0) / D.step));
    const to = Math.min(D.c.length, Math.floor(((P.to ?? Infinity) - D.t0) / D.step) || D.c.length);
    const all = signals(D, S, P.bb, P.stoch);
    const sig = { idx: [], side: [], price: [] };
    all.idx.forEach((t, n) => { if (t >= from && t < to) { sig.idx.push(t); sig.side.push(all.side[n]); sig.price.push(all.price[n]); } });
    // dane do `to`, żeby pozycje zamykały się w wybranym okresie
    const hi = D.h.subarray ? D.h.subarray(0, to) : D.h.slice(0, to), lo = D.l.subarray ? D.l.subarray(0, to) : D.l.slice(0, to);
    const trades = simulate(hi, lo, sig, P.tp, P.sl, P.weights, P.dists);
    const sum = summarize(trades, D, P.fee, P.capital);
    sum.list = trades; sum.signals = sig.idx.length;
    sum.random = randomBenchmark({ ...D, h: hi, l: lo }, P.tp, P.sl, P.weights, P.dists, P.fee, P.capital, from, to);
    return sum;
  }

  const api = { backtest, prepareState, signals, simulate };
  if (typeof module !== "undefined") module.exports = api; else root.BT = api;
})(this);
