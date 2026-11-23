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

  // dir: "both" | "long" | "short". Przy "both" short ma pierwszeństwo, jak w bocie.
  function signals(D, S, bb, stoch, dir = "both") {
    const idx = [], side = [], price = [], doS = dir !== "long", doL = dir !== "short";
    for (let t = 1; t < D.c.length; t++) {
      if (doS && condShort(S, t, D.h[t], bb, stoch)) {
        idx.push(t); side.push(-1); price.push(firstPrice(p => condShort(S, t, p, bb, stoch), D.o[t], D.h[t]));
      } else if (doL && condLong(S, t, D.l[t], bb, stoch)) {
        idx.push(t); side.push(1); price.push(firstPrice(p => condLong(S, t, p, bb, stoch), D.o[t], D.l[t]));
      }
    }
    return { idx, side, price };
  }

  function firstHit(hi, lo, start, up, down) {
    for (let j = start; j < hi.length; j++) if (hi[j] >= up || lo[j] <= down) return j;
    return -1;
  }

  // Jedna pozycja od sygnału (i, side, p1) do wyjścia. null, gdy pozycja jest otwarta do końca danych.
  function simulateOne(hi, lo, i, side, p1, tp, sl, weights, dists) {
    const nW = weights.length;
    let filled = weights[0], qty = weights[0] / p1, lastP = p1, k = 1, j = i, allowTp = false;
    for (;;) {
      const avg = filled / qty;
      let exitP = null, win = false;
      if (side === 1) {
        const tpL = avg * (1 + tp);
        if (k < nW) {
          const nxt = p1 * (1 - dists[k - 1]);
          if (lo[j] <= nxt) { filled += weights[k]; qty += weights[k] / nxt; lastP = nxt; k++; allowTp = false; continue; }
          if (allowTp && hi[j] >= tpL) { exitP = tpL; win = true; }
          else j = firstHit(hi, lo, j + 1, tpL, nxt);
        } else {
          const slL = lastP * (1 - sl);
          if (lo[j] <= slL) exitP = slL;
          else if (allowTp && hi[j] >= tpL) { exitP = tpL; win = true; }
          else j = firstHit(hi, lo, j + 1, tpL, slL);
        }
      } else {
        const tpL = avg * (1 - tp);
        if (k < nW) {
          const nxt = p1 * (1 + dists[k - 1]);
          if (hi[j] >= nxt) { filled += weights[k]; qty += weights[k] / nxt; lastP = nxt; k++; allowTp = false; continue; }
          if (allowTp && lo[j] <= tpL) { exitP = tpL; win = true; }
          else j = firstHit(hi, lo, j + 1, nxt, tpL);
        } else {
          const slL = lastP * (1 + sl);
          if (hi[j] >= slL) exitP = slL;
          else if (allowTp && lo[j] <= tpL) { exitP = tpL; win = true; }
          else j = firstHit(hi, lo, j + 1, slL, tpL);
        }
      }
      if (exitP !== null) {
        const gross = side === 1 ? qty * exitP - filled : filled - qty * exitP;
        return { i, j, side, entry: p1, avg: filled / qty, exit: exitP, gross, filled, fills: k, win };
      }
      if (j === -1) return null;          // pozycja otwarta do końca danych
      allowTp = true;
    }
  }

  // simulate_avg dla jednego coina: jedna pozycja naraz, następny sygnał dopiero po wyjściu.
  function simulate(hi, lo, sig, tp, sl, weights, dists) {
    const out = [];
    let s = 0;
    while (s < sig.idx.length) {
      const t = simulateOne(hi, lo, sig.idx[s], sig.side[s], sig.price[s], tp, sl, weights, dists);
      if (!t) break;
      out.push(t);
      while (s < sig.idx.length && sig.idx[s] <= t.j) s++;   // searchsorted(..., j, side="right")
    }
    return out;
  }

  const stateCache = new WeakMap();
  const sigCache = new WeakMap();
  function coinSignals(D, bb, stoch, dir) {
    if (!stateCache.has(D)) stateCache.set(D, prepareState(D.c));
    if (!sigCache.has(D)) sigCache.set(D, {});
    const key = `${bb}|${stoch}|${dir}`, c = sigCache.get(D);
    return c[key] ||= signals(D, stateCache.get(D), bb, stoch, dir);
  }
  const idxAt = (D, ms) => Math.ceil((ms - D.t0) / D.step);

  // Portfel: wiele coinów, wspólny kapitał. Każdy coin ma najwyżej jedną pozycję naraz.
  // Przy pierwszym wejściu rezerwujemy całą kwotę pozycji (suma wszystkich wejść), żeby dokładki zawsze
  // miały pokrycie. Sygnał, na który nie ma wolnego kapitału, jest pomijany (liczymy takie przypadki).
  // P: {bb, stoch, dir, tp, sl, amounts:[USD na wejście], dists, fee, capital, from, to (ms)}
  function portfolio(coins, P) {
    const size = P.amounts.reduce((a, b) => a + b, 0), weights = P.amounts.map(a => a / size);
    const cands = [], per = {};
    for (const { name, D } of coins) {
      const sig = coinSignals(D, P.bb, P.stoch, P.dir);
      const from = Math.max(1, idxAt(D, P.from ?? D.t0)), to = Math.min(D.c.length, P.to == null || P.to === Infinity ? D.c.length : idxAt(D, P.to));
      const hi = D.h.subarray(0, to), lo = D.l.subarray(0, to);
      per[name] = { name, D, hi, lo, from, to, busy: -1, trades: 0, wins: 0, pnl: 0, signals: 0, skipped: 0 };
      sig.idx.forEach((t, n) => { if (t >= from && t < to) cands.push({ name, t, ms: D.t0 + t * D.step, side: sig.side[n], price: sig.price[n] }); });
    }
    cands.sort((a, b) => a.ms - b.ms);

    let cash = P.capital, open = [], maxOpen = 0, skipped = 0;
    const closed = [], openAtEnd = [];
    const release = ms => {
      open = open.filter(o => { if (o.exitMs < ms) { cash += size + o.pnl; return false; } return true; });
    };
    for (const c of cands) {
      const st = per[c.name];
      if (c.t <= st.busy) continue;                      // ten coin ma jeszcze otwartą pozycję
      st.signals++;
      release(c.ms);
      if (cash < size - 1e-9) { skipped++; st.skipped++; continue; }
      const tr = simulateOne(st.hi, st.lo, c.t, c.side, c.price, P.tp, P.sl, weights, P.dists);
      if (!tr) { st.busy = Infinity; openAtEnd.push(c); cash -= size; continue; }
      const pnl = (tr.gross - 2 * P.fee * tr.filled) * size;
      const rec = { ...tr, coin: c.name, entryMs: c.ms, exitMs: st.D.t0 + tr.j * st.D.step, pnl };
      st.busy = tr.j; st.trades++; st.pnl += pnl; if (pnl > 0) st.wins++;
      cash -= size; open.push(rec); closed.push(rec);
      maxOpen = Math.max(maxOpen, open.length + openAtEnd.length);
    }

    // krzywa kapitału: kapitał startowy + zrealizowany wynik, w chwilach zamknięcia pozycji
    closed.sort((a, b) => a.exitMs - b.exitMs);
    const startMs = cands.length ? cands[0].ms : Math.min(...coins.map(x => x.D.t0));
    let eq = P.capital, peak = P.capital, maxDd = 0, maxDdPct = 0;
    const curve = [[startMs, eq]], years = {};
    for (const t of closed) {
      eq += t.pnl; peak = Math.max(peak, eq);
      maxDd = Math.min(maxDd, eq - peak); maxDdPct = Math.min(maxDdPct, (eq - peak) / peak);
      curve.push([t.exitMs, eq]);
      const y = new Date(t.exitMs).getUTCFullYear();
      (years[y] ||= { pnl: 0, trades: 0 }); years[y].pnl += t.pnl; years[y].trades++;
    }
    const wins = closed.filter(t => t.pnl > 0), losses = closed.filter(t => t.pnl <= 0);
    const sum = a => a.reduce((s, t) => s + t.pnl, 0);

    // losowe wejścia z tymi samymi zasadami wyjścia, każdy coin osobno: średni wynik na transakcję
    let rN = 0, rPnl = 0, rWins = 0;
    for (const st of Object.values(per)) {
      for (let seed = 1; seed <= 2; seed++) {
        let x = (seed * 2654435761 + st.name.length * 97) % 4294967296;
        const rnd = () => (x = (x * 1664525 + 1013904223) % 4294967296) / 4294967296;
        const idx = [], side = [], price = [];
        for (let t = st.from; t < st.to; t++) if (rnd() < 1 / 3) { idx.push(t); side.push(P.dir === "long" ? 1 : P.dir === "short" ? -1 : (rnd() < .5 ? 1 : -1)); price.push(st.D.o[t]); }
        for (const tr of simulate(st.hi, st.lo, { idx, side, price }, P.tp, P.sl, weights, P.dists)) {
          const v = (tr.gross - 2 * P.fee * tr.filled) * size; rN++; rPnl += v; if (v > 0) rWins++;
        }
      }
    }

    return {
      size, trades: closed.length, wins: wins.length, net: eq - P.capital, end: eq, maxDd, maxDdPct, curve, years,
      avgWin: wins.length ? sum(wins) / wins.length : null, avgLoss: losses.length ? sum(losses) / losses.length : null,
      avgFills: closed.length ? closed.reduce((a, t) => a + t.fills, 0) / closed.length : 0,
      maxOpen, skipped, openAtEnd: openAtEnd.length, list: closed, perCoin: Object.values(per),
      random: { perTrade: rN ? rPnl / rN : 0, winRate: rN ? rWins / rN : 0 },
    };
  }

  const api = { portfolio, prepareState, signals, simulate, simulateOne };
  if (typeof module !== "undefined") module.exports = api; else root.BT = api;
})(this);
