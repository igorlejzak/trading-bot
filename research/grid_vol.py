"""Grid z parametrami DOPASOWANYMI DO ZMIENNOŚCI (zamiast sztywnych procentów).

Pomysł: zamiast "wejście 1.5% za BB, TP 2.2%, SL 2%" mówimy "wejście k_bb zmienności za BB, TP k_tp zmienności...".
Jednostka zmienności (u) = odchylenie standardowe zwrotów 15-minutowych z ostatniego okna (np. tygodnia),
liczone WYŁĄCZNIE z zamkniętych świeczek. Gdy rynek jest spokojny, u jest małe i wszystkie odległości się
zwężają; gdy rynek szaleje, u rośnie i odległości same się rozszerzają. Mnożniki k są stałe - nic się nie
"douczanie" na bieżąco, więc nie ma ryzyka dopasowania do szumu z ostatnich dni.

Wartości w procentach zamrażamy w chwili wejścia w pozycję.
"""
import itertools
import time

import numpy as np
import pandas as pd

from livemode import prepare_state, STEP_15M
from grid import ind, metrics, STATE_COLS
from grid_avg import simulate_avg, _random_entries

GRID_VOL = {
    "vol_window": [672, 1344, 2880],        # okno zmienności w świeczkach 15m: tydzień, 2 tygodnie, miesiąc
    "k_bb": [2, 3, 4, 5, 6],                # wejście: ile jednostek zmienności za BB
    "mode": ["stoch", "bb_only"],
    "k_tp": [2, 3, 4, 5, 6, 7, 8],          # TP od średniej
    "k_sl": [2, 3, 4, 6, 8, 10, 12],        # SL (przy 2 wejściach: od ostatniego wejścia, po dołożeniu)
    # (nazwa, wagi, mnożnik odstępu dokładki) - 1x bez dokładki
    "entries": [("1x", [1.0], None)] + [
        (f"2x {round(w * 100)}/{round((1 - w) * 100)} d{d}", [w, 1 - w], d)
        for w in (0.3, 0.5, 0.7) for d in (4, 6, 8, 10, 12)
    ],
}

GRID_VOL_SMALL = {
    "vol_window": [672],
    "k_bb": [3, 4],
    "mode": ["stoch"],
    "k_tp": [3, 5],
    "k_sl": [3, 5],
    "entries": [("1x", [1.0], None), ("2x 30/70 d8", [0.3, 0.7], 8)],
}


def vol_unit(m1_state, window):
    """Zmienność (odchylenie standardowe zwrotów 15m) z ostatnich `window` ZAMKNIĘTYCH świeczek, dla każdej minuty."""
    c15 = m1_state.groupby("bucket")["close"].last()
    sigma = np.log(c15).diff().rolling(window, min_periods=window // 2).std()
    sigma.index = sigma.index + STEP_15M            # zmienność do świeczki t-1 używana w trakcie świeczki t
    return m1_state["bucket"].map(sigma).to_numpy(dtype=np.float64)


def cond_short_arr(S, p, bb_pct, mode, st_hi=80):
    bb_h, _, k, d, _ = ind(S, p)
    with np.errstate(invalid="ignore"):
        ok = p > bb_h * (1 + bb_pct)
        if mode == "stoch":
            ok &= (k > st_hi) & (d > st_hi)
    return ok


def cond_long_arr(S, p, bb_pct, mode, st_lo=20):
    _, bb_l, k, d, _ = ind(S, p)
    with np.errstate(invalid="ignore"):
        ok = p < bb_l * (1 - bb_pct)
        if mode == "stoch":
            ok &= (k < st_lo) & (d < st_lo)
    return ok


def build_signals_vol(S, op, hi, lo, bb_arr, mode, steps=30):
    """Jak grid.build_signals, ale odległość za BB jest inna w każdej minucie (bb_arr)."""
    idx_s = np.flatnonzero(cond_short_arr(S, hi, bb_arr, mode))
    idx_l = np.flatnonzero(cond_long_arr(S, lo, bb_arr, mode))
    idx_l = idx_l[~np.isin(idx_l, idx_s)]

    def first_prices(cond, idx, start, end):
        X = {c: S[c][idx] for c in STATE_COLS}
        b = bb_arr[idx]
        start, end = start.copy(), end.copy()
        at_open = cond(X, start, b, mode)
        for _ in range(steps):
            mid = (start + end) / 2
            ok = cond(X, mid, b, mode)
            end = np.where(ok, mid, end)
            start = np.where(ok, start, mid)
        return np.where(at_open, start, end)

    p_s = first_prices(cond_short_arr, idx_s, op[idx_s], hi[idx_s])
    p_l = first_prices(cond_long_arr, idx_l, op[idx_l], lo[idx_l])
    idx = np.concatenate([idx_s, idx_l])
    side = np.concatenate([-np.ones(len(idx_s), dtype=np.int8), np.ones(len(idx_l), dtype=np.int8)])
    price = np.concatenate([p_s, p_l])
    order = np.argsort(idx, kind="stable")
    return idx[order], side[order], price[order]


def run_grid_vol(m1, test_start, grid=GRID_VOL, out_path="grid_vol.csv"):
    t0 = time.time()
    print("przygotowuję stan z 15m...")
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    units = {w: vol_unit(s, w) for w in grid["vol_window"]}
    del s
    for w, u in units.items():
        q = np.nanpercentile(u, [10, 50, 90]) * 100
        print(f"  okno {w} świeczek: 1 jednostka zmienności = {q[1]:.2f}% (10-90%: {q[0]:.2f}-{q[2]:.2f}%)")

    mgmt = list(itertools.product(grid["k_tp"], grid["k_sl"], grid["entries"]))
    signal_cfgs = list(itertools.product(grid["vol_window"], grid["k_bb"], grid["mode"]))
    total = len(signal_cfgs) * len(mgmt)
    print(f"kombinacji: {total}")

    # losowe wejścia z tym samym, zmiennościowym zarządzaniem pozycją - punkt odniesienia
    print("liczę losowy punkt odniesienia...")
    randoms = _random_entries(op, n_seeds=2)
    base = {}
    for w in grid["vol_window"]:
        u = units[w]
        for k_tp, k_sl, (ename, wts, k_d) in mgmt:
            g_tr, g_te = [], []
            for idx, side, price in randoms:
                ok = ~np.isnan(u[idx])
                idx_, side_, price_, uu = idx[ok], side[ok], price[ok], u[idx[ok]]
                dist_arr = (k_d * uu)[:, None] if k_d else None
                e, g, *_ = simulate_avg(hi, lo, idx_, side_, price_, 0, 0, wts, [], "last",
                                        tp_arr=k_tp * uu, sl_arr=k_sl * uu, dist_arr=dist_arr)
                g_tr.append(g[e < split]); g_te.append(g[e >= split])
            g_tr, g_te = np.concatenate(g_tr), np.concatenate(g_te)
            base[(w, k_tp, k_sl, ename)] = (g_tr.mean() * 100 if len(g_tr) else 0.0,
                                           g_te.mean() * 100 if len(g_te) else 0.0)
    print(f"  gotowe ({time.time() - t0:.0f}s)")

    rows, done = [], 0
    for w, k_bb, mode in signal_cfgs:
        u = units[w]
        sig_idx, sig_side, sig_price = build_signals_vol(S, op, hi, lo, k_bb * u, mode)
        us = u[sig_idx]
        for k_tp, k_sl, (ename, wts, k_d) in mgmt:
            dist_arr = (k_d * us)[:, None] if k_d else None
            e_idx, gross, filled, fills, win = simulate_avg(
                hi, lo, sig_idx, sig_side, sig_price, 0, 0, wts, [], "last",
                tp_arr=k_tp * us, sl_arr=k_sl * us, dist_arr=dist_arr)
            tr = e_idx < split
            ue = u[e_idx] * 100                      # zmienność w chwili wejścia - do przeliczenia na %
            row = {"vol_window": w, "k_bb": k_bb, "mode": mode, "k_tp": k_tp, "k_sl": k_sl, "entries": ename,
                   # jakie to były procenty w praktyce (mediana i zakres 10-90% po transakcjach)
                   "bb_%_med": round(float(np.median(k_bb * ue)), 2) if len(ue) else np.nan,
                   "tp_%_med": round(float(np.median(k_tp * ue)), 2) if len(ue) else np.nan,
                   "tp_%_p10": round(float(np.percentile(k_tp * ue, 10)), 2) if len(ue) else np.nan,
                   "tp_%_p90": round(float(np.percentile(k_tp * ue, 90)), 2) if len(ue) else np.nan,
                   "sl_%_med": round(float(np.median(k_sl * ue)), 2) if len(ue) else np.nan,
                   "dist_%_med": round(float(np.median(k_d * ue)), 2) if (k_d and len(ue)) else np.nan}
            row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
            row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))
            b_tr, b_te = base[(w, k_tp, k_sl, ename)]
            row["train_edge_%"] = round(row.get("train_gross_%", np.nan) - b_tr, 3)
            row["test_edge_%"] = round(row.get("test_gross_%", np.nan) - b_te, 3)
            rows.append(row)
        done += len(mgmt)
        el = time.time() - t0
        print(f"  okno {w:5d} k_bb {k_bb} {mode:8s} sygnałów: {len(sig_idx):6d}   {done}/{total}   "
              f"{el:5.0f}s, zostało ~{el / done * (total - done):4.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    print(f"zapisano {out_path} ({len(res)} wierszy, {time.time() - t0:.0f}s)")
    return res