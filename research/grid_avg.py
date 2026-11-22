"""Grid dla strategii z UŚREDNIANIEM (2 albo 3 wejścia).

Zasady (tak jak opisałeś):
- Pierwsze wejście: sygnał BB (+ StochRSI w trybie "stoch"), dokładnie jak bot live.
- Kolejne wejścia: zlecenia limit co X% od PIERWSZEGO wejścia, przeciw nam (long niżej, short wyżej).
  Bollinger nie ma już znaczenia - dokładamy tylko po procentach.
- TP liczony od ŚREDNIEJ ceny wejścia - po każdym dołożeniu przesuwa się sam.
- SL NIE ISTNIEJE, dopóki nie wejdzie ostatnie dołożenie. Dopiero wtedy ustawiamy SL:
    sl_ref = "last" -> X% dalej od ceny OSTATNIEGO wejścia
    sl_ref = "avg"  -> X% od ŚREDNIEJ ceny wejścia
- Ostrożnie: w minucie, w której wchodzi dołożenie, TP się nie liczy; SL w tej samej minucie się liczy.
- Kapitał 100 USD na całą pozycję (wszystkie wejścia razem). Niewykorzystana część kapitału nie płaci prowizji.
"""
import itertools
import time

import numpy as np
import pandas as pd

from livemode import prepare_state
from grid import build_signals, first_hit, metrics, STATE_COLS, FEE_TAKER

FULL_YEARS = [2024, 2025, 2026]     # pełne (prawie) lata w danych; 2023 to tylko listopad-grudzień

GRID_AVG2 = {
    "bb_pct": [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02],
    "mode": ["stoch", "bb_only"],
    "dist": [[round(d, 4)] for d in np.arange(0.005, 0.0501, 0.005)],   # drugie wejście: 0.5% ... 5.0% od pierwszego
    "w1": [round(w, 2) for w in np.arange(0.1, 0.91, 0.1)],              # część kapitału na 1. wejście (reszta na 2.)
    "tp": [round(t, 4) for t in np.arange(0.006, 0.0301, 0.002)],        # TP 0.6% ... 3.0% od średniej
    "sl": [round(s, 4) for s in np.arange(0.01, 0.0501, 0.005)],         # SL 1% ... 5% (po ostatnim dołożeniu)
    "sl_ref": ["last"],
}

# ETAP 2: TP zmniejszany po dołożeniu. Wpisz tu najlepszy obszar z etapu 1 (BB, tryb, odstęp, podział),
# a grid sprawdzi pary (TP po 1. wejściu, TP po 2. wejściu) z TP2 <= TP1, razem z SL.
GRID_AVG2_TP2 = {
    "bb_pct": [0.015, 0.0175, 0.02],          # <- uzupełnij po etapie 1
    "mode": ["stoch"],                       # <- uzupełnij po etapie 1
    "dist": [[0.02], [0.03], [0.04]],        # <- uzupełnij po etapie 1
    "w1": [0.3, 0.5],                        # <- uzupełnij po etapie 1
    "tp": [(round(t1, 4), round(t2, 4))
           for t1 in np.arange(0.006, 0.0301, 0.002)
           for t2 in np.arange(0.003, 0.0301, 0.001) if t2 <= t1 + 1e-9],
    "sl": [round(s, 4) for s in np.arange(0.01, 0.0501, 0.005)],
    "sl_ref": ["last"],
}

def splits3(step=10, lo=10, hi=60):
    """Podziały kapitału na 3 wejścia co 10 USD (suma 100), każde wejście od lo do hi USD - bez skrajności."""
    out = []
    for a in range(lo, hi + 1, step):
        for b in range(lo, hi + 1, step):
            c = 100 - a - b
            if lo <= c <= hi:
                out.append((a / 100, b / 100, c / 100))
    return out


# 3 WEJŚCIA: 2. wejście d1 od pierwszego, 3. wejście kolejne d2 dalej (np. 3% i potem +5% = 8% od pierwszego)
GRID_AVG3 = {
    "bb_pct": [0.0125, 0.015, 0.0175],
    "mode": ["stoch", "bb_only"],
    "dist": [[d1 / 100, (d1 + d2) / 100] for d1 in (1, 2, 3, 4, 5) for d2 in (1, 2, 3, 4, 5)],
    "w1": splits3(),                                                    # 27 podziałów, np. 30/30/40, 20/30/50
    "tp": [round(t, 4) for t in np.arange(0.010, 0.0301, 0.002)],      # TP 1% ... 3% od średniej
    "sl": [round(s, 4) for s in np.arange(0.020, 0.0501, 0.005)],      # SL 2% ... 5% od ostatniego wejścia
    "sl_ref": ["last"],
}

GRID_AVG3_SMALL = {
    "bb_pct": [0.015],
    "mode": ["stoch"],
    "dist": [[0.03, 0.08], [0.02, 0.05]],
    "w1": [(0.3, 0.3, 0.4), (0.2, 0.3, 0.5)],
    "tp": [0.01, 0.02],
    "sl": [0.02, 0.03],
    "sl_ref": ["last"],
}

GRID_AVG2_SMALL = {
    "bb_pct": [0.01, 0.015],
    "mode": ["stoch"],
    "dist": [[0.02], [0.03]],
    "w1": [0.3, 0.5],
    "tp": [0.01, 0.02],
    "sl": [0.02, 0.03],
    "sl_ref": ["last"],
}


def weights_for(w1, n_entries):
    """Rozkład kapitału. 2 wejścia: [w1, 1-w1]. Dla 3 wejść podajemy listę wag wprost."""
    if isinstance(w1, (list, tuple)):
        return list(w1)
    return [w1, 1 - w1] if n_entries == 2 else [1.0]


def simulate_avg(hi, lo, sig_idx, sig_side, sig_price, tp, sl, weights, dists, sl_ref="last",
                 tp_arr=None, sl_arr=None, dist_arr=None):
    """weights: udział kapitału na kolejne wejścia (suma 1). dists: odległości kolejnych dołożeń od PIERWSZEGO
    wejścia (np. [0.02] albo [0.02, 0.04]). Zwraca te same tablice co grid.simulate + liczbę wypełnionych wejść.
    tp_arr / sl_arr / dist_arr (opcjonalnie): osobne TP, SL i odstępy DLA KAŻDEGO SYGNAŁU - używane przez
    wersję dopasowaną do zmienności (grid_vol.py). Wartość jest zamrażana w chwili wejścia."""
    n, n_sig, n_w = len(hi), len(sig_idx), len(weights)
    # tp: jedna liczba (ten sam TP od średniej przez całą pozycję) albo lista/krotka: TP po 1., 2., ... wejściu
    tps = list(tp) if isinstance(tp, (list, tuple)) else [tp] * n_w
    out_idx, out_gross, out_filled, out_fills, out_win = [], [], [], [], []
    s = 0
    while s < n_sig:
        i, side, p1 = int(sig_idx[s]), int(sig_side[s]), float(sig_price[s])
        if tp_arr is not None:
            tps = [float(tp_arr[s])] * n_w
        sl_s = float(sl_arr[s]) if sl_arr is not None else sl
        d_s = dist_arr[s] if dist_arr is not None else dists
        filled, qty, last_p = weights[0], weights[0] / p1, p1
        k, j, allow_tp, exit_p, win = 1, i, False, None, False

        while True:
            avg = filled / qty
            if side == 1:                                     # LONG
                tp_l = avg * (1 + tps[k - 1])
                if k < n_w:                                   # jeszcze czekamy na dołożenie - bez SL
                    nxt = p1 * (1 - d_s[k - 1])
                    if lo[j] <= nxt:
                        filled += weights[k]; qty += weights[k] / nxt; last_p = nxt
                        k += 1; allow_tp = False
                        continue                              # ta sama minuta: kolejne dołożenie albo SL
                    if allow_tp and hi[j] >= tp_l:
                        exit_p, win = tp_l, True; break
                    j = first_hit(hi, lo, j + 1, tp_l, nxt, n)
                else:                                         # wszystkie wejścia są - działa SL
                    ref = last_p if sl_ref == "last" else avg
                    sl_l = ref * (1 - sl_s)
                    if lo[j] <= sl_l:
                        exit_p, win = sl_l, False; break
                    if allow_tp and hi[j] >= tp_l:
                        exit_p, win = tp_l, True; break
                    j = first_hit(hi, lo, j + 1, tp_l, sl_l, n)
            else:                                             # SHORT
                tp_l = avg * (1 - tps[k - 1])
                if k < n_w:
                    nxt = p1 * (1 + d_s[k - 1])
                    if hi[j] >= nxt:
                        filled += weights[k]; qty += weights[k] / nxt; last_p = nxt
                        k += 1; allow_tp = False
                        continue
                    if allow_tp and lo[j] <= tp_l:
                        exit_p, win = tp_l, True; break
                    j = first_hit(hi, lo, j + 1, nxt, tp_l, n)
                else:
                    ref = last_p if sl_ref == "last" else avg
                    sl_l = ref * (1 + sl_s)
                    if hi[j] >= sl_l:
                        exit_p, win = sl_l, False; break
                    if allow_tp and lo[j] <= tp_l:
                        exit_p, win = tp_l, True; break
                    j = first_hit(hi, lo, j + 1, sl_l, tp_l, n)
            if j == -1:
                break                                         # pozycja otwarta do końca danych - pomijamy
            allow_tp = True

        if exit_p is None:
            break
        gross = (qty * exit_p - filled) if side == 1 else (filled - qty * exit_p)
        out_idx.append(i); out_gross.append(gross); out_filled.append(filled)
        out_fills.append(k); out_win.append(win)
        s = int(np.searchsorted(sig_idx, j, side="right"))

    return (np.array(out_idx, dtype=np.int64), np.array(out_gross), np.array(out_filled),
            np.array(out_fills), np.array(out_win, dtype=bool))


def yearly(e_idx, gross, filled, years, fee=FEE_TAKER):
    """Wynik (USD na pozycji 100 USD, po prowizji) osobno dla każdego roku + ile pełnych lat było na plusie."""
    net = (gross - 2 * fee * filled) * 100
    yrs = years[e_idx] if len(e_idx) else np.array([], dtype=int)
    out = {}
    for y in [2023] + FULL_YEARS:
        m = yrs == y
        out[f"pnl_{y}"] = round(float(net[m].sum()), 1)
        out[f"trades_{y}"] = int(m.sum())
    full = [out[f"pnl_{y}"] for y in FULL_YEARS]
    out["lata_na_plusie"] = int(sum(v > 0 for v in full))
    out["najgorszy_rok_usd"] = min(full)
    out["pnl_3lata_usd"] = round(float(net.sum()), 1)
    return out


def _random_entries(op, n_seeds=2, every=40):
    n = len(op)
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(2000, n), size=n // every, replace=False))
        side = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx))
        out.append((idx, side, op[idx]))
    return out


def run_grid_avg(m1, test_start, grid=GRID_AVG2, n_entries=2, out_path="grid_avg2.csv", baseline_seeds=2):
    t0 = time.time()
    print("przygotowuję stan z 15m...")
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    years = s["time"].dt.year.to_numpy()
    del s

    mgmt = list(itertools.product(grid["dist"], grid["w1"], grid["tp"], grid["sl"], grid["sl_ref"]))
    signal_cfgs = list(itertools.product(grid["bb_pct"], grid["mode"]))
    total = len(signal_cfgs) * len(mgmt)
    print(f"kombinacji: {total}  (sygnałów: {len(signal_cfgs)}, zarządzania pozycją: {len(mgmt)})")

    # losowe wejścia przez ten sam silnik - punkt odniesienia dla każdego ustawienia zarządzania
    print("liczę losowy punkt odniesienia...")
    randoms = _random_entries(op, n_seeds=baseline_seeds)
    base = {}
    for dist, w1, tp, sl, ref in mgmt:
        w = weights_for(w1, n_entries)
        g_tr, g_te = [], []
        for idx, side, price in randoms:
            e, g, *_ = simulate_avg(hi, lo, idx, side, price, tp, sl, w, dist, ref)
            g_tr.append(g[e < split]); g_te.append(g[e >= split])
        g_tr, g_te = np.concatenate(g_tr), np.concatenate(g_te)
        base[(tuple(dist), str(w1), tp, sl, ref)] = (g_tr.mean() * 100 if len(g_tr) else 0.0,
                                                     g_te.mean() * 100 if len(g_te) else 0.0)
    print(f"  gotowe ({time.time() - t0:.0f}s)")

    rows, done = [], 0
    for bb_pct, mode in signal_cfgs:
        sig = build_signals(S, op, hi, lo, bb_pct, mode)
        for dist, w1, tp, sl, ref in mgmt:
            w = weights_for(w1, n_entries)
            e_idx, gross, filled, fills, win = simulate_avg(hi, lo, *sig, tp, sl, w, dist, ref)
            tr = e_idx < split
            row = {"bb_%": bb_pct * 100, "mode": mode,
                   "dist_%": "/".join(f"{d * 100:g}" for d in dist),
                   "weights_%": "/".join(f"{x * 100:.0f}" for x in w),
                   "tp_%": ("/".join(f"{t * 100:g}" for t in tp) if isinstance(tp, (list, tuple)) else round(tp * 100, 2)), "sl_%": round(sl * 100, 2), "sl_ref": ref}
            row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
            row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))
            b_tr, b_te = base[(tuple(dist), str(w1), tp, sl, ref)]
            row["train_edge_%"] = round(row.get("train_gross_%", np.nan) - b_tr, 3)
            row["test_edge_%"] = round(row.get("test_gross_%", np.nan) - b_te, 3)
            row.update(yearly(e_idx, gross, filled, years))
            rows.append(row)
        done += len(mgmt)
        el = time.time() - t0
        print(f"  BB {bb_pct * 100:.2f}% {mode:8s} sygnałów: {len(sig[0]):6d}   {done}/{total}   "
              f"{el:5.0f}s, zostało ~{el / done * (total - done):5.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    print(f"zapisano {out_path} ({len(res)} wierszy, {time.time() - t0:.0f}s)")
    return res
