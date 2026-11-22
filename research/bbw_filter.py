"""Filtr szerokości Bollinger Bands: nie bierzemy wybić, gdy wstęgi są bardzo wąskie (rynek w bok / akumulacja).

Szerokość liczymy z 20 ZAMKNIĘTYCH świeczek 15m PRZED wybiciem (zanim samo wybicie rozepchnie wstęgi):
    szerokość = (górna - dolna) / środkowa

Dwie wersje progu:
- BEZWZGLĘDNA: odcinamy X% najwęższych sytuacji (percentyl liczony tylko na TRAIN),
- WZGLĘDNA: odcinamy, gdy wstęgi są węższe niż X razy ich typowa szerokość z ostatniego miesiąca
  (np. 0.7 = wstęgi o 30% węższe niż zwykle). Ta wersja sama dopasowuje się do coina i okresu.

Najpierw tabela "jak wypadają transakcje wg szerokości" (bez filtra), potem wyniki z filtrem przy różnych progach.
"""
import numpy as np
import pandas as pd

from livemode import prepare_state, STEP_15M
from grid import build_signals, metrics, STATE_COLS
from grid_avg import simulate_avg

FEE = 0.0005
STRATS = {
    "1x BB1.9 TP2.0 SL4.6": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046, "weights": [1.0], "dist": []},
    "2x BB1.5 4% 30/70 TP2.2 SL2": {"bb_pct": 0.015, "mode": "stoch", "tp": 0.022, "sl": 0.020,
                                     "weights": [0.3, 0.7], "dist": [0.04]},
}
ABS_PCT = [0, 5, 10, 15, 20, 25, 30, 40]          # odetnij X% najwęższych (percentyl z train)
REL = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]           # odetnij, gdy szerokość < X * typowa z ostatniego miesiąca


def bb_width(s, window=20, dev=2, rel_window=2880):
    c15 = s.groupby("bucket")["close"].last()
    mid = c15.rolling(window).mean()
    sd = c15.rolling(window).std(ddof=0)
    w = 2 * dev * sd / mid
    w.index = w.index + STEP_15M                   # szerokość do świeczki t-1, używana w trakcie świeczki t
    rel = w / w.rolling(rel_window, min_periods=rel_window // 4).median()
    return s["bucket"].map(w).to_numpy(dtype=np.float64), s["bucket"].map(rel).to_numpy(dtype=np.float64)


def run(sig, keep, hi, lo, c, split):
    idx, side, price = (a[keep] for a in sig)
    e_idx, gross, filled, fills, win = simulate_avg(hi, lo, idx, side, price, c["tp"], c["sl"], c["weights"], c["dist"])
    tr = e_idx < split
    row = {}
    row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
    row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))
    return row, e_idx, gross - 2 * FEE * filled


def bbw_test(m1, test_start, strats=STRATS):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    width, rel = bb_width(s)

    cols = ["train_trades", "train_win_%", "train_pnl_usd", "train_maxdd_usd",
            "test_trades", "test_win_%", "test_pnl_usd", "test_maxdd_usd"]
    for name, c in strats.items():
        sig = build_signals(S, op, hi, lo, c["bb_pct"], c["mode"])
        w_sig, r_sig = width[sig[0]], rel[sig[0]]
        print(f"\n################ {name} ################")

        # 1) bez filtra: jak wypadają transakcje wg szerokości wstęg przed wybiciem
        _, e_idx, net = run(sig, np.ones(len(sig[0]), bool), hi, lo, c, split)
        t = pd.DataFrame({"okres": np.where(e_idx < split, "train", "test"), "pnl_usd": net * 100,
                          "szer_%": width[e_idx] * 100, "wzgl": rel[e_idx]})
        t["szer_grupa"] = pd.qcut(t["szer_%"], 5, labels=["1 najwęższe", "2", "3", "4", "5 najszersze"])
        print("\n--- bez filtra: wynik wg szerokości wstęg przed wybiciem (5 równych grup) ---")
        print(t.groupby(["szer_grupa", "okres"], observed=True).agg(
            transakcji=("pnl_usd", "size"), wynik_usd=("pnl_usd", "sum"), srednio_usd=("pnl_usd", "mean"),
            win_pct=("pnl_usd", lambda x: round((x > 0).mean() * 100)),
            szer_od=("szer_%", "min"), szer_do=("szer_%", "max")).round(2).unstack("okres").to_string())

        # 2) filtr bezwzględny
        train_sig = sig[0] < split
        rows = []
        for p in ABS_PCT:
            thr = np.nanpercentile(w_sig[train_sig], p) if p > 0 else -np.inf
            keep = ~(w_sig < thr)
            row, *_ = run(sig, keep, hi, lo, c, split)
            rows.append({"odcinam_najwezsze_%": p, "prog_szer_%": round(thr * 100, 2) if p > 0 else 0,
                         "sygnalow_zostaje_%": round(keep.mean() * 100), **{k: row.get(k) for k in cols}})
        print("\n--- filtr BEZWZGLĘDNY (percentyl szerokości z train) ---")
        print(pd.DataFrame(rows).to_string(index=False))

        # 3) filtr względny
        rows = []
        for r in REL:
            keep = ~(r_sig < r) if r > 0 else np.ones(len(r_sig), bool)
            row, *_ = run(sig, keep, hi, lo, c, split)
            rows.append({"odcinam_gdy_wezsze_niz_x_typowej": r, "sygnalow_zostaje_%": round(keep.mean() * 100),
                         **{k: row.get(k) for k in cols}})
        print("\n--- filtr WZGLĘDNY (szerokość / typowa z ostatniego miesiąca) ---")
        print(pd.DataFrame(rows).to_string(index=False))