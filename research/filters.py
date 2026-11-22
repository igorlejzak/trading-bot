"""Czy dwa proste filtry z diagnozy ADA i XRP pomagają na WSZYSTKICH coinach?

1. Przerwa po stop lossie: po SL nie otwieramy nowej pozycji na tym coinie przez X minut.
   Powód: w krachach (3.01.2024, 2.03.2025, 10.10.2025) strategia łapała SL i od razu wchodziła
   ponownie, kilka razy z rzędu w kilkanaście minut.
2. Tylko longi: na ADA i XRP shorty przegrywały, longi zarabiały.

Wartości są ustalone Z GÓRY (okrągłe), bez szukania najlepszej. Filtr uznajemy za sensowny tylko wtedy,
gdy pomaga na większości coinów, a ostatecznie na BNB, HYPE i live.

python filters.py                -> SOL BTC ETH XRP ADA
python filters.py BNB HYPE       -> wybrane coiny
"""
import sys

import numpy as np
import pandas as pd

from cross_coin import START, TEST_START
from data import exchange, load_candles
from grid import FEE_TAKER, STATE_COLS, build_signals
from grid_avg import simulate_avg
from livemode import prepare_state

BASE = {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046}
VARIANTS = {
    "bazowa (jak dotąd)":       {"cooldown": 0,    "long_only": False},
    "przerwa 1h po SL":         {"cooldown": 60,   "long_only": False},
    "przerwa 4h po SL":         {"cooldown": 240,  "long_only": False},
    "przerwa 24h po SL":        {"cooldown": 1440, "long_only": False},
    "tylko long":               {"cooldown": 0,    "long_only": True},
    "tylko long + przerwa 4h":  {"cooldown": 240,  "long_only": True},
}


def run_coin(symbol, m1=None, test_start=TEST_START):
    if m1 is None:
        m1 = load_candles(symbol, "1m", exchange.parse8601(START), exchange.milliseconds())
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    years = s["time"].dt.year.to_numpy()
    idx, side, price = build_signals(S, op, hi, lo, BASE["bb_pct"], BASE["mode"])

    rows = []
    for name, v in VARIANTS.items():
        keep = side == 1 if v["long_only"] else np.ones(len(idx), dtype=bool)
        e, gross, filled, fills, win = simulate_avg(hi, lo, idx[keep], side[keep], price[keep], BASE["tp"], BASE["sl"],
                                                    [1.0], [], cooldown=v["cooldown"])
        net = (gross - 2 * FEE_TAKER * filled) * 100
        tr = e < split
        eq = np.cumsum(net)
        row = {"coin": symbol.split("/")[0], "wariant": name, "transakcji": len(e),
               "win_%": round(win.mean() * 100, 1) if len(e) else np.nan,
               "train_usd": round(net[tr].sum(), 1), "test_usd": round(net[~tr].sum(), 1),
               "razem_usd": round(net.sum(), 1),
               "maxdd_usd": round((eq - np.maximum.accumulate(np.maximum(eq, 0))).min(), 1) if len(e) else 0}
        for y in sorted(set(years[e])):
            row[f"rok_{y}"] = round(net[years[e] == y].sum(), 1)
        rows.append(row)
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    coins = [a.upper() for a in sys.argv[1:]] or ["SOL", "BTC", "ETH", "XRP", "ADA"]
    rows = []
    for c in coins:
        print("liczę", c, "...")
        rows.extend(run_coin(c + "/USDT"))
    df = pd.DataFrame(rows)
    df.to_csv("filters.csv", index=False)

    order = list(VARIANTS)
    for col in ["razem_usd", "train_usd", "test_usd", "maxdd_usd"]:
        t = df.pivot(index="wariant", columns="coin", values=col).loc[order, [c for c in coins]]
        if col != "maxdd_usd":
            t["SUMA"] = t.sum(axis=1)
            t["coinów_na_plusie"] = (t[coins] > 0).sum(axis=1)
        print(f"\n=== {col} (USD na pozycję 100 USD, po opłatach) ===")
        print(t.to_string())

    print("\n=== rok po roku, suma wszystkich coinów ===")
    ycols = sorted(c for c in df.columns if c.startswith("rok_"))
    print(df.groupby("wariant")[ycols].sum().loc[order].round(1).to_string())
    print("\n=== liczba transakcji ===")
    print(df.pivot(index="wariant", columns="coin", values="transakcji").loc[order, coins].to_string())
    print("\nzapisano filters.csv")