"""Wszystkie transakcje wybranych strategii na wszystkich coinach - każda z wynikiem (wygrana / przegrana)
i warunkami rynku w chwili wejścia. Zapis do trades_all.csv + podsumowanie.

python trades_all.py               -> SOL BTC ETH XRP ADA
python trades_all.py BNB HYPE      -> wybrane coiny
"""
import sys

import numpy as np
import pandas as pd

from cross_coin import START, TEST_START
from data import exchange, load_candles
from diagnose import trades_for
from grid import STATE_COLS
from grid_vol import vol_unit
from livemode import prepare_state

STRATS = {
    # najlepsza na SOL wg zysk / drawdown na train (najgładsza krzywa)
    "2x BB1.5 stoch, dokładka 4%, 30/70, TP2.2 SL2": {"bb_pct": 0.015, "mode": "stoch", "tp": 0.022, "sl": 0.020,
                                                     "weights": [0.3, 0.7], "dist": [0.04]},
    # najlepsza na SOL wg najgorszego roku (stabilność rok po roku)
    "2x BB1.25 stoch, dokładka 1%, 90/10, TP2.2 SL1": {"bb_pct": 0.0125, "mode": "stoch", "tp": 0.022, "sl": 0.010,
                                                      "weights": [0.9, 0.1], "dist": [0.01]},
    # dla porównania: 1 wejście, najlepiej przenosi się na inne coiny
    "1x BB1.9, TP2.0 SL4.6": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
                              "weights": [1.0], "dist": []},
}


def coin_trades(symbol):
    m1 = load_candles(symbol, "1m", exchange.parse8601(START), exchange.milliseconds())
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo, cl = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low", "close"])
    vol = vol_unit(s, 2880) * 100                                   # zmienność 15m z ostatniego miesiąca, %
    ch = {}
    for name, lag in [("zmiana_1d_%", 1440), ("zmiana_7d_%", 7 * 1440)]:
        a = np.full(len(cl), np.nan)
        a[lag:] = (cl[lag:] / cl[:-lag] - 1) * 100
        ch[name] = a
    out = []
    for name, c in STRATS.items():
        t = trades_for(s, S, op, hi, lo, c)
        t.insert(0, "strategia", name)
        t.insert(0, "coin", symbol.split("/")[0])
        t["wynik"] = np.where(t["win"], "wygrana", "przegrana")
        t["okres"] = np.where(t["time"] < TEST_START, "train", "test")
        t["rok"] = t["time"].dt.year
        t["zmiennosc_%"] = vol[t["i"]].round(3)
        for k, a in ch.items():
            t[k] = a[t["i"]].round(2)
        t["godzina_utc"] = t["time"].dt.hour
        t["dzien_tyg"] = t["time"].dt.dayofweek
        out.append(t.drop(columns="i"))
    return pd.concat(out, ignore_index=True)


def summary(df, coins):
    def agg(g):
        return pd.Series({"transakcji": len(g), "wygrane_%": round(g["win"].mean() * 100, 1),
                          "wynik_usd": round(g["pnl_usd"].sum(), 1),
                          "dokładek_%": round((g["fills"] > 1).mean() * 100, 1)})

    for name in STRATS:
        d = df[df["strategia"] == name]
        print(f"\n################ {name} ################")
        t = d.groupby("coin").apply(agg, include_groups=False).reindex(coins)
        t.loc["SUMA"] = [len(d), round(d["win"].mean() * 100, 1), round(d["pnl_usd"].sum(), 1),
                         round((d["fills"] > 1).mean() * 100, 1)]
        print(t.to_string())
        for col in ["okres", "side", "rok"]:
            p = d.pivot_table(index="coin", columns=col, values="pnl_usd", aggfunc="sum").reindex(coins).round(1)
            p.loc["SUMA"] = p.sum()
            print(f"\nwynik USD wg: {col}")
            print(p.to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    coins = [a.upper() for a in sys.argv[1:]] or ["SOL", "BTC", "ETH", "XRP", "ADA"]
    parts = []
    for c in coins:
        print("liczę", c, "...")
        parts.append(coin_trades(c + "/USDT"))
    df = pd.concat(parts, ignore_index=True)
    out = "trades_all.csv" if not sys.argv[1:] else f"trades_{'_'.join(coins)}.csv"
    df.to_csv(out, index=False)
    summary(df, coins)
    print(f"\nzapisano {out}: {len(df)} transakcji, każda z wynikiem i warunkami rynku w chwili wejścia")
