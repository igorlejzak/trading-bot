"""Backtest na 20 coinach ze świeczek 15m zamiast 1m - 15 razy mniej danych do pobrania (~2 min na coin).

Silnik jest ten sam (prepare_state + build_signals + simulate_avg), tylko jeden "krok" to świeczka 15m,
a nie minuta. Wejście nadal liczone jest dokładnie (pierwsza cena w świeczce, przy której bot by wszedł).
Różnica: gdy w JEDNEJ świeczce 15m wypada i TP, i SL, zakładamy gorzej (SL), a TP w świeczce wejścia
nie jest liczony. Dlatego wyniki są trochę ostrożniejsze niż na 1m.

python scan15.py kalibracja   -> SOL BTC ETH XRP ADA: porównanie 15m z 1m na danych, które już mamy
python scan15.py              -> wszystkie coiny z listy: pobranie 15m + backtest
python scan15.py LINK DOGE    -> wybrane coiny
"""
import sys

import numpy as np
import pandas as pd

from cross_coin import START, TEST_START
from data import exchange, load_candles
from grid import FEE_TAKER, STATE_COLS, build_signals
from grid_avg import simulate_avg
from livemode import STEP_15M, prepare_state

COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "HYPE", "PEPE", "LINK", "DOT", "DOGE", "AVAX", "ALGO", "SUI",
         "LTC", "XLM", "UNI", "NEAR", "AAVE", "TAO", "ONDO"]
CALIB = ["SOL", "BTC", "ETH", "XRP", "ADA"]
STRATS = {
    "1x BB1.9 TP2 SL4.6": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
                           "weights": [1.0], "dist": []},
    "2x BB1.5 stoch 4% 30/70 TP2.2 SL2": {"bb_pct": 0.015, "mode": "stoch", "tp": 0.022, "sl": 0.020,
                                          "weights": [0.3, 0.7], "dist": [0.04]},
}


def download_15m(symbol):
    """Miesiąc po miesiącu od najnowszego wstecz - działa też dla coinów notowanych od niedawna."""
    now = exchange.milliseconds()
    for m in pd.date_range("2023-11-01", pd.Timestamp.now().strftime("%Y-%m-01"), freq="MS")[::-1]:
        load_candles(symbol, "15m", int(m.timestamp() * 1000), now)
    return load_candles(symbol, "15m", exchange.parse8601(START), now)


def to_15m(m1):
    """Świeczki 15m zbudowane z 1m (do kalibracji - te same dane, inna rozdzielczość)."""
    g = m1.assign(b=m1["ts"] // STEP_15M * STEP_15M).groupby("b")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()}).reset_index().rename(columns={"b": "ts"})
    d["time"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d


def backtest(df, coin, res):
    """df: świeczki 1m albo 15m. Zwraca tabelę transakcji wszystkich strategii."""
    s = prepare_state(df).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo, cl = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low", "close"])
    times = s["time"].to_numpy()
    per_day = 1440 if res == "1m" else 96
    ch7 = np.full(len(cl), np.nan)
    ch7[7 * per_day:] = (cl[7 * per_day:] / cl[:-7 * per_day] - 1) * 100
    out = []
    for name, c in STRATS.items():
        idx, side, price = build_signals(S, op, hi, lo, c["bb_pct"], c["mode"])
        e, gross, filled, fills, win = simulate_avg(hi, lo, idx, side, price, c["tp"], c["sl"], c["weights"], c["dist"])
        sd = side[np.searchsorted(idx, e)]
        out.append(pd.DataFrame({"coin": coin, "strategia": name, "rozdzielczosc": res, "time": times[e],
                                 "side": np.where(sd == 1, "long", "short"), "fills": fills,
                                 "wynik": np.where(win, "wygrana", "przegrana"), "win": win,
                                 "pnl_usd": (gross - 2 * FEE_TAKER * filled) * 100, "zmiana_7d_%": ch7[e]}))
    t = pd.concat(out, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t["okres"] = np.where(t["time"] < TEST_START, "train", "test")
    t["rok"] = t["time"].dt.year
    return t


def table(t, index):
    g = t.groupby(index)
    r = pd.DataFrame({"transakcji": g.size(), "wygrane_%": (g["win"].mean() * 100).round(1),
                      "razem_usd": g["pnl_usd"].sum().round(1)})
    for col, vals in [("okres", ["train", "test"]), ("side", ["long", "short"])]:
        p = t.pivot_table(index=index, columns=col, values="pnl_usd", aggfunc="sum").reindex(columns=vals)
        r = r.join(p.round(1))
    yrs = t.pivot_table(index=index, columns="rok", values="pnl_usd", aggfunc="sum").round(1)
    yrs.columns = [f"rok_{y}" for y in yrs.columns]
    r = r.join(yrs)
    r["od"] = g["time"].min().dt.strftime("%Y-%m")
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 200)
    args = [a.upper() for a in sys.argv[1:]]

    if args == ["KALIBRACJA"]:
        parts = []
        for c in CALIB:
            print("liczę", c, "(1m i 15m) ...")
            m1 = load_candles(c + "/USDT", "1m", exchange.parse8601(START), exchange.milliseconds())
            parts += [backtest(m1, c, "1m"), backtest(to_15m(m1), c, "15m")]
        t = pd.concat(parts)
        for name in STRATS:
            d = t[t["strategia"] == name]
            p = d.pivot_table(index="coin", columns="rozdzielczosc", values="pnl_usd", aggfunc="sum").round(1)
            n = d.pivot_table(index="coin", columns="rozdzielczosc", values="pnl_usd", aggfunc="size")
            p = p.join(n, rsuffix="_transakcji").reindex(CALIB)
            p.loc["SUMA"] = p.sum()
            print(f"\n=== {name}: wynik USD i liczba transakcji, 1m vs 15m ===")
            print(p.to_string())
        sys.exit()

    markets = exchange.load_markets()
    coins = args or COINS
    parts, skipped = [], []
    for c in coins:
        sym = c + "/USDT"
        if sym not in markets or not markets[sym].get("spot"):
            skipped.append(c)
            continue
        print(f"=== {c}: pobieram 15m i liczę ...")
        try:
            df = download_15m(sym)
        except Exception as e:
            print(f"  błąd pobierania {c}: {e} - pomijam (odpal jeszcze raz później)")
            skipped.append(c)
            continue
        if len(df) < 3000:
            print(f"  za mało danych ({len(df)} świeczek) - pomijam")
            skipped.append(c)
            continue
        parts.append(backtest(df, c, "15m"))
    if skipped:
        print("\npominięte:", ", ".join(skipped))

    t = pd.concat(parts, ignore_index=True)
    t.to_csv("trades_15m.csv", index=False)
    for name in STRATS:
        d = t[t["strategia"] == name]
        r = table(d, "coin").sort_values("razem_usd", ascending=False)
        print(f"\n################ {name} (15m, USD na pozycję 100 USD, po opłatach) ################")
        print(r.to_string())
        print(f"\ncoinów na plusie: {(r['razem_usd'] > 0).sum()} z {len(r)} | suma {r['razem_usd'].sum():.1f} USD"
              f" | train {r['train'].sum():.1f} | test {r['test'].sum():.1f}"
              f" | longi {r['long'].sum():.1f} | shorty {r['short'].sum():.1f}"
              f" | test na plusie: {(r['test'] > 0).sum()} z {len(r)}")
    print(f"\nzapisano trades_15m.csv: {len(t)} transakcji")
