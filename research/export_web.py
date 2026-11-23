"""Eksport danych do backtestu na stronie Research.

Z lokalnej bazy candles.db (świeczki 1m) robi świeczki 15m i zapisuje je jako małe pliki JSON
w data/web/<COIN>.json. Strona liczy backtest w przeglądarce na tych plikach, więc serwer na Railway
nie musi trzymać bazy ani niczego liczyć.

    cd research
    python export_web.py              # SOL, BTC, ETH, XRP
    python export_web.py SOL ADA      # wybrane coiny

Nie pobiera nic z giełdy: bierze tylko to, co już jest w bazie (najpierw `python cross_coin.py download`).
"""
import json
import os
import sqlite3
import sys

import pandas as pd

from data import DB_PATH, EXCHANGE_ID

STEP_15M = 15 * 60_000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "web")
DOMYSLNE = ["SOL", "BTC", "ETH", "XRP"]


def r6(x):
    """6 cyfr znaczących - wystarczy do symulacji, a plik jest ~2x mniejszy."""
    return float(f"{x:.6g}")


def eksport(coin):
    symbol = f"{coin}/USDT"
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        m1 = pd.read_sql_query(
            "SELECT ts, open, high, low, close FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = '1m' "
            "ORDER BY ts", conn, params=(EXCHANGE_ID, symbol))
    finally:
        conn.close()
    if m1.empty:
        print(f"{symbol}: brak świeczek 1m w bazie, pomijam")
        return

    m1["bucket"] = m1["ts"] // STEP_15M * STEP_15M
    c15 = m1.groupby("bucket").agg(open=("open", "first"), high=("high", "max"),
                                   low=("low", "min"), close=("close", "last"))
    # tylko pełne świeczki: odcinamy ostatnią, jeśli jest niedomknięta
    if m1["ts"].iloc[-1] < c15.index[-1] + STEP_15M - 60_000:
        c15 = c15.iloc[:-1]
    # ciągła oś czasu co 15 min: dziury (przerwy giełdy) wypełniamy płaską świeczką z ostatnim close
    full = pd.RangeIndex(int(c15.index[0]), int(c15.index[-1]) + STEP_15M, STEP_15M)
    dziury = len(full) - len(c15)
    c15 = c15.reindex(full)
    c15["close"] = c15["close"].ffill()
    for col in ("open", "high", "low"):
        c15[col] = c15[col].fillna(c15["close"])

    out = {"symbol": symbol, "t0": int(c15.index[0]), "step": STEP_15M,
           "o": [r6(v) for v in c15["open"]], "h": [r6(v) for v in c15["high"]],
           "l": [r6(v) for v in c15["low"]], "c": [r6(v) for v in c15["close"]]}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{coin}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    od = pd.to_datetime(out["t0"], unit="ms").date()
    do = pd.to_datetime(int(c15.index[-1]), unit="ms").date()
    print(f"{symbol}: {len(c15)} świeczek 15m, {od} - {do}, uzupełnione dziury: {dziury}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB -> {os.path.relpath(path)}")


if __name__ == "__main__":
    for c in [a.upper() for a in sys.argv[1:]] or DOMYSLNE:
        eksport(c)
    # lista coinów dla strony
    coiny = sorted(f[:-5] for f in os.listdir(OUT_DIR) if f.endswith(".json") and f != "index.json")
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(coiny, f)
    print("Dostępne na stronie:", ", ".join(coiny))
