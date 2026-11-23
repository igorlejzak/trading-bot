"""Eksport danych do backtestu na stronie Research.

Zapisuje świeczki 15m jako małe pliki JSON w data/web/<COIN>.json. Strona liczy backtest w przeglądarce
na tych plikach, więc serwer na Railway nie musi trzymać bazy ani niczego liczyć.

    cd research
    python export_web.py              # wszystkie coiny, które mają świeczki 1m w candles.db
    python export_web.py SOL ADA      # wybrane coiny
    python export_web.py --top20      # 20 popularnych coinów (brakujące dociąga z OKX)

Źródło danych dla każdego coina:
- są świeczki 1m w candles.db -> składamy z nich 15m (to samo, na czym liczy research),
- nie ma -> pobieramy świeczki 15m z OKX od START (zapisują się w candles.db, następnym razem bez pobierania).
  Trzy lata 15m to ~1100 zapytań, czyli kilka minut na coin.
"""
import json
import os
import sqlite3
import sys

import pandas as pd

from data import DB_PATH, EXCHANGE_ID, init_db, load_candles, exchange

STEP_15M = 15 * 60_000
START = "2023-11-01T00:00:00Z"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "web")
TOP20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "LINK", "AVAX",
         "DOT", "LTC", "BCH", "XLM", "UNI", "NEAR", "AAVE", "SUI", "HYPE", "PEPE"]


def r6(x):
    """6 cyfr znaczących - wystarczy do symulacji, a plik jest ~2x mniejszy."""
    return float(f"{x:.6g}")


def coiny_z_baza():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        rows = conn.execute("SELECT DISTINCT symbol FROM candles WHERE exchange = ? AND timeframe = '1m'",
                            (EXCHANGE_ID,)).fetchall()
    finally:
        conn.close()
    return sorted(s.split("/")[0] for (s,) in rows if s.endswith("/USDT"))


def swieczki_15m(symbol):
    """DataFrame (index = początek świeczki w ms) z kolumnami open/high/low/close."""
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        m1 = pd.read_sql_query(
            "SELECT ts, open, high, low, close FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = '1m' "
            "ORDER BY ts", conn, params=(EXCHANGE_ID, symbol))
    finally:
        conn.close()
    if not m1.empty:
        m1["bucket"] = m1["ts"] // STEP_15M * STEP_15M
        c15 = m1.groupby("bucket").agg(open=("open", "first"), high=("high", "max"),
                                       low=("low", "min"), close=("close", "last"))
        if m1["ts"].iloc[-1] < c15.index[-1] + STEP_15M - 60_000:    # ostatnia niedomknięta - odcinamy
            c15 = c15.iloc[:-1]
        return c15, "1m z bazy"
    print(f"{symbol}: brak 1m w bazie, pobieram 15m z OKX (to może potrwać kilka minut)...")
    df = load_candles(symbol, "15m", exchange.parse8601(START), exchange.milliseconds())
    if df.empty:
        return None, None
    return df.set_index("ts")[["open", "high", "low", "close"]], "15m z OKX"


def eksport(coin):
    symbol = f"{coin}/USDT"
    try:
        c15, zrodlo = swieczki_15m(symbol)
    except Exception as e:
        print(f"{symbol}: błąd ({e}), pomijam")
        return
    if c15 is None or len(c15) < 500:
        print(f"{symbol}: za mało danych, pomijam")
        return
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
    print(f"{symbol}: {len(c15)} świeczek 15m ({zrodlo}), {od} - {do}, uzupełnione dziury: {dziury}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    init_db()
    args = [a.upper() for a in sys.argv[1:]]
    if args == ["--TOP20"]:
        coiny = TOP20
    else:
        coiny = args or coiny_z_baza()
    for c in coiny:
        eksport(c)
    # lista coinów dla strony
    gotowe = sorted(f[:-5] for f in os.listdir(OUT_DIR) if f.endswith(".json") and f != "index.json")
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(gotowe, f)
    print(f"Zapisano w {os.path.abspath(OUT_DIR)}")
    print("Dostępne na stronie:", ", ".join(gotowe))
