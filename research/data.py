import sqlite3

import ccxt
import pandas as pd

DB_PATH = "candles.db"
EXCHANGE_ID = "okx"
exchange = ccxt.okx({"enableRateLimit": True})


def init_db():
    """Tworzy tabelę candles, jeśli nie istnieje."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            exchange  TEXT    NOT NULL,
            symbol    TEXT    NOT NULL,
            timeframe TEXT    NOT NULL,
            ts        INTEGER NOT NULL,
            open      REAL    NOT NULL,
            high      REAL    NOT NULL,
            low       REAL    NOT NULL,
            close     REAL    NOT NULL,
            volume    REAL    NOT NULL,
            PRIMARY KEY (exchange, symbol, timeframe, ts)
        )
    """)
    conn.commit()
    conn.close()


def get_cached_range(conn, symbol, timeframe):
    """Zwraca (najstarszy_ts, najnowszy_ts) dla danej pary albo (None, None)."""
    return conn.execute(
        """
        SELECT MIN(ts), MAX(ts)
        FROM candles
        WHERE exchange = ? AND symbol = ? AND timeframe = ?
        """,
        (EXCHANGE_ID, symbol, timeframe),
    ).fetchone()


def download(symbol, timeframe, since, until):
    """Pobiera ZAMKNIĘTE świeczki z giełdy od since do until (ms), w pętli po paczkach."""
    step = exchange.parse_timeframe(timeframe) * 1000   # długość świeczki w ms
    now = exchange.milliseconds()
    all_candles = []

    while since < until:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
        if not batch:                   # pusta paczka = brak danych dalej
            break
        all_candles.extend(batch)

        last_ts = batch[-1][0]
        if last_ts < since:             # zabezpieczenie: giełda nie poszła do przodu
            break
        since = last_ts + step          # następna paczka zaczyna się po ostatniej świeczce
        if len(all_candles) % 10000 < 100:
            print(f"  pobrano {len(all_candles)} świeczek {symbol} {timeframe}...")

    # zostaw tylko świeczki w zakresie i już ZAMKNIĘTE (ts + step <= teraz)
    return [c for c in all_candles if c[0] < until and c[0] + step <= now]


def save(conn, symbol, timeframe, candles):
    """Zapisuje świeczki do bazy, pomijając duplikaty."""
    rows = [
        (EXCHANGE_ID, symbol, timeframe, ts, o, h, l, c, v)
        for ts, o, h, l, c, v in candles
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def load_candles(symbol, timeframe, since, until):
    """Główna funkcja: dociąga brakujące dane i zwraca DataFrame z bazy."""
    step = exchange.parse_timeframe(timeframe) * 1000
    conn = sqlite3.connect(DB_PATH)
    try:
        oldest, newest = get_cached_range(conn, symbol, timeframe)

        if oldest is None:
            # nic nie mamy dla tej pary -> pobierz cały zakres
            save(conn, symbol, timeframe, download(symbol, timeframe, since, until))
        else:
            if since < oldest:
                # brakuje POCZĄTKU
                save(conn, symbol, timeframe, download(symbol, timeframe, since, oldest))
            if newest + step < until:
                # brakuje KOŃCA
                save(conn, symbol, timeframe, download(symbol, timeframe, newest + step, until))

        df = pd.read_sql_query(
            """
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
              AND ts >= ? AND ts < ?
            ORDER BY ts
            """,
            conn,
            params=(EXCHANGE_ID, symbol, timeframe, since, until),
        )
    finally:
        conn.close()

    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


if __name__ == "__main__":
    init_db()
    start = exchange.parse8601("2026-08-01T00:00:00Z")
    df = load_candles("SOL/USDT", "15m", start, exchange.milliseconds())
    print(len(df))
    print(df.head())
    print(df["ts"].diff().value_counts())   # dziury w danych? powinno być tylko 900000