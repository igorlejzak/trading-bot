import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator


DEFAULT_PARAMS = {
    "adx_window": 14,
    "adx_max": None,      # None = filtr wyłączony
    "bb_window": 20,
    "bb_dev": 2,
    "bb_pct": 0.01,       # wyjście 0,3% za wstęgę
    "rsi_window": 14,
    "rsi_long": 100,        # long tylko gdy RSI <= 40
    "rsi_short": 0,       # short tylko gdy RSI >= 60
    "stoch_window": 14,
    "stoch_smooth1": 3,
    "stoch_smooth2": 3,
    "stoch_low": 20,       # strefa wyprzedania: K i D <= 20
    "stoch_high": 80,      # strefa wykupienia: K i D >= 80
    "use_cross": False,     # False = wystarczy strefa StochRSI, bez czekania na przecięcie
    "cross_lookback": 3,   # przecięcie K/D w strefie w ostatnich N świeczkach
}


def add_indicators(df, p):
    """Dopisuje do df kolumny: bb_high, bb_low, rsi, stoch_k, stoch_d."""
    bb = BollingerBands(close=df["close"], window=p["bb_window"], window_dev=p["bb_dev"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["rsi"] = RSIIndicator(close=df["close"], window=p["rsi_window"]).rsi()

    st = StochRSIIndicator(
        close=df["close"],
        window=p["stoch_window"],
        smooth1=p["stoch_smooth1"],
        smooth2=p["stoch_smooth2"],
    )
    df["stoch_k"] = st.stochrsi_k() * 100   # ta zwraca 0-1, przeliczamy na 0-100
    df["stoch_d"] = st.stochrsi_d() * 100
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=p["adx_window"]).adx()
    return df



def add_signals(df, p):
    """Dopisuje kolumny gotowości do wejścia (na zamkniętej świeczce) i poziomy wejścia."""
    k, d = df["stoch_k"], df["stoch_d"]

    # obie linie StochRSI w strefie
    in_low = (k <= p["stoch_low"]) & (d <= p["stoch_low"])
    in_high = (k >= p["stoch_high"]) & (d >= p["stoch_high"])

    # przecięcie liczy się tylko wtedy, gdy zaszło W strefie
    df["bull_cross"] = (k > d) & (k.shift(1) <= d.shift(1)) & in_low
    df["bear_cross"] = (k < d) & (k.shift(1) >= d.shift(1)) & in_high

    # czy takie przecięcie było w ostatnich N świeczkach
    n = p["cross_lookback"]
    df["bull_cross_recent"] = df["bull_cross"].rolling(n).sum() > 0
    df["bear_cross_recent"] = df["bear_cross"].rolling(n).sum() > 0

    if p["use_cross"]:
        long_stoch, short_stoch = df["bull_cross_recent"], df["bear_cross_recent"]
    else:
        long_stoch, short_stoch = in_low, in_high

    df["long_ready"] = (df["rsi"] <= p["rsi_long"]) & long_stoch
    df["short_ready"] = (df["rsi"] >= p["rsi_short"]) & short_stoch

    if p["adx_max"] is not None:
        sideways = df["adx"] < p["adx_max"]
        df["long_ready"] = df["long_ready"] & sideways
        df["short_ready"] = df["short_ready"] & sideways

    df["long_level"] = df["bb_low"] * (1 - p["bb_pct"])
    df["short_level"] = df["bb_high"] * (1 + p["bb_pct"])
    return df


def prepare(df, params=None):
    """Główna funkcja: łączy domyślne parametry z podanymi i liczy wszystko."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = add_indicators(df.copy(), p)
    df = add_signals(df, p)
    return df


if __name__ == "__main__":
    from data import exchange, load_candles

    start = exchange.parse8601("2026-08-01T00:00:00Z")
    df = load_candles("SOL/USDT", "15m", start, exchange.milliseconds())

    for adx in [None, 25, 20]:
        params = {**MY_STRAT, "adx_max": adx}
        trades = run_backtest(train_df, params)
        print("ADX <", adx, summarize(trades, BT_PARAMS))
    out = prepare(df)
    cols = ["time", "close", "bb_low", "long_level", "rsi", "stoch_k", "stoch_d", "long_ready"]
    print(out[cols].tail(10))