import numpy as np
import pandas as pd

STEP_15M = 15 * 60_000
BB_WINDOW, BB_DEV, RSI_WINDOW, SMOOTH = 20, 2, 14, 3


def prepare_state(m1):
    """Do każdej minuty dokleja stan z zamkniętych świeczek 15m (sumy do BB, stan RSI, poprzednie StochRSI).
    Dzięki temu wskaźniki "jak w bocie" policzymy dla DOWOLNEJ aktualnej ceny p."""
    m1 = m1.copy()
    m1["bucket"] = m1["ts"] // STEP_15M * STEP_15M
    c15 = m1.groupby("bucket")["close"].last()

    a = 1 / RSI_WINDOW
    diff = c15.diff()
    ag = diff.clip(lower=0).ewm(alpha=a, adjust=False, min_periods=RSI_WINDOW).mean()
    al = (-diff).clip(lower=0).ewm(alpha=a, adjust=False, min_periods=RSI_WINDOW).mean()
    rsi = 100 - 100 / (1 + ag / al)
    srsi = (rsi - rsi.rolling(RSI_WINDOW).min()) / (rsi.rolling(RSI_WINDOW).max() - rsi.rolling(RSI_WINDOW).min())
    k = srsi.rolling(SMOOTH).mean()

    prev = pd.DataFrame({
        "s_sum": c15.rolling(BB_WINDOW - 1).sum(),
        "s_sq": (c15 ** 2).rolling(BB_WINDOW - 1).sum(),
        "last_close": c15, "ag": ag, "al": al,
        "rsi_min": rsi.rolling(RSI_WINDOW - 1).min(),
        "rsi_max": rsi.rolling(RSI_WINDOW - 1).max(),
        "srsi1": srsi, "srsi2": srsi.shift(1), "k1": k, "k2": k.shift(1),
    })
    prev.index = prev.index + STEP_15M          # stan świeczki t-1 używany w trakcie świeczki t
    return m1.join(prev, on="bucket")


def indicators_at(s, p):
    """BB i StochRSI (K, D) dla aktualnej ceny p. s = wiersz albo cała tabela z prepare_state."""
    mean = (s["s_sum"] + p) / BB_WINDOW
    std = np.sqrt(np.maximum((s["s_sq"] + p ** 2) / BB_WINDOW - mean ** 2, 0))
    a = 1 / RSI_WINDOW
    d = p - s["last_close"]
    ag = (1 - a) * s["ag"] + a * np.maximum(d, 0)
    al = (1 - a) * s["al"] + a * np.maximum(-d, 0)
    rsi = 100 - 100 / (1 + ag / al)
    lo, hi = np.minimum(s["rsi_min"], rsi), np.maximum(s["rsi_max"], rsi)
    srsi = (rsi - lo) / (hi - lo)
    k = (s["srsi2"] + s["srsi1"] + srsi) / SMOOTH
    dd = (s["k2"] + s["k1"] + k) / SMOOTH
    return mean + BB_DEV * std, mean - BB_DEV * std, k * 100, dd * 100


def cond_short(s, p, bb_pct, hi_lvl):
    bb_h, _, k, d = indicators_at(s, p)
    return (p > bb_h * (1 + bb_pct)) & (k > hi_lvl) & (d > hi_lvl)


def cond_long(s, p, bb_pct, lo_lvl):
    _, bb_l, k, d = indicators_at(s, p)
    return (p < bb_l * (1 - bb_pct)) & (k < lo_lvl) & (d < lo_lvl)


def first_price(cond, s, start, end, steps=30):
    """Pierwsza cena między start a end, przy której warunek jest spełniony (bisekcja).
    Warunek jest monotoniczny: im dalej cena ucieka, tym łatwiej go spełnić."""
    if cond(s, start):
        return start
    for _ in range(steps):
        mid = (start + end) / 2
        if cond(s, mid):
            end = mid
        else:
            start = mid
    return end


def run_live_mode(m1, bb_pct=0.01, stoch_low=20, stoch_high=80, tp=0.01, sl=0.03, fee=0.0, size=100.0):
    """Symulacja bota live na 1m: wejście, gdy cena W TRAKCIE minuty (knot) spełni warunek bota."""
    s = prepare_state(m1)
    # szybki filtr: czy w tej minucie w ogóle był moment spełniający warunek (sprawdzamy ekstremum)
    s["can_short"] = cond_short(s, s["high"], bb_pct, stoch_high).fillna(False)
    s["can_long"] = cond_long(s, s["low"], bb_pct, stoch_low).fillna(False)

    trades, pos = [], None
    for r in s.to_dict("records"):
        if pos is None:
            if r["can_short"]:
                entry = first_price(lambda x, p: cond_short(x, p, bb_pct, stoch_high), r, r["open"], r["high"])
                pos = {"side": "short", "entry": entry, "entry_time": r["time"],
                       "tp": entry * (1 - tp), "sl": entry * (1 + sl)}
            elif r["can_long"]:
                entry = first_price(lambda x, p: cond_long(x, p, bb_pct, stoch_low), r, r["open"], r["low"])
                pos = {"side": "long", "entry": entry, "entry_time": r["time"],
                       "tp": entry * (1 + tp), "sl": entry * (1 - sl)}
            continue

        if pos["side"] == "long":
            hit_sl, hit_tp = r["low"] <= pos["sl"], r["high"] >= pos["tp"]
        else:
            hit_sl, hit_tp = r["high"] >= pos["sl"], r["low"] <= pos["tp"]
        if not (hit_sl or hit_tp):
            continue
        exit_price = pos["sl"] if hit_sl else pos["tp"]
        pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
        if pos["side"] == "short":
            pnl_pct = -pnl_pct
        pnl_pct -= 2 * fee
        trades.append({"side": pos["side"], "entry_time": pos["entry_time"], "exit_time": r["time"],
                       "entry": pos["entry"], "exit": exit_price, "reason": "SL" if hit_sl else "TP",
                       "pnl_pct": pnl_pct, "pnl_usd": size * pnl_pct, "ambiguous": bool(hit_sl and hit_tp)})
        pos = None
    return trades


def load_live(live_path, symbol):
    """Tabela z bota live. Uwaga: Time = czas ZAMKNIĘCIA, bot nie zapisuje czasu wejścia."""
    live = pd.read_csv(live_path, sep=None, engine="python")
    live = live[live["Pair"] == symbol].copy()
    live["time"] = pd.to_datetime(live["Time"], utc=True).astype("datetime64[ns, UTC]")
    live["side"] = live["Direction"].str.lower()
    live["live_result"] = live["PnL"].apply(lambda x: "TP" if x > 0 else "SL")
    return live.sort_values("time").reset_index(drop=True)


def match_with_live(live, bt, price_tol=0.005, max_hours=24):
    """Paruje po kierunku + cenie wejścia (±0.5%). Wejście w backteście musi być przed wyjściem live
    (max 24h wcześniej). Każda transakcja z backtestu może być użyta tylko raz."""
    bt = bt.copy().reset_index(drop=True)
    bt["entry_time"] = bt["entry_time"].astype("datetime64[ns, UTC]")
    bt["exit_time"] = bt["exit_time"].astype("datetime64[ns, UTC]")
    used, pairs = set(), []

    for _, lv in live.iterrows():
        cand = bt[(bt["side"] == lv["side"])
                  & (~bt.index.isin(used))
                  & (bt["entry_time"] <= lv["time"] + pd.Timedelta("5min"))
                  & (bt["entry_time"] >= lv["time"] - pd.Timedelta(hours=max_hours))
                  & ((bt["entry"] - lv["Entry"]).abs() / lv["Entry"] <= price_tol)]
        if cand.empty:
            pairs.append(None)
            continue
        j = (cand["entry"] - lv["Entry"]).abs().idxmin()
        used.add(j)
        pairs.append(j)

    live = live.copy()
    live["bt_idx"] = pairs
    found = live["bt_idx"].notna()
    m = live[found].join(bt, on="bt_idx", rsuffix="_bt")
    same = m["live_result"] == m["reason"]

    print(f"live: {len(live)}, backtest: {len(bt)}")
    print(f"sparowane: {found.sum()}, z tym samym wynikiem: {same.sum()}")
    print("\nSPAROWANE:")
    print(m[["time", "entry_time", "exit_time", "side", "Entry", "entry", "live_result", "reason"]]
          .rename(columns={"time": "live_exit", "Entry": "entry_live", "entry": "entry_bt"}))
    print("\nLIVE BEZ PARY:")
    print(live.loc[~found, ["time", "side", "Entry", "PnL"]])
    print("\nBACKTEST BEZ PARY:")
    print(bt.loc[~bt.index.isin(used), ["entry_time", "exit_time", "side", "entry", "reason"]])


def compare_live_mode(live_path, symbol="SOL/USDT"):
    from data import load_candles
    live = load_live(live_path, symbol)
    start = live["time"].min() - pd.Timedelta(days=5)
    end = live["time"].max() + pd.Timedelta(days=1)
    m1 = load_candles(symbol, "1m", int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    bt = pd.DataFrame(run_live_mode(m1))
    bt = bt[bt["exit_time"] >= live["time"].min() - pd.Timedelta(hours=1)]
    match_with_live(live, bt)

    wins = (bt["pnl_usd"] > 0).mean() * 100
    print(f"\nTRYB LIVE: {len(bt)} transakcji, win rate {wins:.1f}%, PnL {bt['pnl_usd'].sum():.2f} USD (bez prowizji)")
    return bt