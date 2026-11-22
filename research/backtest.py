import pandas as pd
import random

from strategy import prepare
from data import load_candles, exchange


BT_PARAMS = {
    "tp": 0.01,       # take profit 1%
    "sl": 0.02,       # stop loss 2%
    "fee": 0.0005,    # prowizja na stronę (sprawdź aktualną stawkę OKX dla swojego rynku)
    "size": 100.0,    # wielkość pozycji w USD
}

def run_random(df, bt_params=None, p_enter=0.05, seed=42):
    """Benchmark: wejście losowe (losowa strona), to samo TP/SL/prowizje co strategia."""
    bt = {**BT_PARAMS, **(bt_params or {})}
    rng = random.Random(seed)
    trades, pos = [], None

    for cur in df.to_dict("records"):
        if pos is None:
            if rng.random() >= p_enter:      # w 95% świeczek nic nie robimy
                continue
            side = rng.choice(["long", "short"])
            entry = cur["open"]
            if side == "long":
                tp, sl = entry * (1 + bt["tp"]), entry * (1 - bt["sl"])
            else:
                tp, sl = entry * (1 - bt["tp"]), entry * (1 + bt["sl"])
            pos = {"side": side, "entry": entry, "entry_time": cur["time"],
                   "tp": tp, "sl": sl, "bb_width": None}

        hit_sl, hit_tp = sl_hit(pos, cur), tp_hit(pos, cur)
        if hit_sl:
            trades.append(close_trade(pos, sl_price(pos, cur), cur["time"], "SL", hit_tp, bt))
            pos = None
        elif hit_tp:
            trades.append(close_trade(pos, pos["tp"], cur["time"], "TP", False, bt))
            pos = None

    return trades

def try_open(prev, cur, bt):
    if prev["long_ready"] and cur["low"] <= prev["long_level"]:
        entry = min(cur["open"], prev["long_level"])

        return {
            "bb_width": prev["bb_width"],
            "side": "long",
            "entry": entry,
            "entry_time": cur["time"],
            "tp": entry * (1 + bt["tp"]),
            "sl": entry * (1 - bt["sl"]),
        }
    elif prev["short_ready"] and cur["high"] >= prev["short_level"]:
        entry = max(cur["open"], prev["short_level"])
        return {
            "bb_width": prev["bb_width"],
            "side": "short",
            "entry": entry,
            "entry_time": cur["time"],
            "tp": entry * (1 - bt["tp"]),
            "sl": entry * (1 + bt["sl"]),
        }
    return None


def sl_hit(pos, cur):
    """Czy knot świeczki dotknął stop lossa."""
    if pos["side"] == "long":
        return cur["low"] <= pos["sl"]
    return cur["high"] >= pos["sl"]


def tp_hit(pos, cur):
    """Czy knot świeczki dotknął take profitu."""
    if pos["side"] == "long":
        return cur["high"] >= pos["tp"]
    return cur["low"] <= pos["tp"]


def sl_price(pos, cur):
    """Cena wyjścia na SL. Jeśli świeczka otworzyła się już za SL (luka), wychodzimy po open."""
    if pos["side"] == "long":
        return min(cur["open"], pos["sl"])
    return max(cur["open"], pos["sl"])


def close_trade(pos, exit_price, exit_time, reason, ambiguous, bt):
    """Buduje słownik zamkniętej transakcji z PnL po prowizjach."""
    if pos["side"] == "long":
        pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
    else:
        pnl_pct = (pos["entry"] - exit_price) / pos["entry"]
    pnl_pct -= 2 * bt["fee"]   # prowizja przy wejściu i przy wyjściu

    return {
        "bb_width": pos["bb_width"],
        "side": pos["side"],
        "entry_time": pos["entry_time"],
        "exit_time": exit_time,
        "entry": pos["entry"],
        "exit": exit_price,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "pnl_usd": bt["size"] * pnl_pct,
        "ambiguous": ambiguous,   # TP i SL w tej samej świeczce -> do sprawdzenia na 1m
    }

def resolve_ambiguous(trades, symbol, bt_params=None):
    """Rozstrzyga transakcje ambiguous na świeczkach 1m (tylko ta jedna świeczka 15m)."""
    bt = {**BT_PARAMS, **(bt_params or {})}
    out, changed_to_tp, unresolved = [], 0, 0

    for t in trades:
        if not t["ambiguous"]:
            out.append(t)
            continue

        ts = int(t["exit_time"].timestamp() * 1000)
        raw = exchange.fetch_ohlcv(symbol, "1m", since=ts, limit=15)
        minutes = [{"time": pd.to_datetime(m[0], unit="ms", utc=True),
                    "open": m[1], "high": m[2], "low": m[3], "close": m[4]}
                   for m in raw if ts <= m[0] < ts + 15 * 60_000]

        side, entry = t["side"], t["entry"]
        pos = {"side": side, "entry": entry, "entry_time": t["entry_time"],
               "bb_width": t.get("bb_width")}
        if side == "long":
            pos["tp"], pos["sl"] = entry * (1 + bt["tp"]), entry * (1 - bt["sl"])
        else:
            pos["tp"], pos["sl"] = entry * (1 - bt["tp"]), entry * (1 + bt["sl"])

        # jeśli wejście było w tej samej świeczce 15m, najpierw szukamy minuty wejścia
        in_position = t["entry_time"] != t["exit_time"]
        result = None
        for m in minutes:
            if not in_position:
                touched = m["low"] <= entry if side == "long" else m["high"] >= entry
                if not touched:
                    continue
                in_position = True
                if sl_hit(pos, m):   # minuta wejścia: konserwatywnie tylko SL
                    result = close_trade(pos, pos["sl"], m["time"], "SL", True, bt)
                    break
                continue
            hit_sl, hit_tp = sl_hit(pos, m), tp_hit(pos, m)
            if hit_sl:
                result = close_trade(pos, sl_price(pos, m), m["time"], "SL", hit_tp, bt)
                break
            if hit_tp:
                result = close_trade(pos, pos["tp"], m["time"], "TP", False, bt)
                changed_to_tp += 1
                break

        if result is None:      # brak danych 1m -> zostaje stary wynik
            result = t
            unresolved += 1
        out.append(result)

    print(f"ambiguous: {sum(t['ambiguous'] for t in trades)}, zamienione na TP: {changed_to_tp}, "
          f"nierozstrzygnięte: {unresolved}")
    return out

def run_backtest(df, strat_params=None, bt_params=None):
    """Zwraca listę zamkniętych transakcji (słowników)."""
    bt = {**BT_PARAMS, **(bt_params or {})}
    df = prepare(df, strat_params)
    rows = df.to_dict("records")

    trades = []
    pos = None

    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]

        if pos is None:
            pos = try_open(prev, cur, bt)
            # wersja ostrożna: na świeczce wejścia sprawdzamy tylko SL
            if pos is not None and sl_hit(pos, cur):
                trades.append(close_trade(pos, pos["sl"], cur["time"], "SL", True, bt))
                pos = None
            continue

        hit_sl = sl_hit(pos, cur)
        hit_tp = tp_hit(pos, cur)

        if hit_sl:
            # oba trafione -> zakładamy, że pierwszy był SL (ostrożnie)
            trades.append(close_trade(pos, sl_price(pos, cur), cur["time"], "SL", hit_tp, bt))
            pos = None
        elif hit_tp:
            trades.append(close_trade(pos, pos["tp"], cur["time"], "TP", False, bt))
            pos = None

    return trades


LIVE_STRAT = {"rsi_long": 100, "rsi_short": 0, "use_cross": False, "bb_pct": 0.01}
LIVE_BT = {"tp": 0.01, "sl": 0.03, "fee": 0.0}   # fee 0, bo w tabeli live też nie ma prowizji


    
def summarize(trades, bt):
    """Podstawowe metryki."""
    if not trades:
        return {"trades": 0}

    t = pd.DataFrame(trades)
    equity = t["pnl_usd"].cumsum()
    drawdown = equity - equity.cummax()
    wins = t["pnl_usd"] > 0

    return {
        "trades": len(t),
        "longs": int((t["side"] == "long").sum()),
        "shorts": int((t["side"] == "short").sum()),
        "win_rate_%": round(float(wins.mean()) * 100, 1),
        "break_even_win_rate_%": round((bt["sl"] + 2 * bt["fee"]) / (bt["tp"] + bt["sl"]) * 100, 1),
        "total_pnl_usd": round(float(t["pnl_usd"].sum()), 2),
        "avg_pnl_%": round(float(t["pnl_pct"].mean()) * 100, 3),
        "max_drawdown_usd": round(float(drawdown.min()), 2),
        "ambiguous": int(t["ambiguous"].sum()),
    }

if __name__ == "__main__":
    from tune_tpsl import run_tune, analyze_tune_years

    now = exchange.milliseconds()
    m1 = load_candles("SOL/USDT", "1m", exchange.parse8601("2023-11-05T00:00:00Z"), now)
    test_start = pd.Timestamp("2025-11-10", tz="UTC")
    from grid_avg import run_grid_avg
    run_grid_avg(m1, test_start)