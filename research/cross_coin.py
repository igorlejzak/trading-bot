"""Test wybranego ustawienia na innych coinach - BEZ żadnego poprawiania parametrów.

Ustawienie wybrano wyłącznie na zbiorze TRAIN dla SOL/USDT (środek najlepszego obszaru grida).
Uruchomienie:
    python cross_coin.py download   # pobiera świeczki 1m dla wszystkich coinów (na noc, ~2.5 h na coin)
    python cross_coin.py            # liczy wyniki (kilkanaście sekund na coin)
"""
import sys

import numpy as np
import pandas as pd

from data import load_candles, exchange
from livemode import prepare_state
from grid import build_signals, metrics, STATE_COLS
from grid_avg import simulate_avg, _random_entries
from grid_vol import vol_unit, build_signals_vol

# Wybrane 19.11.2026 WYŁĄCZNIE na TRAIN SOL/USDT, przed zobaczeniem innych coinów. Nie zmieniać.
STRATEGIES = {
    # 1 wejście: środek najlepszego obszaru (sąsiedztwo BB/SL/TP)
    "1x  BB1.9 TP2.0 SL4.6":        {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
                                     "weights": [1.0], "dist": []},
    # 2 wejścia: najlepszy zysk/drawdown na train
    #"2x  BB1.5 4% 30/70 TP2.2 SL2":  {"bb_pct": 0.015, "mode": "stoch", "tp": 0.022, "sl": 0.020,
                                     #"weights": [0.3, 0.7], "dist": [0.04]},
    # 3 wejścia: najlepszy zysk/drawdown na train (ta sama reguła wyboru)
    #"3x  BB1.75 5/9 60/10/30 TP1.4 SL3": {"bb_pct": 0.0175, "mode": "stoch", "tp": 0.014, "sl": 0.030,
                                          #"weights": [0.6, 0.1, 0.3], "dist": [0.05, 0.09]},
    # DOPASOWANE DO ZMIENNOŚCI (mnożniki jednostki zmienności, a nie procenty) - najlepszy zysk/drawdown na train
    #"1x VOL tydz. k_bb5 TP6 SL2":    {"vol_window": 672, "k_bb": 5, "mode": "bb_only", "k_tp": 6, "k_sl": 2,
                                     #"weights": [1.0], "k_d": None},
    #"2x VOL mies. k_bb3 30/70 d4 TP7 SL8": {"vol_window": 2880, "k_bb": 3, "mode": "stoch", "k_tp": 7, "k_sl": 8,
                                            #"weights": [0.3, 0.7], "k_d": 4},
    # 1 wejście + ŁAGODNE dostrajanie TP/SL (alfa) - zarabiało w każdym roku na SOL (tune_tpsl, rok po roku)
    "1x BB1.9 TP2.0 SL4.6 + dostrajanie a1.0 2tyg": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
                                                    "weights": [1.0], "dist": [], "alpha": 1.0, "alpha_window": 1344},
    # 2 wejścia wybrane wg NAJGORSZEGO ROKU (stabilność rok po roku, analyze_years) - środek obszaru
    #"2x BB1.25 stoch 1% 90/10 TP2.2 SL1 (rok/rok)": {"bb_pct": 0.0125, "mode": "stoch", "tp": 0.022, "sl": 0.010,
                                                     #"weights": [0.9, 0.1], "dist": [0.01]},
}

COINS = ["SOL/USDT", "BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "BNB/USDT", "HYPE/USDT"]
START = "2023-11-05T00:00:00Z"
TEST_START = pd.Timestamp("2025-11-10", tz="UTC")


def download(symbol):
    """Pobiera 1m miesiąc po miesiącu, od najnowszego wstecz - każdy miesiąc od razu w bazie."""
    now = exchange.milliseconds()
    months = pd.date_range("2023-11-01", pd.Timestamp.now().strftime("%Y-%m-01"), freq="MS")[::-1]
    for m in months:
        print(f"=== {symbol}: od {m.date()}")
        load_candles(symbol, "1m", int(m.timestamp() * 1000), now)


def evaluate(symbol, m1=None, test_start=TEST_START):
    """Zwraca wiersz wyników dla każdej strategii na danym coinie."""
    if m1 is None:
        m1 = load_candles(symbol, "1m", exchange.parse8601(START), exchange.milliseconds())
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    randoms = _random_entries(op, n_seeds=2)
    units = {}

    def run(c, idx, side, price):
        """Symulacja jednej strategii - sztywne procenty, mnożniki zmienności albo łagodne dostrajanie (alfa)."""
        if "alpha" in c:
            w = c["alpha_window"]
            if w not in units:
                units[w] = vol_unit(s, w)
            ref = np.nanmedian(units[w][:split])            # typowa zmienność TEGO coina, tylko z train
            u = units[w][idx]
            mult = np.clip(np.where(np.isnan(u), 1.0, u / ref) ** c["alpha"], 0.5, 2.0)
            return simulate_avg(hi, lo, idx, side, price, 0, 0, c["weights"], c["dist"], "last",
                                tp_arr=c["tp"] * mult, sl_arr=c["sl"] * mult)
        if "vol_window" not in c:
            return simulate_avg(hi, lo, idx, side, price, c["tp"], c["sl"], c["weights"], c["dist"])
        u = units[c["vol_window"]][idx]
        ok = ~np.isnan(u)
        idx, side, price, u = idx[ok], side[ok], price[ok], u[ok]
        dist_arr = (c["k_d"] * u)[:, None] if c["k_d"] else None
        return simulate_avg(hi, lo, idx, side, price, 0, 0, c["weights"], [], "last",
                            tp_arr=c["k_tp"] * u, sl_arr=c["k_sl"] * u, dist_arr=dist_arr)

    rows = []
    for name, c in STRATEGIES.items():
        if "vol_window" in c:
            if c["vol_window"] not in units:
                units[c["vol_window"]] = vol_unit(s, c["vol_window"])
            sig = build_signals_vol(S, op, hi, lo, c["k_bb"] * units[c["vol_window"]], c["mode"])
        else:
            sig = build_signals(S, op, hi, lo, c["bb_pct"], c["mode"])
        e_idx, gross, filled, fills, win = run(c, *sig)
        tr = e_idx < split
        row = {"coin": symbol, "strategia": name}
        row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
        row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))

        # losowe wejścia z identycznym zarządzaniem pozycją na TYM coinie - punkt odniesienia
        g_tr, g_te = [], []
        for idx, side, price in randoms:
            e, g, *_ = run(c, idx, side, price)
            g_tr.append(g[e < split]); g_te.append(g[e >= split])
        row["train_edge_%"] = round(row.get("train_gross_%", np.nan) - np.concatenate(g_tr).mean() * 100, 3)
        row["test_edge_%"] = round(row.get("test_gross_%", np.nan) - np.concatenate(g_te).mean() * 100, 3)
        rows.append(row)
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        for coin in COINS:
            download(coin)
    else:
        wybrane = [a.upper() + "/USDT" for a in sys.argv[1:]] or COINS
        rows = []
        for coin in wybrane:
            print("liczę", coin, "...")
            rows.extend(evaluate(coin))
        res = pd.DataFrame(rows).set_index(["strategia", "coin"]).sort_index()
        cols = ["train_trades", "train_win_%", "train_edge_%", "train_net_%", "train_pnl_usd", "train_maxdd_usd",
                "test_trades", "test_win_%", "test_edge_%", "test_net_%", "test_pnl_usd", "test_maxdd_usd"]
        print(res[cols].to_string())
