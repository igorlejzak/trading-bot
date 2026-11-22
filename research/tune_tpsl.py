"""Łagodne dostrajanie TYLKO TP i SL (i ewentualnie odstępu dokładki) do zmienności.

Wejście zostaje bez zmian (BB i StochRSI już same się dopasowują). Zmieniamy tylko procenty TP/SL:

    TP = TP_bazowe * (zmienność_teraz / zmienność_typowa) ** alfa

alfa = 0    -> zwykłe sztywne procenty (to co mamy)
alfa = 0.5  -> połowiczne dostrajanie: gdy zmienność jest 2x większa, TP rośnie ~1.4x
alfa = 1    -> pełne dostrajanie: TP rośnie tyle samo co zmienność
Mnożnik ograniczamy do 0.5-2.0, żeby nigdy nie odjechał za daleko.
"Zmienność typowa" = mediana z okresu TRAIN, więc test nie przecieka do ustawień.

Porównanie jest "w parach": dla tych samych bazowych TP/SL sprawdzamy, czy alfa > 0 poprawia wynik względem alfa = 0.
"""
import itertools
import time

import numpy as np
import pandas as pd

from livemode import prepare_state
from grid import build_signals, metrics, STATE_COLS
from grid_avg import simulate_avg, yearly
from grid_vol import vol_unit

BASES = {
    # nazwa: wejście + bazowe zarządzanie (najlepsze sztywne ustawienia z gridów)
    "1x": {"bb_pct": 0.019, "mode": "bb_only", "weights": [1.0], "dist": [],
           "tp": [0.016, 0.018, 0.020, 0.022, 0.024], "sl": [0.036, 0.041, 0.046, 0.051, 0.056]},
    "2x": {"bb_pct": 0.015, "mode": "stoch", "weights": [0.3, 0.7], "dist": [0.04],
           "tp": [0.018, 0.020, 0.022, 0.024, 0.026], "sl": [0.015, 0.0175, 0.020, 0.0225, 0.025]},
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOWS = [672, 1344, 2880]
CLIP = (0.5, 2.0)


def run_tune(m1, test_start, bases=BASES, out_path="tune_tpsl.csv", fee=0.0005):
    t0 = time.time()
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    years = s["time"].dt.year.to_numpy()
    units = {w: vol_unit(s, w) for w in WINDOWS}
    ref = {w: np.nanmedian(units[w][:split]) for w in WINDOWS}      # typowa zmienność - tylko z TRAIN
    del s

    rows = []
    for name, b in bases.items():
        sig_idx, sig_side, sig_price = build_signals(S, op, hi, lo, b["bb_pct"], b["mode"])
        for w in WINDOWS:
            u = units[w][sig_idx]
            ratio = np.where(np.isnan(u), 1.0, u / ref[w])
            for alpha, scale_dist in itertools.product(ALPHAS, [False, True] if b["dist"] else [False]):
                if alpha == 0 and scale_dist:
                    continue
                mult = np.clip(ratio ** alpha, *CLIP)
                dist_arr = (np.array(b["dist"])[None, :] * mult[:, None]) if (b["dist"] and scale_dist) else None
                for tp, sl in itertools.product(b["tp"], b["sl"]):
                    e_idx, gross, filled, fills, win = simulate_avg(
                        hi, lo, sig_idx, sig_side, sig_price, 0, 0, b["weights"], b["dist"], "last",
                        tp_arr=tp * mult, sl_arr=sl * mult, dist_arr=dist_arr)
                    tr = e_idx < split
                    row = {"strategia": name, "okno": w, "alfa": alpha, "skaluj_odstep": scale_dist,
                           "tp_baza_%": round(tp * 100, 2), "sl_baza_%": round(sl * 100, 2)}
                    row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
                    row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))
                    row.update(yearly(e_idx, gross, filled, years))
                    rows.append(row)
        print(f"  {name} gotowe ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    print(f"zapisano {out_path} ({len(res)} wierszy)")
    return res


def analyze_tune(res):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    key = ["strategia", "tp_baza_%", "sl_baza_%"]
    fixed = res[res["alfa"] == 0].groupby(key)[["train_pnl_usd", "test_pnl_usd", "train_maxdd_usd", "test_maxdd_usd"]].first()
    fixed.columns = [c + "_sztywne" for c in fixed.columns]
    r = res[res["alfa"] > 0].join(fixed, on=key)
    r["lepiej_train"] = r["train_pnl_usd"] > r["train_pnl_usd_sztywne"]
    r["lepiej_test"] = r["test_pnl_usd"] > r["test_pnl_usd_sztywne"]
    r["lepiej_oba"] = r["lepiej_train"] & r["lepiej_test"]
    r["d_train"] = r["train_pnl_usd"] - r["train_pnl_usd_sztywne"]
    r["d_test"] = r["test_pnl_usd"] - r["test_pnl_usd_sztywne"]
    r["d_dd_test"] = r["test_maxdd_usd"] - r["test_maxdd_usd_sztywne"]     # > 0 = mniejszy drawdown

    print("\n=== Czy dostrajanie TP/SL pomaga? (porównanie z TYMI SAMYMI bazowymi TP/SL bez dostrajania) ===")
    print("lepiej_* = % par, w których dostrajanie dało więcej USD; d_* = średnia różnica w USD")
    print(r.groupby(["strategia", "okno", "skaluj_odstep", "alfa"]).agg(
        par=("lepiej_oba", "size"),
        lepiej_train_pct=("lepiej_train", lambda x: round(x.mean() * 100)),
        lepiej_test_pct=("lepiej_test", lambda x: round(x.mean() * 100)),
        lepiej_oba_pct=("lepiej_oba", lambda x: round(x.mean() * 100)),
        d_train_usd=("d_train", "mean"), d_test_usd=("d_test", "mean"),
        d_dd_test_usd=("d_dd_test", "mean")).round(1).to_string())


def analyze_tune_years(res):
    """Rok po roku: czy dostrajanie TP/SL daje wynik na plusie w większej liczbie lat niż sztywne procenty."""
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    ycols = ["pnl_2024", "pnl_2025", "pnl_2026"]
    print("\n=== ROK PO ROKU: średnio po wszystkich bazowych TP/SL (sztywne = alfa 0) ===")
    print(res.groupby(["strategia", "okno", "skaluj_odstep", "alfa"]).agg(
        **{c: (c, "mean") for c in ycols},
        wszystkie_3_lata_na_plusie_pct=("lata_na_plusie", lambda x: round((x == 3).mean() * 100)),
        najgorszy_rok=("najgorszy_rok_usd", "mean"), razem_3_lata=("pnl_3lata_usd", "mean")).round(1).to_string())
