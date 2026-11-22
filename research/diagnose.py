"""Kiedy strategia traci? Najgorsze okresy, serie strat i stan rynku SOL w tych momentach.

Dla wybranej strategii liczy wszystkie transakcje na całych 3 latach i pokazuje:
1. największe spadki kapitału (od szczytu do dołka) z datami,
2. najdłuższe serie strat z rzędu,
3. wyniki miesiąc po miesiącu obok zachowania SOL (zmiana ceny, zmienność, cena),
4. wynik zależnie od warunków rynku w chwili wejścia: zmienność, trend z ostatnich 7 dni,
   oraz czy gramy z trendem, czy przeciw niemu,
5. wykres diagnose.png: cena SOL + krzywa kapitału z zaznaczonymi najgorszymi okresami.
"""
import numpy as np
import pandas as pd

from livemode import prepare_state
from grid import build_signals, STATE_COLS
from grid_avg import simulate_avg
from grid_vol import vol_unit

FEE = 0.0005
STRATS = {
    "1x BB1.9 TP2.0 SL4.6": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046, "weights": [1.0], "dist": []},
    "2x BB1.5 4% 30/70 TP2.2 SL2": {"bb_pct": 0.015, "mode": "stoch", "tp": 0.022, "sl": 0.020,
                                     "weights": [0.3, 0.7], "dist": [0.04]},
}


def trades_for(s, S, op, hi, lo, c):
    sig = build_signals(S, op, hi, lo, c["bb_pct"], c["mode"])
    e_idx, gross, filled, fills, win = simulate_avg(hi, lo, *sig, c["tp"], c["sl"], c["weights"], c["dist"])
    side = sig[1][np.searchsorted(sig[0], e_idx)]
    t = pd.DataFrame({"i": e_idx, "time": s["time"].to_numpy()[e_idx], "side": np.where(side == 1, "long", "short"),
                      "fills": fills, "win": win, "pnl_usd": (gross - 2 * FEE * filled) * 100})
    return t


def drawdowns(t, top=5):
    eq = t["pnl_usd"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    out, i = [], 0
    n = len(t)
    while i < n:
        if dd.iloc[i] < 0:
            start = i - 1 if i > 0 else 0
            j = i
            while j < n and dd.iloc[j] < 0:
                j += 1
            trough = dd.iloc[i:j].idxmin()
            seg = t.iloc[i:j]
            out.append({"od": t["time"].iloc[start], "dołek": t["time"].iloc[trough],
                        "odrobione": t["time"].iloc[j] if j < n else "nie odrobione",
                        "spadek_usd": round(dd.iloc[trough], 1), "transakcji": j - start,
                        "strat": int((~seg["win"]).sum())})
            i = j
        else:
            i += 1
    return pd.DataFrame(out).sort_values("spadek_usd").head(top)


def loss_streaks(t, top=5):
    out, run = [], []
    for k, w in enumerate(t["win"]):
        if not w:
            run.append(k)
        else:
            if run:
                out.append(run)
            run = []
    if run:
        out.append(run)
    rows = [{"od": t["time"].iloc[r[0]], "do": t["time"].iloc[r[-1]], "strat_z_rzedu": len(r),
             "suma_usd": round(t["pnl_usd"].iloc[r].sum(), 1)} for r in out]
    return pd.DataFrame(rows).sort_values(["strat_z_rzedu", "suma_usd"], ascending=[False, True]).head(top)


def diagnose(m1, test_start, strats=STRATS, png="diagnose.png"):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo, cl = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low", "close"])
    vol = vol_unit(s, 2880) * 100                       # zmienność 15m z ostatniego miesiąca, w %
    lag = 7 * 1440
    ret7 = np.full(len(cl), np.nan)
    ret7[lag:] = (cl[lag:] / cl[:-lag] - 1) * 100       # zmiana ceny SOL w ostatnich 7 dniach, w %

    # miesięczne zachowanie SOL
    mdf = pd.DataFrame({"time": s["time"], "close": cl, "vol": vol}).set_index("time")
    sol_m = pd.DataFrame({"sol_zmiana_%": mdf["close"].resample("MS").last().pct_change() * 100,
                          "sol_cena": mdf["close"].resample("MS").mean(),
                          "zmiennosc_%": mdf["vol"].resample("MS").mean()}).round(2)

    all_t = {}
    for name, c in strats.items():
        t = trades_for(s, S, op, hi, lo, c)
        t["vol"] = vol[t["i"]]
        t["trend7_%"] = ret7[t["i"]]
        t["przeciw_trendowi"] = ((t["side"] == "short") & (t["trend7_%"] > 0)) | ((t["side"] == "long") & (t["trend7_%"] < 0))
        all_t[name] = t
        print(f"\n######## {name}: {len(t)} transakcji, wynik {t['pnl_usd'].sum():.1f} USD ########")

        print("\n--- 5 największych spadków kapitału ---")
        print(drawdowns(t).to_string(index=False))
        print("\n--- 5 najdłuższych serii strat ---")
        print(loss_streaks(t).to_string(index=False))

        print("\n--- miesiąc po miesiącu (najgorsze 8) ---")
        tm = t.set_index("time").resample("MS").agg(transakcji=("pnl_usd", "size"), wynik_usd=("pnl_usd", "sum"),
                                                     win_pct=("win", lambda x: round(x.mean() * 100) if len(x) else np.nan))
        tm = tm.join(sol_m)
        print(tm.sort_values("wynik_usd").head(8).round(1).to_string())

        print("\n--- wynik wg warunków rynku w chwili wejścia ---")
        t["zmiennosc_grupa"] = pd.qcut(t["vol"], 3, labels=["niska", "średnia", "wysoka"])
        t["trend_grupa"] = pd.qcut(t["trend7_%"].abs(), 3, labels=["boczny", "umiarkowany", "silny"])
        for g in ["zmiennosc_grupa", "trend_grupa", "przeciw_trendowi", "side"]:
            print(t.groupby(g, observed=True).agg(transakcji=("pnl_usd", "size"), wynik_usd=("pnl_usd", "sum"),
                                                  srednio_usd=("pnl_usd", "mean"),
                                                  win_pct=("win", lambda x: round(x.mean() * 100))).round(2).to_string())
            print()

    # wykres
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(all_t) + 1, 1, figsize=(14, 4 + 3 * len(all_t)), sharex=True)
        daily = mdf["close"].resample("D").last()
        axes[0].plot(daily.index, daily.values, color="black", lw=1)
        axes[0].set_title("Cena SOL/USDT")
        axes[0].axvline(test_start, color="gray", ls="--")
        for ax, (name, t) in zip(axes[1:], all_t.items()):
            eq = t["pnl_usd"].cumsum().to_numpy()
            ax.plot(t["time"], eq, lw=1.2)
            ax.fill_between(t["time"], eq, np.maximum.accumulate(eq), color="red", alpha=0.3)
            ax.set_title(f"{name}: kapitał (USD), na czerwono spadki od szczytu")
            ax.axvline(test_start, color="gray", ls="--")
            for _, r in drawdowns(t, 3).iterrows():
                axes[0].axvspan(r["od"], r["dołek"], color="red", alpha=0.08)
        plt.tight_layout()
        plt.savefig(png, dpi=110)
        print(f"zapisano wykres {png}")
    except ImportError:
        print("brak matplotlib - pip install matplotlib, żeby dostać wykres")
    return all_t
