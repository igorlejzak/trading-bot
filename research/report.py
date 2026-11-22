"""Raport graficzny do README: jak zachowuje się kapitał na tle ceny SOL dla 6 ustawień
(od najlepszego do najgorszego) i co by się działo z kontem 10 000 zł przy różnych dźwigniach.

Wybór 6 ustawień - WYŁĄCZNIE wg TRAIN (zysk / drawdown) z grida 2 wejść (grid_avg2.csv):
  najlepsze (= nasze wybrane), średnio dobre (75. percentyl), średnie (mediana),
  średnio słabe (25. percentyl), najgorsze + nasze wybrane 1 wejście dla porównania.

Dla każdego ustawienia jeden obrazek docs/report_X.png:
  lewa strona  - cena SOL; tło zielone/czerwone = miesiąc, w którym strategia zarobiła/straciła;
                 na drugiej osi krzywa kapitału (USD na pozycji 100 USD)
  prawa strona - konto 10 000 zł przy dźwigni 1x/2x/3x/5x/10x (procent składany, skala logarytmiczna)
Do tego docs/report_summary.md z tabelą liczb.

Uproszczenia dźwigni: każda transakcja angażuje cały kapitał konta x dźwignia; strata >= 100% konta = konto zerowe.
Nie liczymy likwidacji w trakcie transakcji ani opłat za finansowanie - w praktyce byłoby gorzej.
"""
import os

import numpy as np
import pandas as pd

from livemode import prepare_state
from grid import build_signals, STATE_COLS
from grid_avg import simulate_avg

FEE = 0.0005
START_PLN = 10_000
LEVERAGES = [1, 2, 3, 5, 10]
ONE_ENTRY = {"nazwa": "1 wejście (wybrane)", "bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
             "weights": [1.0], "dist": []}


def pick_six(grid_csv="grid_avg2.csv", min_trades=100):
    res = pd.read_csv(grid_csv, dtype={"dist_%": str, "weights_%": str})
    res = res[res["train_trades"] >= min_trades].copy()
    res["score"] = res["train_pnl_usd"] / res["train_maxdd_usd"].abs()
    res = res.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(res) - 1
    picks = [("2 wejścia: najlepsze (wybrane)", 0), ("2 wejścia: średnio dobre (75%)", int(n * 0.25)),
             ("2 wejścia: średnie (mediana)", int(n * 0.5)), ("2 wejścia: średnio słabe (25%)", int(n * 0.75)),
             ("2 wejścia: najgorsze", n)]
    out = []
    for name, i in picks:
        r = res.iloc[i]
        w = [float(x) / 100 for x in r["weights_%"].split("/")]
        out.append({"nazwa": name, "bb_pct": r["bb_%"] / 100, "mode": r["mode"], "tp": r["tp_%"] / 100,
                    "sl": r["sl_%"] / 100, "weights": w, "dist": [float(x) / 100 for x in r["dist_%"].split("/")]})
    return [out[0], ONE_ENTRY] + out[1:]


def label(c):
    if len(c["weights"]) == 1:
        return f"BB {c['bb_pct'] * 100:.2f}% {c['mode']}, TP {c['tp'] * 100:.1f}%, SL {c['sl'] * 100:.1f}%"
    w = "/".join(f"{x * 100:.0f}" for x in c["weights"])
    return (f"BB {c['bb_pct'] * 100:.2f}% {c['mode']}, dokładka {c['dist'][0] * 100:g}%, {w}, "
            f"TP {c['tp'] * 100:.1f}%, SL {c['sl'] * 100:.1f}%")


def account_curve(r, lev):
    """r = zwrot transakcji jako ułamek kapitału (po prowizjach). Zwraca saldo konta po każdej transakcji."""
    eq, bal = [], START_PLN
    for x in r:
        bal = max(bal * (1 + lev * x), 0.0)
        if bal < START_PLN * 0.01:          # poniżej 1% kapitału traktujemy konto jako wyzerowane
            bal = 0.0
        eq.append(bal)
    return np.array(eq)


def max_dd_pct(eq):
    eq = np.concatenate([[START_PLN], eq])
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def make_report(m1, test_start, grid_csv="grid_avg2.csv", out_dir="docs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    times = s["time"]
    daily = pd.Series(s["close"].to_numpy(), index=times).resample("D").last()

    rows = []
    for k, c in enumerate(pick_six(grid_csv)):
        sig = build_signals(S, op, hi, lo, c["bb_pct"], c["mode"])
        e_idx, gross, filled, fills, win = simulate_avg(hi, lo, *sig, c["tp"], c["sl"], c["weights"], c["dist"])
        r = gross - 2 * FEE * filled                     # zwrot jako ułamek kapitału pozycji
        t = pd.DataFrame({"time": times.to_numpy()[e_idx], "usd": r * 100})
        eq_usd = t["usd"].cumsum().to_numpy()
        monthly = t.set_index("time")["usd"].resample("MS").sum()
        n_month = t.set_index("time")["usd"].resample("MS").size()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 2]})
        # lewa: cena + miesiące zielone/czerwone + kapitał
        for m, v in monthly.items():
            if n_month[m] == 0:
                continue
            ax1.axvspan(m, m + pd.offsets.MonthBegin(1), color="green" if v > 0 else "red", alpha=0.12, lw=0)
        ax1.plot(daily.index, daily.values, color="black", lw=1, label="cena SOL")
        ax1.axvline(test_start, color="gray", ls="--", lw=1)
        ax1.set_ylabel("cena SOL (USD)")
        ax1b = ax1.twinx()
        ax1b.plot(t["time"], eq_usd, color="tab:blue", lw=1.6, label="kapitał strategii")
        ax1b.set_ylabel("zysk strategii (USD na pozycji 100 USD)", color="tab:blue")
        ax1.set_title(f"{c['nazwa']}\n{label(c)}   (tło: zielony = miesiąc na plusie, czerwony = na minusie)",
                      fontsize=10)

        # prawa: konto 10 000 zł przy dźwigniach
        lev_stats = {}
        for lev in LEVERAGES:
            acc = account_curve(r, lev)
            lev_stats[lev] = (acc[-1] if len(acc) else START_PLN, max_dd_pct(acc) if len(acc) else 0.0)
            end = "KONTO WYZEROWANE" if lev_stats[lev][0] == 0 else f"{lev_stats[lev][0]:,.0f} zł".replace(",", " ")
            ax2.plot(t["time"], np.where(acc > 0, acc, np.nan), lw=1.3,
                     label=f"{lev}x: {end}, max spadek {lev_stats[lev][1]:.0f}%")
        ax2.set_yscale("log")
        ax2.axhline(START_PLN, color="gray", lw=0.8)
        ax2.axvline(test_start, color="gray", ls="--", lw=1)
        ax2.set_title("Konto 10 000 zł przy różnej dźwigni (skala log.)", fontsize=10)
        ax2.legend(fontsize=8, loc="upper left")
        fig.autofmt_xdate()
        plt.tight_layout()
        png = os.path.join(out_dir, f"report_{k + 1}.png")
        plt.savefig(png, dpi=110)
        plt.close(fig)

        tr = t["time"] < test_start
        rows.append({"ustawienie": c["nazwa"], "parametry": label(c), "transakcji": len(t),
                     "win_%": round(float(win.mean() * 100), 1) if len(win) else 0,
                     "zysk_train_usd": round(t.loc[tr, "usd"].sum(), 1), "zysk_test_usd": round(t.loc[~tr, "usd"].sum(), 1),
                     "max_spadek_usd": round(float((eq_usd - np.maximum.accumulate(np.concatenate([[0], eq_usd]))[1:]).min()), 1),
                     "zyskowne_miesiace": f"{int((monthly[n_month > 0] > 0).sum())}/{int((n_month > 0).sum())}",
                     **{f"{lev}x_konto_zl": round(v[0]) for lev, v in lev_stats.items()},
                     **{f"{lev}x_max_spadek_%": round(v[1]) for lev, v in lev_stats.items()}})
        print(f"zapisano {png}")

    summ = pd.DataFrame(rows)
    with open(os.path.join(out_dir, "report_summary.md"), "w", encoding="utf-8") as f:
        try:
            f.write(summ.to_markdown(index=False))
        except ImportError:
            summ.to_csv(os.path.join(out_dir, "report_summary.csv"), sep=";", decimal=",", index=False)
            f.write("brak biblioteki tabulate - tabela zapisana w report_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(summ.to_string(index=False))
    return summ
