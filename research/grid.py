"""Grid search dla strategii "jak bot live" na świeczkach 1m.

Pomysł na szybkość (trzy poziomy):
1. Stan z zamkniętych świeczek 15m liczymy RAZ dla całej historii.
2. Dla każdego ustawienia SYGNAŁU (odległość od BB + tryb StochRSI/RSI) wyznaczamy raz listę minut,
   w których bot mógłby wejść, i cenę wejścia (wektorowa bisekcja na wszystkich kandydatach naraz).
3. Dla każdego ustawienia ZARZĄDZANIA pozycją (TP, SL, liczba wejść, podział kapitału, odstęp)
   symulujemy już tylko skoki "od wejścia do następnego zdarzenia" - nie idziemy minuta po minucie.
"""
import itertools
import time

import numpy as np
import pandas as pd

from livemode import prepare_state, BB_WINDOW, BB_DEV, RSI_WINDOW, SMOOTH

# ------------------------------------------------------------------ siatka parametrów
GRID = {
    "bb_pct": [round(float(x), 4) for x in np.arange(0.003, 0.0201, 0.001)],   # 0.3% ... 2.0% co 0.1 - jak daleko za BB
    "mode": ["stoch", "stoch_rsi", "rsi", "bb_only"],                    # warunek oscylatora
    "tp": [round(float(x), 4) for x in np.arange(0.010, 0.0301, 0.002)],       # 1.0% ... 3.0% co 0.2
    "sl": [round(float(x), 4) for x in np.arange(0.020, 0.0501, 0.002)],       # 2.0% ... 5.0% co 0.2
    # (nazwa, wagi kapitału na kolejne wejścia - suma = 1)
    "entries": [
        ("1x", [1.0]),
        ("2x rowno", [1 / 2, 1 / 2]),
        ("2x podwajanie", [1 / 3, 2 / 3]),
        ("3x rowno", [1 / 3, 1 / 3, 1 / 3]),
        ("3x podwajanie", [1 / 7, 2 / 7, 4 / 7]),
    ],
    "dist": [0.01, 0.02],          # odstęp między kolejnymi wejściami (liczony od pierwszego wejścia)
}

# mała siatka do szybkiego testu, czy wszystko działa (kilkanaście sekund)
GRID_SMALL = {
    "bb_pct": [0.005, 0.01, 0.015],
    "mode": ["stoch", "bb_only"],
    "tp": [0.01, 0.02],
    "sl": [0.03, 0.05],
    "entries": [("1x", [1.0]), ("3x podwajanie", [1 / 7, 2 / 7, 4 / 7])],
    "dist": [0.02],
}

# progi oscylatorów dla każdego trybu: (stoch_low, stoch_high, rsi_long, rsi_short); None = nie używamy
MODES = {
    "stoch": (20, 80, None, None),
    "stoch_rsi": (20, 80, 40, 60),
    "rsi": (None, None, 30, 70),
    "bb_only": (None, None, None, None),
}

CAPITAL = 100.0          # USD na całą pozycję (wszystkie wejścia razem)
FEE_TAKER = 0.0005       # 0.05% na stronę - zlecenia market
FEE_MAKER = 0.0002       # 0.02% na stronę - zlecenia limit
MIN_TRADES = 100         # mniej transakcji na train = nie bierzemy do rankingu


# ------------------------------------------------------------------ wskaźniki "jak w bocie"
STATE_COLS = ["s_sum", "s_sq", "last_close", "ag", "al", "rsi_min", "rsi_max", "srsi1", "srsi2", "k1", "k2"]


def ind(S, p):
    """BB górna/dolna, StochRSI K i D oraz RSI dla aktualnej ceny p (tablice numpy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (S["s_sum"] + p) / BB_WINDOW
        std = np.sqrt(np.maximum((S["s_sq"] + p * p) / BB_WINDOW - mean * mean, 0))
        a = 1 / RSI_WINDOW
        d = p - S["last_close"]
        ag = (1 - a) * S["ag"] + a * np.maximum(d, 0)
        al = (1 - a) * S["al"] + a * np.maximum(-d, 0)
        rsi = 100 - 100 / (1 + ag / al)
        lo, hi = np.minimum(S["rsi_min"], rsi), np.maximum(S["rsi_max"], rsi)
        srsi = (rsi - lo) / (hi - lo)
        k = (S["srsi2"] + S["srsi1"] + srsi) / SMOOTH
        dd = (S["k2"] + S["k1"] + k) / SMOOTH
    return mean + BB_DEV * std, mean - BB_DEV * std, k * 100, dd * 100, rsi


def cond_short(S, p, bb_pct, mode):
    st_lo, st_hi, rsi_long, rsi_short = MODES[mode]
    bb_h, _, k, d, rsi = ind(S, p)
    with np.errstate(invalid="ignore"):
        ok = p > bb_h * (1 + bb_pct)
        if st_hi is not None:
            ok &= (k > st_hi) & (d > st_hi)
        if rsi_short is not None:
            ok &= rsi >= rsi_short
    return ok


def cond_long(S, p, bb_pct, mode):
    st_lo, st_hi, rsi_long, rsi_short = MODES[mode]
    _, bb_l, k, d, rsi = ind(S, p)
    with np.errstate(invalid="ignore"):
        ok = p < bb_l * (1 - bb_pct)
        if st_lo is not None:
            ok &= (k < st_lo) & (d < st_lo)
        if rsi_long is not None:
            ok &= rsi <= rsi_long
    return ok


def first_prices(cond, S_sub, start, end, steps=30):
    """Wektorowa bisekcja: dla każdej minuty-kandydata pierwsza cena między start a end,
    przy której warunek jest spełniony. Warunek jest monotoniczny w cenie."""
    start, end = start.copy(), end.copy()
    at_open = cond(S_sub, start)
    for _ in range(steps):
        mid = (start + end) / 2
        ok = cond(S_sub, mid)
        end = np.where(ok, mid, end)
        start = np.where(ok, start, mid)
    return np.where(at_open, start, end)   # jeśli warunek był już na open -> wejście po open


def build_signals(S, op, hi, lo, bb_pct, mode):
    """Zwraca (indeksy minut, strona +1 long / -1 short, cena wejścia) dla jednego ustawienia sygnału."""
    idx_s = np.flatnonzero(cond_short(S, hi, bb_pct, mode))
    idx_l = np.flatnonzero(cond_long(S, lo, bb_pct, mode))
    idx_l = idx_l[~np.isin(idx_l, idx_s)]                 # jak w bocie: short sprawdzany pierwszy

    def sub(idx):
        return {c: S[c][idx] for c in STATE_COLS}

    price_s = first_prices(lambda X, p: cond_short(X, p, bb_pct, mode), sub(idx_s), op[idx_s], hi[idx_s])
    price_l = first_prices(lambda X, p: cond_long(X, p, bb_pct, mode), sub(idx_l), op[idx_l], lo[idx_l])

    idx = np.concatenate([idx_s, idx_l])
    side = np.concatenate([-np.ones(len(idx_s), dtype=np.int8), np.ones(len(idx_l), dtype=np.int8)])
    price = np.concatenate([price_s, price_l])
    order = np.argsort(idx, kind="stable")
    return idx[order], side[order], price[order]


# ------------------------------------------------------------------ symulacja pozycji
def first_hit(hi, lo, start, up, down, n):
    """Pierwsza minuta >= start, w której high >= up albo low <= down. -1 gdy brak do końca danych."""
    size = 64
    while start < n:
        end = start + size
        m = (hi[start:end] >= up) | (lo[start:end] <= down)
        k = int(m.argmax())
        if m[k]:
            return start + k
        start = end
        if size < 65536:
            size *= 4
    return -1


def simulate(hi, lo, sig_idx, sig_side, sig_price, tp, sl, weights, dist):
    """Symuluje wszystkie transakcje dla jednego ustawienia. Zwraca tablice:
    indeks minuty wejścia, wynik brutto (ułamek kapitału), wypełniona część kapitału, liczba wejść, czy TP.

    Zasady: jedna pozycja naraz. Kolejne wejścia to zlecenia limit w odstępach `dist` od pierwszego wejścia.
    TP i SL liczone od ŚREDNIEJ ceny wejścia. Gdy w jednej minucie pada i zdarzenie niekorzystne
    (dokładka / SL), i TP - zakładamy, że niekorzystne było pierwsze (ostrożnie)."""
    n, n_sig, n_w = len(hi), len(sig_idx), len(weights)
    e_idx, e_gross, e_filled, e_fills, e_win = [], [], [], [], []
    s = 0
    while s < n_sig:
        i = int(sig_idx[s])
        side = int(sig_side[s])
        p1 = float(sig_price[s])
        filled = weights[0]
        qty = weights[0] / p1
        k = 1                      # ile wejść już wypełnionych
        j = i
        allow_tp = False           # na minucie wejścia nie liczymy TP
        result = None              # cena wyjścia

        while True:
            avg = filled / qty
            if side == 1:          # LONG: niekorzystnie = w dół
                tp_l = avg * (1 + tp)
                sl_l = avg * (1 - sl)
                nxt = p1 * (1 - dist * k) if k < n_w else -1.0
                if lo[j] <= sl_l and sl_l >= nxt:
                    result, win = sl_l, False
                    break
                if k < n_w and lo[j] <= nxt:           # dokładka
                    filled += weights[k]
                    qty += weights[k] / nxt
                    k += 1
                    allow_tp = False
                    continue                            # ta sama minuta: może kolejna dokładka albo SL
                if allow_tp and hi[j] >= tp_l:
                    result, win = tp_l, True
                    break
                j = first_hit(hi, lo, j + 1, tp_l, max(sl_l, nxt), n)
            else:                  # SHORT: niekorzystnie = w górę
                tp_l = avg * (1 - tp)
                sl_l = avg * (1 + sl)
                nxt = p1 * (1 + dist * k) if k < n_w else np.inf
                if hi[j] >= sl_l and sl_l <= nxt:
                    result, win = sl_l, False
                    break
                if k < n_w and hi[j] >= nxt:
                    filled += weights[k]
                    qty += weights[k] / nxt
                    k += 1
                    allow_tp = False
                    continue
                if allow_tp and lo[j] <= tp_l:
                    result, win = tp_l, True
                    break
                j = first_hit(hi, lo, j + 1, min(sl_l, nxt), tp_l, n)
            if j == -1:
                break               # pozycja otwarta do końca danych - pomijamy
            allow_tp = True

        if result is None:
            break
        gross = (qty * result - filled) if side == 1 else (filled - qty * result)
        e_idx.append(i)
        e_gross.append(gross)
        e_filled.append(filled)
        e_fills.append(k)
        e_win.append(win)
        s = int(np.searchsorted(sig_idx, j, side="right"))   # następny sygnał po zamknięciu

    return (np.array(e_idx, dtype=np.int64), np.array(e_gross), np.array(e_filled),
            np.array(e_fills), np.array(e_win, dtype=bool))


# ------------------------------------------------------------------ metryki
def metrics(gross, filled, fills, win, prefix):
    n = len(gross)
    if n == 0:
        return {f"{prefix}_trades": 0}
    net_t = gross - 2 * FEE_TAKER * filled          # po prowizji taker (ułamek kapitału)
    net_m = gross - 2 * FEE_MAKER * filled          # po prowizji maker
    equity = np.cumsum(net_t) * CAPITAL
    dd = float((equity - np.maximum.accumulate(equity)).min())
    sd = net_t.std(ddof=1) if n > 1 else 0.0
    return {
        f"{prefix}_trades": n,
        f"{prefix}_win_%": round(win.mean() * 100, 1),
        f"{prefix}_gross_%": round(gross.mean() * 100, 3),      # średnio na transakcję PRZED prowizją
        f"{prefix}_net_%": round(net_t.mean() * 100, 3),        # po prowizji 0.05%/stronę
        f"{prefix}_net_maker_%": round(net_m.mean() * 100, 3),  # po prowizji 0.02%/stronę
        f"{prefix}_pnl_usd": round(float(net_t.sum()) * CAPITAL, 1),
        f"{prefix}_tstat": round(float(net_t.mean() / sd * np.sqrt(n)), 2) if sd > 0 else 0.0,
        f"{prefix}_maxdd_usd": round(dd, 1),
        f"{prefix}_avg_fills": round(float(fills.mean()), 2),
    }


def random_baseline(op, hi, lo, split, grid, variants, seeds=(0, 1, 2)):
    """Losowe wejścia (losowa minuta, losowa strona, cena = open) przepuszczone przez TEN SAM silnik.
    Zwraca średni wynik brutto dla każdego ustawienia zarządzania pozycją - punkt odniesienia "zero umiejętności".
    Kilka losowań (seeds) łączymy razem, żeby punkt odniesienia był mniej zaszumiony."""
    n = len(op)
    randoms = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(2000, n), size=n // 30, replace=False))   # średnio co 30 minut
        side = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx))
        randoms.append((idx, side, op[idx]))

    base = {}
    for tp, sl, (ename, w, dist) in itertools.product(grid["tp"], grid["sl"], variants):
        g_train, g_test = [], []
        for idx, side, price in randoms:
            e_idx, gross, *_ = simulate(hi, lo, idx, side, price, tp, sl, w, dist)
            g_train.append(gross[e_idx < split])
            g_test.append(gross[e_idx >= split])
        g_train, g_test = np.concatenate(g_train), np.concatenate(g_test)
        base[(tp, sl, ename, dist)] = (float(g_train.mean()) * 100 if len(g_train) else 0.0,
                                       float(g_test.mean()) * 100 if len(g_test) else 0.0)
    return base


# ------------------------------------------------------------------ główna pętla
def entry_variants(grid):
    """Lista (nazwa, wagi, odstęp). Dla jednego wejścia odstęp nie ma znaczenia."""
    out = []
    for name, w in grid["entries"]:
        if len(w) == 1:
            out.append((name, w, 0.0))
        else:
            out.extend((name, w, d) for d in grid["dist"])
    return out


def run_grid(m1, test_start, grid=GRID, out_path="grid_results.csv"):
    """m1 = świeczki 1m z load_candles. test_start = pd.Timestamp początku zbioru testowego."""
    t0 = time.time()
    print("przygotowuję stan z 15m...")
    s = prepare_state(m1).reset_index(drop=True)
    S = {c: s[c].to_numpy(dtype=np.float64) for c in STATE_COLS}
    op, hi, lo = (s[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low"])
    split = int(np.searchsorted(s["ts"].to_numpy(), int(test_start.timestamp() * 1000)))
    del s

    variants = entry_variants(grid)
    signal_cfgs = list(itertools.product(grid["bb_pct"], grid["mode"]))
    total = len(signal_cfgs) * len(grid["tp"]) * len(grid["sl"]) * len(variants)
    print(f"kombinacji: {total}  (sygnałów: {len(signal_cfgs)}, zarządzania pozycją: {total // len(signal_cfgs)})")

    print("liczę losowy punkt odniesienia...")
    base = random_baseline(op, hi, lo, split, grid, variants)
    b = np.array(list(base.values()))
    print(f"  gotowe ({time.time() - t0:.0f}s). Losowe wejście średnio brutto: "
          f"train {b[:, 0].mean():+.3f}%, test {b[:, 1].mean():+.3f}% na transakcję (powinno być blisko 0)")

    rows, done = [], 0
    for bb_pct, mode in signal_cfgs:
        sig_idx, sig_side, sig_price = build_signals(S, op, hi, lo, bb_pct, mode)
        for tp, sl, (ename, w, dist) in itertools.product(grid["tp"], grid["sl"], variants):
            e_idx, gross, filled, fills, win = simulate(hi, lo, sig_idx, sig_side, sig_price, tp, sl, w, dist)
            tr = e_idx < split
            row = {"bb_%": bb_pct * 100, "mode": mode, "tp_%": tp * 100, "sl_%": sl * 100,
                   "entries": ename, "dist_%": dist * 100}
            row.update(metrics(gross[tr], filled[tr], fills[tr], win[tr], "train"))
            row.update(metrics(gross[~tr], filled[~tr], fills[~tr], win[~tr], "test"))
            b_train, b_test = base[(tp, sl, ename, dist)]
            # przewaga nad losowym wejściem z tym samym TP/SL/siatką (w % kapitału na transakcję, przed prowizją)
            row["train_edge_%"] = round(row.get("train_gross_%", np.nan) - b_train, 3)
            row["test_edge_%"] = round(row.get("test_gross_%", np.nan) - b_test, 3)
            rows.append(row)
        done += total // len(signal_cfgs)
        el = time.time() - t0
        print(f"  BB {bb_pct * 100:.2f}% {mode:10s} sygnałów: {len(sig_idx):6d}   "
              f"{done}/{total}   {el:5.0f}s, zostało ~{el / done * (total - done):4.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)
    print(f"zapisano {out_path} ({len(res)} wierszy, {time.time() - t0:.0f}s)")
    return res


def show_top(res, n=20, min_trades=MIN_TRADES):
    """Ranking po t-stat na TRAIN (wynik po prowizji taker). Kolumny test = jedno uczciwe sprawdzenie."""
    ok = res[res["train_trades"] >= min_trades].sort_values("train_tstat", ascending=False)
    cols = ["bb_%", "mode", "tp_%", "sl_%", "entries", "dist_%",
            "train_trades", "train_win_%", "train_edge_%", "train_net_%", "train_pnl_usd", "train_tstat",
            "train_maxdd_usd", "test_trades", "test_win_%", "test_edge_%", "test_net_%", "test_pnl_usd"]
    print(f"\nTOP {n} wg train_tstat (min {min_trades} transakcji na train):")
    print(ok[cols].head(n).to_string(index=False))
    return ok


def show_marginals(res, min_trades=MIN_TRADES):
    """Średni wynik (po prowizji) dla każdej wartości parametru, uśredniony po wszystkich pozostałych.
    To jest odporniejsze niż patrzenie na jeden najlepszy wiersz: pokazuje, KTÓRE parametry w ogóle mają znaczenie."""
    ok = res[res["train_trades"] >= min_trades]
    for col in ["bb_%", "mode", "tp_%", "sl_%", "entries", "dist_%"]:
        t = ok.groupby(col).agg(kombinacji=("train_net_%", "size"),
                                train_net=("train_net_%", "mean"), test_net=("test_net_%", "mean"),
                                train_edge=("train_edge_%", "mean"), test_edge=("test_edge_%", "mean"),
                                train_trades=("train_trades", "mean")).round(3)
        print(f"\n--- średnio wg {col} ---")
        print(t.to_string())