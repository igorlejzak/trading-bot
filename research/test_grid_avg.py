"""Testy uśredniania na ręcznie ułożonych świeczkach (pytest test_grid_avg.py)."""
import numpy as np
import pytest

from grid import simulate
from grid_avg import simulate_avg


def run(hi, lo, side, price, tp, sl, weights, dists, ref="last"):
    hi, lo = np.array(hi, dtype=float), np.array(lo, dtype=float)
    return simulate_avg(hi, lo, np.array([0]), np.array([side], dtype=np.int8), np.array([float(price)]),
                        tp, sl, weights, dists, ref)


def test_no_sl_before_second_entry():
    # long 100, dołożenie na 97 (3%). Cena spada do 98 (2%) - BEZ SL, więc nie ma straty; potem TP od 100
    idx, gross, filled, fills, win = run([100.2, 99, 101.1], [100, 98, 99.5], 1, 100, 0.01, 0.02, [0.5, 0.5], [0.03])
    assert win[0] and fills[0] == 1 and gross[0] == pytest.approx(0.5 * 0.01)


def test_average_then_tp_from_average():
    w = [0.3, 0.7]
    avg = 1 / (w[0] / 100 + w[1] / 97)
    idx, gross, filled, fills, win = run([100.2, 98, avg * 1.011], [100, 96.9, 97.5], 1, 100, 0.01, 0.02, w, [0.03])
    assert fills[0] == 2 and win[0] and gross[0] == pytest.approx(0.01)


def test_sl_from_last_entry():
    # dołożenie na 97, SL 2% od 97 = 95.06
    idx, gross, filled, fills, win = run([100.2, 98, 97.5], [100, 96.9, 95.0], 1, 100, 0.01, 0.02, [0.5, 0.5], [0.03])
    avg = 1 / (0.5 / 100 + 0.5 / 97)
    assert fills[0] == 2 and not win[0]
    assert gross[0] == pytest.approx((95.06 - avg) / avg)


def test_sl_from_average():
    avg = 1 / (0.5 / 100 + 0.5 / 97)
    sl_l = avg * 0.98
    idx, gross, filled, fills, win = run([100.2, 98, 97.5], [100, 96.9, sl_l - 0.01], 1, 100, 0.01, 0.02,
                                         [0.5, 0.5], [0.03], ref="avg")
    assert not win[0] and gross[0] == pytest.approx(-0.02)


def test_sl_same_minute_as_fill_is_counted():
    # w minucie dołożenia cena od razu leci przez SL -> strata (ostrożnie)
    idx, gross, filled, fills, win = run([100.2, 99], [100, 94], 1, 100, 0.01, 0.02, [0.5, 0.5], [0.03])
    assert fills[0] == 2 and not win[0]


def test_short_two_entries_tp():
    w = [0.5, 0.5]
    avg = 1 / (w[0] / 100 + w[1] / 102)
    idx, gross, filled, fills, win = run([100.5, 102.2, 101.5], [100, 101, avg * 0.989], -1, 100, 0.01, 0.03, w, [0.02])
    assert fills[0] == 2 and win[0] and gross[0] == pytest.approx(0.01)


def test_three_entries():
    w = [1 / 7, 2 / 7, 4 / 7]
    idx, gross, filled, fills, win = run([100.2, 99, 98, 99.9], [100, 97.9, 95.9, 97], 1, 100, 0.01, 0.05, w, [0.02, 0.04])
    assert fills[0] == 3 and win[0] and gross[0] == pytest.approx(0.01)


def test_single_entry_matches_old_engine():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 20000)))
    hi, lo = c * 1.001, c * 0.999
    idx = np.sort(rng.choice(np.arange(100, 19000), 300, replace=False))
    side = rng.choice(np.array([-1, 1], dtype=np.int8), 300)
    a = simulate(hi, lo, idx, side, c[idx], 0.01, 0.02, [1.0], 0.0)
    b = simulate_avg(hi, lo, idx, side, c[idx], 0.01, 0.02, [1.0], [], "last")
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_smaller_tp_after_second_entry():
    # TP 1% przed dołożeniem, 0.5% po dołożeniu (od średniej)
    w = [0.5, 0.5]
    avg = 1 / (w[0] / 100 + w[1] / 97)
    idx, gross, filled, fills, win = run([100.2, 98, avg * 1.006], [100, 96.9, 97.5], 1, 100, (0.01, 0.005), 0.02,
                                         w, [0.03])
    assert fills[0] == 2 and win[0] and gross[0] == pytest.approx(0.005)


def test_tp_tuple_first_value_used_before_add():
    idx, gross, filled, fills, win = run([100.2, 101.1], [100, 99.9], 1, 100, (0.01, 0.005), 0.02, [0.5, 0.5], [0.03])
    assert fills[0] == 1 and win[0] and gross[0] == pytest.approx(0.5 * 0.01)


def test_per_signal_arrays_match_scalars():
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 20000)))
    hi, lo = c * 1.001, c * 0.999
    idx = np.sort(rng.choice(np.arange(100, 19000), 300, replace=False))
    side = rng.choice(np.array([-1, 1], dtype=np.int8), 300)
    a = simulate_avg(hi, lo, idx, side, c[idx], 0.012, 0.025, [0.4, 0.6], [0.03])
    n = len(idx)
    b = simulate_avg(hi, lo, idx, side, c[idx], 0, 0, [0.4, 0.6], [], "last",
                     tp_arr=np.full(n, 0.012), sl_arr=np.full(n, 0.025), dist_arr=np.full((n, 1), 0.03))
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])