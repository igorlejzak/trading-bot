"""Testy silnika grida na ręcznie ułożonych świeczkach (uruchom: pytest test_grid.py)."""
import numpy as np
import pytest

from grid import simulate, first_hit


def run(hi, lo, side, price, tp, sl, weights, dist):
    hi, lo = np.array(hi, dtype=float), np.array(lo, dtype=float)
    return simulate(hi, lo, np.array([0]), np.array([side], dtype=np.int8), np.array([float(price)]),
                    tp, sl, weights, dist)


def test_first_hit():
    hi = np.array([1.0, 1.0, 2.0, 1.0]); lo = np.array([0.9, 0.9, 0.9, 0.1])
    assert first_hit(hi, lo, 0, 2.0, 0.5, 4) == 2      # najpierw high >= 2
    assert first_hit(hi, lo, 3, 2.0, 0.5, 4) == 3      # potem low <= 0.5
    assert first_hit(hi, lo, 0, 5.0, 0.0, 4) == -1     # nic nie trafione


def test_long_tp_single_entry():
    idx, gross, filled, fills, win = run([100.2, 100.5, 101.1], [100, 99.9, 100.4], 1, 100, 0.01, 0.03, [1.0], 0.0)
    assert win[0] and fills[0] == 1 and gross[0] == pytest.approx(0.01)


def test_short_sl_single_entry():
    idx, gross, filled, fills, win = run([100, 101, 103.5], [99.8, 99.5, 101], -1, 100, 0.01, 0.03, [1.0], 0.0)
    assert not win[0] and gross[0] == pytest.approx(-0.03)


def test_no_tp_on_entry_minute():
    # TP dotknięte w minucie wejścia się nie liczy; dopiero w następnej
    idx, gross, filled, fills, win = run([101.5, 100.2, 101.2], [100, 99.9, 100], 1, 100, 0.01, 0.03, [1.0], 0.0)
    assert win[0] and len(gross) == 1


def test_long_averaging_then_tp():
    # wejście 1/3 po 100, dokładka 2/3 po 98 -> średnia 98.658, TP 1% od średniej = 99.645
    w = [1 / 3, 2 / 3]
    idx, gross, filled, fills, win = run([100.2, 99.0, 99.7], [100, 97.9, 98.5], 1, 100, 0.01, 0.03, w, 0.02)
    avg = 1 / (w[0] / 100 + w[1] / 98)
    assert fills[0] == 2 and filled[0] == pytest.approx(1.0) and win[0]
    assert avg == pytest.approx(98.658, abs=1e-3)
    assert gross[0] == pytest.approx(0.01)            # 1% od średniej na całym kapitale


def test_fill_and_tp_same_minute_is_conservative():
    # w minucie 1 pada dokładka (98) i "TP" - zakładamy, że TP nie było; zamyka się dopiero w minucie 2
    w = [0.5, 0.5]
    idx, gross, filled, fills, win = run([100.2, 101.5, 100.5], [100, 97.9, 98.5], 1, 100, 0.01, 0.05, w, 0.02)
    assert fills[0] == 2 and win[0] and gross[0] == pytest.approx(0.01)


def test_sl_before_next_entry():
    # SL 2% od średniej (98) wypada wyżej niż dokładka (97) -> stop bez dokładki, strata tylko na 1/2 kapitału
    w = [0.5, 0.5]
    idx, gross, filled, fills, win = run([100.2, 99.0], [100, 96.0], 1, 100, 0.01, 0.02, w, 0.03)
    assert fills[0] == 1 and not win[0] and gross[0] == pytest.approx(-0.02 * 0.5)


def test_short_three_entries_then_sl():
    # short 100, dokładki 102 i 104 (podwajanie 1:2:4), potem SL 3% od średniej
    w = [1 / 7, 2 / 7, 4 / 7]
    avg = 1 / (w[0] / 100 + w[1] / 102 + w[2] / 104)
    idx, gross, filled, fills, win = run([100.5, 102.1, 104.2, 108.0], [100, 100.5, 102, 104], -1, 100, 0.01, 0.03, w, 0.02)
    assert fills[0] == 3 and not win[0]
    assert gross[0] == pytest.approx(-0.03)           # pełny kapitał, 3% od średniej
    assert avg == pytest.approx(102.83, abs=0.01)


def test_one_position_at_a_time():
    # drugi sygnał w trakcie otwartej pozycji jest ignorowany
    hi = np.array([100.2, 100.3, 100.4, 101.2, 100.0]); lo = np.array([100, 99.9, 99.9, 100.0, 99.0])
    idx, gross, *_ = simulate(hi, lo, np.array([0, 2]), np.array([1, 1], dtype=np.int8), np.array([100.0, 100.0]),
                              0.01, 0.03, [1.0], 0.0)
    assert len(idx) == 1