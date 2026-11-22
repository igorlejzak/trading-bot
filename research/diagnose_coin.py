"""Diagnoza strategii 1x na dowolnym coinie: python research/diagnose_coin.py ADA"""
import sys

from cross_coin import START, TEST_START
from data import load_candles, exchange
from diagnose import diagnose

coin = (sys.argv[1] if len(sys.argv) > 1 else "XRP").upper() + "/USDT"
m1 = load_candles(coin, "1m", exchange.parse8601(START), exchange.milliseconds())
strats = {"1x BB1.9 TP2.0 SL4.6": {"bb_pct": 0.019, "mode": "bb_only", "tp": 0.020, "sl": 0.046,
                                    "weights": [1.0], "dist": []}}
diagnose(m1, TEST_START, strats=strats, png=f"diagnose_{coin.split('/')[0]}.png", symbol=coin)