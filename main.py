import ccxt
import pandas as pd
import numpy as np

from ta.volatility import BollingerBands
from ta.momentum import StochRSIIndicator 



exchange = ccxt.bybit()
swieczki = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=100)
df = pd.DataFrame(swieczki, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
indicator_stochrsi= StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3, fillna=False)

df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()

df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
df['stochrsi_k'] = indicator_stochrsi.stochrsi_k()
df['stochrsi_d'] = indicator_stochrsi.stochrsi_d()

print(df)