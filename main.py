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


df['stochrsi_k'] = indicator_stochrsi.stochrsi_k()
df['stochrsi_d'] = indicator_stochrsi.stochrsi_d()

df['stochrsi_k'] = df['stochrsi_k'] * 100
df['stochrsi_d'] = df['stochrsi_d'] * 100

ostatnia = df.iloc[-1]

if ostatnia['close'] > ostatnia['bb_bbh'] * 1.01 and ostatnia['stochrsi_k'] > 80 and ostatnia['stochrsi_d'] > 80:
    print("SYGNAŁ SHORT")

elif ostatnia['close'] < ostatnia['bb_bbl'] * 0.99 and ostatnia['stochrsi_k'] < 20 and ostatnia['stochrsi_d'] < 20:
    print("SYGNAŁ LONG")

else:
    print("brak sygnału")

print(df)