import ccxt
import pandas as pd

exchange = ccxt.bybit()
swieczki = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
df = pd.DataFrame(swieczki, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
print(df)