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

pozycja = {"otwarta": False, "typ": None, "entry": None}

cena = df['close'].iloc[-1]   # aktualna cena (ostatnia świeczka)
ostatnia = df.iloc[-1]

if not pozycja["otwarta"]:
    # NIE MAM POZYCJI → szukam sygnału wejścia
    if cena > ostatnia['bb_bbh'] * 1.01 and ostatnia['stochrsi_k'] > 80 and ostatnia['stochrsi_d'] > 80:
        pozycja["otwarta"] = True
        pozycja["typ"] = "short"
        pozycja["entry"] = cena
        print(f"OTWARTO SHORT po cenie {cena}")

    elif cena < ostatnia['bb_bbl'] * 0.99 and ostatnia['stochrsi_k'] < 20 and ostatnia['stochrsi_d'] < 20:
        pozycja["otwarta"] = True
        pozycja["typ"] = "long"
        pozycja["entry"] = cena
        print(f"OTWARTO LONG po cenie {cena}")

else:   # MAM POZYCJĘ → szukam sygnału wyjścia
    entry= pozycja["entry"]
    if pozycja["typ"] == "short":
        tp = entry * 0.99
        sl = entry * 1.03
        if cena <= tp:
            pozycja["otwarta"] = False
            print(f"ZAMKNIĘTO SHORT z zyskiem po cenie {cena}")
        if cena >= sl:
            pozycja["otwarta"] = False
            print(f"ZAMKNIĘTO SHORT ze stratą po cenie {cena}")

    elif pozycja["typ"] == "long":
        tp = entry * 1.01
        sl = entry * 0.97
        if cena >= tp:
            pozycja["otwarta"] = False
            print(f"ZAMKNIĘTO LONG z zyskiem po cenie {cena}")
        if cena <= sl:
            pozycja["otwarta"] = False
            print(f"ZAMKNIĘTO LONG ze stratą po cenie {cena}")