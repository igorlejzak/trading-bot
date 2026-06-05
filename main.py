
import ccxt
import pandas as pd
import numpy as np
import time
import json

from ta.volatility import BollingerBands
from ta.momentum import StochRSIIndicator 

def pobierz_dane(symbol):
    
    swieczki = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
    df = pd.DataFrame(swieczki, columns=['timestamp','open','high','low','close','volume'])
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
    
    return df


kapital= 1000.0
rozmiar = 100.0







exchange = ccxt.bybit()
pozycja = {"otwarta": False, "typ": None, "entry": None}

coiny = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "HYPE/USDT"]
pozycje = {
    "BTC/USDT": {"otwarta": False, "typ": None, "entry": None},
    "ETH/USDT": {"otwarta": False, "typ": None, "entry": None},
    "SOL/USDT": {"otwarta": False, "typ": None, "entry": None},
    "XRP/USDT": {"otwarta": False, "typ": None, "entry": None},
    "HYPE/USDT": {"otwarta": False, "typ": None, "entry": None}
}






while True:

    stan = {
    "kapital": kapital,
    "coiny": {}
    }

    for coin in coiny:
        df = pobierz_dane(coin)           # pobierz dane TEGO coina
        pozycja = pozycje[coin]

        ostatnia = df.iloc[-1]
        cena = df['close'].iloc[-1]

        stan["coiny"][coin] = {
        "cena": float(cena),
        "bb_gorna": float(ostatnia['bb_bbh']),
        "bb_dolna": float(ostatnia['bb_bbl']),
        "stochrsi": float(ostatnia['stochrsi_k']),
        "pozycja_otwarta": pozycja["otwarta"],
        "pozycja_typ": pozycja["typ"],
        "pozycja_entry": pozycja["entry"]
        }

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

        else:   #w pozycji
            entry= pozycja["entry"]
            if pozycja["typ"] == "short":
                tp = entry * 0.99
                sl = entry * 1.03
                if cena <= tp or cena >= sl:          
                    pozycja["otwarta"] = False
                    zmiana = (entry - cena) / entry
                    zysk = rozmiar * zmiana
                    kapital += zysk
                    print(f"ZAMKNIĘTO SHORT {coin} | wynik: {zysk:.2f}$ | kapitał: {kapital:.2f}$")

            elif pozycja["typ"] == "long":
                tp = entry * 1.01
                sl = entry * 0.97
                if cena >= tp or cena <= sl:          
                    pozycja["otwarta"] = False
                    zmiana = (cena - entry) / entry
                    zysk = rozmiar * zmiana
                    kapital += zysk
                    print(f"ZAMKNIĘTO LONG {coin} | wynik: {zysk:.2f}$ | kapitał: {kapital:.2f}$")

        print(f"{coin} | cena: {cena}, BB dolna: {ostatnia['bb_bbl']:.1f}, StochRSI: {ostatnia['stochrsi_k']:.1f}")
    with open("stan.json", "w") as f:
        json.dump(stan, f, indent=2)

    time.sleep(0.1)
    print(f"Kapitał: {kapital:.2f}$")