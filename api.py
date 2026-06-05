from fastapi import FastAPI
from fastapi.responses import FileResponse
import json
import threading
import ccxt
import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import StochRSIIndicator
import time

app = FastAPI()

# --- stan globalny ---
kapital = 1000.0
rozmiar = 100.0
exchange = ccxt.okx()
coiny = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
pozycje = {c: {"otwarta": False, "typ": None, "entry": None} for c in coiny}

def pobierz_dane(symbol):
    swieczki = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
    df = pd.DataFrame(swieczki, columns=['timestamp','open','high','low','close','volume'])
    indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    indicator_stochrsi = StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3, fillna=False)
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    df['stochrsi_k'] = indicator_stochrsi.stochrsi_k() * 100
    df['stochrsi_d'] = indicator_stochrsi.stochrsi_d() * 100
    return df

def bot_loop():
    global kapital
    while True:
        stan = {"kapital": kapital, "coiny": {}}
        for coin in coiny:
            try:
                df = pobierz_dane(coin)
                pozycja = pozycje[coin]
                ostatnia = df.iloc[-1]
                cena = float(df['close'].iloc[-1])

                stan["coiny"][coin] = {
                    "cena": cena,
                    "bb_gorna": float(ostatnia['bb_bbh']),
                    "bb_dolna": float(ostatnia['bb_bbl']),
                    "stochrsi": float(ostatnia['stochrsi_k']),
                    "pozycja_otwarta": pozycja["otwarta"],
                    "pozycja_typ": pozycja["typ"],
                    "pozycja_entry": pozycja["entry"]
                }

                if not pozycja["otwarta"]:
                    if cena > float(ostatnia['bb_bbh']) * 1.01 and ostatnia['stochrsi_k'] > 80 and ostatnia['stochrsi_d'] > 80:
                        pozycja["otwarta"] = True
                        pozycja["typ"] = "short"
                        pozycja["entry"] = cena
                    elif cena < float(ostatnia['bb_bbl']) * 0.99 and ostatnia['stochrsi_k'] < 20 and ostatnia['stochrsi_d'] < 20:
                        pozycja["otwarta"] = True
                        pozycja["typ"] = "long"
                        pozycja["entry"] = cena
                else:
                    entry = pozycja["entry"]
                    if pozycja["typ"] == "short":
                        if cena <= entry * 0.99 or cena >= entry * 1.03:
                            zmiana = (entry - cena) / entry
                            kapital += rozmiar * zmiana
                            pozycja["otwarta"] = False
                    elif pozycja["typ"] == "long":
                        if cena >= entry * 1.01 or cena <= entry * 0.97:
                            zmiana = (cena - entry) / entry
                            kapital += rozmiar * zmiana
                            pozycja["otwarta"] = False

            except Exception as e:
                print(f"Błąd dla {coin}: {e}")

        with open("stan.json", "w") as f:
            json.dump(stan, f, indent=2)
        time.sleep(1)

# --- uruchom bota w tle przy starcie ---
thread = threading.Thread(target=bot_loop, daemon=True)
thread.start()

# --- endpointy ---
@app.get("/api/stan")
def get_stan():
    with open("stan.json", "r") as f:
        return json.load(f)

@app.get("/")
def index():
    return FileResponse("index.html")