"""Bot paper-tradingowy: wiele coinów i kilka strategii naraz, żeby zebrać jak najwięcej danych live.

Każda strategia ma na każdym coinie własną, niezależną pozycję (100 USD). Wynik zapisywany jest brutto
i netto (po opłacie 0.05% za każdą stronę, jak w backteście). Historia i otwarte pozycje trafiają na
osobną gałąź `bot-data` na GitHubie: main zostaje czysty, a zapis nie wywołuje redeployu na Railway.

Pętla co ~2 s: JEDNO zapytanie o ceny wszystkich coinów (fetch_tickers). Zamknięte świeczki 15m są
w pamięci i dociągane tylko wtedy, gdy zamknie się nowa. Wskaźniki liczone są na zamkniętych świeczkach
plus aktualnej cenie (niezamknięta świeczka) - tak samo jak wcześniej i jak w livemode.py.
"""
import base64
import json
import os
import threading
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from ta.momentum import StochRSIIndicator
from ta.volatility import BollingerBands

# ---------------------------------------------------------------- ustawienia
STRATEGIE = {
    # zasady, na których bot chodził od czerwca - zostają, żeby historia była ciągła
    "original": {"name": "Original rules", "desc": "1% beyond the band + StochRSI, TP 1%, SL 3%",
                 "bb": 0.010, "stoch": True, "tp": 0.010, "sl": 0.030, "wagi": [1.0], "odstepy": []},
    # najlepiej przenosząca się na inne coiny w badaniu (cross_coin.py)
    "1x": {"name": "1 entry, BB 1.9%", "desc": "1.9% beyond the band, TP 2%, SL 4.6%",
           "bb": 0.019, "stoch": False, "tp": 0.020, "sl": 0.046, "wagi": [1.0], "odstepy": []},
    # najlepsza na SOL (najgładsza krzywa), ale słaba na ETH - live na wielu coinach to rozstrzygnie
    "2x": {"name": "2 entries, averaging", "desc": "1.5% beyond the band + StochRSI, add 4% further, 30/70, "
                                                   "TP 2.2% from average, SL 2% after the add",
           "bb": 0.015, "stoch": True, "tp": 0.022, "sl": 0.020, "wagi": [0.3, 0.7], "odstepy": [0.04]},
}
CHCIANE_COINY = ["BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "HYPE", "PEPE", "LINK", "DOT", "DOGE", "AVAX",
                 "ALGO", "SUI", "LTC", "XLM", "UNI", "NEAR", "AAVE", "TAO", "ONDO"]
ROZMIAR = 100.0           # USD na pozycję (przy 2 wejściach: suma obu)
FEE = 0.0005              # OKX taker, za każdą stronę
STOCH_HI, STOCH_LO = 80, 20
KAPITAL_START = 1000.0
TF_MS = 15 * 60 * 1000

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "IgorLejzak/trading-bot")
GALAZ = os.environ.get("DATA_BRANCH", "bot-data")
PLIK_HISTORIA, PLIK_POZYCJE = "historia.csv", "pozycje.json"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
API = f"https://api.github.com/repos/{GITHUB_REPO}"
NAGLOWEK = "Time,Pair,Direction,Entry,Close,PnL,PnL_net,Strategy,Opened,Fills"

app = FastAPI()
exchange = ccxt.okx({"enableRateLimit": True})


def teraz():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- GitHub (gałąź bot-data)
def gh_upewnij_galaz():
    """Tworzy gałąź bot-data z main, jeśli jej jeszcze nie ma. Zwraca True, gdy gałąź istnieje."""
    r = requests.get(f"{API}/branches/{GALAZ}", headers=HEADERS, timeout=20)
    if r.status_code == 200:
        return True
    main = requests.get(f"{API}/git/ref/heads/main", headers=HEADERS, timeout=20)
    if main.status_code != 200:
        print(f"BŁĄD: nie widzę gałęzi main ({main.status_code}) - sprawdź GITHUB_TOKEN")
        return False
    sha = main.json()["object"]["sha"]
    c = requests.post(f"{API}/git/refs", headers=HEADERS, timeout=20,
                      json={"ref": f"refs/heads/{GALAZ}", "sha": sha})
    print(f"Tworzę gałąź {GALAZ}: {c.status_code}")
    return c.status_code == 201


def gh_czytaj(plik):
    """Zwraca (tekst, sha) albo (None, None), gdy pliku nie ma. Rzuca wyjątek przy innym błędzie."""
    r = requests.get(f"{API}/contents/{plik}", params={"ref": GALAZ}, headers=HEADERS, timeout=20)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    return base64.b64decode(r.json()["content"]).decode("utf-8"), r.json()["sha"]


def gh_zapisz(plik, tekst, opis):
    _, sha = gh_czytaj(plik)
    payload = {"message": opis, "content": base64.b64encode(tekst.encode()).decode(), "branch": GALAZ}
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{API}/contents/{plik}", json=payload, headers=HEADERS, timeout=20)
    if r.status_code not in (200, 201):
        print(f"BŁĄD zapisu {plik}: {r.status_code} {r.text[:200]}")


def wczytaj_historie(tekst):
    """Czyta zarówno stary format (6 kolumn, tylko zasady 'original', bez opłat), jak i nowy."""
    out = []
    for line in tekst.strip().split("\n"):
        p = line.strip().split(",")
        if not line.strip() or p[0] in ("Time", "") or line.startswith("#"):
            continue
        if len(p) == 6:   # stary bot: PnL brutto, 100 USD, 1 wejście
            pnl = float(p[5])
            out.append({"time": p[0], "pair": p[1], "direction": p[2], "entry": float(p[3]), "close": float(p[4]),
                        "pnl": pnl, "pnl_net": round(pnl - ROZMIAR * FEE * 2, 2), "strategy": "original",
                        "opened": "", "fills": 1})
        elif len(p) == 10:
            out.append({"time": p[0], "pair": p[1], "direction": p[2], "entry": float(p[3]), "close": float(p[4]),
                        "pnl": float(p[5]), "pnl_net": float(p[6]), "strategy": p[7], "opened": p[8],
                        "fills": int(p[9])})
    return out


def historia_csv(historia):
    lines = [NAGLOWEK] + [f"{h['time']},{h['pair']},{h['direction']},{h['entry']},{h['close']},{h['pnl']},"
                          f"{h['pnl_net']},{h['strategy']},{h['opened']},{h['fills']}" for h in historia]
    return "\n".join(lines) + "\n"


def wczytaj_stan_z_github():
    """Zwraca (historia, pozycje, ok). Przy błędzie ok=False i bot NIE nadpisuje danych na GitHubie."""
    try:
        if not gh_upewnij_galaz():
            return [], {}, False
        tekst, _ = gh_czytaj(PLIK_HISTORIA)
        historia = wczytaj_historie(tekst) if tekst else []
        tekst_p, _ = gh_czytaj(PLIK_POZYCJE)
        pozycje = json.loads(tekst_p) if tekst_p else {}
        print(f"Wczytano {len(historia)} transakcji i {len(pozycje)} otwartych pozycji z gałęzi {GALAZ}")
        return historia, pozycje, True
    except Exception as e:
        print(f"BŁĄD wczytywania danych z GitHub: {e}")
        return [], {}, False


# ---------------------------------------------------------------- stan w pamięci
historia, pozycje, DANE_OK = wczytaj_stan_z_github()   # pozycje: {"1x|SOL/USDT": {...}}
zmiany = threading.Event()                              # coś się zmieniło -> zapisz na GitHub
swieczki = {}                                           # coin -> DataFrame zamkniętych świeczek 15m
STAN = {"updated": None, "strategies": {}, "coins": {}, "skipped": []}
lock = threading.Lock()


def klucz(strat, coin):
    return f"{strat}|{coin}"


def dostepne_coiny():
    rynki = exchange.load_markets()
    ok, brak = [], []
    for c in CHCIANE_COINY:
        s = f"{c}/USDT"
        (ok if s in rynki and rynki[s].get("spot") and rynki[s].get("active", True) else brak).append(s)
    if brak:
        print("Brak na OKX spot, pomijam:", ", ".join(brak))
    return ok, brak


def odswiez_swieczki(coin, now_ms):
    """Dociąga świeczki tylko, gdy od ostatniej zamkniętej minęło >= 15 min (zamknęła się nowa)."""
    df = swieczki.get(coin)
    if df is not None and now_ms < int(df["ts"].iloc[-1]) + 2 * TF_MS + 3000:
        return df
    raw = exchange.fetch_ohlcv(coin, timeframe="15m", limit=100)
    d = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    d = d[d["ts"] + TF_MS <= now_ms].reset_index(drop=True)          # tylko zamknięte
    swieczki[coin] = d
    return d


def wskazniki(df, cena):
    close = pd.concat([df["close"], pd.Series([cena])], ignore_index=True)   # + niezamknięta świeczka
    bb = BollingerBands(close=close, window=20, window_dev=2)
    st = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3, fillna=False)
    return (float(bb.bollinger_hband().iloc[-1]), float(bb.bollinger_lband().iloc[-1]),
            float(st.stochrsi_k().iloc[-1] * 100), float(st.stochrsi_d().iloc[-1] * 100))


def sygnal(s, cena, bbh, bbl, k, d):
    if cena > bbh * (1 + s["bb"]) and (not s["stoch"] or (k > STOCH_HI and d > STOCH_HI)):
        return "short"
    if cena < bbl * (1 - s["bb"]) and (not s["stoch"] or (k < STOCH_LO and d < STOCH_LO)):
        return "long"
    return None


def podsumuj(p):
    usd = sum(w["usd"] for w in p["wejscia"])
    qty = sum(w["usd"] / w["cena"] for w in p["wejscia"])
    return usd, qty, usd / qty


def obsluz_pozycje(name, s, coin, cena, bbh, bbl, k, d):
    """Jedna strategia na jednym coinie: wejście, dołożenie, TP / SL. Zasady jak w grid_avg.simulate_avg."""
    key = klucz(name, coin)
    p = pozycje.get(key)
    if p is None:
        kier = sygnal(s, cena, bbh, bbl, k, d)
        if kier:
            pozycje[key] = {"typ": kier, "otwarta": teraz(),
                            "wejscia": [{"cena": cena, "usd": ROZMIAR * s["wagi"][0], "czas": teraz()}]}
            zmiany.set()
        return

    znak = 1 if p["typ"] == "long" else -1
    n = len(p["wejscia"])
    p1 = p["wejscia"][0]["cena"]
    if n < len(s["wagi"]):                                  # czekamy na dołożenie - bez SL
        poziom = p1 * (1 - znak * s["odstepy"][n - 1])
        if (cena - poziom) * znak <= 0:
            p["wejscia"].append({"cena": cena, "usd": ROZMIAR * s["wagi"][n], "czas": teraz()})
            zmiany.set()
            return
    usd, qty, avg = podsumuj(p)
    tp = avg * (1 + znak * s["tp"])
    trafiony_tp = (cena - tp) * znak >= 0
    trafiony_sl = False
    if len(p["wejscia"]) == len(s["wagi"]):                 # SL dopiero po ostatnim wejściu, od jego ceny
        sl = p["wejscia"][-1]["cena"] * (1 - znak * s["sl"])
        trafiony_sl = (cena - sl) * znak <= 0
    if not (trafiony_tp or trafiony_sl):
        return

    brutto = znak * (qty * cena - usd)
    oplaty = FEE * (usd + qty * cena)
    historia.append({"time": teraz(), "pair": coin, "direction": p["typ"].upper(), "entry": round(avg, 6),
                     "close": round(cena, 6), "pnl": round(brutto, 2), "pnl_net": round(brutto - oplaty, 2),
                     "strategy": name, "opened": p["otwarta"], "fills": len(p["wejscia"])})
    del pozycje[key]
    zmiany.set()


def bot_loop():
    coiny, brak = dostepne_coiny()
    STAN["skipped"] = brak
    STAN["strategies"] = {k: {kk: v for kk, v in s.items() if kk not in ("wagi", "odstepy")} | {
        "weights": s["wagi"], "adds": s["odstepy"]} for k, s in STRATEGIE.items()}
    while True:
        start = time.time()
        try:
            ceny = exchange.fetch_tickers(coiny)
        except Exception as e:
            print(f"Błąd pobierania cen: {e}")
            time.sleep(5)
            continue
        now_ms = exchange.milliseconds()
        for coin in coiny:
            try:
                cena = float(ceny[coin]["last"])
                df = odswiez_swieczki(coin, now_ms)
                bbh, bbl, k, d = wskazniki(df, cena)
                with lock:
                    for name, s in STRATEGIE.items():
                        obsluz_pozycje(name, s, coin, cena, bbh, bbl, k, d)
                    STAN["coins"][coin] = {
                        "price": cena, "bb_upper": bbh, "bb_lower": bbl, "stoch_k": k, "stoch_d": d,
                        "positions": {name: pozycje.get(klucz(name, coin)) for name in STRATEGIE}}
            except Exception as e:
                print(f"Błąd dla {coin}: {e}")
        STAN["updated"] = teraz()
        time.sleep(max(0.5, 2 - (time.time() - start)))


def zapis_loop():
    """Co minutę zapisuje historię i otwarte pozycje, jeśli coś się zmieniło (mniej commitów)."""
    while True:
        zmiany.wait()
        time.sleep(60)
        zmiany.clear()
        if not DANE_OK:
            print("Nie zapisuję: przy starcie nie udało się wczytać danych z GitHub (nadpisałbym je)")
            continue
        try:
            with lock:
                h, p = historia_csv(historia), json.dumps(pozycje, indent=1)
            gh_zapisz(PLIK_HISTORIA, h, "bot: update history")
            gh_zapisz(PLIK_POZYCJE, p, "bot: update open positions")
        except Exception as e:
            print(f"BŁĄD zapisu na GitHub: {e}")
            zmiany.set()


threading.Thread(target=bot_loop, daemon=True).start()
threading.Thread(target=zapis_loop, daemon=True).start()


@app.get("/api/stan")
def get_stan():
    with lock:
        return json.loads(json.dumps(STAN))


@app.get("/api/historia")
def get_historia():
    with lock:
        return list(historia)


@app.get("/")
def index():
    return FileResponse("index.html")
