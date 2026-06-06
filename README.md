# 🤖 Trading Bot

A live cryptocurrency trading bot with a real-time web dashboard. The bot monitors multiple trading pairs simultaneously, detects entry signals based on technical indicators, and tracks paper trading performance — all visible through a browser-based dashboard.

**Live demo:** `[https://trading-bot-production-ba29.up.railway.app]`

---

## Strategy

The bot uses a **mean-reversion strategy** — the idea that when price moves too far from its average, it tends to return. Entries are triggered when two conditions are met at the same time:

- Price has moved **more than 1% beyond** the Bollinger Band (upper or lower)
- StochRSI confirms the market is **overbought or oversold**

This double confirmation reduces false signals — the bot only acts when the market is at an extreme on both measures.

### Bollinger Bands (BB)

Bollinger Bands consist of three lines calculated from the last 20 candles:
- **Middle band** — simple moving average (SMA)
- **Upper band** — SMA + 2 standard deviations
- **Lower band** — SMA − 2 standard deviations

When price moves outside the bands, it statistically suggests an extreme move that may reverse. The wider the bands, the more volatile the market.

### Stochastic RSI (StochRSI)

StochRSI is a momentum oscillator ranging from 0 to 100. It applies the Stochastic formula to RSI values, making it more sensitive to recent price action.

- **Above 80** → overbought (potential reversal down → SHORT signal)
- **Below 20** → oversold (potential reversal up → LONG signal)

Both K and D lines must confirm the signal before the bot acts.

---

## Entry & Exit Logic

| Direction | Entry condition | Take Profit | Stop Loss |
|-----------|----------------|-------------|-----------|
| **LONG** | Price < Lower BB × 0.99 AND StochRSI K & D < 20 | Entry × 1.01 (+1%) | Entry × 0.97 (−3%) |
| **SHORT** | Price > Upper BB × 1.01 AND StochRSI K & D > 80 | Entry × 0.99 (−1%) | Entry × 1.03 (+3%) |

Position size: **$100 per trade** on a starting capital of **$1,000** (paper trading).

---

## Dashboard

The web dashboard updates in real time and shows:

- **Current capital** (starting at $1,000, updated after each closed position)
- **Watching List** — live price, Bollinger Bands values, and StochRSI for each monitored pair
- **Active positions** — open trades with entry price and direction
- **Candlestick chart** — interactive TradingView chart (15-minute timeframe) with pair selector

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Bot logic | Python |
| Market data | ccxt (OKX exchange) |
| Indicators | ta (technical analysis library) |
| Data processing | Pandas, NumPy |
| Backend API | FastAPI |
| Web server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Charts | TradingView Widget |
| Deployment | Railway |

---

## Monitored Pairs

- BTC/USDT
- ETH/USDT
- SOL/USDT
- XRP/USDT

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/IgorLejzak/trading-bot.git
cd trading-bot
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the application**
```bash
uvicorn api:app --reload
```

**5. Open in browser**
```
http://127.0.0.1:8000
```

The bot starts automatically in the background when the server launches.

---

## Project Structure

```
trading-bot/
├── api.py              # FastAPI backend + bot logic
├── main.py             # Standalone bot (local testing)
├── index.html          # Dashboard frontend
├── stan.json           # Live state (auto-generated)
├── requirements.txt    # Python dependencies
├── Procfile            # Railway deployment config
└── README.md
```

---

## Disclaimer

This project is for **educational purposes only**. All trading is simulated (paper trading) — no real funds are used. Past performance of the strategy does not guarantee future results.
