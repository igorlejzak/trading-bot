import sqlite3
import pandas as pd

conn = sqlite3.connect("candles.db")
df = pd.read_sql_query("""
    SELECT symbol, timeframe, COUNT(*) AS swieczek, MIN(ts) AS od, MAX(ts) AS do
    FROM candles
    GROUP BY symbol, timeframe
    ORDER BY symbol, timeframe
""", conn)
conn.close()

df["od"] = pd.to_datetime(df["od"], unit="ms")
df["do"] = pd.to_datetime(df["do"], unit="ms")
print(df.to_string(index=False))