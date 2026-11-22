"""Analiza grida dopasowanego do zmienności (grid_vol.csv) i porównanie ze sztywnymi procentami."""
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

res = pd.read_csv("grid_vol.csv")
res = res[res["train_trades"] >= 100].copy()
res["oba_plus"] = (res["train_pnl_usd"] > 0) & (res["test_pnl_usd"] > 0)
res["train_pnl_do_dd"] = (res["train_pnl_usd"] / res["train_maxdd_usd"].abs()).round(2)
res["wejscia"] = res["entries"].str[:2]

agg = dict(kombinacji=("oba_plus", "size"),
           train_plus_pct=("train_pnl_usd", lambda x: round((x > 0).mean() * 100)),
           test_plus_pct=("test_pnl_usd", lambda x: round((x > 0).mean() * 100)),
           oba_plus_pct=("oba_plus", lambda x: round(x.mean() * 100)),
           train_edge=("train_edge_%", "mean"), test_edge=("test_edge_%", "mean"))

# 1) Które parametry mają znaczenie
for col in ["vol_window", "k_bb", "mode", "k_tp", "k_sl", "wejscia", "entries"]:
    print(f"\n--- wg {col} ---")
    print(res.groupby(col).agg(**agg).round(3).to_string())

# 2) Ranking zarobek / drawdown na TRAIN (test tylko do podglądu) + jakie to były procenty w praktyce
cols = ["vol_window", "k_bb", "mode", "k_tp", "k_sl", "entries",
        "bb_%_med", "tp_%_med", "tp_%_p10", "tp_%_p90", "sl_%_med", "dist_%_med",
        "train_trades", "train_win_%", "train_pnl_usd", "train_maxdd_usd", "train_pnl_do_dd",
        "test_trades", "test_win_%", "test_pnl_usd", "test_maxdd_usd"]
for n_in in ["1x", "2x"]:
    sub = res[res["wejscia"] == n_in]
    print(f"\n=== {n_in}: TOP 20 wg zarobek / drawdown na TRAIN ===")
    print(sub.sort_values("train_pnl_do_dd", ascending=False)[cols].head(20).to_string(index=False))

# 3) Porównanie ze sztywnymi procentami (najlepsze ustawienia z wcześniejszych gridów, SOL)
print("""
=== dla porównania: sztywne procenty (SOL) ===
1x  BB1.9 TP2.0 SL4.6              train +51.7 USD, test  +5.2 USD, drawdown ok. -15
2x  BB1.5 4% 30/70 TP2.2 SL2       train +56.2 USD, test +10.3 USD, drawdown -8 / -8
""")

res.sort_values("train_pnl_do_dd", ascending=False)[cols].to_csv(
    "grid_vol_ranking.csv", sep=";", decimal=",", index=False)
print("zapisano grid_vol_ranking.csv - wszystkie kombinacje, otwórz w Excelu")