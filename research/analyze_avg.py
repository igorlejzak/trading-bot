"""Analiza wyników grida z uśrednianiem (grid_avg2.csv albo grid_avg3.csv - ustaw PLIK)."""
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

PLIK = "grid_avg3.csv"          # <- grid_avg2.csv dla 2 wejść, grid_avg3.csv dla 3 wejść

res = pd.read_csv(PLIK, dtype={"dist_%": str, "weights_%": str})
res = res[res["train_trades"] >= 100].copy()            # za mało transakcji = wynik niepewny
res["oba_plus"] = (res["train_pnl_usd"] > 0) & (res["test_pnl_usd"] > 0)

agg = dict(kombinacji=("oba_plus", "size"),
           train_plus_pct=("train_pnl_usd", lambda x: round((x > 0).mean() * 100)),
           test_plus_pct=("test_pnl_usd", lambda x: round((x > 0).mean() * 100)),
           oba_plus_pct=("oba_plus", lambda x: round(x.mean() * 100)),
           train_edge=("train_edge_%", "mean"), test_edge=("test_edge_%", "mean"),
           srednio_wejsc=("train_avg_fills", "mean"))

# 1) Który parametr ma znaczenie? (każdy osobno, uśrednione po reszcie)
for col in ["bb_%", "mode", "dist_%", "weights_%", "tp_%", "sl_%"]:
    print(f"\n--- wg {col} ---")
    print(res.groupby(col).agg(**agg).round(3).to_string())

# 2) Mapa: odstęp drugiego wejścia x podział kapitału -> % kombinacji (TP, SL, BB) na plusie w obu okresach
print("\n=== % kombinacji na plusie w OBU okresach: wiersze = odstęp 2. wejścia, kolumny = podział kapitału ===")
print(res.pivot_table(index="dist_%", columns="weights_%", values="oba_plus", aggfunc="mean").mul(100).round(0).to_string())

# 3) Dla każdej pary (odstęp, podział) najlepsze TP/SL/BB wg TRAIN + jak wyszło na TEST
best = (res.sort_values("train_tstat", ascending=False)
           .groupby(["dist_%", "weights_%"], as_index=False).head(1)
           .sort_values("train_tstat", ascending=False))
cols = ["dist_%", "weights_%", "bb_%", "mode", "tp_%", "sl_%", "train_trades", "train_win_%", "train_avg_fills",
        "train_pnl_usd", "train_maxdd_usd", "test_trades", "test_win_%", "test_pnl_usd"]
print("\n=== najlepsze TP/SL/BB dla każdej pary (odstęp, podział) wg train - TOP 25 ===")
print(best[cols].head(25).to_string(index=False))

# 4) Pełne rankingi: po zarobku i po zarobku na dolara drawdownu (wybieramy po TRAIN, test tylko podglądamy)
full = res.copy()
full["train_pnl_do_dd"] = (full["train_pnl_usd"] / full["train_maxdd_usd"].abs()).round(2)
cols = ["bb_%", "mode", "dist_%", "weights_%", "tp_%", "sl_%",
        "train_trades", "train_win_%", "train_avg_fills", "train_pnl_usd", "train_maxdd_usd", "train_pnl_do_dd",
        "test_trades", "test_win_%", "test_pnl_usd", "test_maxdd_usd"]

print("\n=== TOP 30 wg zarobku na TRAIN ===")
print(full.sort_values("train_pnl_usd", ascending=False)[cols].head(30).to_string(index=False))

print("\n=== TOP 30 wg zarobek / drawdown na TRAIN ===")
print(full.sort_values("train_pnl_do_dd", ascending=False)[cols].head(30).to_string(index=False))

# cała tabela do Excela (średnik i przecinek, żeby polski Excel dobrze otworzył)
out = PLIK.replace(".csv", "_ranking.csv")
full.sort_values("train_pnl_do_dd", ascending=False)[cols].to_csv(out, sep=";", decimal=",", index=False)
print("\nzapisano", out, "- wszystkie kombinacje, otwórz w Excelu")
