
import pandas as pd

pd.set_option("display.width", 250)
res = pd.read_csv("grid_tp2.csv", dtype={"tp_%": str, "dist_%": str, "weights_%": str})
res = res[res["train_trades"] >= 100].copy()
res[["tp1", "tp2"]] = res["tp_%"].str.split("/", expand=True).astype(float)

# 1) Dla każdego TP1: co się dzieje, gdy po dołożeniu zmniejszamy TP (średnio po odstępach, podziałach i SL)
for t1 in [1.8, 2.2, 2.6]:
    sub = res[res["tp1"] == t1]
    print(f"\n=== TP1 = {t1}%: wynik wg TP po dołożeniu (TP2) ===")
    print(sub.groupby("tp2").agg(train_usd=("train_pnl_usd", "mean"), test_usd=("test_pnl_usd", "mean"),
                                 win_train=("train_win_%", "mean"),
                                 dd_train=("train_maxdd_usd", "mean"), dd_test=("test_maxdd_usd", "mean"))
             .round(1).to_string())

# 2) Nasza wersja (4%, 30/70, SL 2%) - bezpośrednie porównanie
ours = res[(res["dist_%"] == "4") & (res["weights_%"] == "30/70") & (res["sl_%"] == 2.0)]
print("\n=== nasza wersja 2x (4%, 30/70, SL 2%): TP1 2.2% i różne TP2 ===")
print(ours[ours["tp1"] == 2.2][["tp_%", "train_trades", "train_win_%", "train_pnl_usd", "train_maxdd_usd",
                                "test_win_%", "test_pnl_usd", "test_maxdd_usd"]].to_string(index=False))