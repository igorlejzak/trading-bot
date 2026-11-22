import pandas as pd

pd.set_option("display.width", 250)
res = pd.read_csv("grid_bb_ext.csv")

BB, MODE, ENTRIES = 1.0, "stoch", "1x"     # zmieniaj te trzy i patrz
sub = res[(res["mode"] == "bb_only") & (res["entries"] == "1x")].copy()
sub["oba_plus"] = (sub["train_pnl_usd"] > 0) & (sub["test_pnl_usd"] > 0)
print(sub.groupby("bb_%").agg(
    train_trades=("train_trades", "mean"),
    kombinacji=("oba_plus", "size"),
    train_plus_pct=("train_pnl_usd", lambda x: round((x > 0).mean() * 100)),
    test_plus_pct=("test_pnl_usd", lambda x: round((x > 0).mean() * 100)),
    oba_plus_pct=("oba_plus", lambda x: round(x.mean() * 100)),
    najlepszy_train=("train_pnl_usd", "max")).to_string())

def okolica(r):
    m = sub[(abs(sub["bb_%"] - r["bb_%"]) <= 0.11) &
            (abs(sub["sl_%"] - r["sl_%"]) <= 0.21) &
            (abs(sub["tp_%"] - r["tp_%"]) <= 0.21)]
    return m["train_pnl_usd"].mean()

sub["train_okolica"] = sub.apply(okolica, axis=1)
cols = ["bb_%", "sl_%", "tp_%", "train_trades", "train_pnl_usd", "train_okolica", "test_pnl_usd"]
print(sub.sort_values("train_okolica", ascending=False)[cols].head(15).to_string(index=False))