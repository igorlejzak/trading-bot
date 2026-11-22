
"""Rok po roku: które ustawienia zarabiały w KAŻDYM roku (2024, 2025, 2026), a nie tylko w sumie."""
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

PLIK = "grid_avg2.csv"
res = pd.read_csv(PLIK, dtype={"dist_%": str, "weights_%": str})
res = res[res["pnl_3lata_usd"].notna() & ((res["trades_2024"] + res["trades_2025"] + res["trades_2026"]) >= 120)].copy()
ycols = ["pnl_2024", "pnl_2025", "pnl_2026"]

# 1) Które parametry dają stabilność rok po roku
for col in ["bb_%", "mode", "dist_%", "weights_%", "tp_%", "sl_%"]:
    print(f"\n--- wg {col}: % kombinacji na plusie we WSZYSTKICH 3 latach + średni wynik każdego roku ---")
    print(res.groupby(col).agg(
        kombinacji=("lata_na_plusie", "size"),
        trzy_lata_plus_pct=("lata_na_plusie", lambda x: round((x == 3).mean() * 100)),
        **{c: (c, "mean") for c in ycols},
        najgorszy_rok=("najgorszy_rok_usd", "mean")).round(1).to_string())

# 2) Ranking: najpierw liczba lat na plusie, potem NAJGORSZY rok (im wyższy, tym stabilniej), potem suma
cols = ["bb_%", "mode", "dist_%", "weights_%", "tp_%", "sl_%"] + \
       ["trades_2024", "trades_2025", "trades_2026"] + ycols + ["najgorszy_rok_usd", "pnl_3lata_usd", "train_maxdd_usd"]
best = res.sort_values(["lata_na_plusie", "najgorszy_rok_usd", "pnl_3lata_usd"], ascending=False)
print("\n=== TOP 30: na plusie w każdym roku, posortowane po NAJGORSZYM roku ===")
print(best[cols].head(30).to_string(index=False))

# 3) Gdzie jest nasze wybrane ustawienie
ours = res[(res["bb_%"] == 1.5) & (res["mode"] == "stoch") & (res["dist_%"] == "4") &
           (res["weights_%"] == "30/70") & (res["tp_%"] == 2.2) & (res["sl_%"] == 2.0)]
print("\n=== nasze wybrane 2x ===")
print(ours[cols].to_string(index=False))
print(f"\nkombinacji na plusie we wszystkich 3 latach: {(res['lata_na_plusie'] == 3).sum()} z {len(res)}")

best[cols].to_csv(PLIK.replace(".csv", "_lata.csv"), sep=";", decimal=",", index=False)
