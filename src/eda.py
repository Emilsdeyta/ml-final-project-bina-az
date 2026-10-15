"""
eda.py - exploratory figures (Problem 1b) + auto-generated numbers for the report.

EDA only looks at TRAIN+VALIDATION rows: the test set is not explored (it is only
counted, for the split table). Figures go to report/figures/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev

FIG_DIR = ev.FIG_DIR


def _plt():
    return ev._plt()


def _explore_frame(S: dict) -> pd.DataFrame:
    df = pd.concat([S["df_train"], S["df_val"]], ignore_index=True)
    df["log_ppm2"] = np.log(df["price"] / df["area_m2"])
    return df


def run_eda(S_A: dict, S_B: dict, out_dir: Path | str = FIG_DIR) -> list[Path]:
    plt = _plt()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _explore_frame(S_A)
    saved = []

    # 1) price and log price ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].hist(df["price"].clip(upper=df["price"].quantile(.995)) / 1e3, bins=70, color="#4c72b0")
    ax[0].set(xlabel="price, thousand AZN (clipped at 99.5%)", ylabel="listings", title="Price: heavy right tail")
    ax[1].hist(np.log10(df["price"]), bins=70, color="#55a868")
    ax[1].set(xlabel="log10 price (AZN)", title=f"log price: skew {np.log(df['price']).skew():.2f} vs raw {df['price'].skew():.1f}")
    saved.append(ev._save(fig, out / "eda_price_hist.png"))

    # 2) area vs price -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    h = ax.hexbin(np.log10(df["area_m2"]), np.log10(df["price"]), gridsize=55, bins="log", mincnt=1, cmap="viridis")
    ax.set(xlabel="log10 area (m$^2$)", ylabel="log10 price (AZN)", title="Area vs price")
    fig.colorbar(h, label="listings")
    saved.append(ev._save(fig, out / "eda_area_vs_price.png"))

    # 3) rooms -----------------------------------------------------------------
    r = df["rooms"].clip(upper=8)
    groups = [(int(k), np.log10(df.loc[r == k, "price"])) for k in sorted(r.dropna().unique())]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.boxplot([g for _, g in groups], showfliers=False)
    ax.set_xticklabels([f"{k}{'+' if k == 8 else ''}" for k, _ in groups])
    ax.set(xlabel="rooms", ylabel="log10 price (AZN)", title="Price by number of rooms")
    saved.append(ev._save(fig, out / "eda_rooms_box.png"))

    # 4) location map ---------------------------------------------------------
    m = df.dropna(subset=["lat", "lng"])
    m = m[m["lat"].between(40.25, 40.55) & m["lng"].between(49.6, 50.25)]
    m = m.sample(min(len(m), 25_000), random_state=42)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    sc = ax.scatter(m["lng"], m["lat"], c=m["log_ppm2"], s=3, cmap="plasma", alpha=.6)
    ax.set(xlabel="longitude", ylabel="latitude", title="Where it is expensive: log(AZN per m$^2$)")
    fig.colorbar(sc, label="log price per m$^2$")
    saved.append(ev._save(fig, out / "eda_location_map.png"))

    # 5) missing values (BEFORE imputation) ---------------------------------
    cols = ["rooms", "floor", "total_floors", "land_sot", "lat", "lng"]
    miss = df[cols].isna().mean().mul(100)
    miss["repair unknown"] = (df["repair"] == "missing").mean() * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#c44e52")
    ax.set(xlabel="% missing", title="Missing values (floor/land gaps are structural: houses vs flats)")
    ax.title.set_fontsize(9)
    saved.append(ev._save(fig, out / "eda_missing.png"))

    # 6) class balance --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.4))
    names = ["train", "val", "test"]
    prem = [S_B["y_train"].mean(), S_B["y_val"].mean(), S_B["y_test"].mean()]
    ax.bar(names, prem, color="#4c72b0")
    ax.axhline(.5, color="k", ls=":", lw=1)
    ax.set(ylim=(0, 1), ylabel="share premium", title=f"Task B balance (threshold {S_B['threshold']:,.0f} AZN, train median)")
    ax.title.set_fontsize(9)
    for i, v in enumerate(prem):
        ax.text(i, v + .02, f"{v:.3f}", ha="center")
    saved.append(ev._save(fig, out / "eda_class_balance.png"))

    # 7) district effect --------------------------------------------------------
    g = df.assign(ppm2=df["price"] / df["area_m2"]).groupby("district")["ppm2"].agg(["median", "size"])
    g = g[g["size"] >= 50].sort_values("median")
    pick = pd.concat([g.head(10), g.tail(10)])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(pick.index, pick["median"], color=["#55a868"] * 10 + ["#c44e52"] * 10)
    ax.set(xlabel="median AZN per m$^2$", title="Cheapest and most expensive districts (n>=50)")
    saved.append(ev._save(fig, out / "eda_district_ppm2.png"))
    print(f"[eda] saved {len(saved)} figures to {out}")
    return saved


def write_data_summary(S_A: dict, S_B: dict, out_dir: Path | str = FIG_DIR) -> dict:
    """Numbers the report quotes, written as LaTeX macros so text and code cannot drift."""
    out = Path(out_dir)
    rep = S_A["clean_report"]
    d = {
        "NRawRows": rep["rows_raw"], "NAfterItemDedup": rep["after_item_dedup"],
        "NAfterResidential": rep["after_residential"], "NAfterValidity": rep["after_validity"],
        "NFinal": rep["after_content_dedup"], "NTrain": len(S_A["y_train"]), "NVal": len(S_A["y_val"]),
        "NTest": len(S_A["y_test"]), "NFeatures": len(S_A["feature_names"]), "NFlats": S_A["n_groups"],
        "NSharedFlats": S_A["groups_shared_across_splits"],
    }
    fmt = {k: f"{v:,}" for k, v in d.items()}
    fmt["TierThreshold"] = f"{S_B['threshold']:,.0f}"
    fmt["PremiumTrain"], fmt["PremiumVal"], fmt["PremiumTest"] = (f"{S_B[k].mean():.3f}" for k in ("y_train", "y_val", "y_test"))
    leak = S_A.get("leakage", {})
    fmt["LeakTotalPriceCorr"] = f"{leak.get('total_price_corr_with_price', float('nan')):.4f}"
    fmt["LeakUnitPriceCorr"] = f"{leak.get('unit_price_x_area_corr_with_price', float('nan')):.4f}"
    tex = "% AUTO-GENERATED by src/eda.py - do not edit by hand\n" + "".join(
        f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in fmt.items())
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_summary.tex").write_text(tex, encoding="utf-8")
    summary = {**d, "threshold_azn": S_B["threshold"], "feature_names": S_A["feature_names"], "leakage_audit": leak,
               "premium_rate": {"train": float(S_B["y_train"].mean()), "val": float(S_B["y_val"].mean()), "test": float(S_B["y_test"].mean())},
               "group_aware_split": S_A["group_aware"], "cleaning_steps": rep}
    ev.save_json(summary, out / "data_summary.json")
    return summary
