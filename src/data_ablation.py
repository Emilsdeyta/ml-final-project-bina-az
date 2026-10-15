"""
data_ablation.py - evidence for the data/leakage decisions (report, Problem 1/Data & design).

    python -m src.data_ablation

Compares preprocessing variants with a fixed scikit-learn tree / ridge used purely
as a DIAGNOSTIC probe (not a core model). Scores are on the VALIDATION split only -
the test set is never touched. Shows:
  * imputation + feature engineering gain (V0 -> V2),
  * what a row-level split hides (V3: re-posted flats on both sides),
  * whether in-sample target encoding inflates scores (V4: leakage probe; the effect is
    modest with k=20 neighbours but the encoding is still invalid, so we never use it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep
from . import evaluate as ev

VARIANTS = [
    ("V0 median impute, one-hot districts", dict(imputation="median", use_encodings=False, onehot_top_k=30), True),
    ("V1 + group-wise imputation", dict(imputation="group", use_encodings=False, onehot_top_k=30), True),
    ("V2 + OOF district-TE + geo-kNN (FINAL)", dict(), True),
    ("V3 final features, row-level split", dict(), False),
    ("V4 in-sample encodings (LEAKY demo)", dict(leaky_insample_encodings=True), True),
]


def main(out_dir: Path | str = ev.FIG_DIR) -> list[dict]:
    rows = []
    for name, kw, grouped in VARIANTS:
        A = data_prep.get_splits("A", group_aware=grouped, preprocessor_kwargs=kw, verbose=False)
        B = data_prep.get_splits("B", group_aware=grouped, preprocessor_kwargs=kw, verbose=False)
        t = DecisionTreeRegressor(max_depth=10, min_samples_leaf=15, random_state=data_prep.SEED).fit(A["X_train"], A["y_train"])
        rg = Ridge(alpha=10.0).fit(A["X_train_scaled"], A["y_train"])
        c = DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, random_state=data_prep.SEED).fit(B["X_train"], B["y_train"])
        rows.append({
            "variant": name, "n_features": A["X_train"].shape[1], "shared_flats": A["groups_shared_across_splits"],
            "tree_r2_train": ev.r2(A["y_train"], t.predict(A["X_train"])), "tree_r2_val": ev.r2(A["y_val"], t.predict(A["X_val"])),
            "ridge_r2_val": ev.r2(A["y_val"], rg.predict(A["X_val_scaled"])),
            "clf_auc_val": roc_auc_score(B["y_val"], c.predict_proba(B["X_val"])[:, 1]),
        })
    hdr = f"{'variant':40s}{'feat':>5s}{'shared':>8s}{'tree R2 tr':>12s}{'tree R2 val':>12s}{'ridge R2 val':>13s}{'clf AUC val':>12s}"
    print("DATA ABLATION (validation only)\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:40s}{r['n_features']:5d}{r['shared_flats']:8d}{r['tree_r2_train']:12.3f}"
              f"{r['tree_r2_val']:12.3f}{r['ridge_r2_val']:13.3f}{r['clf_auc_val']:12.3f}")
    body = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Variant & Feat. & Shared flats & Tree $R^2$ tr/val & Ridge $R^2$ val & AUC val \\", r"\midrule"]
    for r in rows:
        body.append(f"{ev._tex_escape(r['variant'])} & {r['n_features']} & {r['shared_flats']} & "
                    f"{r['tree_r2_train']:.3f}/{r['tree_r2_val']:.3f} & {r['ridge_r2_val']:.3f} & {r['clf_auc_val']:.3f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_data.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    ev.save_json(rows, out / "ablation_data.json")
    return rows


if __name__ == "__main__":
    main()
