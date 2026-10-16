"""
run_all.py - ONE command reproduces every number and figure.

    python -m src.run_all                 # full run
    python -m src.run_all --only-data     # data + EDA only (while teammates finish)
    python -m src.run_all --fast          # 3-fold CV, 100 bootstrap draws

Pipeline: load -> clean -> leakage-safe features -> grouped split -> EDA ->
baselines -> teammates' experiments (tree, SVM) -> UNIFORM re-evaluation of every
model with the same metric code -> bootstrap SE + nested CV -> error analysis ->
results table (console + report/figures/results_table.tex + results.json).

Teammates' modules are called through the frozen interface (TEAM_PLAN section 3):
    experiments_tree.run(splits_a, splits_b) -> dict
    experiments_svm.run(splits_b)            -> dict
Each model entry may carry  "factory": zero-arg callable returning a FRESH model with
the FINAL config, and "scaled": bool (True = fit on X_*_scaled). With a factory this
script re-fits and re-scores the model itself, so every row of the results table
comes from the same evaluate.py code (fair comparison, nested CV, error analysis).
Entries without a factory are shown with the numbers the teammate reported.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import random
import time
from pathlib import Path

import numpy as np

from . import data_prep, eda
from . import evaluate as ev
from .linear_baseline import (MajorityClassifier, MeanRegressor, RidgeRegressionScratch,
                              select_ridge_alpha)

SEED = data_prep.SEED


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true", help="3-fold CV and 100 bootstrap draws")
    p.add_argument("--cv-folds", type=int, default=5, help="nested CV folds (<=1 disables)")
    p.add_argument("--bootstrap", type=int, default=300, help="bootstrap draws for test-set std")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--only-data", action="store_true", help="stop after data summary + EDA")
    p.add_argument("--no-group-split", action="store_true", help="row-level split (ablation only!)")
    p.add_argument("--strict", action="store_true", help="fail if a teammate module is missing")
    p.add_argument("--out-dir", default=str(ev.FIG_DIR), help="figures/tables directory")
    a = p.parse_args(argv)
    if a.fast:
        a.cv_folds, a.bootstrap = 3, 100
    return a


class Run:
    """Holds results of one pipeline execution."""

    def __init__(self, args, S_A, S_B):
        self.args, self.S = args, {"A": S_A, "B": S_B}
        self.out = Path(args.out_dir)
        self.results, self.extra, self.preds, self.notes = {}, {}, {}, []
        self._folds = {}

    def folds(self, task):
        if task not in self._folds:                       # built once, reused by every model
            self._folds[task] = list(data_prep.get_cv_folds(
                task, k=self.args.cv_folds, seed=SEED, group_aware=not self.args.no_group_split))
        return self._folds[task]

    def register(self, key, task, factory, scaled=False, reported=None):
        """Fit on train, score on test with OUR metrics, bootstrap SE, nested CV."""
        S = self.S[task]
        r = ev.fit_eval(task, factory, S, scaled=scaled, split="test")
        se = ev.bootstrap_se(task, S["y_test"], r["y_pred"], r["scores"], n_boot=self.args.bootstrap)
        cv = {}
        if self.args.cv_folds > 1:
            c = ev.cross_val_nested(factory, task, scaled=scaled, folds=self.folds(task))
            cv = {"mean": c["mean"], "std": c["std"]}
        rec = {"task": task, "test": r["metrics"], "test_se": se, "cv": cv, "source": "run_all",
               "fit_time": r["fit_time"], "predict_time": r["predict_time"], "scaled": scaled}
        if reported:                                       # cross-check the teammate's own numbers
            rec["reported_by_owner"] = {k: v for k, v in ev.normalize_result(reported)["test"].items()}
            for k, v in rec["reported_by_owner"].items():
                mine = r["metrics"].get(k)
                if mine is not None and abs(mine - v) > 0.02 * max(1.0, abs(mine)) and k not in ("fit_time", "predict_time"):
                    self.notes.append(f"[check] {key}: {k} reported {v:.4f} vs recomputed {mine:.4f}")
        self.results[key] = rec
        self.preds[key] = {"task": task, "y_pred": r["y_pred"], "scores": r["scores"]}
        main = ("rmse", "r2") if task == "A" else ("f1", "roc_auc")
        print(f"[model] {key:16s} task={task} " + " ".join(f"{m}={r['metrics'][m]:.4f}" for m in main)
              + f" fit={r['fit_time']:.2f}s")

    def absorb(self, out: dict, label: str):
        """Merge a teammate's run() output."""
        for key, val in out.items():
            if isinstance(val, dict) and callable(val.get("factory")):
                rec = ev.normalize_result(val)
                task = val.get("task") or (rec["task"] if rec else None)
                if task not in ("A", "B"):
                    self.notes.append(f"[warn] {label}.{key}: has a factory but no 'task' - skipped")
                    continue
                self.register(key, task, val["factory"], scaled=bool(val.get("scaled", False)), reported=val)
            elif ev.normalize_result(val) is not None:
                self.results[key] = {**ev.normalize_result(val), "source": f"reported by {label}"}
                self.notes.append(f"[note] {key}: no factory -> teammate's own numbers shown, no nested CV / error analysis")
            else:
                self.extra[key] = val


def _call_module(run: Run, name: str, call):
    try:
        mod = importlib.import_module(f".{name}", package=__package__)
        out = call(mod)
    except (ModuleNotFoundError, NotImplementedError) as e:
        if run.args.strict:
            raise
        print(f"[skip] {name}: not ready yet ({type(e).__name__}: {e})")
        return
    run.absorb(out, name)


def _error_analysis(run: Run):
    out = run.out
    A, B = run.S["A"], run.S["B"]
    a_keys = [k for k, p in run.preds.items() if p["task"] == "A"]
    b_keys = [k for k, p in run.preds.items() if p["task"] == "B"]
    if a_keys:                                              # best regression model by test RMSE (log)
        best = min(a_keys, key=lambda k: run.results[k]["test"]["rmse"])
        yp = run.preds[best]["y_pred"]
        we = ev.worst_errors(A["y_test"], yp, A["df_test"], n=20)
        we.to_csv(out / "worst_errors_taskA.csv", index=False)
        by_d = ev.error_by_group(A["y_test"], yp, A["df_test"]["district"])
        by_b = ev.error_by_group(A["y_test"], yp, ev.price_bucket(A["price_test"]), min_n=1)
        by_c = ev.error_by_group(A["y_test"], yp, A["df_test"]["building_category"], min_n=1)
        by_d.to_csv(out / "error_by_district_taskA.csv")
        by_b.to_csv(out / "error_by_price_bucket_taskA.csv")
        by_c.to_csv(out / "error_by_category_taskA.csv")
        ev.plot_pred_vs_true(A["price_test"], np.expm1(yp), out / "pred_vs_true_taskA.png", title=f"Task A: {ev.DISPLAY.get(best, best)}")
        ev.plot_error_table(by_d, out / "error_by_district_taskA.png", "mdape", "Hardest districts (median abs % error)")
        run.extra["error_analysis_taskA"] = {
            "model": best, "mdape_by_price_bucket": by_b["mdape"].round(4).to_dict(),
            "bias_pct_by_price_bucket": by_b["bias_pct"].round(2).to_dict(),
            "mdape_by_category": by_c["mdape"].round(4).to_dict(),
            "worst_district": None if by_d.empty else str(by_d.index[0])}
    if b_keys:
        ev.plot_roc_pr({ev.DISPLAY.get(k, k): (B["y_test"], run.preds[k]["scores"]) for k in b_keys}, out / "roc_pr_taskB.png")
        ev.plot_confusion_matrices({ev.DISPLAY.get(k, k): ev.confusion_matrix(B["y_test"], run.preds[k]["y_pred"]) for k in b_keys},
                                   out / "confusion_taskB.png")
        best = max(b_keys, key=lambda k: run.results[k]["test"].get("f1", 0))
        err = ev.classification_error_by_group(B["y_test"], run.preds[best]["y_pred"], B["df_test"]["district"])
        err.to_csv(out / "error_by_district_taskB.csv")
        by_b = ev.classification_error_by_group(B["y_test"], run.preds[best]["y_pred"], ev.price_bucket(B["price_test"]), min_n=1)
        by_b.to_csv(out / "error_by_price_bucket_taskB.csv")
        run.extra["error_analysis_taskB"] = {"model": best, "error_rate_by_price_bucket": by_b["error_rate"].round(4).to_dict()}


def main(argv=None) -> None:
    args = _parse(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    print(f"=== bina.az price project | seed={SEED} | python {platform.python_version()} | numpy {np.__version__} ===")
    ga = not args.no_group_split
    S_A = data_prep.get_splits("A", group_aware=ga)
    S_B = data_prep.get_splits("B", group_aware=ga)
    run = Run(args, S_A, S_B)
    out = run.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[clean] rows: raw={S_A['clean_report']['rows_raw']:,} -> final={S_A['clean_report']['after_content_dedup']:,} "
          f"| leakage audit: {S_A['leakage']}")
    eda.write_data_summary(S_A, S_B, out)
    if not args.skip_eda:
        eda.run_eda(S_A, S_B, out)
    if args.only_data:
        print(f"[done] data stage only, {time.time() - t0:.1f}s")
        return

    # --- reference baselines (ours, from scratch) -------------------------------
    run.register("mean_baseline", "A", MeanRegressor)
    alpha = select_ridge_alpha(S_A["X_train_scaled"], S_A["y_train"], S_A["X_val_scaled"], S_A["y_val"])
    run.register("ridge_scratch", "A", lambda: RidgeRegressionScratch(alpha), scaled=True)
    run.extra["ridge_alpha_selected_on_val"] = alpha
    run.register("majority_baseline", "B", MajorityClassifier)

    # --- teammates' experiments (frozen interface) ----------------------------
    _call_module(run, "experiments_tree", lambda m: m.run(S_A, S_B))
    _call_module(run, "experiments_svm", lambda m: m.run(S_B))

    _error_analysis(run)
    print()
    ev.compare(run.results, out_dir=out)
    for n in run.notes:
        print(n)
    ev.save_json({"seed": SEED, "group_aware_split": ga, "cv_folds": args.cv_folds, "bootstrap": args.bootstrap,
                  "results": run.results, "extra": run.extra, "notes": run.notes}, out / "results.json")
    print(f"\n[done] {time.time() - t0:.1f}s | tables: {out / 'results_table.tex'} | numbers: {out / 'results.json'}")


if __name__ == "__main__":
    main()
