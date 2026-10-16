"""
evaluate.py - metrics (NumPy only), uncertainty, error analysis, figures, results table.

scikit-learn is NOT imported here; the tests compare every metric against it.

Conventions
-----------
Task A models predict y = log1p(price). ``regression_metrics`` reports both the
log scale (what the model optimises) and AZN (what a buyer cares about).
Task B: label 1 = premium. ``scores`` = any monotone score (SVM margin, tree P(1)).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

SEED = 42
FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


# --------------------------------------------------------------------------- #
# Regression (Task A)
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    d = _a(y_true) - _a(y_pred)
    return float(np.sqrt(np.mean(d * d)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_a(y_true) - _a(y_pred))))


def r2(y_true, y_pred) -> float:
    yt, yp = _a(y_true), _a(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true_log, y_pred_log) -> dict:
    """Metrics on log1p(price) (suffix-free keys) and on AZN (``*_price``)."""
    yt, yp = _a(y_true_log), _a(y_pred_log)
    pt, pp = np.expm1(yt), np.expm1(np.clip(yp, None, 30.0))
    return {
        "rmse": rmse(yt, yp), "mae": mae(yt, yp), "r2": r2(yt, yp),
        "rmse_price": rmse(pt, pp), "mae_price": mae(pt, pp), "r2_price": r2(pt, pp),
        "mdape": float(np.median(np.abs(pp - pt) / pt)),       # median absolute % error
    }


# --------------------------------------------------------------------------- #
# Classification (Task B)
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """[[TN, FP], [FN, TP]] for labels {0, 1}."""
    yt, yp = _a(y_true).astype(int), _a(y_pred).astype(int)
    return np.array([[np.sum((yt == 0) & (yp == 0)), np.sum((yt == 0) & (yp == 1))],
                     [np.sum((yt == 1) & (yp == 0)), np.sum((yt == 1) & (yp == 1))]], dtype=int)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(_a(y_true).astype(int) == _a(y_pred).astype(int)))


def precision_recall_f1(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the average rank."""
    a = _a(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    new = np.ones(len(a), dtype=bool)
    new[1:] = sa[1:] != sa[:-1]
    block = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(a))
    ranks = np.empty(len(a))
    ranks[order] = ((starts + ends + 1) / 2.0)[block]
    return ranks


def roc_auc(y_true, scores) -> float:
    """Mann-Whitney U / rank formulation (exact with ties)."""
    y, s = _a(y_true).astype(int), _a(scores)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cum_counts(y_true, scores):
    y, s = _a(y_true).astype(int), _a(scores)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.flatnonzero(np.diff(s)), len(y) - 1]        # end of each tie block
    return tps[last].astype(float), fps[last].astype(float)


def roc_curve(y_true, scores):
    tps, fps = _cum_counts(y_true, scores)
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def pr_curve(y_true, scores):
    tps, fps = _cum_counts(y_true, scores)
    return np.r_[0.0, tps / tps[-1]], np.r_[1.0, tps / (tps + fps)]    # recall, precision


def pr_auc(y_true, scores) -> float:
    """Average precision (step-wise area under the PR curve)."""
    if int(_a(y_true).sum()) == 0:
        return float("nan")
    rec, prec = pr_curve(y_true, scores)
    return float(np.sum(np.diff(rec) * prec[1:]))


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    p, r, f = precision_recall_f1(y_true, y_pred)
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    out = {"accuracy": accuracy(y_true, y_pred), "precision": p, "recall": r, "f1": f,
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if scores is not None:
        out["roc_auc"], out["pr_auc"] = roc_auc(y_true, scores), pr_auc(y_true, scores)
    return out


# --------------------------------------------------------------------------- #
# Model helpers (anything with fit/predict works - ours and sklearn's)
# --------------------------------------------------------------------------- #
def model_scores(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        s = np.asarray(model.predict_proba(X))
        s = s[:, 1] if s.ndim == 2 else s
    else:
        s = model.predict(X)
    return _a(s)


def evaluate_arrays(task: str, y_true, y_pred, scores=None) -> dict:
    return regression_metrics(y_true, y_pred) if task == "A" else classification_metrics(y_true, y_pred, scores)


def fit_eval(task: str, factory, S: dict, scaled: bool = False, split: str = "test") -> dict:
    """Fit a fresh model on the train split, evaluate on ``split`` ("val" or "test").
    Times fit and predict. Returns metrics + predictions (for error analysis)."""
    sfx = "_scaled" if scaled else ""
    Xtr, Xev, yev = S["X_train" + sfx], S[f"X_{split}{sfx}"], S[f"y_{split}"]
    model = factory()
    t0 = time.perf_counter()
    model.fit(Xtr, S["y_train"])
    t1 = time.perf_counter()
    y_pred = _a(model.predict(Xev))
    t2 = time.perf_counter()
    scores = model_scores(model, Xev) if task == "B" else None
    return {"metrics": evaluate_arrays(task, yev, y_pred, scores), "fit_time": t1 - t0,
            "predict_time": t2 - t1, "y_pred": y_pred, "scores": scores, "model": model}


# --------------------------------------------------------------------------- #
# Uncertainty: bootstrap (test set) and cross-validation (train+val)
# --------------------------------------------------------------------------- #
def bootstrap_se(task: str, y_true, y_pred, scores=None, n_boot: int = 300, seed: int = SEED) -> dict:
    """Std and 95% percentile interval of every metric over bootstrap resamples."""
    yt, yp = _a(y_true), _a(y_pred)
    sc = None if scores is None else _a(scores)
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        runs.append(evaluate_arrays(task, yt[i], yp[i], None if sc is None else sc[i]))
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) > 1:
            out[k] = {"std": float(v.std(ddof=1)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return out


def kfold_indices(n: int, k: int = 5, seed: int = SEED, y=None, groups=None):
    """List of (train_idx, val_idx). Stratified when ``y`` is binary and no groups;
    group-aware (a group never straddles folds) when ``groups`` is given."""
    rng = np.random.default_rng(seed)
    if groups is not None:
        ug, inv = np.unique(groups, return_inverse=True)
        fold = (rng.permutation(len(ug)) % k)[inv]
    elif y is not None and len(np.unique(y)) == 2:
        fold = np.empty(n, dtype=int)
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(np.asarray(y) == c))
            fold[idx] = np.arange(len(idx)) % k
    else:
        fold = rng.permutation(n) % k
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def summarize_folds(fold_metrics: list[dict]) -> dict:
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float, np.floating))]
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) for k in keys}
    return {"mean": mean, "std": std, "folds": fold_metrics}


def cross_val_mean_std(model_factory, X, y, k: int = 5, seed: int = SEED, task: str | None = None,
                       groups=None) -> dict:
    """Plain K-fold on ready-made arrays -> mean +- std per metric.

    NOTE: X was already preprocessed on the whole training set, so target-based
    features saw other folds' labels. Fine for model-selection sweeps; for the
    headline numbers prefer ``cross_val_nested`` which re-fits the preprocessing
    inside every fold.
    """
    X, y = np.asarray(X), np.asarray(y)
    task = task or ("B" if set(np.unique(y)) <= {0, 1} else "A")
    res = []
    for tr, va in kfold_indices(len(y), k, seed, y=y if task == "B" else None, groups=groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        res.append(evaluate_arrays(task, y[va], m.predict(X[va]), model_scores(m, X[va]) if task == "B" else None))
    return summarize_folds(res)


def cross_val_nested(model_factory, task: str, k: int = 5, seed: int = SEED, scaled: bool = False,
                     folds=None, **fold_kwargs) -> dict:
    """Leakage-proof CV on train+val: ``data_prep.get_cv_folds`` re-fits imputation,
    clipping, OOF encodings, tier threshold and scaler inside every fold."""
    from . import data_prep
    sfx = "_scaled" if scaled else ""
    res = []
    for F in (folds if folds is not None else data_prep.get_cv_folds(task, k=k, seed=seed, **fold_kwargs)):
        m = model_factory()
        m.fit(F["X_train" + sfx], F["y_train"])
        Xv = F["X_val" + sfx]
        res.append(evaluate_arrays(task, F["y_val"], m.predict(Xv), model_scores(m, Xv) if task == "B" else None))
    return summarize_folds(res)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
PRICE_EDGES = (0, 100_000, 200_000, 400_000, 800_000, np.inf)


def price_bucket(price, edges=PRICE_EDGES) -> np.ndarray:
    labels = [f"{int(lo / 1000)}k-{'inf' if np.isinf(hi) else str(int(hi / 1000)) + 'k'}" for lo, hi in zip(edges[:-1], edges[1:])]
    return np.array(labels)[np.clip(np.digitize(_a(price), edges[1:-1]), 0, len(labels) - 1)]


def worst_errors(y_true_log, y_pred_log, df, n: int = 20):
    """The n listings with the largest absolute error in AZN (rows of ``df`` must align with y)."""
    pt, pp = np.expm1(_a(y_true_log)), np.expm1(np.clip(_a(y_pred_log), None, 30.0))
    out = df.reset_index(drop=True).copy()
    out["price_true"], out["price_pred"] = pt, pp
    out["abs_err"], out["pct_err"] = np.abs(pp - pt), (pp - pt) / pt * 100.0
    cols = ["item_id", "district", "building_category", "area_m2", "rooms",
            "price_true", "price_pred", "abs_err", "pct_err"]
    return out.sort_values("abs_err", ascending=False).head(n)[[c for c in cols if c in out]].reset_index(drop=True)


def error_by_group(y_true_log, y_pred_log, groups, min_n: int = 30):
    """Per-group count, MAE (AZN), median absolute % error and signed bias (%)."""
    import pandas as pd
    pt, pp = np.expm1(_a(y_true_log)), np.expm1(np.clip(_a(y_pred_log), None, 30.0))
    d = pd.DataFrame({"g": np.asarray(groups).astype(str), "abs": np.abs(pp - pt),
                      "ape": np.abs(pp - pt) / pt, "bias": (pp - pt) / pt * 100.0})
    t = d.groupby("g").agg(n=("abs", "size"), mae_azn=("abs", "mean"),
                           mdape=("ape", "median"), bias_pct=("bias", "mean"))
    return t[t["n"] >= min_n].sort_values("mdape", ascending=False)


def classification_error_by_group(y_true, y_pred, groups, min_n: int = 30):
    import pandas as pd
    d = pd.DataFrame({"g": np.asarray(groups).astype(str), "err": _a(y_true).astype(int) != _a(y_pred).astype(int),
                      "fn": (_a(y_true) == 1) & (_a(y_pred) == 0), "fp": (_a(y_true) == 0) & (_a(y_pred) == 1)})
    t = d.groupby("g").agg(n=("err", "size"), error_rate=("err", "mean"), fn_rate=("fn", "mean"), fp_rate=("fp", "mean"))
    return t[t["n"] >= min_n].sort_values("error_rate", ascending=False)


# --------------------------------------------------------------------------- #
# Figures (matplotlib imported lazily; headless-safe)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            if path.exists():
                path.unlink()
            fig.savefig(path, dpi=160, bbox_inches="tight")
            break
        except OSError as e:
            if attempt == 3:
                print(f"[warn] could not write {path.name}: {e}")
            else:
                time.sleep(0.7)
    _plt().close(fig)
    return path


def plot_pred_vs_true(price_true, price_pred, path, title: str = ""):
    plt = _plt()
    pt, pp = _a(price_true), np.clip(_a(price_pred), 1.0, None)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    h = ax[0].hexbin(np.log10(pt), np.log10(pp), gridsize=55, bins="log", mincnt=1, cmap="viridis")
    lo, hi = np.log10(min(pt.min(), pp.min())), np.log10(max(pt.max(), pp.max()))
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[0].set(xlabel="true price, log10 AZN", ylabel="predicted price, log10 AZN", title=title or "Predicted vs true")
    fig.colorbar(h, ax=ax[0], label="count")
    res = np.log(pp) - np.log(pt)
    ax[1].hist(res, bins=60, color="#4c72b0")
    ax[1].axvline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="log residual (pred - true)", ylabel="count", title="Residuals")
    return _save(fig, path)


def plot_roc_pr(curves: dict, path):
    """curves: {name: (y_true, scores)}"""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for name, (y, s) in curves.items():
        fpr, tpr = roc_curve(y, s)
        rec, prec = pr_curve(y, s)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc(y, s):.3f})")
        ax[1].plot(rec, prec, label=f"{name} (AP={pr_auc(y, s):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    ax[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=.3)
    return _save(fig, path)


def plot_confusion_matrices(cms: dict, path):
    plt = _plt()
    n = len(cms)
    fig, ax = plt.subplots(1, n, figsize=(3.1 * n, 3.1), squeeze=False)
    for a, (name, cm) in zip(ax[0], cms.items()):
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["standard", "premium"], yticklabels=["standard", "premium"],
              xlabel="predicted", ylabel="true", title=name)
        a.title.set_fontsize(8)
    return _save(fig, path)


def plot_error_table(table, path, value: str, title: str, top: int = 15):
    plt = _plt()
    t = table.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(t) + 1.2))
    ax.barh(t.index, t[value], color="#c44e52")
    ax.set(xlabel=value, title=title)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Results table (console + LaTeX)
# --------------------------------------------------------------------------- #
DISPLAY = {
    "ridge_scratch": "Ridge, normal eq. (ours)", "mean_baseline": "Mean predictor",
    "majority_baseline": "Majority class",
    "tree_reg": "Decision tree (ours)", "sk_tree_reg": "sklearn DecisionTreeRegressor",
    "tree_clf": "Decision tree (ours)", "sk_tree_clf": "sklearn DecisionTreeClassifier",
    "svm": "Pegasos SVM (ours)", "sk_svc": "sklearn SVC", "sk_sgd": "sklearn SGD (hinge)",
}
ALIASES = {"auc": "roc_auc", "acc": "accuracy", "train_time": "fit_time", "fit_s": "fit_time",
           "r2_log": "r2", "rmse_log": "rmse", "mae_log": "mae"}
A_COLS = [("rmse", "RMSE (log)"), ("mae", "MAE (log)"), ("r2", r"$R^2$ (log)"),
          ("r2_price", r"$R^2$ (AZN)"), ("mae_price", "MAE (AZN)"), ("fit_time", "Fit (s)")]
B_COLS = [("accuracy", "Acc."), ("precision", "Prec."), ("recall", "Rec."), ("f1", "F1"),
          ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("fit_time", "Fit (s)")]
A_CV, B_CV = ["rmse", "mae", "r2"], ["accuracy", "f1", "roc_auc"]
LOWER_BETTER = {"rmse", "mae", "rmse_price", "mae_price", "mdape", "fit_time"}
_A_KEYS = {"rmse", "mae", "r2", "rmse_price", "mae_price", "r2_price"}
_B_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def normalize_result(raw) -> dict | None:
    """Accept a teammate's flat metric dict OR our canonical record; return
    {"task","test","test_se","cv"} or None when ``raw`` is not a metric dict."""
    if not isinstance(raw, dict):
        return None
    src = raw.get("test", raw)
    norm = {}
    for k, v in src.items():
        key = ALIASES.get(str(k).lower(), str(k).lower())
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v):
            norm[key] = float(v)
    for t in ("fit_time", "predict_time"):
        if t in raw and isinstance(raw[t], (int, float)):
            norm[t] = float(raw[t])
    task = raw.get("task") or ("A" if _A_KEYS & set(norm) else "B" if _B_KEYS & set(norm) else None)
    if task is None:
        return None
    return {"task": task, "test": norm, "test_se": raw.get("test_se", {}), "cv": raw.get("cv", {})}


def _fmt(key, v, se=None):
    if v is None or not math.isfinite(v):
        return "--"
    d = 0 if key in ("mae_price", "rmse_price") else (2 if key == "fit_time" else 3)
    s = f"{v:,.{d}f}" if d == 0 else f"{v:.{d}f}"
    if se is not None and math.isfinite(se) and key != "fit_time":
        s += f" $\\pm$ {se:,.{d}f}" if d == 0 else f" $\\pm$ {se:.{d}f}"
    return s


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _tabular(rows: list[tuple], cols, bold_best: bool = True) -> str:
    """rows: (display_name, test_dict, se_dict) ; returns a booktabs tabular."""
    best = {}
    for key, _ in cols:
        vals = [(r[1].get(key), i) for i, r in enumerate(rows) if r[1].get(key) is not None and math.isfinite(r[1][key])]
        if vals and key != "fit_time":
            best[key] = (min if key in LOWER_BETTER else max)(vals)[1]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for i, (name, test, se) in enumerate(rows):
        cells = []
        for key, _ in cols:
            c = _fmt(key, test.get(key), (se.get(key) or {}).get("std") if isinstance(se.get(key), dict) else se.get(key))
            cells.append(r"\textbf{" + c + "}" if bold_best and best.get(key) == i else c)
        lines.append(_tex_escape(name) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _console(title, rows, cols) -> str:
    head = f"{'Model':34s}" + "".join(f"{h.replace('$', '').replace(chr(92), ''):>22s}" for _, h in cols)
    out = [title, head, "-" * len(head)]
    for name, test, se in rows:
        cells = []
        for key, _ in cols:
            c = _fmt(key, test.get(key), (se.get(key) or {}).get("std") if isinstance(se.get(key), dict) else se.get(key))
            cells.append(f"{c.replace(chr(36) + chr(92) + 'pm' + chr(36), '+-'):>22s}")
        out.append(f"{name[:33]:34s}" + "".join(cells))
    return "\n".join(out)


def compare(results: dict, out_dir: Path | str = FIG_DIR, verbose: bool = True) -> str:
    """Build the results tables from {key: metrics-dict}. Non-metric entries
    (feature importances, margin info, ...) are skipped. Writes
    report/figures/results_table.tex and returns the console text."""
    A, B, cvA, cvB = [], [], [], []
    for key, raw in results.items():
        rec = normalize_result(raw)
        if rec is None:
            continue
        name = DISPLAY.get(key, key)
        (A if rec["task"] == "A" else B).append((name, rec["test"], rec["test_se"]))
        cv = rec["cv"]
        if cv and cv.get("mean"):
            row = (name, cv["mean"], cv.get("std", {}))
            (cvA if rec["task"] == "A" else cvB).append(row)

    blocks, tex = [], []
    if A:
        blocks.append(_console("TASK A - price regression (test set; +- = bootstrap std)", A, A_COLS))
        tex.append(("tab:results_a", "Task A (price regression), test set. $\\pm$: bootstrap std; best per column in bold.", _tabular(A, A_COLS)))
    if B:
        blocks.append(_console("TASK B - price tier classification (test set; +- = bootstrap std)", B, B_COLS))
        tex.append(("tab:results_b", "Task B (price tier), test set. $\\pm$: bootstrap std; best per column in bold.", _tabular(B, B_COLS)))
    if cvA:
        cols = [c for c in A_COLS if c[0] in A_CV]
        blocks.append(_console("TASK A - 5-fold nested CV on train+val (mean +- std)", cvA, cols))
        tex.append(("tab:cv_a", "Task A, nested cross-validation on train+val (mean $\\pm$ std over folds).", _tabular(cvA, cols)))
    if cvB:
        cols = [c for c in B_COLS if c[0] in B_CV]
        blocks.append(_console("TASK B - 5-fold nested CV on train+val (mean +- std)", cvB, cols))
        tex.append(("tab:cv_b", "Task B, nested cross-validation on train+val (mean $\\pm$ std over folds).", _tabular(cvB, cols)))

    text = "\n\n".join(blocks) if blocks else "(no metric results to compare)"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join("\\begin{table*}[t]\n\\centering\\small\n\\caption{%s}\\label{%s}\n%s\n\\end{table*}" % (cap, lab, tab)
                       for lab, cap, tab in tex)
    (out_dir / "results_table.tex").write_text(body + "\n", encoding="utf-8")
    if verbose:
        print(text)
    return text


# --------------------------------------------------------------------------- #
def to_jsonable(o):
    """JSON-safe copy: numpy -> python, drops callables/models/large arrays."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not callable(v) and k not in ("model", "y_pred", "scores", "factory")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 200 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
