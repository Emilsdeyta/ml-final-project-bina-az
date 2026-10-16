"""
experiments_svm.py — SVM (Pegasos) experiments, tuning, and sklearn comparison.

Interface (frozen, TEAM_PLAN §3):
    run(splits_b: dict) -> dict

Returned dict keys:
    "svm"    — our best Pegasos SVM metrics (test split)
    "sk_svc" — sklearn SVC baseline
    "sk_sgd" — sklearn SGDClassifier(loss="hinge") baseline
    "margin" — margin / feature-weight analysis

Figures saved to report/figures/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.svm import PegasosSVM, RandomFourierFeatures
from src import evaluate as ev

FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
SEED = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clf_metrics(svm: PegasosSVM, Xs: np.ndarray, y: np.ndarray) -> dict:
    y_pred = svm.predict(Xs)
    scores = svm.decision_function(Xs)
    return ev.classification_metrics(y, y_pred, scores)


# ---------------------------------------------------------------------------
# 1. lambda_ sweep — pick best val-F1  (Problem 4b)
# ---------------------------------------------------------------------------
def _lambda_sweep(Xs_tr, y_tr, Xs_va, y_va, n_iters: int = 15_000) -> dict:
    lambdas = np.logspace(-5, 0, 18)
    val_f1, val_auc = [], []
    for lam in lambdas:
        svm = PegasosSVM(lambda_=float(lam), n_iters=n_iters, random_state=SEED)
        svm.fit(Xs_tr, y_tr)
        m = _clf_metrics(svm, Xs_va, y_va)
        val_f1.append(m["f1"])
        val_auc.append(m.get("roc_auc", float("nan")))
    best_idx = int(np.argmax(val_f1))
    return {
        "lambdas": lambdas.tolist(),
        "val_f1": val_f1,
        "val_roc_auc": val_auc,
        "best_lambda": float(lambdas[best_idx]),
        "best_val_f1": float(val_f1[best_idx]),
    }


def _plot_lambda_sweep(sweep: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(sweep["lambdas"], sweep["val_f1"], "b-o", ms=4, label="Val F1")
    ax.semilogx(sweep["lambdas"], sweep["val_roc_auc"], "g--s", ms=4, label="Val ROC-AUC")
    ax.axvline(sweep["best_lambda"], color="r", ls=":", lw=1.5,
               label=f"Best λ={sweep['best_lambda']:.1e}")
    ax.set(xlabel="λ (regularisation)", ylabel="Score",
           title="Pegasos SVM — λ sweep (validation set)")
    ax.legend(); ax.grid(alpha=0.3)
    ev._save(fig, FIG_DIR / "svm_lambda_sweep.pdf")


# ---------------------------------------------------------------------------
# 2. Convergence plot  (Problem 5b)
# ---------------------------------------------------------------------------
def _plot_convergence(svm: PegasosSVM) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = svm.objective_history_
    iters = np.arange(1, len(history) + 1) * svm.log_every
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, history, lw=1.5, color="#2171b5")
    ax.set(xlabel="Iteration", ylabel="Objective J(w, b)",
           title="Pegasos objective convergence")
    ax.grid(alpha=0.3)
    ev._save(fig, FIG_DIR / "svm_convergence.pdf")


# ---------------------------------------------------------------------------
# 3. Learning-rate schedule comparison
# ---------------------------------------------------------------------------
def _schedule_comparison(Xs_tr, y_tr, Xs_va, y_va,
                          best_lambda: float, n_iters: int) -> dict:
    results = {}
    for sched in ("pegasos", "constant"):
        svm = PegasosSVM(lambda_=best_lambda, n_iters=n_iters,
                         schedule=sched, random_state=SEED)
        svm.fit(Xs_tr, y_tr)
        m = _clf_metrics(svm, Xs_va, y_va)
        results[sched] = {
            "val_f1": m["f1"],
            "val_roc_auc": m.get("roc_auc", float("nan")),
        }
    return results


# ---------------------------------------------------------------------------
# 4. RFF bandwidth sweep  (Problem 3d)
# ---------------------------------------------------------------------------
def _rff_sweep(Xs_tr, y_tr, Xs_va, y_va,
               best_lambda: float, n_iters: int) -> dict:
    gammas = np.logspace(-3, 2, 12)
    val_f1 = []
    for gamma in gammas:
        rff = RandomFourierFeatures(gamma=float(gamma), n_components=200, seed=SEED)
        Xr_tr = rff.fit_transform(Xs_tr)
        Xr_va = rff.transform(Xs_va)
        svm = PegasosSVM(lambda_=best_lambda, n_iters=n_iters, random_state=SEED)
        svm.fit(Xr_tr, y_tr)
        m = _clf_metrics(svm, Xr_va, y_va)
        val_f1.append(m["f1"])
    best_idx = int(np.argmax(val_f1))
    return {
        "gammas": gammas.tolist(),
        "val_f1": val_f1,
        "best_gamma": float(gammas[best_idx]),
        "best_val_f1": float(val_f1[best_idx]),
    }


def _plot_rff_sweep(rff_res: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(rff_res["gammas"], rff_res["val_f1"], "m-o", ms=4)
    ax.axvline(rff_res["best_gamma"], color="r", ls=":", lw=1.5,
               label=f"Best γ={rff_res['best_gamma']:.1e}")
    ax.set(xlabel="γ (RFF bandwidth)", ylabel="Val F1",
           title="RFF bandwidth sweep (validation set)")
    ax.legend(); ax.grid(alpha=0.3)
    ev._save(fig, FIG_DIR / "svm_rff_sweep.pdf")


# ---------------------------------------------------------------------------
# 5. Feature weight analysis  (Problem 6)
# ---------------------------------------------------------------------------
def _plot_feature_weights(svm: PegasosSVM,
                           feature_names: list[str], top_k: int = 15) -> dict:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = svm.w
    idx = np.argsort(np.abs(w))[::-1][:top_k]
    top_names = [feature_names[i] for i in idx]
    top_vals = w[idx]
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in top_vals]
    fig, ax = plt.subplots(figsize=(7, 0.4 * top_k + 1.5))
    ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Weight w_i", title=f"Top-{top_k} SVM feature weights (|w_i|)")
    ax.grid(alpha=0.3, axis="x")
    ev._save(fig, FIG_DIR / "svm_feature_weights.pdf")
    return {"top_features": top_names, "top_weights": top_vals.tolist()}


# ---------------------------------------------------------------------------
# 6. sklearn baselines  (Problem 5b)
# ---------------------------------------------------------------------------
def _sklearn_baselines(Xs_tr, y_tr, Xs_te, y_te, best_lambda: float) -> dict:
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier

    # SVC (X already scaled, no extra scaler needed)
    svc = SVC(C=1.0 / best_lambda, kernel="rbf",
              probability=True, random_state=SEED)
    t0 = time.perf_counter()
    svc.fit(Xs_tr, y_tr)
    svc_fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    svc_pred = svc.predict(Xs_te)
    svc_pred_t = time.perf_counter() - t0
    svc_scores = svc.predict_proba(Xs_te)[:, 1]

    # SGDClassifier — linear SVM via SGD (comparable to Pegasos)
    sgd = SGDClassifier(loss="hinge", alpha=best_lambda,
                        max_iter=1000, random_state=SEED, tol=1e-4)
    t0 = time.perf_counter()
    sgd.fit(Xs_tr, y_tr)
    sgd_fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    sgd_pred = sgd.predict(Xs_te)
    sgd_pred_t = time.perf_counter() - t0
    sgd_scores = sgd.decision_function(Xs_te)

    return {
        "sk_svc": {
            **ev.classification_metrics(y_te, svc_pred, svc_scores),
            "fit_time": svc_fit_t, "predict_time": svc_pred_t,
        },
        "sk_sgd": {
            **ev.classification_metrics(y_te, sgd_pred, sgd_scores),
            "fit_time": sgd_fit_t, "predict_time": sgd_pred_t,
        },
    }


# ---------------------------------------------------------------------------
# Main entry point  (frozen interface — TEAM_PLAN §3)
# ---------------------------------------------------------------------------
def run(splits_b: dict) -> dict:
    """Run all SVM experiments. Returns metrics dict for evaluate.compare().

    Uses only X_*_scaled arrays (SVM requires standardised input).
    Final hyperparameters chosen on val set only — test set never seen during
    selection (team rule #8).
    """
    Xs_tr = splits_b["X_train_scaled"]
    Xs_va = splits_b["X_val_scaled"]
    Xs_te = splits_b["X_test_scaled"]
    y_tr  = splits_b["y_train"]
    y_va  = splits_b["y_val"]
    y_te  = splits_b["y_test"]
    feat  = splits_b["feature_names"]

    # 1. Lambda sweep
    print("\n[svm] 1/6 — λ sweep …")
    sweep = _lambda_sweep(Xs_tr, y_tr, Xs_va, y_va)
    _plot_lambda_sweep(sweep)
    best_lam = sweep["best_lambda"]
    print(f"[svm]   best λ={best_lam:.2e}  val-F1={sweep['best_val_f1']:.4f}")

    # 2. Final model (val selection done, now train with best params)
    print("[svm] 2/6 — Final model …")
    best_svm = PegasosSVM(lambda_=best_lam, n_iters=20_000,
                           log_every=200, random_state=SEED)
    t0 = time.perf_counter()
    best_svm.fit(Xs_tr, y_tr)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = best_svm.predict(Xs_te)
    pred_time = time.perf_counter() - t0

    val_m  = _clf_metrics(best_svm, Xs_va, y_va)
    test_m = _clf_metrics(best_svm, Xs_te, y_te)
    print(f"[svm]   val  F1={val_m['f1']:.4f}  AUC={val_m.get('roc_auc',0):.4f}")
    print(f"[svm]   test F1={test_m['f1']:.4f}  AUC={test_m.get('roc_auc',0):.4f}")
    _plot_convergence(best_svm)

    # 3. Schedule comparison
    print("[svm] 3/6 — Schedule comparison …")
    schedules = _schedule_comparison(Xs_tr, y_tr, Xs_va, y_va, best_lam, 15_000)

    # 4. RFF sweep
    print("[svm] 4/6 — RFF γ sweep …")
    rff_res = _rff_sweep(Xs_tr, y_tr, Xs_va, y_va, best_lam, 15_000)
    _plot_rff_sweep(rff_res)
    print(f"[svm]   best γ={rff_res['best_gamma']:.2e}  RFF val-F1={rff_res['best_val_f1']:.4f}")

    # 5. Feature weights + margin
    print("[svm] 5/6 — Feature weights & margin …")
    weight_info = _plot_feature_weights(best_svm, feat)
    margin_info = {
        "norm_w": best_svm.norm_w(),
        "margin_width": best_svm.margin_width(),
        "n_support_vectors_train": int(
            best_svm.support_vector_mask(Xs_tr, y_tr).sum()),
        **weight_info,
    }

    # 6. sklearn baselines
    print("[svm] 6/6 — sklearn baselines …")
    sk = _sklearn_baselines(Xs_tr, y_tr, Xs_te, y_te, best_lam)
    print(f"[svm]   SVC F1={sk['sk_svc']['f1']:.4f}  SGD F1={sk['sk_sgd']['f1']:.4f}")

    # ROC + PR
    ev.plot_roc_pr(
        {"Pegasos SVM": (y_te, best_svm.decision_function(Xs_te))},
        FIG_DIR / "svm_roc_pr.pdf",
    )

    # Capture best_lam in a default arg to avoid late-binding closure issue
    results = {
        "svm": {
            **test_m,
            "fit_time": fit_time, "predict_time": pred_time,
            "val_f1": val_m["f1"],
            "best_lambda": best_lam,
            "schedule_comparison": schedules,
            "lambda_sweep": sweep,
            "rff_sweep": rff_res,
            # run_all.py uses these 3 keys to re-fit + re-score uniformly
            "factory": (lambda lam: lambda: PegasosSVM(
                lambda_=lam, n_iters=20_000, log_every=200, random_state=SEED
            ))(best_lam),
            "task": "B",
            "scaled": True,
        },
        "sk_svc": {
            **sk["sk_svc"],
            "factory": (lambda lam: lambda: __import__(
                "sklearn.svm", fromlist=["SVC"]
            ).SVC(C=1.0 / lam, kernel="rbf", probability=True, random_state=SEED))(best_lam),
            "task": "B",
            "scaled": True,
        },
        "sk_sgd": {
            **sk["sk_sgd"],
            "factory": (lambda lam: lambda: __import__(
                "sklearn.linear_model", fromlist=["SGDClassifier"]
            ).SGDClassifier(loss="hinge", alpha=lam, max_iter=1000,
                            random_state=SEED, tol=1e-4))(best_lam),
            "task": "B",
            "scaled": True,
        },
        "margin": margin_info,
    }

    ev.save_json(results, FIG_DIR / "svm_results.json")
    print("[svm] Done. Figures → report/figures/")
    return results
