"""Decision-tree experiments, validation tuning, and sklearn comparison."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src import evaluate as ev
from src.decision_tree import DecisionTree

FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
SEED = 42
TUNING_SAMPLES = 8_000


REG_CONFIGS = [
    {"criterion": "mse", "max_depth": 6, "min_samples_leaf": 15},
    {"criterion": "mse", "max_depth": 8, "min_samples_leaf": 15},
    {"criterion": "mse", "max_depth": 10, "min_samples_leaf": 15},
    {"criterion": "mse", "max_depth": 12, "min_samples_leaf": 15},
    {"criterion": "mse", "max_depth": 10, "min_samples_leaf": 30},
]

CLF_CONFIGS = [
    {"criterion": "gini", "max_depth": 6, "min_samples_leaf": 15},
    {"criterion": "gini", "max_depth": 8, "min_samples_leaf": 15},
    {"criterion": "gini", "max_depth": 10, "min_samples_leaf": 15},
    {"criterion": "gini", "max_depth": 12, "min_samples_leaf": 15},
    {"criterion": "gini", "max_depth": 10, "min_samples_leaf": 30},
    {"criterion": "entropy", "max_depth": 10, "min_samples_leaf": 15},
]


def _metrics(model, X, y, task):
    """Calculate the correct metrics for regression or classification."""
    y_pred = model.predict(X)

    if task == "regression":
        return ev.regression_metrics(y, y_pred)

    scores = model.predict_proba(X)[:, 1]
    return ev.classification_metrics(y, y_pred, scores)


def _tune(X_train, y_train, X_val, y_val, task, configs):
    """Choose hyperparameters using validation data only."""
    rng = np.random.default_rng(SEED)
    n = min(TUNING_SAMPLES, len(X_train))
    indices = rng.choice(len(X_train), size=n, replace=False)

    X_tune = X_train[indices]
    y_tune = y_train[indices]

    metric_name = "r2" if task == "regression" else "roc_auc"
    records = []

    for config in configs:
        model = DecisionTree(task=task, random_state=SEED, **config)

        start = time.perf_counter()
        model.fit(X_tune, y_tune)
        fit_time = time.perf_counter() - start

        metrics = _metrics(model, X_val, y_val, task)
        score = float(metrics[metric_name])

        records.append(
            {
                "config": dict(config),
                "validation_score": score,
                "fit_time": fit_time,
            }
        )

        print(
            f"[tree:{task}] {config} "
            f"val-{metric_name}={score:.4f}"
        )

    best = max(records, key=lambda result: result["validation_score"])
    return dict(best["config"]), records


def _custom_factory(task, config):
    """Return a callable that creates a fresh custom tree."""
    frozen_config = dict(config)

    return lambda: DecisionTree(
        task=task,
        random_state=SEED,
        **frozen_config,
    )


def _sklearn_regression_factory(config):
    from sklearn.tree import DecisionTreeRegressor

    return lambda: DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=SEED,
    )


def _sklearn_classification_factory(config):
    from sklearn.tree import DecisionTreeClassifier

    return lambda: DecisionTreeClassifier(
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=SEED,
    )


def _fit_and_measure(model, X_train, y_train, X_eval, y_eval, task):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_eval)
    predict_time = time.perf_counter() - start

    if task == "regression":
        metrics = ev.regression_metrics(y_eval, y_pred)
    else:
        scores = model.predict_proba(X_eval)[:, 1]
        metrics = ev.classification_metrics(y_eval, y_pred, scores)

    return metrics, fit_time, predict_time


def _feature_importance_info(reg_tree, clf_tree, feature_names):
    names = list(feature_names)
    reg_values = reg_tree.feature_importances_
    clf_values = clf_tree.feature_importances_

    reg_order = np.argsort(reg_values)[::-1]
    clf_order = np.argsort(clf_values)[::-1]

    return {
        "regression": {
            "feature_names": names,
            "values": reg_values.tolist(),
            "top_features": [names[i] for i in reg_order[:15]],
        },
        "classification": {
            "feature_names": names,
            "values": clf_values.tolist(),
            "top_features": [names[i] for i in clf_order[:15]],
        },
    }


def run(splits_a: dict, splits_b: dict) -> dict:
    """Run regression and classification decision-tree experiments."""

    # Raw, unscaled features are correct for decision trees.
    Xa_tr, ya_tr = splits_a["X_train"], splits_a["y_train"]
    Xa_va, ya_va = splits_a["X_val"], splits_a["y_val"]
    Xa_te, ya_te = splits_a["X_test"], splits_a["y_test"]

    Xb_tr, yb_tr = splits_b["X_train"], splits_b["y_train"]
    Xb_va, yb_va = splits_b["X_val"], splits_b["y_val"]
    Xb_te, yb_te = splits_b["X_test"], splits_b["y_test"]

    feature_names = splits_a["feature_names"]

    # Hyperparameter selection: train subset -> validation set.
    print("\n[tree] 1/5 — Regression validation tuning")
    best_reg, reg_tuning = _tune(
        Xa_tr, ya_tr, Xa_va, ya_va, "regression", REG_CONFIGS
    )

    print("\n[tree] 2/5 — Classification validation tuning")
    best_clf, clf_tuning = _tune(
        Xb_tr, yb_tr, Xb_va, yb_va, "classification", CLF_CONFIGS
    )

    # Final custom models trained on the full training split.
    print("\n[tree] 3/5 — Final custom trees")
    reg_tree = DecisionTree(
        task="regression", random_state=SEED, **best_reg
    )
    clf_tree = DecisionTree(
        task="classification", random_state=SEED, **best_clf
    )

    reg_test, reg_fit_time, reg_predict_time = _fit_and_measure(
        reg_tree, Xa_tr, ya_tr, Xa_te, ya_te, "regression"
    )
    clf_test, clf_fit_time, clf_predict_time = _fit_and_measure(
        clf_tree, Xb_tr, yb_tr, Xb_te, yb_te, "classification"
    )

    reg_val = _metrics(reg_tree, Xa_va, ya_va, "regression")
    clf_val = _metrics(clf_tree, Xb_va, yb_va, "classification")

    print(
        f"[tree] regression: val-R2={reg_val['r2']:.4f}, "
        f"test-R2={reg_test['r2']:.4f}"
    )
    print(
        f"[tree] classification: val-AUC={clf_val['roc_auc']:.4f}, "
        f"test-AUC={clf_test['roc_auc']:.4f}"
    )

    # sklearn comparison with the same selected complexity.
    print("\n[tree] 4/5 — sklearn baselines")
    sk_reg_factory = _sklearn_regression_factory(best_reg)
    sk_clf_factory = _sklearn_classification_factory(best_clf)

    sk_reg, sk_reg_fit, sk_reg_predict = _fit_and_measure(
        sk_reg_factory(), Xa_tr, ya_tr, Xa_te, ya_te, "regression"
    )
    sk_clf, sk_clf_fit, sk_clf_predict = _fit_and_measure(
        sk_clf_factory(), Xb_tr, yb_tr, Xb_te, yb_te, "classification"
    )

    print(
        f"[tree] sklearn regression test-R2={sk_reg['r2']:.4f}"
    )
    print(
        f"[tree] sklearn classification test-AUC="
        f"{sk_clf['roc_auc']:.4f}"
    )

    print("\n[tree] 5/5 — Feature importance")
    importance = _feature_importance_info(
        reg_tree, clf_tree, feature_names
    )

    results = {
        "tree_reg": {
            **reg_test,
            "fit_time": reg_fit_time,
            "predict_time": reg_predict_time,
            "val_r2": reg_val["r2"],
            "best_config": best_reg,
            "tuning": reg_tuning,
            "factory": _custom_factory("regression", best_reg),
            "task": "A",
            "scaled": False,
        },
        "tree_clf": {
            **clf_test,
            "fit_time": clf_fit_time,
            "predict_time": clf_predict_time,
            "val_roc_auc": clf_val["roc_auc"],
            "best_config": best_clf,
            "tuning": clf_tuning,
            "factory": _custom_factory("classification", best_clf),
            "task": "B",
            "scaled": False,
        },
        "sk_tree_reg": {
            **sk_reg,
            "fit_time": sk_reg_fit,
            "predict_time": sk_reg_predict,
            "factory": sk_reg_factory,
            "task": "A",
            "scaled": False,
        },
        "sk_tree_clf": {
            **sk_clf,
            "fit_time": sk_clf_fit,
            "predict_time": sk_clf_predict,
            "factory": sk_clf_factory,
            "task": "B",
            "scaled": False,
        },
        "feature_importances": importance,
    }

    ev.save_json(results, FIG_DIR / "tree_results.json")
    print("[tree] Done.")
    return results