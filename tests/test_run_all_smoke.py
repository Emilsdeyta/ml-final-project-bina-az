"""End-to-end smoke test of run_all with synthetic data and FAKE teammate modules
(sklearn stand-ins, test-only), proving the frozen interface glue works."""
import sys
import types

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import run_all


def _fake_tree():
    m = types.ModuleType("src.experiments_tree")
    m.run = lambda a, b: {
        "tree_reg": {"task": "A", "rmse": 0.3, "factory": lambda: DecisionTreeRegressor(max_depth=8, random_state=0)},
        "tree_clf": {"task": "B", "factory": lambda: DecisionTreeClassifier(max_depth=8, random_state=0)},
        "feature_importances": {"area_m2": 0.5},
    }
    return m


def _fake_svm():
    m = types.ModuleType("src.experiments_svm")
    m.run = lambda b: {"svm": {"task": "B", "scaled": True, "factory": lambda: LinearSVC()},
                       "sk_svc": {"f1": 0.9, "roc_auc": 0.95, "accuracy": 0.9, "precision": .9, "recall": .9},
                       "margin": {"width": 1.0}}
    return m


def test_full_pipeline(synth_env, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.experiments_tree", _fake_tree())
    monkeypatch.setitem(sys.modules, "src.experiments_svm", _fake_svm())
    run_all.main(["--fast", "--bootstrap", "30", "--out-dir", str(tmp_path)])
    for f in ("results_table.tex", "results.json", "data_summary.tex", "eda_price_hist.png",
              "worst_errors_taskA.csv", "roc_pr_taskB.png", "confusion_taskB.png", "pred_vs_true_taskA.png"):
        assert (tmp_path / f).exists(), f
    tex = (tmp_path / "results_table.tex").read_text()
    assert "Decision tree (ours)" in tex and "Pegasos SVM (ours)" in tex and "tab:cv_a" in tex


def test_runs_without_teammate_modules(synth_env, tmp_path, monkeypatch):
    for n in ("src.experiments_tree", "src.experiments_svm"):
        monkeypatch.setitem(sys.modules, n, None)          # import raises -> graceful skip
    run_all.main(["--fast", "--bootstrap", "20", "--skip-eda", "--out-dir", str(tmp_path)])
    assert "Ridge" in (tmp_path / "results_table.tex").read_text()
