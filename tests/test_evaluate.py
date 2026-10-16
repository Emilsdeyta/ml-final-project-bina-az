"""Metrics are checked against scikit-learn (allowed for sanity checks only)."""
import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.linear_model import Ridge

from src import evaluate as ev
from src.linear_baseline import MajorityClassifier, MeanRegressor, RidgeRegressionScratch


@pytest.fixture()
def reg():
    r = np.random.default_rng(0)
    y = r.normal(12, 1, 500)
    return y, y + r.normal(0, .3, 500)


@pytest.fixture()
def clf():
    r = np.random.default_rng(1)
    y = (r.random(800) < .45).astype(int)
    s = np.round(y * .8 + r.normal(0, 1, 800), 1)             # rounding -> many tied scores
    return y, (s > .4).astype(int), s


def test_regression_metrics_match_sklearn(reg):
    y, p = reg
    assert ev.rmse(y, p) == pytest.approx(np.sqrt(skm.mean_squared_error(y, p)))
    assert ev.mae(y, p) == pytest.approx(skm.mean_absolute_error(y, p))
    assert ev.r2(y, p) == pytest.approx(skm.r2_score(y, p))
    m = ev.regression_metrics(np.log1p(np.exp(y)), np.log1p(np.exp(p)))
    assert m["r2_price"] == pytest.approx(skm.r2_score(np.exp(y), np.exp(p)), rel=1e-6)


def test_classification_metrics_match_sklearn(clf):
    y, yp, s = clf
    assert np.array_equal(ev.confusion_matrix(y, yp), skm.confusion_matrix(y, yp))
    p, r, f = ev.precision_recall_f1(y, yp)
    assert (p, r, f) == pytest.approx((skm.precision_score(y, yp), skm.recall_score(y, yp), skm.f1_score(y, yp)))
    assert ev.accuracy(y, yp) == pytest.approx(skm.accuracy_score(y, yp))
    assert ev.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))        # ties handled
    assert ev.pr_auc(y, s) == pytest.approx(skm.average_precision_score(y, s))


def test_curves_endpoints_and_auc_consistency(clf):
    y, _, s = clf
    fpr, tpr = ev.roc_curve(y, s)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    assert trap(tpr, fpr) == pytest.approx(ev.roc_auc(y, s), abs=1e-9)


def test_degenerate_cases():
    assert np.isnan(ev.roc_auc([1, 1, 1], [.1, .2, .3]))
    assert ev.precision_recall_f1([0, 0], [0, 0]) == (0.0, 0.0, 0.0)
    assert ev.r2([3, 3, 3], [3, 3, 3]) == 1.0
    assert ev.roc_auc([0, 1, 0, 1], [.5, .5, .5, .5]) == 0.5


def test_ridge_matches_sklearn():
    r = np.random.default_rng(2)
    X = r.normal(size=(300, 6)); y = X @ r.normal(size=6) + 3 + r.normal(0, .1, 300)
    a = RidgeRegressionScratch(alpha=3.0).fit(X, y)
    b = Ridge(alpha=3.0).fit(X, y)
    assert np.allclose(a.coef_, b.coef_) and a.intercept_ == pytest.approx(b.intercept_)
    assert np.allclose(a.predict(X), b.predict(X))


def test_bootstrap_se_and_cv(reg, clf):
    y, p = reg
    se = ev.bootstrap_se("A", y, p, n_boot=100)
    assert 0 < se["rmse"]["std"] < 0.1 and se["rmse"]["lo"] < ev.rmse(y, p) < se["rmse"]["hi"]
    yy, yp, s = clf
    cv = ev.cross_val_mean_std(MajorityClassifier, np.zeros((len(yy), 1)), yy, k=5, task="B")
    assert cv["mean"]["accuracy"] == pytest.approx(max(yy.mean(), 1 - yy.mean()), abs=0.03)
    assert cv["mean"]["roc_auc"] == pytest.approx(0.5)
    folds = ev.kfold_indices(100, 5, y=(np.arange(100) % 2))
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(100))


def test_group_kfold_keeps_groups_together():
    g = np.repeat(np.arange(40), 5)
    for tr, va in ev.kfold_indices(len(g), 4, groups=g):
        assert set(g[tr]).isdisjoint(g[va])


def test_compare_writes_tex_and_skips_non_metrics(tmp_path):
    res = {"tree_reg": {"rmse": .3, "mae": .2, "r2": .8, "fit_time": 1.5},
           "sk_tree_reg": {"RMSE": .31, "MAE": .21, "R2": .79},
           "svm": {"f1": .9, "precision": .88, "recall": .92, "auc": .96, "accuracy": .91},
           "feature_importances": {"area_m2": .5}, "margin": {"width": 1.2}}
    txt = ev.compare(res, out_dir=tmp_path, verbose=False)
    assert "Decision tree (ours)" in txt and "Pegasos SVM" in txt and "feature" not in txt
    tex = (tmp_path / "results_table.tex").read_text()
    assert r"\toprule" in tex and r"\textbf{0.300}" in tex and "tab:results_b" in tex


def test_end_to_end_with_baselines():
    r = np.random.default_rng(3)
    X = r.normal(size=(400, 3)); y = X[:, 0] * 2 + r.normal(0, .5, 400)
    m = MeanRegressor().fit(X, y)
    assert ev.r2(y, m.predict(X)) == pytest.approx(0, abs=1e-9)
