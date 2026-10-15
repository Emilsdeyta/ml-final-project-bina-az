"""
tests/test_svm.py — unit tests for PegasosSVM and RandomFourierFeatures.

All tests use synthetic toy data (numpy only, no Kaggle CSV needed).
Seed: np.random.default_rng(0) as per team rules.
"""
import numpy as np
import pytest
from src.svm import PegasosSVM, RandomFourierFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def linearly_separable():
    """Two well-separated Gaussian blobs — linear SVM should get ~100%."""
    rng = np.random.default_rng(0)
    X_pos = rng.normal([2, 2], 0.4, (60, 2))
    X_neg = rng.normal([-2, -2], 0.4, (60, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 60 + [0] * 60)
    return X, y


@pytest.fixture(scope="module")
def xor_data():
    """XOR-like data — not linearly separable, needs RFF."""
    rng = np.random.default_rng(0)
    corners = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    X = rng.normal(0, 0.25, (200, 2)) + corners[rng.integers(4, size=200)]
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Test 1 — Linearly separable: accuracy ~100%
# ---------------------------------------------------------------------------
def test_accuracy_on_separable(linearly_separable):
    X, y = linearly_separable
    svm = PegasosSVM(lambda_=1e-3, n_iters=8000, random_state=42)
    svm.fit(X, y)
    acc = (svm.predict(X) == y).mean()
    assert acc >= 0.98, f"Expected >=0.98 accuracy, got {acc:.3f}"


# ---------------------------------------------------------------------------
# Test 2 — Objective function decreases during training
# ---------------------------------------------------------------------------
def test_objective_decreases(linearly_separable):
    X, y = linearly_separable
    svm = PegasosSVM(lambda_=1e-3, n_iters=6000, log_every=200, random_state=42)
    svm.fit(X, y)
    assert len(svm.objective_history_) > 5, "Not enough objective log entries"
    n = len(svm.objective_history_)
    first_avg = np.mean(svm.objective_history_[: n // 4])
    last_avg  = np.mean(svm.objective_history_[3 * n // 4 :])
    assert first_avg > last_avg, (
        f"Objective should decrease: start={first_avg:.4f}, end={last_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Larger lambda -> smaller ||w|| (wider margin)
# ---------------------------------------------------------------------------
def test_larger_lambda_smaller_norm(linearly_separable):
    X, y = linearly_separable
    lambdas = [1e-4, 1e-3, 1e-2]
    norms = []
    for lam in lambdas:
        s = PegasosSVM(lambda_=lam, n_iters=5000, random_state=42)
        s.fit(X, y)
        norms.append(s.norm_w())
    assert norms[0] > norms[-1], (
        f"||w|| should decrease as lambda grows: {list(zip(lambdas, norms))}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Support vectors are inside/on the margin band
# ---------------------------------------------------------------------------
def test_support_vectors_inside_margin():
    # Use overlapping Gaussians so some points fall inside the margin band.
    # With well-separated data every y*(w·x+b) > 1, leaving no SVs — correct
    # SVM behaviour but makes the "exists" assertion vacuously fail.
    rng = np.random.default_rng(0)
    X_pos = rng.normal([1, 1], 1.2, (80, 2))
    X_neg = rng.normal([-1, -1], 1.2, (80, 2))
    X_ov = np.vstack([X_pos, X_neg])
    y_ov = np.array([1] * 80 + [0] * 80)

    svm = PegasosSVM(lambda_=1e-2, n_iters=5000, random_state=42)
    svm.fit(X_ov, y_ov)
    sv_mask = svm.support_vector_mask(X_ov, y_ov)
    assert sv_mask.sum() > 0, "No support vectors found — overlapping data should have some"
    # Every flagged point must satisfy y*(w·x+b) <= 1 + tolerance
    y_pm1 = np.where(y_ov == svm.classes_[1], 1.0, -1.0)
    margins = y_pm1 * svm.decision_function(X_ov)
    assert np.all(margins[sv_mask] <= 1.0 + 1e-6), \
        "Some 'support vectors' have margin > 1 — mask is wrong"


# ---------------------------------------------------------------------------
# Test 5 — margin_width = 2 / ||w||
# ---------------------------------------------------------------------------
def test_margin_width_formula(linearly_separable):
    X, y = linearly_separable
    svm = PegasosSVM(lambda_=1e-3, n_iters=5000, random_state=42)
    svm.fit(X, y)
    expected = 2.0 / svm.norm_w()
    assert abs(svm.margin_width() - expected) < 1e-10


# ---------------------------------------------------------------------------
# Test 6 — predict_proba_approx rows sum to 1
# ---------------------------------------------------------------------------
def test_predict_proba_sums_to_one(linearly_separable):
    X, y = linearly_separable
    svm = PegasosSVM(lambda_=1e-3, n_iters=3000, random_state=42)
    svm.fit(X, y)
    proba = svm.predict_proba_approx(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ---------------------------------------------------------------------------
# Test 7 — schedule="constant" also converges
# ---------------------------------------------------------------------------
def test_constant_schedule(linearly_separable):
    X, y = linearly_separable
    svm = PegasosSVM(lambda_=1e-3, n_iters=5000, schedule="constant", random_state=42)
    svm.fit(X, y)
    acc = (svm.predict(X) == y).mean()
    assert acc >= 0.95, f"Constant schedule accuracy too low: {acc:.3f}"


# ---------------------------------------------------------------------------
# Test 8 — RandomFourierFeatures solves XOR (non-linear data)
# ---------------------------------------------------------------------------
def test_rff_solves_xor(xor_data):
    X, y = xor_data
    # Linear SVM fails on XOR
    svm_lin = PegasosSVM(lambda_=1e-3, n_iters=5000, random_state=42)
    svm_lin.fit(X, y)
    acc_lin = (svm_lin.predict(X) == y).mean()
    assert acc_lin < 0.75, f"Linear SVM should fail on XOR, got {acc_lin:.3f}"

    # RFF + linear SVM solves it
    rff = RandomFourierFeatures(gamma=1.0, n_components=200, seed=42)
    X_rff = rff.fit_transform(X)
    svm_rff = PegasosSVM(lambda_=1e-3, n_iters=8000, random_state=42)
    svm_rff.fit(X_rff, y)
    acc_rff = (svm_rff.predict(X_rff) == y).mean()
    assert acc_rff >= 0.85, f"RFF+SVM should solve XOR, got {acc_rff:.3f}"


# ---------------------------------------------------------------------------
# Test 9 — Determinism: same seed -> same weights
# ---------------------------------------------------------------------------
def test_determinism(linearly_separable):
    X, y = linearly_separable
    s1 = PegasosSVM(lambda_=1e-3, n_iters=3000, random_state=7).fit(X, y)
    s2 = PegasosSVM(lambda_=1e-3, n_iters=3000, random_state=7).fit(X, y)
    assert np.allclose(s1.w, s2.w), "Same seed must give identical weights"
    assert np.allclose(s1.b, s2.b), "Same seed must give identical bias"
