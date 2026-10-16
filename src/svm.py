"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

Pegasos minimises the regularised hinge-loss primal by stochastic
sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i   (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                    (only regularise)

Non-linear extension: Random Fourier Features (RFF).
  We map X -> phi(X) via a random cosine embedding, then run linear Pegasos.
  This approximates an RBF kernel (Rahimi & Recht, 2007).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features before training.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Random Fourier Features  (non-linear extension — Problem 3d)
# ---------------------------------------------------------------------------
class RandomFourierFeatures:
    """Approximate RBF kernel via random cosine features.

    Maps X (n, d) -> phi(X) (n, 2*n_components) so that
        phi(x) . phi(x') ≈ exp(-gamma * ||x - x'||^2)

    Then running LINEAR Pegasos on phi(X) gives a non-linear classifier.

    Parameters
    ----------
    gamma : float
        RBF bandwidth.  Larger gamma = more local / more complex boundary.
    n_components : int
        Number of random features (2*n_components output columns).
    seed : int
        Reproducibility.
    """

    def __init__(self, gamma: float = 1.0, n_components: int = 100, seed: int = 42):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def fit(self, X: np.ndarray) -> "RandomFourierFeatures":
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        # Sample random frequencies from N(0, 2*gamma)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=(self.n_components,))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W_ + self.b_          # (n, n_components)
        # [cos(z), sin(z)] * sqrt(1/n_components)  — unbiased estimator
        scale = np.sqrt(1.0 / self.n_components)
        return scale * np.concatenate([np.cos(Z), np.sin(Z)], axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Pegasos SVM
# ---------------------------------------------------------------------------
class PegasosSVM:
    """Soft-margin SVM trained with the Pegasos sub-gradient algorithm.

    Parameters
    ----------
    lambda_ : float
        Regularisation strength.  Larger = wider margin, less fit.
    n_iters : int
        Number of sub-gradient steps (each step uses 1 random sample).
    schedule : {"pegasos", "constant"}
        Learning-rate schedule.
        "pegasos"  : eta_t = 1 / (lambda_ * t)   — the original schedule
        "constant" : eta_t = 1 / sqrt(n_iters)   — alternative
    log_every : int
        Record objective value every this many iterations.
    random_state : int
        Seed for reproducibility (sample selection).
    """

    def __init__(
        self,
        lambda_: float = 1e-4,
        n_iters: int = 20_000,
        schedule: str = "pegasos",
        log_every: int = 100,
        random_state: int = 42,
    ):
        assert schedule in ("pegasos", "constant"), "schedule must be 'pegasos' or 'constant'"
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.schedule = schedule
        self.log_every = log_every
        self.random_state = random_state

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.classes_: np.ndarray | None = None   # original labels, e.g. [0, 1]
        self.objective_history_: list[float] = []

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    def _to_pm1(self, y: np.ndarray) -> np.ndarray:
        """{0,1} (or any 2-class labels) -> {-1, +1}.

        Also stores self.classes_ = sorted original labels so predict()
        can map back.
        """
        self.classes_ = np.array(sorted(np.unique(y)))
        assert len(self.classes_) == 2, "PegasosSVM is binary only"
        # classes_[0] -> -1,  classes_[1] -> +1
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _from_pm1(self, y_pm1: np.ndarray) -> np.ndarray:
        """Map {-1, +1} back to original labels."""
        return np.where(y_pm1 >= 0, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # Objective (for monitoring convergence)
    # ------------------------------------------------------------------
    def _objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """J(w,b) = (lambda/2)||w||^2 + (1/n) sum max(0, 1 - y*(w.x+b))"""
        margins = y * self.decision_function(X)
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        reg = 0.5 * self.lambda_ * float(self.w @ self.w)
        return reg + hinge

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PegasosSVM":
        """Run Pegasos.  X must already be standardised.

        Returns self.
        """
        X = np.asarray(X, dtype=float)
        y_pm1 = self._to_pm1(np.asarray(y))
        n, d = X.shape

        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)
        self.b = 0.0
        self.objective_history_ = []

        eta_const = 1.0 / np.sqrt(self.n_iters)   # used when schedule="constant"

        for t in range(1, self.n_iters + 1):
            # --- learning rate ---
            if self.schedule == "pegasos":
                eta_t = 1.0 / (self.lambda_ * t)
            else:
                eta_t = eta_const

            # --- sample one example ---
            i = rng.integers(n)
            xi, yi = X[i], y_pm1[i]

            # --- sub-gradient step ---
            margin = yi * (self.w @ xi + self.b)
            if margin < 1.0:                         # hinge is active
                self.w = (1.0 - eta_t * self.lambda_) * self.w + eta_t * yi * xi
                self.b = self.b + eta_t * yi
            else:                                    # only regularise
                self.w = (1.0 - eta_t * self.lambda_) * self.w

            # --- project onto L2 ball (Pegasos stability trick) ---
            norm_w = np.sqrt(self.w @ self.w)
            ball_radius = 1.0 / np.sqrt(self.lambda_)
            if norm_w > ball_radius:
                self.w *= ball_radius / norm_w

            # --- log objective ---
            if t % self.log_every == 0:
                self.objective_history_.append(self._objective(X, y_pm1))

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed margin  w . x + b  for each row."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels in the original label space."""
        return self._from_pm1(np.sign(self.decision_function(X)))

    def predict_proba_approx(self, X: np.ndarray) -> np.ndarray:
        """Platt-style sigmoid squashing of the decision score (approximate).

        Returns (n, 2) array: column 0 = P(class_0), column 1 = P(class_1).
        """
        score = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p1, p1])

    # ------------------------------------------------------------------
    # Margin analysis  (Problem 3c)
    # ------------------------------------------------------------------
    def norm_w(self) -> float:
        """||w||  — inversely proportional to the margin width."""
        return float(np.sqrt(self.w @ self.w))

    def margin_width(self) -> float:
        """Geometric margin = 2 / ||w||."""
        nw = self.norm_w()
        return 2.0 / nw if nw > 0 else float("inf")

    def support_vector_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask: True for points inside or on the margin band.

        A point is a support vector candidate when y*(w.x+b) <= 1.
        """
        y_pm1 = np.where(
            np.asarray(y) == self.classes_[1], 1.0, -1.0
        )
        return y_pm1 * self.decision_function(X) <= 1.0 + 1e-9
