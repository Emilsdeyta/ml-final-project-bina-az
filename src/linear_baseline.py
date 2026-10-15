"""
linear_baseline.py - from-scratch reference models (NumPy only).

* RidgeRegressionScratch: closed-form ridge, w = (Xc'Xc + a I)^-1 Xc'yc with an
  UNPENALISED intercept (bonus Problem 8, Task A baseline).
* MeanRegressor / MajorityClassifier: "know-nothing" floors every real model must beat.
"""

from __future__ import annotations

import numpy as np


class RidgeRegressionScratch:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)          # normal equations (A is SPD for alpha > 0)
        self.intercept_ = float(ym - xm @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def select_ridge_alpha(X_tr, y_tr, X_va, y_va, grid=(0.01, 0.1, 1, 10, 100, 1000)):
    """Pick alpha on the VALIDATION split only (never the test set)."""
    best = None
    for a in grid:
        err = float(np.sqrt(np.mean((RidgeRegressionScratch(a).fit(X_tr, y_tr).predict(X_va) - y_va) ** 2)))
        if best is None or err < best[1]:
            best = (a, err)
    return best[0]


class MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MajorityClassifier:
    def fit(self, X, y):
        self.label_ = int(np.mean(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)

    def decision_function(self, X):
        return np.zeros(len(X))
