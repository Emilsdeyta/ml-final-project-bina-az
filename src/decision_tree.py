"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)
        self.proba = None
        self.n_samples = 0
        self.impurity_decrease = 0.0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.n_leaves_ = 0
        self.depth_ = 0

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        y = np.asarray(y)

        if y.size == 0:
            return 0.0

        if self.criterion == "mse":
            mean_y = np.mean(y)
            return float(np.mean((y - mean_y) ** 2))

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities ** 2))

        if self.criterion == "entropy":
            return float(-np.sum(probabilities * np.log2(probabilities)))

        raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_feature = None
        best_threshold = None
        best_decrease = 0.0

        if self.task == "classification":
            _, y_encoded = np.unique(y, return_inverse=True)
            n_classes = int(np.max(y_encoded)) + 1
            total_counts = np.bincount(
                y_encoded,
                minlength=n_classes,
            )

        else:
            total_sum = np.sum(y)
            total_squared_sum = np.sum(y ** 2)

        for feature_idx in range(n_features):
            order = np.argsort(X[:, feature_idx], kind="mergesort")

            x_sorted = X[order, feature_idx]

            if self.task == "classification":
                y_sorted = y_encoded[order]
                left_counts = np.zeros(n_classes, dtype=int)
                right_counts = total_counts.copy()

            else:
                y_sorted = y[order]
                left_sum = 0.0
                left_squared_sum = 0.0

            for split_position in range(1, n_samples):
                moved_target = y_sorted[split_position - 1]

                if self.task == "classification":
                    left_counts[moved_target] += 1
                    right_counts[moved_target] -= 1

                else:
                    left_sum += moved_target
                    left_squared_sum += moved_target ** 2

                n_left = split_position
                n_right = n_samples - split_position

                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue

                if x_sorted[split_position - 1] == x_sorted[split_position]:
                    continue

                if self.task == "classification":
                    left_probabilities = left_counts / n_left
                    right_probabilities = right_counts / n_right

                    if self.criterion == "gini":
                        left_impurity = 1.0 - np.sum(left_probabilities ** 2)
                        right_impurity = 1.0 - np.sum(
                            right_probabilities ** 2
                        )

                    else:
                        positive_left = left_probabilities[
                            left_probabilities > 0
                        ]
                        positive_right = right_probabilities[
                            right_probabilities > 0
                        ]

                        left_impurity = -np.sum(
                            positive_left * np.log2(positive_left)
                        )
                        right_impurity = -np.sum(
                            positive_right * np.log2(positive_right)
                        )

                else:
                    right_sum = total_sum - left_sum
                    right_squared_sum = (
                        total_squared_sum - left_squared_sum
                    )

                    left_impurity = (
                        left_squared_sum / n_left
                        - (left_sum / n_left) ** 2
                    )
                    right_impurity = (
                        right_squared_sum / n_right
                        - (right_sum / n_right) ** 2
                    )

                    left_impurity = max(0.0, left_impurity)
                    right_impurity = max(0.0, right_impurity)

                weighted_impurity = (
                    n_left / n_samples * left_impurity
                    + n_right / n_samples * right_impurity
                )

                decrease = parent_impurity - weighted_impurity

                if decrease > best_decrease:
                    best_feature = feature_idx
                    best_threshold = (
                        x_sorted[split_position - 1]
                        + x_sorted[split_position]
                    ) / 2.0
                    best_decrease = decrease

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_decrease

    # --- grow / predict -----------------------------------------------------

    def _make_leaf(self, y) -> Node:
        """Create a leaf containing its prediction and class probabilities."""
        node = Node()
        node.n_samples = y.size

        if self.task == "regression":
            node.value = float(np.mean(y))
            return node

        counts = np.array(
            [np.sum(y == class_label) for class_label in self.classes_],
            dtype=float,
        )

        node.proba = counts / counts.sum()
        node.value = self.classes_[np.argmax(counts)]

        return node

    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        n_samples = y.size
        impurity = self._impurity(y)

        depth_limit_reached = (
            self.max_depth is not None and depth >= self.max_depth
        )
        too_few_samples = n_samples < self.min_samples_split
        pure_node = impurity <= 1e-12

        if depth_limit_reached or too_few_samples or pure_node:
            return self._make_leaf(y)

        split = self._best_split(X, y)

        if split is None:
            return self._make_leaf(y)

        feature_idx, threshold, decrease = split

        if decrease < self.min_impurity_decrease:
            return self._make_leaf(y)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if (
            np.sum(left_mask) < self.min_samples_leaf
            or np.sum(right_mask) < self.min_samples_leaf
        ):
            return self._make_leaf(y)

        node = Node()
        node.feature = int(feature_idx)
        node.threshold = float(threshold)
        node.n_samples = n_samples
        node.impurity_decrease = float(decrease)

        node.left = self._grow(
            X[left_mask],
            y[left_mask],
            depth + 1,
        )

        node.right = self._grow(
            X[right_mask],
            y[right_mask],
            depth + 1,
        )

        return node

    def _collect_tree_stats(self, node, depth, importances):
        """Return leaf count and maximum depth while collecting importances."""
        if node.is_leaf():
            return 1, depth

        importances[node.feature] += (
            node.n_samples * node.impurity_decrease
        )

        left_leaves, left_depth = self._collect_tree_stats(
            node.left,
            depth + 1,
            importances,
        )
        right_leaves, right_depth = self._collect_tree_stats(
            node.right,
            depth + 1,
            importances,
        )

        n_leaves = left_leaves + right_leaves
        max_depth = max(left_depth, right_depth)

        return n_leaves, max_depth

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.n_features_in_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None

        self.root = self._grow(X, y, depth=0)

        importances = np.zeros(self.n_features_in_, dtype=float)

        self.n_leaves_, self.depth_ = self._collect_tree_stats(
            self.root,
            depth=0,
            importances=importances,
        )

        total_importance = importances.sum()

        if total_importance > 0.0:
            importances /= total_importance

        self.feature_importances_ = importances

        return self

    def _leaf_for_one(self, x, node: Node) -> Node:
        """Traverse the tree and return the leaf reached by one sample."""
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node

    def _predict_one(self, x, node: Node):
        leaf = self._leaf_for_one(x, node)
        return leaf.value

    def predict(self, X):
        if self.root is None:
            raise ValueError("The tree must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incorrect number of features.")

        return np.asarray(
            [self._predict_one(x, self.root) for x in X]
        )

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification.")

        if self.root is None:
            raise ValueError("The tree must be fitted before prediction.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incorrect number of features.")

        probabilities = [
            self._leaf_for_one(x, self.root).proba
            for x in X
        ]

        return np.vstack(probabilities)
