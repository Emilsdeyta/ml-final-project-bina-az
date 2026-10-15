import numpy as np
import pytest

from src.decision_tree import DecisionTree


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("gini", 0.375),
        ("entropy", 0.8112781244591328),
    ],
)
def test_classification_impurity(criterion, expected):
    tree = DecisionTree(
        task="classification",
        criterion=criterion,
    )
    y = np.array([0, 0, 0, 1])

    assert tree._impurity(y) == pytest.approx(expected)


def test_regression_impurity():
    tree = DecisionTree(
        task="regression",
        criterion="mse",
    )
    y = np.array([1.0, 2.0, 3.0])

    assert tree._impurity(y) == pytest.approx(2.0 / 3.0)


def test_best_classification_split():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(
        task="classification",
        criterion="gini",
    )

    feature, threshold, decrease = tree._best_split(X, y)

    assert feature == 0
    assert threshold == pytest.approx(5.5)
    assert decrease == pytest.approx(0.5)


def test_classification_fit_predict_and_proba():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(
        task="classification",
        criterion="gini",
        max_depth=3,
    ).fit(X, y)

    predictions = tree.predict([[2.5], [8.5]])
    probabilities = tree.predict_proba([[2.5], [8.5]])

    np.testing.assert_array_equal(predictions, [0, 1])
    np.testing.assert_allclose(
        probabilities,
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert tree.depth_ == 1
    assert tree.n_leaves_ == 2
    np.testing.assert_allclose(tree.feature_importances_, [1.0])


def test_regression_fit_and_predict():
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

    tree = DecisionTree(
        task="regression",
        criterion="mse",
        max_depth=3,
    ).fit(X, y)

    predictions = tree.predict([[2.5], [8.5]])

    np.testing.assert_allclose(predictions, [1.0, 10.0])

@pytest.mark.parametrize(
    "stopping_parameter",
    [
        {"max_depth": 0},
        {"min_samples_split": 7},
        {"min_samples_leaf": 4},
        {"min_impurity_decrease": 0.6},
    ],
)
def test_stopping_rules_create_root_leaf(stopping_parameter):
    X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(
        task="classification",
        criterion="gini",
        **stopping_parameter,
    ).fit(X, y)

    assert tree.root.is_leaf()
    assert tree.depth_ == 0
    assert tree.n_leaves_ == 1
    np.testing.assert_allclose(tree.feature_importances_, [0.0])


def test_predict_before_fit_raises_error():
    tree = DecisionTree(
        task="classification",
        criterion="gini",
    )

    with pytest.raises(ValueError, match="fitted"):
        tree.predict([[1.0]])


def test_regression_predict_proba_raises_error():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])

    tree = DecisionTree(
        task="regression",
        criterion="mse",
    ).fit(X, y)

    with pytest.raises(ValueError, match="classification"):
        tree.predict_proba([[2.0]])


def test_fit_rejects_different_sample_counts():
    tree = DecisionTree(
        task="classification",
        criterion="gini",
    )

    with pytest.raises(ValueError, match="same number of samples"):
        tree.fit(
            [[1.0], [2.0]],
            [0],
        )


def test_predict_rejects_wrong_feature_count():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1])

    tree = DecisionTree(
        task="classification",
        criterion="gini",
    ).fit(X, y)

    with pytest.raises(ValueError, match="incorrect number of features"):
        tree.predict([[1.0, 2.0]])