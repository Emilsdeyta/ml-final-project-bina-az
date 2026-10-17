# Contribution Report

| Member | GitHub handle | Main responsibilities |
|--------|---------------|-----------------------|
| Emil | @Emilsdeyta | Data cleaning + leakage-safe preprocessing, splits, metrics/evaluation, run_all pipeline, EDA, README, report sections Data/Results |
| Gasham | @gashamhuseynli | Decision tree from scratch, tree experiments, sklearn tree comparison |
| Orxan | @orxannuriyev | Pegasos SVM from scratch, RFF kernel, SVM experiments, sklearn SVM comparison |

## Per-member summary
**Emil —** Wrote `src/data_prep.py` (cleaning, `Preprocessor` with out-of-fold target/geo-kNN encodings, grouped stratified split, nested CV folds), `src/evaluate.py` (all metrics in NumPy, bootstrap, error analysis, results table), `src/eda.py`, `src/run_all.py`, `src/data_ablation.py`, `src/linear_baseline.py` (bonus: ridge normal equations) and the tests in `tests/test_data_prep.py`, `tests/test_evaluate.py`, `tests/test_run_all_smoke.py`. (Add PR numbers.)

**Gasham —** Implemented the from-scratch CART decision tree in `src/decision_tree.py` for both regression and classification, including MSE, Gini and entropy criteria, recursive split construction, stopping parameters, prediction probabilities and feature importance. Built `src/experiments_tree.py` for validation-based hyperparameter tuning, final evaluation, feature-importance analysis and comparison with scikit-learn decision trees. Added and verified decision-tree tests in `tests/test_decision_tree.py`, and debugged split selection, tree depth/leaf behaviour, probability outputs, runtime performance and compatibility with the shared `run_all` pipeline. Also revised the project README and added the final report and presentation sources/PDFs. Main contributions were submitted through PR #9, PR #16 and the final documentation PR.

**Orxan —** Implemented the Pegasos SVM pipeline in three PRs. PR #11 (`orxan/svm-core`) added `src/svm.py`: the `PegasosSVM` class (stochastic sub-gradient descent with L2-ball projection, two learning-rate schedules, objective history logging, support-vector mask and margin-width helpers) and `RandomFourierFeatures` (random cosine embedding to approximate the RBF kernel). PR #12 (`orxan/svm-tests`) added `tests/test_svm.py`: nine unit tests covering accuracy on separable data, objective decrease, λ vs ‖w‖ trade-off, support-vector identification, margin-width formula, approximate probabilities, constant learning-rate schedule, RFF solving XOR, and determinism. Fixed a subtle test failure where perfectly-separated data has no support vectors (correct SVM behaviour, wrong test assumption). PR #13 (`orxan/svm-experiments`) added `src/experiments_svm.py`: 18-point λ sweep on validation F1, final model at 20k iterations, Pegasos vs constant schedule comparison, 12-point RFF γ sweep, top-15 feature-weight bar chart, and scikit-learn SVC/SGDClassifier baselines. Each model entry in the returned dict carries `"factory"`, `"task": "B"` and `"scaled": True` so `run_all.py` can re-fit and cross-check metrics automatically.

## Rough effort split
| Member | Approx. % |
|--------|-----------|
| Emil | ~33 |
| Gasham | ~33 |
| Orxan | ~33 |
