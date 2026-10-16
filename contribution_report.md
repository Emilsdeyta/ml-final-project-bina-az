# Contribution Report

| Member | GitHub handle | Main responsibilities |
|--------|---------------|-----------------------|
| Emil | <handle> | Data cleaning + leakage-safe preprocessing, splits, metrics/evaluation, run_all pipeline, EDA, README, report sections Data/Results |
| Gasham | <handle> | Decision tree from scratch, tree experiments, sklearn tree comparison |
| Orxan | <handle> | Pegasos SVM from scratch, RFF kernel, SVM experiments, sklearn SVM comparison |

## Per-member summary
**Emil —** Wrote `src/data_prep.py` (cleaning, `Preprocessor` with out-of-fold target/geo-kNN encodings, grouped stratified split, nested CV folds), `src/evaluate.py` (all metrics in NumPy, bootstrap, error analysis, results table), `src/eda.py`, `src/run_all.py`, `src/data_ablation.py`, `src/linear_baseline.py` (bonus: ridge normal equations) and the tests in `tests/test_data_prep.py`, `tests/test_evaluate.py`, `tests/test_run_all_smoke.py`. (Add PR numbers.)

**Gasham —** (write your own paragraph: files, PRs, what you debugged.)

**Orxan —** (write your own paragraph: files, PRs, what you debugged.)

## Rough effort split
| Member | Approx. % |
|--------|-----------|
| Emil | ~33 |
| Gasham | ~33 |
| Orxan | ~33 |
