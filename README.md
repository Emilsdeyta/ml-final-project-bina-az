# MLE-AI-201 Final Project - bina.az price prediction

Decision tree and linear SVM (Pegasos) written from scratch in NumPy, benchmarked against scikit-learn
on the bina.az sale listings. Task A: regression of `log1p(price)`. Task B: premium vs standard tier.

## Reproduce everything (one command)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # Python 3.10-3.12
# download the Kaggle CSV into data/house_sale.csv  (see data/README.md)
python -m src.run_all                                  # all numbers, tables, figures
pytest -v                                              # 34 tests, no dataset needed
```
Useful flags: `--only-data` (data + EDA), `--fast` (3-fold CV), `--strict` (fail if a teammate module is missing).
Optional: `python -m src.data_ablation` (evidence for the preprocessing choices; validation split only).

Outputs: `report/figures/` -> `results_table.tex`, `results.json`, `data_summary.tex` (numbers quoted in the paper),
`eda_*.png`, `roc_pr_taskB.png`, `confusion_taskB.png`, `pred_vs_true_taskA.png`, `worst_errors_taskA.csv`, `error_by_*.csv`.

## Leakage safeguards (all covered by tests)
* `unit_price`, `total_price`, identifiers and free text are never features; `clean()` asserts it.
* Imputation values, clip bounds, scaler, tier threshold and encodings come from TRAIN only; `transform` never reads `price`.
* Target-based features (district TE, geo-kNN) are out-of-fold with flat-grouped folds: a row never sees its own label.
* The split keeps re-posted copies of one flat together; headline CV re-fits all preprocessing inside every fold.
* Seed 42 everywhere; no test-set use for any choice.

## Layout
`src/data_prep.py` data + features + splits | `src/evaluate.py` metrics, CI, error analysis, tables |
`src/eda.py` figures | `src/run_all.py` pipeline | `src/decision_tree.py`, `src/svm.py` models |
`src/experiments_*.py` sweeps | `DECISIONS.md` every cleaning decision | `TEAM_NOTES.md` interface notes.
