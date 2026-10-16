# bina.az Price Prediction

Predicting real-estate prices in Baku with a **decision tree** and a **linear SVM (Pegasos)**, both written
from scratch in NumPy and benchmarked against scikit-learn.

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)
![Core models](https://img.shields.io/badge/core%20models-NumPy%20only-informational)

Final project for **MLE-AI-201 Machine Learning**, built by Emil Mammadov, Gasham Huseynli and Orkhan Nuriyev.
The data comes from [bina.az](https://bina.az), Azerbaijan's largest property portal.

We solve two problems on the same listings:

| Task | Question | Target |
|------|----------|--------|
| **A. Regression** | How much is this listing worth? | `log1p(price)`; invert with `expm1` |
| **B. Classification** | Is it a *premium* or a *standard* listing? | `price > train median` (208,000 AZN), so classes are 50/50 |

---

## Results at a glance

Held-out test set (8,622 listings). The ± is a bootstrap standard deviation over 300 resamples.
Every model, ours or scikit-learn's, is scored by the same code in `src/evaluate.py`.

**Task A: price regression**

| Model | RMSE (log) | R² (log) | MAE (AZN) |
|-------|-----------:|---------:|----------:|
| Mean predictor (floor) | 0.629 ± 0.006 | -0.000 | 127,850 |
| **Ridge, normal equations (ours)** | **0.226 ± 0.003** | **0.870 ± 0.004** | **48,280** |
| Decision tree (ours) | 0.239 ± 0.004 | 0.856 ± 0.004 | 49,516 |
| scikit-learn `DecisionTreeRegressor` | 0.238 ± 0.004 | 0.856 ± 0.004 | 49,353 |

**Task B: premium vs. standard**

| Model | Accuracy | F1 | ROC-AUC | Fit time |
|-------|---------:|---:|--------:|---------:|
| Majority class (floor) | 0.500 | 0.000 | 0.500 | 0 s |
| Decision tree (ours) | 0.914 ± 0.003 | 0.914 ± 0.003 | 0.972 ± 0.002 | 65 s |
| scikit-learn `DecisionTreeClassifier` | 0.914 ± 0.003 | 0.914 ± 0.003 | 0.972 ± 0.002 | 0.7 s |
| Pegasos SVM (ours) | 0.906 ± 0.003 | 0.908 ± 0.003 | 0.968 ± 0.002 | 1 s |
| scikit-learn `SVC` (RBF) | **0.924 ± 0.003** | **0.924 ± 0.003** | **0.981 ± 0.001** | 306 s |
| scikit-learn `SGDClassifier` (hinge) | 0.912 ± 0.003 | 0.913 ± 0.003 | 0.969 ± 0.002 | 0.1 s |

Nested 5-fold cross-validation tells the same story (for example tree AUC 0.972 ± 0.001, ridge R² 0.874 ± 0.004).

**What we learned**

- Our from-scratch tree matches scikit-learn's to three decimals on both tasks, which is the best evidence
  that the implementation is correct.
- Pegasos lands within 1.6 F1 points of the exact RBF `SVC` and trains about 300x faster.
- Ridge beats the tree on Task A. Most of the price signal is close to linear once the features are engineered.
- `area_m2` is by far the strongest feature (importance ≈ 0.62 for regression, 0.65 for classification).
  Next come the out-of-fold neighbourhood price and the district target encoding.
- Where the models struggle: listings above 800k AZN (median error 31.6%, under-predicted by 26.7%) and
  house/garden listings (22.7%). Task B errors cluster around the tier threshold (11.4% error in the 200k to 400k bucket).

---

## Quick start

Needs **Python 3.10, 3.11 or 3.12**. NumPy 1.26.4 has no wheel for 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# put the Kaggle CSV at data/house_sale.csv  (see data/README.md)

pytest -v                          # 61 tests, no dataset needed
python -m src.run_all --strict     # reproduces every number, table and figure
```

A full run takes about 45 minutes, mostly the exact `SVC` baseline and nested cross-validation.

| Command | What it does |
|---------|--------------|
| `python -m src.run_all --strict` | Full pipeline. Fails if a model module is missing. Use this for final numbers. |
| `python -m src.run_all --fast` | 3-fold CV and 100 bootstrap draws. Good for a quick sanity check. |
| `python -m src.run_all --only-data` | Cleaning, split and EDA only. |
| `python -m src.run_all --out-dir DIR` | Write outputs somewhere else. |
| `python -m src.data_ablation` | Evidence for the preprocessing choices (validation split only). |

---

## How it fits together

The whole project is one pipeline. `run_all.py` is the only entry point, and every other module is a
stage it calls.

```
 data/house_sale.csv
        │
        ▼
 ┌─────────────────────┐
 │ 1. data_prep.py     │  clean → leakage-safe features → flat-grouped 70/15/15 split
 └──────────┬──────────┘  → splits for Task A and Task B (+ CV folds)
            │
            ▼
 ┌─────────────────────┐
 │ 2. eda.py           │  figures and data summary (train + val only, never test)
 └──────────┬──────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. Models, each trained and tuned on validation only     │
 │                                                          │
 │   linear_baseline.py     mean / majority / ridge         │
 │   experiments_tree.py ─► decision_tree.py   (A and B)    │
 │   experiments_svm.py  ─► svm.py             (B)          │
 └──────────┬───────────────────────────────────────────────┘
            │  each model hands back a "factory" that builds a fresh instance
            ▼
 ┌─────────────────────┐
 │ 4. evaluate.py      │  re-fit every model the same way → test metrics, bootstrap std,
 └──────────┬──────────┘  nested CV, error analysis, figures
            │
            ▼
   report/figures/   results_table.tex · results.json · *.png · *.csv
```

### Module responsibilities

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Orchestration | `run_all.py` | Runs the stages in order and collects results. Contains no modelling logic. |
| Data | `data_prep.py` | Cleaning, feature engineering, encodings, splits, CV folds. The only place that touches raw data. |
| Data | `eda.py` | Exploratory figures and the numbers quoted in the report. |
| Models | `decision_tree.py` | CART from scratch: Gini, entropy and MSE criteria. |
| Models | `svm.py` | Pegasos SVM and Random Fourier Features from scratch. |
| Models | `linear_baseline.py` | Ridge (normal equations) and the mean / majority floors. |
| Experiments | `experiments_tree.py`, `experiments_svm.py` | Hyperparameter tuning, scikit-learn comparisons, feature importance, margin analysis. |
| Evaluation | `evaluate.py` | All metrics, bootstrap, nested CV, error analysis, result tables. No scikit-learn import. |
| Diagnostics | `data_ablation.py` | Shows what each preprocessing decision is worth. |

### The contract between modules

Models never talk to each other. Each experiment module exposes a `run(...)` function that returns a
dictionary, and every model entry in it carries three extra keys:

```python
"factory": lambda: DecisionTree(**final_config),   # builds a fresh, untrained model
"task":    "A",                                    # or "B"
"scaled":  False,                                  # True if it trains on standardised features
```

`run_all.py` uses the factory to **re-fit and re-score every model with identical code**, so each row in a
results table is comparable. It also warns if a number reported by the model's owner differs from the
recomputed one. Models only need `fit(X, y)` and `predict(X)`, plus `predict_proba` or `decision_function`
for Task B. The full interface notes are in [`TEAM_NOTES.md`](TEAM_NOTES.md).

### Where to change what

| I want to... | Edit |
|--------------|------|
| Change a cleaning rule or add a feature | `src/data_prep.py` (and note it in `DECISIONS.md`) |
| Try a new hyperparameter grid for the tree | `REG_CONFIGS` / `CLF_CONFIGS` in `src/experiments_tree.py` |
| Change the SVM search or add a kernel variant | `src/experiments_svm.py`, `src/svm.py` |
| Add a new model to the comparison | Write `fit`/`predict`, return it from a `run()` with `factory`, `task`, `scaled` |
| Add a metric | `src/evaluate.py`, plus a test against scikit-learn in `tests/test_evaluate.py` |

---

## Methodology

### Data

The dataset is the Kaggle [binaaz-sale-project](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project):
100,775 scraped rows with Azerbaijani column names. After cleaning and de-duplication, 57,480 listings remain,
split 70 / 15 / 15 into 40,236 train, 8,622 validation and 8,622 test rows. The CSV is not committed
(see [`data/README.md`](data/README.md)).

Cleaning in short: keep the latest scrape of each ad, drop ads whose price jumped by more than 2x,
keep residential listings, apply fixed sanity bounds, parse numeric strings with regular expressions, and
impute structurally (houses and flats have different missing-value patterns). All 20 decisions, with the
evidence behind each, are in [`DECISIONS.md`](DECISIONS.md).

### Preventing leakage

This was the part we cared about most, and every point below is covered by a test.

- `unit_price` and `total_price` restate the target (correlation ≈ 1.0), so they are dropped and `clean()` asserts it.
- Identifiers, free text and post-listing signals (views, VIP, featured) are never used as features.
- Imputation values, clip bounds, scaler, tier threshold and encodings are fitted on **train only**.
- Target-based features (district target encoding, nearest-neighbour price) are computed **out-of-fold**,
  so a row never sees its own label.
- The split is **grouped by physical flat**. Re-posted copies of the same flat never land on both sides of a split.
- Headline cross-validation re-fits all preprocessing inside every fold. The test set is never used to make a choice.
- Seed 42 everywhere.

### Models

- **Decision tree** (`src/decision_tree.py`): CART with Gini or entropy for classification and MSE for
  regression. Best-split search over thresholds, depth and leaf-size regularisation, impurity-based feature importance.
- **Pegasos SVM** (`src/svm.py`): soft-margin SVM trained by stochastic sub-gradient descent on the regularised
  hinge-loss primal with step size 1/(λt). Random Fourier Features give an RBF-kernel approximation for the
  non-linear variant.
- **Baselines**: ridge regression in closed form, mean and majority predictors as floors, and scikit-learn's
  tree, `SVC` and `SGDClassifier` as reference points. scikit-learn is never used inside our own models.

### Evaluation

- Metrics written from scratch and unit-tested against scikit-learn: RMSE, MAE, R², MdAPE, accuracy,
  precision, recall, F1, ROC-AUC, PR-AUC.
- Uncertainty: bootstrap standard deviation and confidence intervals on the test set, plus nested k-fold CV.
- Error analysis by district, price bucket and building category, plus the worst individual predictions.

---

## Repository layout

```
.
├── src/                      # all code (see "How it fits together")
├── tests/                    # 61 pytest tests: metrics vs scikit-learn, leakage, models, smoke test
├── report/                   # LaTeX report: report.tex, sections/, figures/
├── presentation/             # slides
├── data/                     # dataset goes here (not committed)
├── DECISIONS.md              # every data decision and why
├── TEAM_NOTES.md             # interface contract between modules
├── contribution_report.md    # who did what
├── requirements.txt
└── pytest.ini
```

## Outputs

`python -m src.run_all` writes to `report/figures/`:

- `results_table.tex`, `results.json`, `data_summary.tex`: every number quoted in the report
- `eda_*.png`: exploratory figures
- `roc_pr_taskB.png`, `confusion_taskB.png`, `pred_vs_true_taskA.png`
- `error_by_*.csv`, `worst_errors_taskA.csv`: error analysis

`.png` and `.pdf` outputs are git-ignored. Re-run the pipeline to regenerate them.

## Reproducibility

- Dependencies are pinned: `numpy==1.26.4`, `pandas==2.2.2`, `scipy==1.13.1`, `matplotlib==3.8.4`, `scikit-learn==1.4.2`.
- Seed 42 for splits, folds, bootstrap and model initialisation.
- The reported numbers come from Python 3.11.9. Fit times vary by machine, metrics do not.

## Team

| Member | Contribution |
|--------|--------------|
| **Emil Mammadov** | Data cleaning and leakage-safe preprocessing, splits, evaluation framework, EDA, ridge baseline, the `run_all` pipeline, tests, documentation |
| **Gasham Huseynli** | Decision tree from scratch, tree experiments, scikit-learn tree comparison |
| **Orkhan Nuriyev** | Pegasos SVM and Random Fourier Features from scratch, SVM experiments, scikit-learn SVM comparison |

Per-member details are in [`contribution_report.md`](contribution_report.md).

## Limitations

- Prices are asking prices from listings, not final sale prices.
- The model sees 27 tabular features and no listing text or photos.
- Listings above 800k AZN are rare and consistently under-predicted.
- "Premium" means above the training median, so Task B is a relative, balanced notion rather than an absolute one.
- Random Fourier Features did not beat the linear Pegasos SVM on validation F1 in our runs (0.849 vs. 0.910).
