# Notes for Gasham and Orxan (interface details that matter)

* `get_splits("A"|"B")` is as frozen in TEAM_PLAN. **Target A is `np.log1p(price)`; invert with `np.expm1`**
  (`data_prep.from_log_price`). Extra keys: `price_*`, `df_*`, `groups_train`, `preprocessor`.
* ~27 features, no NaN, constant columns removed. SVM: use `X_*_scaled`; tree: raw `X_*`.
* Do your own CV with `evaluate.cross_val_mean_std(factory, X, y, k, task=..., groups=S["groups_train"])`
  (group-aware). Headline CV is done by `run_all` (nested).
* **Please add to every model entry returned by `run()`** (3 lines, makes the table uniform):
  `"factory": lambda: DecisionTree(**final_cfg)`, `"task": "A"|"B"`, `"scaled": True|False` (SVM: True).
  `run_all` then re-fits it, scores with `evaluate.py`, adds bootstrap std + nested CV + error analysis, and
  warns if your reported numbers differ from the recomputed ones.
* Models need `fit(X, y)` / `predict(X)`; for Task B also `predict_proba` (tree) or `decision_function` (SVM).
* Non-metric entries (`feature_importances`, `margin`) go in the returned dict untouched; they are stored in results.json.
* Never fit/tune on `X_test`. Use `X_val` or CV; final config from validation only.
