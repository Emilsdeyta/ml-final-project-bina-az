"""Data-prep tests - synthetic data only. Run: pytest tests/test_data_prep.py -v"""
import numpy as np
import pandas as pd
import pytest

from src import data_prep as dp


@pytest.fixture()
def clean_df(raw_df):
    return dp.clean(raw_df)


# ------------------------------------------------------------------ cleaning
def test_leakage_and_identifier_columns_are_gone(clean_df):
    assert not set(clean_df.columns) & set(dp.LEAKAGE_COLUMNS + dp.DROPPED_IDENTIFIERS)


def test_only_residential_and_sane_values(clean_df):
    assert set(clean_df["building_category"]) <= set(dp.RESIDENTIAL)
    assert clean_df["area_m2"].between(dp.AREA_MIN, dp.AREA_MAX).all()
    assert (clean_df["price"] > 0).all()
    ppm2 = clean_df["price"] / clean_df["area_m2"]
    assert ppm2.between(dp.PPM2_MIN, dp.PPM2_MAX).all()
    assert clean_df["item_id"].is_unique


def test_parsing_and_structural_missing(clean_df):
    houses = clean_df[clean_df["building_category"] == dp.HOUSE]
    flats = clean_df[clean_df["building_category"] != dp.HOUSE]
    assert houses["floor"].isna().all() and houses["land_sot"].notna().all()
    assert flats["land_sot"].isna().all()
    assert (flats["floor"] <= flats["total_floors"]).all()


def test_bad_coordinates_become_nan_not_rows(raw_df, clean_df):
    assert clean_df["lat"].dropna().between(*dp.AZ_LAT).all()
    assert clean_df["lat"].isna().sum() > 0


def test_latest_scrape_is_kept(raw_df):
    twice = raw_df[raw_df.duplicated("estate_rel_url", keep=False)].sort_values("datetime_scrape_x")
    last = twice.groupby("estate_rel_url").tail(1).set_index("estate_rel_url")["price"]
    c = dp.clean(raw_df).set_index("item_id")["price"]
    sample = [(u, p) for u, p in last.items() if int(u.split("/")[-1]) in c.index][:50]
    assert sample and all(abs(c[int(u.split("/")[-1])] - p) < 1e-6 for u, p in sample)


def test_clean_values_do_not_depend_on_other_rows(raw_df):
    """Every cleaned VALUE is a function of its own raw row (no data-derived statistic).
    Only row *membership* can differ (de-duplication compares rows by definition)."""
    c_all = dp.clean(raw_df).set_index("item_id")
    ids = raw_df["estate_rel_url"].str.extract(r"/items/(\d+)")[0].astype(int)
    c_half = dp.clean(raw_df[ids % 2 == 0]).set_index("item_id")
    common = c_half.index.intersection(c_all.index)
    assert len(common) > 0.9 * len(c_half)
    pd.testing.assert_frame_equal(c_half.loc[common], c_all.loc[common], check_dtype=False)


def test_leakage_audit_shows_columns_are_leaky(raw_df):
    a = dp.leakage_audit(raw_df)
    assert a["total_price_corr_with_price"] > 0.999
    assert a["unit_price_x_area_corr_with_price"] > 0.99


def test_currency_conversion(raw_df):
    r = raw_df.head(300).copy()
    r["currency_x"] = "USD"
    a, b = dp.clean(raw_df.head(300)), dp.clean(r)
    both = a.merge(b, on="item_id", suffixes=("_a", "_b"))
    assert np.allclose(both["price_b"], both["price_a"] * 1.7)


# -------------------------------------------------------------------- splits
def test_split_interface_and_shapes(synth_env):
    s = dp.get_splits("B", verbose=False)
    for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "X_train_scaled",
              "X_val_scaled", "X_test_scaled", "feature_names", "threshold"):
        assert k in s
    assert s["X_train"].dtype == np.float64
    for k in ("X_train", "X_val", "X_test", "X_train_scaled", "X_val_scaled", "X_test_scaled"):
        assert np.isfinite(s[k]).all()
    assert s["X_train"].shape[1] == len(s["feature_names"]) == s["X_test"].shape[1]
    n = len(s["y_train"]) + len(s["y_val"]) + len(s["y_test"])
    assert abs(len(s["y_test"]) / n - 0.15) < 0.03 and abs(len(s["y_val"]) / n - 0.15) < 0.03


def test_scaler_uses_train_statistics_only(synth_env):
    s = dp.get_splits("A", verbose=False)
    assert np.allclose(s["X_train_scaled"].mean(0), 0, atol=1e-8)
    assert np.allclose(s["X_train_scaled"].std(0), 1, atol=1e-8)
    assert not np.allclose(s["X_test_scaled"].mean(0), 0, atol=1e-8)     # test is NOT centred on itself


def test_task_targets_and_threshold_from_train(synth_env):
    a, b = dp.get_splits("A", verbose=False), dp.get_splits("B", verbose=False)
    assert a["threshold"] is None
    assert np.allclose(a["y_train"], np.log1p(a["price_train"]))
    assert np.allclose(dp.from_log_price(a["y_test"]), a["price_test"])
    assert b["threshold"] == pytest.approx(np.median(b["price_train"]))
    assert set(np.unique(b["y_train"])) == {0, 1}
    for k in ("y_train", "y_val", "y_test"):
        assert abs(b[k].mean() - 0.5) < 0.06
    assert np.array_equal(b["y_test"], (b["price_test"] > b["threshold"]).astype(int))


def test_group_split_never_shares_a_flat(synth_env):
    s = dp.get_splits("A", verbose=False)
    assert s["groups_shared_across_splits"] == 0
    assert set(s["item_id_train"]).isdisjoint(s["item_id_test"])
    r = dp.get_splits("A", group_aware=False, verbose=False)
    assert r["groups_shared_across_splits"] > 0         # a row-level split DOES leak reposts


def test_deterministic(synth_env):
    a = dp.get_splits("B", verbose=False)
    dp.clear_cache()
    b = dp.get_splits("B", verbose=False)
    for k in ("X_train", "X_test", "y_train", "y_test"):
        assert np.array_equal(a[k], b[k])
    c = dp.get_splits("B", seed=7, verbose=False)
    assert not np.array_equal(a["item_id_test"], c["item_id_test"])


def test_no_leaky_feature_names(synth_env):
    names = dp.get_splits("A", verbose=False)["feature_names"]
    assert not any(n in names for n in ("price", "unit_price", "total_price", "log_price"))


def test_cache_returns_copies(synth_env):
    a = dp.get_splits("A", verbose=False)
    a["X_train"][:] = 0
    b = dp.get_splits("A", verbose=False)
    assert b["X_train"].std() > 0


# -------------------------------------------------- leakage-focused guarantees
def _split_frames(synth_env):
    s = dp.get_splits("A", verbose=False)
    return s["df_train"], s["df_val"], s["df_test"], dp.make_groups(s["df_train"])


def test_transform_ignores_price_of_new_rows(synth_env):
    d_tr, d_va, _, g = _split_frames(synth_env)
    pre = dp.Preprocessor()
    pre.fit_transform(d_tr, g)
    X1, _ = pre.transform(d_va)
    X2, _ = pre.transform(d_va.assign(price=d_va["price"].to_numpy()[::-1] * 7.0))
    X3, _ = pre.transform(d_va.drop(columns="price"))
    assert np.array_equal(X1, X2) and np.array_equal(X1, X3)


def test_training_encodings_are_out_of_fold(synth_env):
    d_tr, _, _, g = _split_frames(synth_env)
    d_tr = d_tr.iloc[:1500].reset_index(drop=True)
    g = g[:1500]
    X1, names = dp.Preprocessor().fit_transform(d_tr, g)
    i = 10
    d2 = d_tr.copy()
    d2.loc[i, "price"] *= 5.0                                   # change ONE training label
    X2, _ = dp.Preprocessor().fit_transform(d2, g)
    enc = [names.index(n) for n in ("district_te", "geo_logppm2_mean", "geo_logppm2_std")]
    assert np.array_equal(X1[i, enc], X2[i, enc])               # the row never sees its own label
    assert (X1[:, enc] != X2[:, enc]).any()                     # ...but it does inform other rows


def test_in_sample_encoding_is_detectably_leaky(synth_env):
    d_tr, _, _, g = _split_frames(synth_env)
    d_tr = d_tr.iloc[:1500].reset_index(drop=True)
    X1, names = dp.Preprocessor(leaky_insample_encodings=True).fit_transform(d_tr, g[:1500])
    d2 = d_tr.copy(); d2.loc[10, "price"] *= 5.0
    X2, _ = dp.Preprocessor(leaky_insample_encodings=True).fit_transform(d2, g[:1500])
    j = names.index("geo_logppm2_mean")
    assert X1[10, j] != X2[10, j]                               # the diagnostic mode really leaks


def test_threshold_ignores_validation_and_test_prices():
    p = np.arange(1, 101, dtype=float)
    y, thr = dp.make_tier_label(p)
    y2, thr2 = dp.make_tier_label(p * 1000, thr)
    assert thr == 50.5 and thr2 == thr and y2.mean() == 1.0


def test_unseen_district_and_injected_nans_are_handled(synth_env):
    d_tr, d_va, _, g = _split_frames(synth_env)
    pre = dp.Preprocessor()
    pre.fit_transform(d_tr, g)
    odd = d_va.head(50).copy()
    odd["district"] = "Never seen r."
    for c in ("rooms", "lat", "lng", "area_m2", "floor", "total_floors", "land_sot"):
        odd.loc[odd.index[:25], c] = np.nan
    X, _ = pre.transform(odd)
    assert X.shape[0] == 50 and np.isfinite(X).all()


def test_imputation_strategies_differ_but_both_finite(synth_env):
    d_tr, d_va, _, g = _split_frames(synth_env)
    d_tr = d_tr.copy(); d_tr.loc[d_tr.index[:40], "rooms"] = np.nan
    Xg, ng = dp.Preprocessor(imputation="group").fit_transform(d_tr, g)
    Xm, nm = dp.Preprocessor(imputation="median").fit_transform(d_tr, g)
    assert np.isfinite(Xg).all() and np.isfinite(Xm).all()
    assert not np.allclose(Xg[:40, ng.index("rooms")], Xm[:40, nm.index("rooms")])


def test_constant_columns_dropped(synth_env):
    names = dp.get_splits("A", verbose=False)["feature_names"]
    assert "floor_missing" not in names              # flats never miss floors in this data


# ----------------------------------------------------------------- nested CV
def test_cv_folds_are_disjoint_and_refit_per_fold(synth_env):
    s = dp.get_splits("B", verbose=False)
    folds = list(dp.get_cv_folds("B", k=4))
    assert len(folds) == 4
    pool = len(s["y_train"]) + len(s["y_val"])
    assert sum(len(f["y_val"]) for f in folds) == pool
    thr = [f["threshold"] for f in folds]
    assert len(set(thr)) > 1                           # threshold differs per fold = re-fitted on fold-train
    for f in folds:
        assert np.isfinite(f["X_train"]).all() and f["X_val"].shape[1] == f["X_train"].shape[1]
