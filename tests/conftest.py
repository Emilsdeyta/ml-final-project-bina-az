"""Shared fixtures: a small synthetic raw dump with the real bina.az schema, so the
test-suite runs WITHOUT the Kaggle dataset (and is deterministic)."""
import numpy as np
import pandas as pd
import pytest

CATS = ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi", "Torpaq", "Obyekt"]


def make_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    n_d = 40
    d_lat, d_lng = 40.30 + r.random(n_d) * 0.2, 49.70 + r.random(n_d) * 0.3
    d_base = r.uniform(900, 3500, n_d)
    d = r.integers(0, n_d, n)
    cat = r.choice(CATS, n, p=[.5, .2, .2, .07, .03])
    house = cat == "Həyət evi/Bağ evi"
    area = np.clip(np.exp(r.normal(4.3, .5, n)).round(), 20, 500)
    ppm2 = d_base[d] * np.where(cat == "Yeni tikili", 1.1, np.where(house, .6, 1.0)) * np.exp(r.normal(0, .15, n))
    price = (area * ppm2).round(-2)
    total = r.integers(5, 20, n)
    fl = np.minimum(r.integers(1, 17, n), total)
    rooms = np.clip((area / 30).round(), 1, 8)
    raw = pd.DataFrame({
        "estate_rel_url": [f"/items/{1000 + i}" for i in range(n)],
        "datetime_scrape_x": "2024-10-20 20:43:11.452947",
        "price": price, "currency_x": "AZN",
        "location": [f"Rayon{i} r." for i in d],
        "Kateqoriya": cat,
        "Sahə": [f"{int(a)} m²" for a in area],
        "Otaq sayı": np.where(r.random(n) < .03, np.nan, rooms),
        "Mərtəbə": np.where(house, None, [f"{a} / {b}" for a, b in zip(fl, total)]),
        "Torpaq sahəsi": np.where(house, [f"{x} sot" for x in r.integers(1, 10, n)], None),
        "İpoteka": np.where(r.random(n) < .3, "var", None),
        "Çıxarış": np.where(r.random(n) < .8, "var", "yoxdur"),
        "Təmir": np.where(r.random(n) < .9, np.where(r.random(n) < .8, "var", "yoxdur"), None),
        "lat": d_lat[d] + r.normal(0, .01, n), "lng": d_lng[d] + r.normal(0, .01, n),
        "unit_price": [f"{int(p / a):,} AZN/m²".replace(",", " ") for p, a in zip(price, area)],
        "total_price": price,
    })
    # same listing scraped twice (later scrape, price drifted 2%) -> must keep the LAST one
    dup = raw.sample(150, random_state=1).copy()
    dup["datetime_scrape_x"] = "2024-10-21 20:44:09.129440"
    dup["price"] = (dup["price"] * 1.02).round(-2)
    dup["total_price"] = dup["price"]
    # re-posted flat under a NEW item id, 3% different price -> same group, not an exact dup
    rep = raw.sample(100, random_state=2).copy()
    rep["estate_rel_url"] = [f"/items/{50000 + i}" for i in range(len(rep))]
    rep["price"] = (rep["price"] * 1.03).round(-2)
    rep["total_price"] = rep["price"]
    # garbage: wrong coordinates, absurd price, non-positive price, price jump > 2x
    bad = raw.sample(60, random_state=3).copy()
    bad["estate_rel_url"] = [f"/items/{90000 + i}" for i in range(len(bad))]
    bad.iloc[:20, bad.columns.get_loc("lat")] = 0.0
    bad.iloc[20:40, bad.columns.get_loc("price")] *= 80
    bad.iloc[40:45, bad.columns.get_loc("price")] = -5
    bad["total_price"] = bad["price"]                      # real dump: total_price == price
    out = pd.concat([raw, dup, rep, bad], ignore_index=True)
    a = out["Sahə"].str.extract(r"(\d+)")[0].astype(float)
    out["unit_price"] = [f"{int(abs(p) / x):,} AZN/m²".replace(",", " ") for p, x in zip(out["price"], a)]
    out.loc[out.index[-3:], "datetime_scrape_x"] = "2024-10-05 22:07:37.60613+00"
    return out


@pytest.fixture(scope="session")
def raw_df():
    return make_raw()


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory, raw_df):
    p = tmp_path_factory.mktemp("data") / "house_sale.csv"
    raw_df.to_csv(p, index=False, encoding="utf-8")
    return p


@pytest.fixture()
def synth_env(monkeypatch, raw_csv):
    from src import data_prep
    monkeypatch.setenv("BINAAZ_CSV", str(raw_csv))
    data_prep.clear_cache()
    yield raw_csv
    data_prep.clear_cache()
