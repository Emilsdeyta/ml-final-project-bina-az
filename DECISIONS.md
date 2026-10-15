# Data decisions (Problem 1) - every choice and why

| # | Decision | Why / evidence |
|---|----------|----------------|
| 1 | Keep only the LATEST scrape per `/items/<id>` (100,775 -> ~63.8k) | the dump re-scrapes the same ad daily; duplicates would leak across splits |
| 2 | Drop ads whose price moved >2x between scrapes | re-listed as another object / typo; label noise |
| 3 | Residential only: Yeni tikili, Köhnə tikili, Həyət evi/Bağ evi | land plots / commercial follow a different price law |
| 4 | Currency: fixed pegs to AZN (data is 100% AZN, so a no-op) | unknown currency would be dropped, never guessed |
| 5 | Parse `Sahə`, `Mərtəbə` ("5 / 12"), `Torpaq sahəsi` (sot = 100 m2) with regex | numeric columns arrive as strings |
| 6 | Sanity bounds: area 15-1500 m2, price/m2 200-15,000 AZN, rooms <= 30 | fixed constants chosen from the audit (no data-derived statistic) |
| 7 | Coordinates outside Azerbaijan -> NaN (row kept) | wrong geocodes; imputed from the district |
| 8 | Drop identical-content ads with different ids | the same ad pasted twice |
| 9 | **Leakage dropped:** `unit_price`, `total_price` | restate the target (audit: corr ~ 1.0); `leakage_audit()` proves it |
| 10 | **Identifiers/text not used:** owner_name, owner_title, shop_name, shop_title, address, description, img_url, ids | identifiers / free text; `views`, `updated`, `vip`, `featured` are post-listing popularity signals, not known at pricing time |
| 11 | `Binanın növü` not used | 99.8% missing |
| 12 | Split 70/15/15, seed 42, **grouped by physical flat** (location 11 m + area + rooms + floors + land + category) | re-posted flats must not sit on both sides of a split |
| 13 | Stratification key = "price > median of ALL rows", used only to allocate rows | needed for a stratified split; threshold itself = TRAIN median |
| 14 | Imputation (train stats only): rooms <- median of (category, area quintile); coords <- district median; houses: storeys = 0 (structural); flats: land = 0; indicator `rooms_missing` | missingness is structural (houses vs flats), not random (audit section 2) |
| 15 | Winsorise area, rooms, floors, land, distance at train 0.5-99.5% | tame the heavy tails for the SVM; trees unaffected |
| 16 | Encodings: one-hot for 3-level columns; district = log-frequency + smoothed target encoding (OOF); geo-kNN mean/std/distance of log(price/m2) of the 20 nearest TRAIN listings (OOF, flat-grouped folds) | high-cardinality location without 140 dummies; row never sees its own label (tested) |
| 17 | Task A target `log1p(price)` (invert `expm1`); Task B premium = price > TRAIN median | skew 6.3 -> 0.3 after log; balanced classes |
| 18 | Scaler mean/std from TRAIN; constant columns dropped on train | SVM needs scaling; no information from val/test |
| 19 | EDA looks at train+val only | the test set stays unseen |
| 20 | Uncertainty: bootstrap std on test + **nested** 5-fold CV on train+val (preprocessing re-fitted per fold) | rubric 5c; a plain CV on pre-encoded X would leak labels across folds |
