# IDX Exchange — California Property Price Prediction

Predicts the final sale price (`ClosePrice`) of California single-family residential
properties from CRMLS/Trestle MLS sold-listing data. Built over an IDX Exchange Data
Science internship as a weekly progression: exploration → preprocessing → baseline
modeling → a pipeline audit that fixed a real data-leakage/duplication bug → domain
feature engineering → walk-forward stability validation → a Streamlit deployment app.

Every number below is from an actual executed run on this repo's data snapshot, not a
placeholder — see [Known Limitations](#known-limitations) for what's explicitly *not*
yet verified.

## Dataset

- **Source:** CRMLS (California Regional Multiple Listing Service) sold-listing
  extracts, accessed via the Trestle API. Licensed/private data — **not included in
  this repository**.
- **Scope:** `PropertyType == "Residential"`, `PropertySubType ==
  "SingleFamilyResidence"` only, per the internship task specification.
- **Snapshot used for all `wk9`/`wk10`/`wk11` results:** 29 monthly `CRMLSSold*.csv`
  files, through `CRMLSSold202605.csv` (May 2026). An earlier pass was missing that
  final month's file, which silently produced different-but-plausible-looking metrics
  for a full debugging session before the snapshot mismatch was found — see the wk9
  notebook's reflection for the full story. Any comparison across notebooks in this
  repo should confirm they're reading the same snapshot.
- **Target variable:** `ClosePrice` (final sale price).

## Repository Structure

```text
Deliverables/
  notebook_01_exploration.ipynb        weeks 1-2: EDA, outliers, distributions
  notebook_02_preprocessing.ipynb      week 3: cleaning, leakage guards, temporal split
  notebook_03_baseline_model.ipynb     week 4: Linear Regression baseline
  notebook_04_model_comparison.ipynb   week 4: Decision Tree / Random Forest
  notebook_05_advanced_models.ipynb    weeks 5-6: XGBoost / LightGBM tuning
  notebook_06_evaluation.ipynb         week 6: feature importance, geo error map, county breakdown
  wk6_feature_engineering.ipynb        week 6: engineered feature pass
  wk6_geocoding_baseline_m3.ipynb      week 6: geocoding fixes, m3 baseline
  wk9_pipeline_overhaul.ipynb          week 9: leakage/encoding/hyperparameter audit (m4)
  wk10_domain_features.ipynb           week 10: temporal + locational features (m5)
  wk11_walk_forward_stability.ipynb    week 11: 23-fold walk-forward validation
  *.csv                                metrics/importance/backtest outputs, one per notebook
models/
  lgbm_m3.pkl, xgb_m3.pkl              m3 checkpoint models
  lgbm_m4.pkl, xgb_m4.pkl              m4 (post pipeline-overhaul) models
  lgbm_m5.pkl, xgb_m5.pkl              m5 (domain-features) models — lgbm_m5.pkl is deployed
  rf_m3.pkl / rf_m4.pkl / rf_m5.pkl    Random Forest checkpoints — gitignored (1.2-1.5GB each,
                                        retrain in well under a minute from any notebook above)
app.py                                 Streamlit deployment app
requirements.txt                       pinned to the exact env lgbm_m5.pkl was trained/pickled with
comps_lookup.json                      fit-on-train ZIP/area price-per-sqft lookup the app needs
dropdown_choices.json                  City/County/MLS-area/school-district/flooring/ZIP option
                                        lists the app's dropdowns read from -- a small (~96KB)
                                        extract, since the full training CSV is gitignored
scripts/                               supporting scripts (geocoding, etc.)
```

## Methodology Highlights

- **Chronological, never random, splits.** Every notebook from `wk9` onward splits by
  calendar month — the most recent complete month is the test set, the one before it
  is validation, and everything before that (last `N_TRAIN_MONTHS`) is training. No
  row from a later month ever leaks into an earlier split.
- **Fit-on-train discipline, applied twice.** Outlier thresholds (0.5th/99.5th
  percentile `ClosePrice`) and the ZIP/area price-per-sqft comps feature are both
  computed from the training split only, then applied unchanged to validation/test —
  the same rule enforced two different ways.
- **Cross-fit target encoding.** High-cardinality categoricals (`City`,
  `CountyOrParish`, `MLSAreaMajor`, `SchoolDistrictJoined`, `Flooring`) use sklearn's
  `TargetEncoder`, which internally cross-fits during `fit_transform` so no row's own
  target leaks into its own encoded value, even within the training set.
- **A real bug, found and fixed, not assumed away.** `wk9` traced a suspiciously high
  Random Forest score back to the school-district spatial join matching against three
  overlapping district-type layers (Elementary/Unified/High) instead of one, silently
  duplicating 40.2% of rows. Fixed by restricting the join to Unified+High and adding
  a dedup-with-assertion safety net — see [Known Limitations](#known-limitations) for
  what that fix changed.
- **Stability checked two different ways**, not just reported once. `wk10` runs a
  rolling-origin backtest (same window length, cutoff shifted back 1-2 months); `wk11`
  runs a full walk-forward validation (expanding window, one month at a time, from
  6 months of training data through the full 29-month snapshot).

## Models Tested & Results

Feature-set versions: **m3** (geocoding baseline) → **m4** (`wk9`, pipeline overhaul —
leakage fixes, cross-fit target encoding, retuned Random Forest) → **m5** (`wk10`,
adds cyclical month encoding, ZIP/area comps, distance-to-employment-center,
missing-indicator flags). All numbers below are `wk9`/`wk10` overall test-set metrics
on the 29-file snapshot.

| Model              | Feature set | Test R² | Test MAE  | Test MAPE | Test MdAPE |
| ------------------ | ----------- | ------- | --------- | --------- | ---------- |
| Linear Regression   | m4          | 0.7183  | $320,433  | 29.60%    | 22.19%     |
| Decision Tree       | m4          | 0.7973  | $218,923  | 16.13%    | 10.65%     |
| Random Forest       | m4          | 0.8896  | $163,075  | 11.93%    | 7.94%      |
| XGBoost             | m4          | 0.8958  | $166,402  | 12.50%    | 8.68%      |
| **LightGBM**        | **m4**      | 0.9053  | $156,418  | 11.78%    | 8.15%      |
| Linear Regression   | m5          | 0.8162  | $252,475  | 23.90%    | 16.64%     |
| Decision Tree       | m5          | 0.7871  | $224,990  | 16.43%    | 10.79%     |
| Random Forest       | m5          | 0.8967  | $156,842  | 11.53%    | 7.57%      |
| XGBoost             | m5          | 0.9023  | $159,349  | 11.96%    | 8.34%      |
| **LightGBM (m5)**   | **m5**      | **0.9094** | **$152,550** | **11.55%** | **7.97%** |

**Best model overall: LightGBM on the m5 feature set** — R² = 0.9094, MAPE = 11.55%,
on the held-out test month. This is the model shipped in `models/lgbm_m5.pkl` and used
by `app.py`.

Notably, **Linear Regression improved the most of any model from m4→m5** (R² +0.098,
MAPE −5.70pp) when a temporal trend feature (`MonthsSinceStart`) was added — evidence
that its earlier weakness was substantially a missing-signal problem, not a fundamental
limitation of the model.

By feature importance, `DistanceToNearestMajorCenter` and `ZipMedianPricePerSqft`
(both new in m5) rank #3 and #4 overall on the winning LightGBM model — real signal,
not noise the model has to filter out.

## Is It Actually Stable Over Time?

`wk11` runs a genuine walk-forward validation: train on all months through month *N*,
test on month *N+1*, then retrain through *N+1* and test on *N+2* — 23 folds spanning
the full snapshot, not one static cutoff.

- **R² is stable**: 0.8942–0.9198 across all 23 folds, with no statistically
  significant trend over time (p = 0.448) — essentially noise.
- **MAPE/MdAPE drift mildly upward**, and that drift *is* statistically significant
  (p = 0.008 and 0.013 respectively) — but small in absolute terms, about 1
  percentage point of MAPE spread over roughly two years.
- The three worst folds are all winter-month tests (Dec/Jan), which also happen to
  have the smallest test sets in the whole walk. The most likely explanation is a mix
  of real winter seasonality in the housing market and noisier estimates from smaller
  winter samples — not a fundamental breakdown.
- `wk10`'s static single-split MAPE (11.55%) lands almost exactly at this walk's mean
  (11.53%), well inside the 23-fold range — reassuring that the headline number wasn't
  a lucky month.

**Practical takeaway:** retrain monthly in production (as this walk simulates) rather
than deploy one static model indefinitely, and expect winter months to run
1-1.5pp worse on MAPE than the rest of the year.

## Streamlit App

`app.py` loads `models/lgbm_m5.pkl`, collects raw property inputs through a form, and
reconstructs every engineered feature the model expects using the same logic as
`wk10_domain_features.ipynb` — cyclical month encoding, property age, distance to the
nearest of 10 major CA employment centers (haversine, computed live), and the
ZIP/area price-per-sqft comps feature (via the pre-computed `comps_lookup.json`
lookup, since that fit-on-train table can't be recomputed from a single input row).
The constructed feature set is checked against the model's expected schema before
`.predict()` is ever called.

An optional address field geocodes to latitude/longitude (and pre-fills ZIP/city/
county where they match) using geopy + Nominatim (OpenStreetMap) — a free alternative
to a paid geocoding API, so the app works without any API key setup. Manual entry
always remains available as a fallback.

### Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. `requirements.txt` is pinned to the exact package
versions `models/lgbm_m5.pkl` was trained and pickled with (scikit-learn 1.8.0,
xgboost 3.2.0, lightgbm 4.7.0, pandas 3.0.2, numpy 2.4.4) — this project hit real
version-mismatch bugs during development (a `TargetEncoder` behavior change, a
30+ minute `geopandas`/`shapely` `sjoin` hang), so the pins are load-bearing, not
cosmetic.

## Known Limitations

Stated explicitly rather than left implicit:

- **`N_TRAIN_MONTHS=24`** was chosen via a sweep on the m4 feature set (`wk9`) and
  reused unchanged for m5 (`wk10`) — plausible that the new comps/temporal features
  shift the accuracy/window-length tradeoff, not re-verified.
- **`MIN_ZIP_TRAIN_N=15`** (minimum training sales before a ZIP gets its own comps
  median) and the choice of exactly 10 major employment centers are reasonable
  judgment calls, not separately validated by an ablation.
- **The m4 pipeline-overhaul fix changed real numbers**, not just code style: before
  the school-district join fix, Random Forest's validation score looked
  artificially strong due to 40.2% duplicated rows. Anyone comparing against
  pre-`wk9` results in this repo's history should treat them as superseded.
- **`wk11`'s walk-forward reuses `wk10`'s already-chosen LightGBM hyperparameters**
  at every fold rather than re-tuning per fold — a deliberate cost tradeoff for a
  check that's about stability of a fixed pipeline, not further optimization.
- **Live app:** _link added once deployed to Streamlit Community Cloud_ — until then,
  run it locally via `streamlit run app.py`.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, GeoPandas, Streamlit, geopy, joblib.

---

_Built as part of the IDX Exchange Data Science Internship Program._
