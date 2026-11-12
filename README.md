# ResaleIQ — Used Two-Wheeler (and Car) Resale Price Prediction

AIML (CS52) course project, M.S. Ramaiah Institute of Technology.
**Title:** *Used Two-Wheeler Resale Price Prediction with EV Depreciation Analysis* (road tax kept as a side check)

An Apple-style web app that predicts the fair listing price of a used **petrol two-wheeler, electric scooter or car** in India and explains the price feature by feature (Tree SHAP), with an EV vs petrol depreciation view.

## Accuracy (20% of ads hidden during training, then priced)

| Main models (live 2026 listings) | Ads | Typical miss | Within ±20% | Noise floor* |
|---|---|---|---|---|
| **Cars** — XGBoost, 3 averaged models (CarWale, 88,800 ads) | 17,760 test | **8.3%** | **83.0%** | 8.7% |
| CarWale's own valuation, same test ads | 16,650 | 11.3% | 73.2% | |
| **Two-wheelers** — XGBoost, 5 averaged models (BikeWale, 28,062 ads) | 5,613 test | **12.4%** | **69.3%** | 11.1% |
| Simple rule (new price × typical % kept), two-wheelers | 4,014 | 14.0% | 65.9% | |

\*How far apart asking prices are for the very same variant, year and seller — roughly the best any model can do.
By segment: electric cars 7.3% / 88.9% · petrol cars 8.3% · diesel 8.6% · petrol motorcycles 12.0% · petrol scooters 13.6% · electric scooters 14.1% / 61.6%.

Course baseline (Kaggle 2020-21 bikes): in Colab, XGBoost R² 0.921 vs the base paper's Decision Tree 0.864 on the same split. The website's copy (`train.py`, with never-rises-with-age/km rules) scores R² 0.908 and only prices 29 older models with no 2026 ads.

**Features of the main models:** brand, model, exact variant, the variant's new price, fuel, body type, gearbox, state, city, seller type (dealer / private), age, km, owners. Monotone rules: the price never rises with age, km or owners, never falls as the new price rises.
**Checked end to end:** `backend/tests/sweep_test.py` prices every one of the 1,104 models in the dropdowns at three years plus 15 edge cases — 0 failures; median gap to real ads of the same model and age 5.4%.

## Run it (2 commands)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
```bash
.venv/bin/uvicorn backend.app.main:app --port 8000
```
Open http://localhost:8000. Models are already trained in `backend/models/`. To rebuild everything (scrape ≈2 min, train ≈35 s):
```bash
.venv/bin/python backend/scrape_catalog.py && .venv/bin/python backend/build_catalog.py && .venv/bin/python backend/scrape_variants.py
```
```bash
.venv/bin/python backend/scrape_used_2026.py bikewale && .venv/bin/python backend/scrape_used_2026.py carwale
```
```bash
cd backend && ../.venv/bin/python train.py && ../.venv/bin/python train_market.py bike && ../.venv/bin/python train_market.py car && ../.venv/bin/python calibrate_2026.py
```
Test everything (server running): `.venv/bin/python backend/tests/sweep_test.py` (prices) and `.venv/bin/python backend/tests/year_check.py` (years); `backend/tests/year_audit.py` compares ad years with official years.
macOS note: XGBoost needs OpenMP — `brew install libomp`.

## How it works
```
Kaggle bikes (32,648 rows) ─┐                                   ┌─> XGBoost (bikes) ─┐
Kaggle CarDekho cars (8,128)┼─> clean ─> features ─> 80/20 split ┤                    ├─> FastAPI /api/predict ─> website
2026 BikeWale EV study ─────┘   (dedupe,  (state, model family,  └─> XGBoost (cars) ──┤      + Tree SHAP factors
                                 IQR per   log km, owners, cc)       EV depreciation ──┘      + depreciation curve
                                 family+age)                        model (Module 6)
```
- **Petrol two-wheelers & cars:** XGBoost on log(price); Kaggle-era prices scaled to 2026 rupees at 5%/yr.
- **EVs:** Kaggle has almost no EVs, so EV prices use the Module 6 same-variant depreciation results (Ola −12.0%/yr, other EVs −7.3%/yr, petrol −6.9%/yr) applied to the variant's new on-road price.
- **Explanation:** XGBoost `pred_contribs` (exact Tree SHAP) grouped back to original features.

## Repo map
| Path | What |
|---|---|
| `PROGRESS.md` | Living context + progress log (updated every step) |
| `docs/ROADMAP.md` | What's left, in order, with time estimates |
| `docs/DATASETS.md` | Every dataset, link, columns, how it's used |
| `docs/LITERATURE_SURVEY.md` | Survey table + research gap |
| `colab/COLAB_GUIDE.md` | Colab Modules 4, 5, 7, 8 code |
| `docs/figures/` | Report figures (actual vs predicted, SHAP importance, model comparison) |
| `backend/` | `train.py`, FastAPI app, EV + road-tax modules |
| `frontend/` | `index.html`, `styles.css`, `app.js` (no build step) |

## Limitations
Seller-quoted listing prices, not sale prices · no condition/accident data · Kaggle prices are 2020-21 (scaled) · EV sample small (424 ads) · EV new prices are approximate 2026 ex-showroom figures (editable in the app).
