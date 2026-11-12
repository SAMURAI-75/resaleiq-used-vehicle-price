# ResaleIQ — Used Two-Wheeler (and Car) Resale Price Prediction

AIML (CS52) course project, M.S. Ramaiah Institute of Technology.
**Title:** *Used Two-Wheeler Resale Price Prediction with Road Tax Features and EV Depreciation Analysis*

An Apple-style web app that predicts the fair listing price of a used **petrol two-wheeler, electric scooter or car** in India and explains the price feature by feature (Tree SHAP), with an EV vs petrol depreciation view.

| Model (test set, 20% hold-out) | Two-wheelers R² | Two-wheelers MAE | Cars R² | Cars MAE |
|---|---|---|---|---|
| Linear Regression | 0.865 | ₹12,816 | 0.915 | ₹74,699 |
| Random Forest | 0.894 | ₹11,767 | 0.917 | ₹76,701 |
| **XGBoost** | **0.932** | **₹10,134** | **0.925** | **₹69,849** |

## Run it (2 commands)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
```bash
.venv/bin/uvicorn backend.app.main:app --port 8000
```
Open http://localhost:8000. Models are already trained in `backend/models/`. To retrain (≈35 s; downloads Kaggle data with `kagglehub`):
```bash
.venv/bin/python backend/train.py
```
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
