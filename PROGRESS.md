# ResaleIQ — Project Context & Progress Log

> Living log. Updated after every prompt / action. Newest entries at the bottom of the **Log** section.

## 1. Project in one line
**Used Two-Wheeler Resale Price Prediction with Road Tax Features and EV Depreciation Analysis** — AIML (CS52) course project, MSRIT Bangalore. Predict the fair resale price of a used two-wheeler (and, in the web app, a used car) and explain *why* that price.

## 2. Base paper & novelty
- **Base paper:** Ferdous, Mahmud, Ahmed, Arman, Tasnim — *Predicting Pre-Owned Bike Values with Machine Learning Techniques in Bangladesh*, 2nd IEEE ICDSNS, 2024. DT / SVR / KNN / SGD; DT best ("97% accuracy" — odd metric for regression). Full PDF still to fetch from IEEE Xplore via college login.
- **Main novelty:** EV vs petrol two-wheeler depreciation in India, same-variant comparison, on 2026 BikeWale listings (424 EV vs 5,270 petrol ads). Claim: *first India two-wheeler EV resale study* (not "first EV resale study" — cars are already studied, e.g. Roberson et al. 2024, US).
- **Side note:** state road tax shifts used prices where the gap is large (Karnataka 18% vs ~6–8%). Spearman 0.78, ≈ ₹1,427 per +1% tax on a Classic 350.
- **Key EV result:** EVs lose ~10.7%/yr vs petrol ~6.9%/yr — but the gap is Ola (−12.0%/yr). Without Ola: 7.3% vs 6.9% (no real difference, p = 0.85). Honest claim: *one brand loses value fast*, not *EVs lose value fast*.

## 3. Repo layout
```
backend/            FastAPI app + training pipeline
  train.py          downloads Kaggle data, cleans, trains Linear/RF/XGBoost, saves models + metrics
  app/main.py       REST API (/api/meta, /api/predict, /api/metrics) and serves the frontend
  app/ev.py         EV depreciation model from Module 6 results
  app/road_tax.py   2026 state road-tax slabs
  models/           saved models + metrics.json (generated)
frontend/           Apple-style single-page site (HTML/CSS/JS, no build step)
colab/              Colab notebook guide (modules 1–8)
docs/               ROADMAP.md, DATASETS.md, LITERATURE_SURVEY.md
```

## 4. Status board
| Area | Status |
|---|---|
| Topic, novelty, abstract | ✅ done (earlier sessions) |
| Colab modules 1–3, 6 (clean, road tax, EV v2) | ✅ done in Colab (earlier sessions) |
| Colab modules 4, 5, 7, 8 | ⏳ to do — code ready in `colab/COLAB_GUIDE.md` |
| Website backend (FastAPI, 3 models, SHAP) | ✅ runs locally, ~5-20 ms per prediction |
| Verified vehicle list (BikeWale + CarWale) | ✅ 541 two-wheeler + 345 car models |
| Website frontend (Apple-style) | ✅ tested in browser: bike, EV, car, all 3 tabs |
| Docs: roadmap, datasets, lit survey | ✅ `docs/` |
| Report + slides | ⏳ |
| Online deployment | ⏳ optional (Render) |

### Website model scores (test set, 20%, after official-name mapping)
| | Bikes R² | Bikes MAE | Cars R² | Cars MAE |
|---|---|---|---|---|
| Linear Regression | 0.621 | ₹13,979 | 0.892 | ₹73,676 |
| Random Forest | 0.952 | ₹10,011 | 0.896 | ₹69,827 |
| XGBoost (served) | 0.946 | **₹9,632** | **0.917** | **₹64,620** |

### Headline results (12 Nov, from `colab/ResaleIQ_pipeline_executed.ipynb`)
| Result | Number |
|---|---|
| XGBoost on Kaggle test split | R² 0.919, MAE ₹10,442, 79.7% within ±20% |
| Base paper's best (Decision Tree) on same split | R² 0.842, MAE ₹14,547, 65.3% within ±20% |
| Old model on 13,598 live 2026 ads | 8% too low overall (16% for 6-12 yr bikes); median error 19.5% → 18.0% after calibration |
| EV vs petrol, same variant, 30,527 ads | EV −12.0%/yr vs petrol −6.3%/yr, gap −5.7 pts, p = 0.001 |
| Brand effect (Ola) | Not stable: Ola steepest on 5 Nov, flattest on 12 Nov → not claimed |
| Road tax | State helps a little (R² 0.940 → 0.945); tax rate adds nothing on top; LOSO: helps in 5 of 11 states → side note only |

### Open questions for Kartik
1. Run `colab/ResaleIQ_pipeline.ipynb` in Colab (steps in `colab/COLAB_GUIDE.md`), save a copy to Drive.
(Resolved: 2026 data re-collected by Claude; base paper cited from IEEE page; title shortened; demo local only.)

## 5. Log
- **2026-11-12 — Session start.** Received full handoff. Request: Apple-style website (2W/car switch, predict, "why this price" tab, overview section, fast, animated), frontend + backend, datasets list, Colab steps, proper literature survey, roadmap, and this progress file in a GitHub repo. Due tomorrow.
- **2026-11-12 — Research.** Confirmed public datasets (Kaggle saisaathvik bikes, nehalbirla CarDekho cars). Found extra survey candidates: Roberson, Pantha & Helveston 2024 (US EV car value retention), Deshbhagat Univ. IJSTD 2024 used-bike NB-KNN paper. No independent 2026 Indian e-scooter resale index exists (blogs/vendors only) — strengthens the gap claim. Ferdous et al. not indexed by web search; must be pulled from IEEE Xplore.
- **2026-11-12 — Repo init.** `git init`, folder layout, this file.
- **2026-11-12 — Env.** Python 3.11 venv; `brew install libomp` for XGBoost on macOS. Kaggle data via kagglehub (bikes, CarDekho cars, motorcycle dataset).
- **2026-11-12 — Training pipeline** (`backend/train.py`, 35 s). Bikes: 32,648 → 7,167 after dedupe/km/IQR-per-family+age (Colab has 6,733 because it drops unmapped cities). Cars: 8,128 → 6,559. XGBoost best for both (R² 0.932 / 0.925). Figures saved to `docs/figures/`.
- **2026-11-12 — Backend** (`backend/app/`). FastAPI: `/api/meta`, `/api/metrics`, `/api/predict`. Bike/car = XGBoost + exact Tree SHAP (`pred_contribs`) grouped by original feature; EV = Module 6 depreciation model (Ola −12.0%/yr, other EVs −7.3%, petrol −6.9%) on approximate 2026 ex-showroom prices + estimated battery health; bike road-tax note from `road_tax.py`. Kaggle-era prices scaled to 2026 at 5%/yr.
- **2026-11-12 — Frontend** (`frontend/`). Apple-style single page: frosted nav, hero, overview stats with count-up, 5-step pipeline, findings chart (EV vs petrol vs Ola), road-tax bars, model table, predictor with 2W/Car segmented control + Petrol/Electric switch, result tabs Price / Why this price / Depreciation, live re-prediction on input change, dark mode, reduced-motion support. Fixed: empty-state overlay (`[hidden]` rule), cache-busting.
- **2026-11-12 — Docs.** `README.md`, `docs/ROADMAP.md` (tonight's plan, report outline, flowchart, demo script, viva answers), `docs/DATASETS.md`, `docs/LITERATURE_SURVEY.md` (16 + 3 new papers, gap paragraph), `colab/COLAB_GUIDE.md` (Modules 4-8 code).
- **2026-11-12 — GitHub.** Created private repo `resaleiq-used-vehicle-price` and pushed everything (private because the log names the student and college; flip to public in repo settings if wanted).
- **2026-11-12 — Prompt 2: model-list bugs + "is it really live?".** Problems: dropdown used crude families ("Pulsar" = a series, not a bike), and was missing anything after 2021 (Pulsar NS400Z) plus all scooters. Fixes:
  - `backend/scrape_catalog.py`: scraped official model lists from BikeWale (61 brands) and CarWale (29 brands) — names, 2026 ex-showroom price, cc, bhp, body type, fuel, on-sale/discontinued.
  - `backend/build_catalog.py`: mapped all 467 Kaggle bike names / 2,058 car names to official names (rules + auto-match; all targets verified official). 98.4% / 99.7% of rows matched. Found the Kaggle bike data has no scooters and nothing after 2021.
  - Dropdown now = on-sale models (official) + discontinued models that are in the training data. 541 2W, 345 cars.
  - Two engines: XGBoost (live, for models in training data) and a depreciation model (`app/depreciation.py`) for EVs, scooters and post-2021 models — Module 6 rates (motorcycles −6.4%/yr, scooters −8.5%, EVs −7.3%, Ola −12%) on official price + state road tax; cars −11.4%/yr fitted within-model on 2,720 CarDekho listings.
  - UI: grouped dropdown (motorcycles / scooters on sale / discontinued), engine hint under Model, engine badge on result ("Predicted live by XGBoost · 7 ms"), "How this was priced" box with server and round-trip time.
  - Retrained: bike XGBoost R² 0.946 / MAE ₹9,632 (better MAE than before); RF has higher R² (0.952) but higher MAE, so "best" = lowest MAE. Fixed: f-string, form validation blocking submit.
  - Answer to "is it live?": yes — every request runs XGBoost inference (or the depreciation formula) on the server; it's fast because a trained tree model only does ~900 small tree lookups.
- **2026-11-12 — Prompt 3/4: tab gone, base paper, Colab, title, "go".**
  - Old BikeWale tab was in an ended cloud session → re-collected 2026 data ourselves: `backend/scrape_used_2026.py` pulls BikeWale's used-search JSON (offset paging, 24/request, 0.4 s apart) → **30,527 ads, 956 EVs, 1,151 cities**.
  - **2026 test** (`backend/calibrate_2026.py`): Kaggle-trained XGBoost on 13,598 ads of 157 known models → 8% too low overall, ~15% too low for 6+ yr bikes, 4% high for <1 yr. R² (log) 0.67 vs 0.95 in-sample → model goes stale. Website now applies an age-band calibration and shows "Similar 2026 ads" (real 2026 medians).
  - **EV result changed with the bigger sample**: EV −12.0%/yr vs petrol −6.3% (p = 0.001), without Ola −13.0%, Ola alone −10.6% (p = 0.13). Per-brand slopes are noisy (30-70 ads each). New honest claim: *used EVs lose value about twice as fast as petrol; no stable brand effect*. Site Findings, depreciation engine (EV 12%, petrol motorcycles 6.1%, scooters 7.6%) and docs updated.
  - Base paper: full text paywalled; read the public IEEE page → `docs/BASE_PAPER.md`. Six authors (Mahmud, Revathi were missing), DOI 10.1109/ICDSNS62112.2024.10691304. We replicate its four models instead.
  - **One-click Colab notebook** `colab/ResaleIQ_pipeline.ipynb` (+ data zip `colab/resaleiq_colab_data.zip`): all modules incl. base-paper replication, 2026 test, EV regression, SHAP. Executed locally in 45 s → `colab/ResaleIQ_pipeline_executed.ipynb`, figures in `docs/figures/colab/`. Base paper's Decision Tree R² 0.842 vs our XGBoost 0.919 on the same split.
  - Title → "Used Two-Wheeler Resale Price Prediction with EV Depreciation Analysis". Demo: local only.
