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
- **2026-11-13 — Colab run + report.** Kartik ran the notebook in Colab; outputs match the local run (EV results identical; others within 0.02 R²). Colab outputs are now the official figures (`docs/figures/colab/`). Colab numbers: XGBoost R² 0.921, MAE ₹10,422, 79.8% within ±20%; base paper's DT R² 0.864, 64.8%.
  - **Report drafted** as a Claude Doc: https://claude.ai/artifact/9pWZr2uQD22ZeVSLTvANsx — abstract, problem, objectives, 19-paper survey + gap, datasets, methodology (flowchart), implementation (architecture diagram), results (model table, 2026 test chart, SHAP chart), EV vs petrol (chart + 4-check table + 5 Nov vs 12 Nov comparison), road tax side check, limitations, conclusion, references. Exportable to Word/PDF.
  - To do (Kartik): read through, fill author lists for survey rows marked "by title", confirm the ⚠️ paper years, export.
- **2026-11-13 — Sanity check (Ather 450 Apex looked high).** Backtested the depreciation engine on 19,578 real 2026 ads: EVs about right at 2-3 yrs (ratio 1.01), but older petrol bikes/scooters came out 15-23% too high (it starts from today's new price). Added an age-band × group correction (`models/dep_calibration_2026.json`, from `calibrate_2026.py`) and "Real 2026 ads" next to every depreciation-model price. Now: 2024 Ather 450X ₹1,09,900 vs ₹1,10,000 real (6 ads); 2018 Activa ₹39,500 vs ₹40,000 (39 ads). 450 Apex still ~₹1.4L vs 2 ads at ₹1.0L/₹1.3L — too few ads to correct per model.
- **2026-11-13 — "the hell is this": EVs were priced by a formula.** Trained XGBoost on the 2026 BikeWale ads (`backend/train_2026.py`): 28,912 ads after cleaning, 664 models, 920 EVs. Test (20%): R² (log) 0.871, MAE ₹15,140, 68.7% within ±20% (EVs 60.5%, 195 test ads); the Kaggle model managed ~52% on the same 2026 ads. It is now the main two-wheeler engine: 283 models priced by it (incl. every major EV and scooter), 23 older models by the calibrated Kaggle model, 235 rare/new models by the depreciation fallback. Features: brand, model, fuel, body type, state, age, km, owners, 2026 new price. "How this was priced" rewritten in plain language (no ms timings); "Similar ads" relabelled "All-India 2026 ads" (Karnataka asks ~10-25% more). Models section has a 2026 tab.
- **2026-11-13 — "make it full proof, for cars and motorcycles, gas and EV; get more data".**
  - **More data:** CarWale used cars, crawled city by city (the all-India list stops at ~1,450 pages): **91,748 ads**, 1,734 EVs, each with its variant's new on-road price and CarWale's own valuation. BikeWale re-crawled with seller type (30,306 ads). Variant new prices for 995 bike variants from 540 model pages. Full city→state map from CarWale's city list (bike ads with a known state 66.8% → 99.8%). BikeDekho checked: only 588 ads, skipped.
  - **New main models** (`backend/train_market.py`): XGBoost with native categories (brand, model, variant, fuel, body, gearbox, state, city, seller), new price of the exact variant, monotone constraints (never up with age/km/owners), seed-averaged ensemble chosen after experiments (loss functions, learning rates, regularisation, dropping features, residual blending — all within ±0.3 pt; seed averaging +0.8 pt within-20%).
  - **Results (20% test):** cars 8.3% typical miss / 83.0% within ±20% vs **CarWale's own valuation 11.3% / 73.2%** on the same ads; two-wheelers 12.4% / 69.3%. Noise floor (same variant, year, seller): cars 8.7%, bikes 11.1% — both models are near it.
  - **Finding:** same variant + year, Karnataka asks +15% for cars and +8% for bikes vs the national median; Delhi −7%, West Bengal −11% (NCR age rules).
  - **Bugs found and fixed via `backend/tests/sweep_test.py`** (prices all 1,104 models × 3 years + 15 edge cases): NaN new prices crashed 18 requests; old models on the Kaggle fallback rose in price with age (now a running minimum); old cars without a variant used today's new price (now the new price of the variants sold that year, 3-year look-back so an older generation with the same name can't leak in, e.g. Grand Vitara); hybrids were parsed as electric. Final run: 0 models flagged, 0 edge cases failed.
  - **UI:** Variant (with years and new price), City (biggest first), Seller (private/dealer), year list limited to years that appear in that model's ads, warning outside that range, Models section shows accuracy per segment and the CarWale benchmark.
  - Car model size: 3 × depth 8 (45 MB) instead of 5 × depth 10 (128 MB, over GitHub's limit) for −0.1 pt.
  - Report updated: new section 7.4 (main 2026 models, CarWale benchmark, noise floor, state premium, sweep test), abstract, datasets and implementation text.
- **2026-11-13 — "why no Apex?" + "road tax is not the same for every vehicle (my 450cc had 40%)".**
  - Variant lists only held variants with 3+ ads; Guerrilla 450 Apex has 2. New `backend/build_variants.py`: every official variant (BikeWale model pages) + every variant seen in any ad (2,046 bike / 8,353 car variants). Variants without a learned effect are priced through their own new price. Colour editions with the same price merge ("Dash"); different prices stay apart ("Apex (Red)", "Apex (Black and Green)"); bracketed years stripped from car names.
  - The 40% is **GST**, not road tax: since 22 Sep 2025 bikes >350cc pay 40% (was 31%), <=350cc 18% (was 28%), EVs 5%; it sits inside the ex-showroom price. Karnataka road tax is 18% + 11% cess on the tax ≈ 20%. Tax note rewritten: GST by engine size and purchase date, road tax with cess and amount, state premium measured on 2026 ads (`backend/state_premium.py`: same variant + year, Karnataka cars +15%, bikes +8%; Delhi bikes −8%). Replaced the old extrapolated "₹X may reflect the tax" line.
  - Fixed: Fuel/Gearbox row showed for two-wheelers (CSS hid only `.field`); NaN in variant lists broke `/api/variants` for 236 models (caught by the sweep). Sweep: 0/1,104 flagged, 0 edge failures. Report section 9 updated.
- **2026-11-13 — "if it's 2026, why are 2024 and 2025 offered?"** The year list followed the model (Guerrilla 450: 2024-2026), not the chosen variant. Now it follows the variant: Apex → 2026 only (launched 27 Mar 2026, confirmed by Autocar India / DriveSpark), Dash → 2024-2026, discontinued variants stop at their last year (old Creta E 1.6 → 2017-2018). The API also warns if a year falls outside the variant's years. Sweep still 0/1,104.
- **2026-11-13 — "verify years for all vehicles" + full test.**
  - Audit (`backend/tests/year_audit.py`): official year ranges from the sites' own names hold for cars (0.9-1.6% of ads outside) but not for single-year labels ("Activa Standard [2022]": 18-56% outside -- they mark updates). Two-wheeler ranges hold less well (~10%).
  - Rules (`backend/years.py`): ranged labels only (+1 year for next-January registration); two-wheelers widened to their ad years; typo years dropped; thin early tails cut only before the site's launch date when volume starts at that date (R15 V4 2019-20 cut; iQube 2021 kept). A model only lists variants of its own fuel (electric Chetak no longer inherits the 1972-2006 petrol Chetak). Model year lists skip gaps between generations (Grand Vitara 2003-2016, 2022-2026).
  - Year test (`backend/tests/year_check.py`): 1,104 models, 9,368 variants; 98.9% (2W) / 99.9% (cars) of real ads have their year selectable; 0 variants outside their model; 26/27 launch-year spot checks (Thar starts 2014: no 2010-13 ads exist).
  - Full test: all Python compiles; price sweep 0/1,104 flagged, 0/15 edge cases; Colab notebook runs end to end; browser flows (petrol bike, EV scooter, petrol car, EV car, 1995 Maruti 800) all priced; phone layout fixed (Models switch made the page 449px wide).
  - Fixes found by the full test: API rejected 1984 (Maruti 800) -> floor 1980; Kaggle fallback could make a newer year cheaper -> retrained with monotone constraints (R² 0.946 -> 0.908, serves 29 models); train.py overwrote metrics.json and wiped the market models' metrics -> it now merges.
- **2026-11-13 — Colab now includes the main 2026 models (Module 9).** Same cleaning, features, monotone rules and seed-averaged XGBoost as `backend/train_market.py`, on the 13 Nov BikeWale (30,306) and CarWale (91,748) collections; zip grows to 5.5 MB (adds car ads, bike variants, car catalogue, full city map). Executed here: cars 8.3% / 83.0% within ±20% (CarWale's valuation 11.3% / 73.2%), two-wheelers 12.4% / 69.3% — identical to the website. Modules 1-8 unchanged (EV study identical; Module 4 differs only by library version vs Colab, as before). Run time ~2 min here, ~10-15 min on Colab. New figure `docs/figures/colab/m9_market_models.png`.
- **2026-11-13 — Two more improvements.**
  - **EV scooter specs:** `scrape_variants.py` now also saves battery kWh, kerb weight (and cc/bhp where listed); battery size known for 56% of EV ads (variant spec or "2.9 kWh" in the name), rest from the model median; model-level cc/bhp from the catalogue. Result: all bikes 12.4% → 12.3%, EV scooters within-20% 61.6% → 63.0% but typical miss 14.1% → 14.4% — within noise (±4 pts on 138 test ads). Left switched off (`EXTRA_NUMS` in `train_market.py`), noted in README.
  - **Per-segment price ranges:** middle 60% of validation misses per vehicle type (electric scooters −25%/+22%, CNG cars −8%/+7%). Test coverage: 59-67% per type vs 56-79% with one shared range. Model scores unchanged (cars 8.3%, bikes 12.4%), so report and Colab still match.
  - Bug caught by the sweep: a variable-name clash broke every market prediction after the change; fixed. Sweep 0/1,104, year test unchanged.
- **2026-11-13 — Report exported** to `docs/report/ResaleIQ_Report.pdf` (15 pages, A4, charts on pp. 6, 8, 9, 10, 12). Re-export after final edits (author name in byline, survey author lists).
