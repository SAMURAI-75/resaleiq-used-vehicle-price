# Literature Survey

Rule from the teacher: papers from **2022 or later**, focused on the research gap, at least 15.
Legend — ✅ verified (title/venue/year seen online this session or earlier) · ⚠️ year taken from a listing page, confirm on the PDF before submitting.

## A. Core table (16 submitted + 3 new finds)

| # | Paper | Year | Data | Method / best | Metric reported | Relevance to our gap | Check |
|---|---|---|---|---|---|---|---|
| 1 | Ferdous, Mahmud, Ahmed, Arman, Tasnim — *Predicting Pre-Owned Bike Values with ML Techniques in Bangladesh*, 2nd IEEE ICDSNS | 2024 | Bangladesh merchant bike data | DT, SVR, KNN, SGD; DT best | "97% accuracy" (regression!) | **Base paper.** No EVs, no tax, no explanation; misuse of accuracy | ✅ abstract; get PDF |
| 2 | *Interpretable ML for Price Prediction in India's Used Two-Wheeler Market using XAI Tools* | 2025 | Indian used bikes | Tree models + SHAP/LIME | R² | Already does XAI → SHAP is *supporting*, not our novelty | ⚠️ |
| 3 | Sathyanarayanan, K. — *Pre-Owned Bike Price Prediction Using ML*, MSc thesis, National College of Ireland | 2023 | ~5,063 Indian Kaggle listings | LR, Elastic Net, SVR, RF, XGBoost; RF & XGB best | R², RMSE | Closest prior work; no EVs, no location-tax | ✅ (norma.ncirl.ie/7264) |
| 4 | *Used Car Price Prediction and Feature Importance Analysis Using XGBoost and SHAP* | 2025 | 7,253 Indian cars | XGBoost + SHAP | R² 0.87 | Same toolkit as ours, cars only | ⚠️ |
| 5 | *Explainable Used Car Pricing: ML Framework with Marginal Effects* (HKU) | 2024-25 | Cars | Explainable ML, marginal effects | R² | XAI for cars | ⚠️ |
| 6 | Luo et al. — *Revolutionizing the Malaysian Used Car Market* (IEEE) | 2023 | Malaysian cars | NN + regression | R², MAE | Cars only | ✅ |
| 7 | Bukvić et al. — *Price Prediction and Classification of Used-Vehicles Using Supervised ML* | 2022 | Croatian cars, 2nd set 3 months later | Multiple | R², accuracy | Shows data goes stale over time | ✅ |
| 8 | *Car Price Prediction Using ML Algorithms* (IJRASET) | 2022 | Cars | 8 models; XGBoost best | R² | Cars only | ✅ |
| 9 | *Prediction of Used Car Prices using ML Techniques* (IRJET) | 2022 | Cars | RF best | "83.62% accuracy" | Same metric misuse as base paper | ✅ |
| 10 | *Price Prediction for Used Cars: A Comparison of ML Regression* (KTH thesis) | 2022 | Cars | RF best | MAE, R² | Cars only | ✅ |
| 11 | *Prediction of the Price of Used Cars Based on ML Algorithms* | 2023 | Cars | SVM, XGBoost, NN | R² | Cars only | ✅ |
| 12 | *Machine Learning-Based Models for Accurate Car Prices Prediction* | 2024 | Cars | RF best | R² | Cars only | ⚠️ |
| 13 | *Machine Learning Optimization and Challenges in Used Car Price Prediction* | 2025 | Cars | Tuned XGB, RF, LR | R² | Cars only | ⚠️ |
| 14 | *Car Price Prediction and Recognition Using Deep Learning and Computer Vision* | 2025 | Car images + tabular | CNN + regression | R² | Images, not location/EV | ⚠️ |
| 15 | *Linear regression for used car price prediction* (IJCESEN, DOI 10.22399/ijcesen.1070505) | 2023 | Turkish cars | LR | R² 0.73 | Linear baseline | ✅ |
| 16 | *Used Car Price Prediction Using ML* (incl. used EV car prices) | 2023 | Cars incl. EVs | RF, gradient boosting | R² | EV resale studied **for cars only** | ⚠️ |
| 17 🆕 | Roberson, Pantha & Helveston — *Battery-powered bargains? Assessing electric vehicle resale value in the United States* | 2024 | US listings, make-model level | Value-retention estimation (BEV, PHEV, HEV, ICE) | Retention rate | Strongest EV-resale paper — **cars, USA**. Supports "first for Indian two-wheelers" | ✅ (DOAJ record) |
| 18 🆕 | *Price Prediction of Used Bike Using Machine Learning*, Int. J. Scientific & Technical Development (Desh Bhagat Univ.), Vol 10 Issue 1 | 2024 | Indian bikes (brand, model, km, owners, condition, city) | Hybrid Naive Bayes-KNN | accuracy | Another bike paper; still no EV / tax | ✅ |
| 19 🆕 | Das Green — *Depreciation Dynamics of Electric vs. ICE Vehicles in the US Car Market*, Univ. of Michigan thesis | 2024-25 | ~150k Craigslist listings, 17 US cities | Retention-rate regression + ML | Retention | EV vs ICE depreciation — cars, USA; same listing-price limitation as ours | ⚠️ year |

**2026 car papers you can swap in for 2022 entries (8, 9, 10):** *Key Influencing Factors of Used Car Prices Driven by ML* (2026, DOI 10.54254/2754-1169/2026.LD30669; RF + SHAP) · D. Nandhini, *Used Car Price Prediction Using ML*, ISJEM, Jan 2026 · *Second Hand Car Price Prediction using ML*, IJRASET, Apr 2026 (6,019 Indian listings, XGBoost) · *Used Car Price Prediction System Using ML*, IJETRM, Apr 2026.

## B. What the survey shows (use in the report)
1. **Cars dominate.** 13 of 19 papers are cars; only 4 study used two-wheelers (1, 2, 3, 18).
2. **No EV two-wheeler resale study.** EV resale is studied only for cars, mostly in the US (16, 17, 19). Industry sources in 2026 also say India has *no authoritative resale index for used e-scooters* and *no standard battery-health certificate*.
3. **Location is a label, never a cost.** Papers use city/state as a category; none models state road tax.
4. **Metrics are inconsistent.** "Accuracy" is reported for regression (1, 9, 18), so results can't be compared or pooled — and no systematic review/meta-analysis exists for used-vehicle price prediction.
5. **Tree ensembles win.** RF/XGBoost are best in most tabular studies (3, 8, 10, 12, 13) — matches our result (XGBoost R² 0.932).

## C. Research gap (one paragraph for the report)
Used two-wheelers are India's largest second-hand vehicle segment, yet ML pricing research concentrates on cars. The few bike studies (Ferdous et al. 2024; Sathyanarayanan 2023; IJSTD 2024; the 2025 XAI study) use only vehicle attributes and treat location as a label. None studies electric two-wheelers, even though EV resale value is an open question for buyers and lenders, and existing EV resale work covers only cars in the US (Roberson et al. 2024). We therefore (i) build an explained XGBoost price model for Indian two-wheelers reported with proper regression metrics, and (ii) give the first same-variant comparison of EV and petrol two-wheeler depreciation in India, with (iii) a short check on whether state road tax carries into used prices.

## D. To do before submission
- [ ] Download the full Ferdous et al. PDF from IEEE Xplore (college login) and fill in their exact dataset size and metrics.
- [ ] Confirm years for rows marked ⚠️ (open each PDF's first page).
- [ ] Optionally swap rows 8-10 (2022) for two 2026 papers listed above.
