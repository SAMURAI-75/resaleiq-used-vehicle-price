# Colab — what to do (about 20 minutes)

## Why Colab at all?
The website already uses trained models, but the teacher grades the **ML work**: data cleaning, model comparison, accuracy numbers, SHAP and charts. The Colab notebook is that evidence: one place where every step runs top to bottom with its outputs visible. The report's tables and figures come out of it.

You do **not** need your old notebook any more. `ResaleIQ_pipeline.ipynb` does all modules (1–10) in one go, including the main 2026 models the website uses. Keep the old one only as a backup.

## Steps
**New (20 Nov): the notebook now includes the India tax history (Module 9 + new Module 9b). Re-upload BOTH files below; the old zip won't work.**

1. Open https://colab.research.google.com → **File → Upload notebook** → pick `colab/ResaleIQ_pipeline.ipynb` from this project folder.
2. **Runtime → Run all.**
3. The first cell shows a **Choose files** button → pick `colab/resaleiq_colab_data.zip` (same folder).
4. Wait about 10–15 minutes. The heavy cells are Module 4 (SVR, Random Forest) and Module 9 (91,748 car ads).
5. At the end, Colab downloads `resaleiq_figures.zip`. Unzip it into `docs/figures/colab/` in this project. I'll use those for the report.
6. **File → Save a copy in Drive.** That's the notebook you show the teacher.

If Colab asks to sign in to Kaggle, it's because `kagglehub` hit a rate limit. Re-run the cell, or sign in at kaggle.com, open *Settings → Create New Token*, and upload `kaggle.json` when asked.

## What each module shows (for the viva)
| Module | Output | One-line explanation |
|---|---|---|
| 1 Load | 32,648 rows, 25,324 duplicates | The Kaggle file repeats most rows |
| 2 Clean | ~7,100 clean bikes, official model names | Messy names mapped to BikeWale's official list |
| 3 Features | road-tax rate per bike | State slab × the model's 2026 new price |
| 4 Models | table + 2 charts | Base paper's 4 models vs LR / RF / XGBoost on the same split |
| 5 Road tax | 2 small tables | Does state/tax improve prices? Mostly no → side note |
| 6 2026 test | calibration factor | The 2020-21 model priced against live 2026 ads |
| 7 EV vs petrol | regression table + chart | Same variant: EVs lose ~12% a year vs ~6% for petrol; no stable brand effect |
| 8 SHAP | 2 charts | Which features push the price up or down |
| 9 Main 2026 models | 2 tables + chart | Cars 8.3% typical miss vs CarWale's own valuation 11.3%; two-wheelers 12.4%; both near the noise floor. Now also uses the tax paid when new |
| 9b Tax history | chart + table | India's tax on a new vehicle 1990-2026 (excise + VAT, then GST, GST 2.0); bike model with vs without it: 12.41% → 12.37%, i.e. no real change |
| 10 Save | figures zip | Everything for the report |

A copy that I already ran on this Mac (outputs included) is `colab/ResaleIQ_pipeline_executed.ipynb`. Open it if Colab gives trouble during the demo.
