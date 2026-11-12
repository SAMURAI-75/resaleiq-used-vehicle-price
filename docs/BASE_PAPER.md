# Base paper — notes (full PDF not accessible; built from the public IEEE Xplore page)

**Citation (IEEE style)**
K. H. Ferdous, M. S. Arman, K. R. Ahmed, N. Tasnim, I. Mahmud and P. Revathi, "Predicting Pre-Owned Bike Values with Machine Learning Techniques in Bangladesh," *2024 International Conference on Data Science and Network Security (ICDSNS)*, Tiptur, India, 2024, doi: 10.1109/ICDSNS62112.2024.10691304.

- Conference: 26–27 July 2024, Tiptur, India · Added to IEEE Xplore: 1 Oct 2024
- **Six authors** (earlier notes listed five — Imran Mahmud and P. Revathi were missing). Mahmud is at Daffodil International University, Dhaka.
- Sections: I Introduction · II Related Works · III Methodology · IV Result and Outcomes · V Conclusion

## What the paper does (from the abstract + introduction)
| Item | Base paper |
|---|---|
| Data | Used motorcycles, collected from motorcycle **merchants in Bangladesh** (size not stated in the abstract) |
| Features mentioned | condition, mileage, brand reputation, model characteristics, age, engine |
| Models | Decision Tree, Support Vector Regression (SVR), K-Nearest Neighbours (KNN), Stochastic Gradient Descent (SGD) regression, "used with a hybrid architecture" |
| Best | Decision Tree, "97% accuracy rate" |
| Framing | Sustainability / IoT |

## Weaknesses we address (use in Problem Definition + Literature Survey)
1. **"Accuracy" for a regression problem.** Price is a continuous number; accuracy needs classes. We report **MAE (₹ off), RMSE and R²** on a held-out test set, plus "% of predictions within ±20%" so the two can be compared fairly.
2. **No ensembles.** Only single models (DT, SVR, KNN, SGD). We add Random Forest and XGBoost and **re-run all four of their models on our data** as a direct replication (Colab notebook, Module 4).
3. **No explanation.** We add Tree SHAP for every prediction.
4. **No time check.** We test a model trained on 2020-21 listings against 30k live 2026 BikeWale ads.
5. **No EVs, no location cost.** We add the EV vs petrol same-model depreciation study and the state road-tax check.

## How to cite it without the PDF
Cite from the IEEE Xplore page (DOI above). In the viva: "We studied the published abstract and method description; the full text is behind IEEE's paywall, so we replicated their four algorithms on our own data instead of relying on their numbers."

Source: https://ieeexplore.ieee.org/document/10691304
