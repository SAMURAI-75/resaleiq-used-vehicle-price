# Roadmap — due tomorrow

## Where things stand (12 Nov 2026)
| Piece | State |
|---|---|
| Topic, abstract, novelty | ✅ |
| Colab M1-3 (load, clean, road tax) | ✅ |
| Colab M6 EV study | ✅ numbers; ⚠️ CSV still in browser tab |
| Website (frontend + backend + trained models) | ✅ runs locally, all 3 vehicle types |
| Datasets doc, lit survey, Colab guide | ✅ in this repo |
| Colab M4, M5, M7, M8 | ⏳ code ready in `colab/COLAB_GUIDE.md` |
| Final report + slides | ⏳ |
| Deploy website online | ⏳ optional |

## Tonight (≈4 h total)
| # | Task | Who | Time | Done when |
|---|---|---|---|---|
| 1 | Download `two_wheelers_2026_same_variant.csv` from the BikeWale tab | Kartik | 10 min | CSV in Drive |
| 2 | Run Colab Module 4 (copy from guide) | Kartik | 20 min | Results table with LR/RF/XGB |
| 3 | Run Module 5 (two small tables) | Kartik | 20 min | R² table + LOSO table |
| 4 | Run Module 7 SHAP | Kartik | 10 min | Beeswarm + waterfall saved |
| 5 | Run Module 8 figures | Kartik | 20 min | 3 PNGs saved |
| 6 | Get Ferdous et al. PDF; confirm ⚠️ paper years | Kartik | 30 min | Survey table final |
| 7 | Write report (template below) | Kartik + Claude | 90 min | PDF/Doc ready |
| 8 | Rehearse website demo (script below) | Kartik | 15 min | Can demo in 3 min |

## Tomorrow morning
- Start the site: `.venv/bin/uvicorn backend.app.main:app --port 8000` → open http://localhost:8000 on the laptop **before** the presentation (works offline once loaded, except Google Fonts — falls back to system font).
- Optional: deploy to Render (free) so it's reachable by URL — needs your GitHub + Render login.

## Report structure (matches teacher's requirements)
1. Abstract · 2. Problem definition · 3. Objectives · 4. Literature survey (`docs/LITERATURE_SURVEY.md`) + research gap
5. Datasets (`docs/DATASETS.md`) · 6. Methodology — **flowchart** (below) · 7. Implementation (Colab modules + website architecture diagram)
8. Results — model table (MAE, RMSE, R²), actual-vs-predicted, SHAP, road-tax side note, **EV vs petrol table + chart**
9. Discussion (Ola drives the EV gap) · 10. Limitations · 11. Conclusion & future work · 12. References

### Flowchart (methodology)
```
[Kaggle bikes 32,648] ─┐
[BikeWale 2026 5,694] ─┤
[Road-tax sources]  ───┤
                       ▼
            Load & inspect (M1)
                       ▼
   Clean: dedupe → km filter → IQR per family+age (M2)
                       ▼
   Features: city→state, road-tax rate, log km, one-hot (M3)
                       ▼
          80/20 split (random_state = 42)
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   Linear Reg.   Random Forest    XGBoost (M4)
         └─────────────┼─────────────┘
                       ▼
      Metrics: MAE, RMSE, R² (expm1 back to ₹)
                       ▼
   ┌───────────────────┼──────────────────────┐
   ▼                   ▼                      ▼
Road-tax check (M5)  EV vs petrol (M6)    SHAP (M7)
                       ▼
          Website: FastAPI + Apple-style UI
```

### Website architecture
```
Browser (index.html + app.js)  ──fetch──▶  FastAPI (backend/app/main.py)
   segmented control 2W/Car                 ├─ /api/meta     brands, models, states
   form → POST /api/predict                 ├─ /api/metrics  test-set scores
   tabs: Price | Why | Depreciation         └─ /api/predict
                                                 ├─ bike → XGBoost + Tree SHAP + road-tax note
                                                 ├─ car  → XGBoost + Tree SHAP
                                                 └─ ev   → Module 6 depreciation model + battery health
```

## 3-minute demo script
1. Hero → "Built on real listings" stats → How it works (5 steps).
2. Findings: "EVs don't lose value faster — one brand does." Point at the orange Ola line.
3. Models table: XGBoost best, R² 0.93 on unseen bikes.
4. Predict: 2022 Royal Enfield Classic, Karnataka → price → **Why this price** (engine, age, state) → road-tax note.
5. Switch to Electric → Ola S1 Pro 2022 → Depreciation tab (Ola vs petrol). Switch brand to Ather → gap shrinks.
6. Switch to Car → Hyundai i20 → show it works for cars too.

## Viva quick answers
- *Why log price?* Prices are skewed (₹15k–₹15L); log makes errors proportional; we convert back with `expm1` before scoring.
- *Why XGBoost won?* Trees capture non-linear age × model effects; boosting fixes previous trees' errors.
- *What's SHAP?* Splits each prediction into each feature's push from the average; we use exact Tree SHAP from XGBoost.
- *How are EVs priced if Kaggle has none?* From our own 2026 same-variant regression; the app says so on screen.
- *Is the EV result causal?* No — listing prices, small sample, and it's mostly Ola. We say "one brand loses value fast".
