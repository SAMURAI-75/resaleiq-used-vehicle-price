"""Train the two-wheeler price model on live 2026 BikeWale ads (petrol, CNG and electric).

  python backend/train_2026.py      (after scrape_used_2026.py, build_catalog.py)

Why: the Kaggle data (2020-21) has no scooters, almost no EVs and nothing launched after 2021, and it
goes stale (see calibrate_2026.py). The 2026 ads cover all of these, so this model prices every
two-wheeler with ML. The Kaggle model stays as the course baseline and as a fallback.

Features: brand, model, fuel, body type, state, age, log(1+km), owners, 2026 ex-showroom price of the model.
Target: log(price). 80/20 split, random_state=42. Output models/bike2026_xgb.joblib + metrics in metrics.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from train import CITY_STATE, feature_groups, iqr_filter, age_group, make_pre, scores

ROOT = Path(__file__).resolve().parent
NOW = 2026
CAT = ["brand", "model_key", "fuel", "body", "state"]
NUM = ["age", "log_km", "owners", "new_price"]
BODY = {"5": "scooter"}


def load():
    u = pd.read_csv(ROOT / "data" / "used_2026" / "bikewale_used_2026.csv")
    n0 = len(u)
    u = u.drop_duplicates("stockId").rename(columns={"makeName": "brand", "priceNumeric": "price", "kmNumeric": "km",
                                                     "ownersId": "owners", "cityName": "city"})
    u["model"] = (u.modelName.astype(str).str.replace(r"\[.*?\]", "", regex=True)
                  .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.strip())
    u["model_key"] = u.brand + " | " + u.model
    u["age"] = NOW - u.makeYear
    u["fuel"] = u.fuel.fillna("Petrol")
    u["body"] = u.bodyStyleId.astype(str).map(BODY).fillna("motorcycle")
    u["state"] = u.city.map(CITY_STATE).fillna("Other")
    u["owners"] = u.owners.clip(1, 4)
    cat = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())["bike"]
    newp = {f"{b} | {m['model']}": m["new_price"] for b, ms in cat.items() for m in ms if m.get("new_price")}
    u["new_price"] = u.model_key.map(newp)            # NaN for models no longer sold: XGBoost handles it
    u = u[(u.price >= 5_000) & (u.price <= 5_000_000) & u.km.between(0, 150_000) & u.age.between(0, 25)]
    u["age_grp"] = age_group(u.age)
    u = iqr_filter(u, ["model_key", "age_grp"], "price")
    u["log_km"] = np.log1p(u.km)
    return u.reset_index(drop=True), {"ads_loaded": n0, "final_rows": int(len(u)), "ev_rows": int((u.fuel == "Electric").sum()),
                                      "models": int(u.model_key.nunique())}


def main():
    u, log = load()
    print(log)
    X, y = u[CAT + NUM], np.log1p(u.price)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr_f, Xte_f = Xtr.fillna({"new_price": -1}), Xte.fillna({"new_price": -1})   # LR / RF need no NaN
    res, fitted = {}, {}
    for name, m, xt, xe in [
        ("Linear Regression", LinearRegression(), Xtr_f, Xte_f),
        ("Random Forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42), Xtr_f, Xte_f),
        ("XGBoost", XGBRegressor(n_estimators=1200, learning_rate=0.04, max_depth=7, subsample=0.85, colsample_bytree=0.8,
                                 min_child_weight=2, random_state=42, n_jobs=-1), Xtr, Xte)]:
        p = Pipeline([("pre", make_pre(CAT, NUM)), ("m", m)]).fit(xt, ytr)
        res[name] = scores(yte, p.predict(xe))
        fitted[name] = p
        print(f"  {name:18s} MAE {res[name]['MAE']:>9,.0f}  R2 {res[name]['R2']:.3f}  R2(log) {res[name]['R2_log']:.3f}")
    xgb = fitted["XGBoost"]
    pred = np.expm1(xgb.predict(Xte))
    ratio = np.expm1(yte.values) / pred
    within = float(np.mean(np.abs(pred / np.expm1(yte.values) - 1) <= .2))
    ev = (Xte.fuel == "Electric").values
    ev_m = {"ev_test_ads": int(ev.sum()),
            "ev_MAE": float(np.mean(np.abs(pred[ev] - np.expm1(yte.values)[ev]))),
            "ev_within_20pct": float(np.mean(np.abs(pred[ev] / np.expm1(yte.values)[ev] - 1) <= .2))}
    print("  within ±20%:", round(within, 3), "| EVs:", ev_m)
    trained = {k: int(n) for k, n in u.model_key.value_counts().items()}
    joblib.dump({"pipe": xgb, "groups": feature_groups(xgb.named_steps["pre"], CAT, NUM),
                 "band": [float(np.quantile(ratio, .2)), float(np.quantile(ratio, .8))],
                 "cat": CAT, "num": NUM, "trained": trained}, ROOT / "models" / "bike2026_xgb.joblib", compress=3)
    mp = ROOT / "models" / "metrics.json"
    metrics = json.loads(mp.read_text())
    metrics["bike2026"] = {"models": res, "train": int(len(Xtr)), "test": int(len(Xte)), "cleaning": log,
                           "xgb_within_20pct": within, **ev_m}
    mp.write_text(json.dumps(metrics, indent=2))
    print("saved models/bike2026_xgb.joblib")


if __name__ == "__main__":
    main()
