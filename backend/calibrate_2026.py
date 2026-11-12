"""Test the Kaggle-trained bike model on live 2026 BikeWale ads and save a calibration.

  python backend/calibrate_2026.py      (after scrape_used_2026.py and train.py)

The XGBoost model learned 2020-21 asking prices. Prices are moved to 2026 with 5 %/yr, but new-bike
prices rose faster than that, so the model may run low. Here we price every 2026 petrol ad whose model
the XGBoost model knows, and measure median(actual / predicted) overall and by age band.
Output models/calibration_2026.json:
  factor_by_age   multiplier the API applies to XGBoost bike prices
  similar         median 2026 asking price per "Brand | Model|age" (shown as "Similar 2026 ads")
  test            before/after metrics on the 2026 ads (report table)
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from train import CITY_STATE

ROOT = Path(__file__).resolve().parent
NOW, GROWTH, REF = 2026, 1.05, 2021
BANDS = [(-1, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 30)]


def band(a):
    return next(f"{lo + 1}-{hi}" for lo, hi in BANDS if lo < a <= hi)


def main():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    m = joblib.load(ROOT / "models" / "bike_xgb.joblib")
    u = pd.read_csv(ROOT / "data" / "used_2026" / "bikewale_used_2026.csv")
    u = u.rename(columns={"makeName": "brand", "priceNumeric": "price", "kmNumeric": "km", "ownersId": "owners", "cityName": "city"})
    u["model"] = (u.modelName.astype(str).str.replace(r"\[.*?\]", "", regex=True)
                  .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.strip())
    u["key"] = u.brand + " | " + u.model
    u["age"] = NOW - u.makeYear
    u = u[(u.price > 5_000) & u.km.between(100, 150_000) & u.age.between(0, 25)].drop_duplicates("stockId")

    sim = u.groupby(["key", "age"]).price.agg(["median", "count"]).reset_index()
    similar = {f"{r.key}|{int(r.age)}": [float(r["median"]), int(r["count"])] for _, r in sim.iterrows() if r["count"] >= 2}

    t = u[(u.fuel == "Petrol") & u.key.isin(meta["bike"]["trained"])].copy()
    t["state"] = t.city.map(CITY_STATE).fillna("Other")
    t["log_km"] = np.log1p(t.km)
    t["owners"] = t.owners.clip(1, 4)
    t["power"] = t.key.map(lambda k: meta["bike"]["trained"][k]["cc"])
    t = t.rename(columns={"key": "model_key"})
    t["pred"] = np.expm1(m["pipe"].predict(t[m["cat"] + m["num"]])) * GROWTH ** (NOW - REF)
    t["ratio"] = t.price / t.pred
    t["band"] = t.age.map(band)
    fb = t.groupby("band").ratio.agg(["median", "count"])
    overall = float(t.ratio.median())
    factor_by_age = {b: (float(r["median"]) if r["count"] >= 30 else overall) for b, r in fb.iterrows()}
    t["pred_cal"] = t.pred * t.band.map(factor_by_age)

    def mets(p):
        return {"MAE": float(mean_absolute_error(t.price, p)), "R2": float(r2_score(t.price, p)),
                "R2_log": float(r2_score(np.log(t.price), np.log(p))),
                "within_20pct": float(np.mean(np.abs(p / t.price - 1) <= .2)),
                "median_abs_pct_err": float(np.median(np.abs(p / t.price - 1)))}
    test = {"ads_total": int(len(u)), "ads_tested": int(len(t)), "models_tested": int(t.model_key.nunique()),
            "before": mets(t.pred), "after": mets(t.pred_cal), "overall_ratio": overall,
            "ratio_by_band": {b: [float(r["median"]), int(r["count"])] for b, r in fb.iterrows()}}
    (ROOT / "models" / "calibration_2026.json").write_text(json.dumps(
        {"factor_by_age": factor_by_age, "bands": BANDS, "similar": similar, "test": test}, indent=1))
    print(json.dumps(test, indent=1))


def dep_calibration():
    """Backtest the depreciation engine (app/depreciation.py) on every 2026 ad it would price and save
    median(actual / predicted) by group (EV / scooter / motorcycle) and age band -> models/dep_calibration_2026.json."""
    import sys
    sys.path.insert(0, str(ROOT.parent))
    from backend.app import depreciation as dep
    cat = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())["bike"]
    info = {(b, m["model"]): m for b, ms in cat.items() for m in ms if m.get("new_price")}
    u = pd.read_csv(ROOT / "data" / "used_2026" / "bikewale_used_2026.csv")
    u["model"] = (u.modelName.astype(str).str.replace(r"\[.*?\]", "", regex=True)
                  .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.strip())
    u["age"] = NOW - u.makeYear
    u = u[(u.priceNumeric > 5_000) & u.kmNumeric.between(0, 150_000) & u.age.between(0, 25)]
    rows = []
    for r in u.itertuples():
        m = info.get((r.makeName, r.model))
        if not m:
            continue
        out = dep.predict_2w(r.makeName, m, m["new_price"], r.age, r.kmNumeric, int(min(max(r.ownersId, 1), 4)), None)
        grp = "ev" if m["fuel"] == "electric" else ("scooter" if m["type"] != "motorcycle" else "motorcycle")
        rows.append((grp, band(r.age), r.priceNumeric / out["price"]))
    d = pd.DataFrame(rows, columns=["grp", "band", "ratio"])
    fac = {}
    for g, gd in d.groupby("grp"):
        overall = float(gd.ratio.median())
        fac[g] = {}
        last = overall
        for lo, hi in BANDS:
            b = f"{lo + 1}-{hi}"
            s = gd[gd.band == b].ratio
            last = float(s.median()) if len(s) >= 30 else last   # thin band -> carry the previous band's factor
            fac[g][b] = last
    test = {g: {"ads": int(len(gd)), "median_ratio": float(gd.ratio.median())} for g, gd in d.groupby("grp")}
    (ROOT / "models" / "dep_calibration_2026.json").write_text(json.dumps({"bands": BANDS, "factor": fac, "test": test}, indent=1))
    print(json.dumps(fac, indent=1))


if __name__ == "__main__":
    main()
    dep_calibration()
