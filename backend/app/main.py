"""ResaleIQ API.

  uvicorn backend.app.main:app --reload     (from the repo root)

GET  /api/meta      verified model list (BikeWale / CarWale) + which engine prices each model
GET  /api/metrics   test-set metrics of every ML model
POST /api/predict   {"type": "2w" | "car", brand, model, year, km, owners, ...}
                    -> price, range, factors, curve, and which engine produced it (+ time taken)
Everything else serves the frontend.

Two pricing engines, chosen per model:
  "ml"  XGBoost trained on Kaggle listings, run live on every request; factors are exact Tree SHAP values.
        Used when the model has >= MIN_TRAIN listings in the training data.
  "dep" Depreciation model (app/depreciation.py) on the official 2026 new price.
        Used for EVs, scooters and anything launched after the training data was collected.
"""
import json
import math
import time
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import depreciation as dep
from .road_tax import SOURCES, cheapest_state, road_tax_rate

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT.parent / "frontend"
NOW = 2026
MIN_TRAIN = 3

META = json.loads((ROOT / "models" / "meta.json").read_text())
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())
CATALOG = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())
MODELS = {k: joblib.load(ROOT / "models" / f"{k}_xgb.joblib") for k in ("bike", "car", "bike2026")}
TRAINED_2026 = MODELS["bike2026"]["trained"]          # model key -> number of 2026 training ads
BOOSTERS = {k: v["pipe"].named_steps["m"].get_booster() for k, v in MODELS.items()}
GROWTH = META["growth"]
_cal_path = ROOT / "models" / "calibration_2026.json"
CAL = json.loads(_cal_path.read_text()) if _cal_path.exists() else None   # from calibrate_2026.py


_dep_path = ROOT / "models" / "dep_calibration_2026.json"
DEP_CAL = json.loads(_dep_path.read_text()) if _dep_path.exists() else None   # from calibrate_2026.py


def dep_factor(info, age):
    """Backtest correction for the depreciation engine: median(2026 ad / prediction) by group and age band."""
    if not DEP_CAL:
        return 1.0
    grp = "ev" if info.get("fuel") == "electric" else ("scooter" if info.get("type") in ("scooter", "moped") else "motorcycle")
    for lo, hi in DEP_CAL["bands"]:
        if lo < age <= hi:
            return DEP_CAL["factor"].get(grp, {}).get(f"{lo + 1}-{hi}", 1.0)
    return 1.0


def cal_factor(mk, age):
    """2026 calibration for the bike model: median(actual 2026 ad / prediction) for this age band."""
    if mk != "bike" or not CAL:
        return 1.0
    for lo, hi in CAL["bands"]:
        if lo < age <= hi:
            return CAL["factor_by_age"].get(f"{lo + 1}-{hi}", 1.0)
    return 1.0
OWNER_WORD = dep.OWNER_WORD

POPULAR_2W = ["Hero", "Honda", "Bajaj", "TVS", "Royal Enfield", "Yamaha", "Suzuki", "KTM", "Ather", "OLA", "VIDA",
              "Jawa", "Yezdi", "Kawasaki", "Harley-Davidson", "Triumph", "BMW", "Ducati"]
POPULAR_CAR = ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Toyota", "Kia", "Honda", "MG", "Skoda", "Volkswagen", "Renault"]


def inr(x):
    """Indian digit grouping: 1234567 -> 12,34,567."""
    s = f"{int(round(x)):d}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _order(brands, popular):
    return sorted(brands, key=lambda b: (popular.index(b) if b in popular else len(popular), b.lower()))


def _build_index():
    """Join the verified catalog with the trained models -> one entry per selectable model."""
    idx = {"2w": {}, "car": {}}
    for brand, models in CATALOG["bike"].items():
        for m in models:
            key = f"{brand} | {m['model']}"
            tr = META["bike"]["trained"].get(key)
            n26 = TRAINED_2026.get(key, 0)
            ml = tr is not None and tr["n"] >= MIN_TRAIN and m["fuel"] != "electric"
            engine = "ml26" if n26 >= MIN_TRAIN else "ml" if ml else ("dep" if m.get("new_price") else None)
            if not engine:
                continue
            idx["2w"].setdefault(brand, {})[m["model"]] = {
                **m, "engine": engine, "cc": m.get("cc") or (tr or {}).get("cc"),
                "train_rows": n26 if engine == "ml26" else (tr or {}).get("n", 0)}
    for brand, models in CATALOG["car"].items():
        for m in models:
            key = f"{brand} | {m['model']}"
            tr = META["car"]["trained"].get(key)
            ml = tr is not None and tr["n"] >= MIN_TRAIN
            engine = "ml" if ml else ("dep" if m.get("new_price") else None)
            if not engine:
                continue
            idx["car"].setdefault(brand, {})[m["model"]] = {**m, **({k: tr[k] for k in ("cc", "power", "seats", "mileage", "fuel", "transmission")} if tr else {}),
                                                         "engine": engine, "train_rows": (tr or {}).get("n", 0)}
    return idx


INDEX = _build_index()

app = FastAPI(title="ResaleIQ", version="2.0")
app.add_middleware(GZipMiddleware, minimum_size=800)


class PredictIn(BaseModel):
    type: Literal["2w", "car"]
    brand: str
    model: str
    year: int = Field(ge=1990, le=NOW)
    km: float = Field(ge=0, le=1_000_000)
    owners: int = Field(1, ge=1, le=4)
    state: Optional[str] = None
    cc: Optional[float] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    seller_type: Optional[str] = "Individual"
    power: Optional[float] = None
    seats: Optional[float] = None
    new_price: Optional[float] = None
    kwh: Optional[float] = None


def _info(kind, brand, model):
    info = INDEX[kind].get(brand, {}).get(model)
    if not info:
        raise HTTPException(422, f"{brand} {model} is not in the verified list")
    return info


# ------------------------------------------------------------------ ML engine
def _frame(kind, p: PredictIn, info, ages):
    n, key = len(ages), f"{p.brand} | {p.model}"
    log_km = math.log1p(p.km)
    if kind == "bike":
        return pd.DataFrame({"brand": [p.brand] * n, "model_key": [key] * n, "state": [p.state or "Other"] * n,
                             "age": ages, "log_km": [log_km] * n, "owners": [p.owners] * n,
                             "power": [p.cc or info["cc"]] * n})
    return pd.DataFrame({
        "brand": [p.brand] * n, "model_key": [key] * n, "fuel": [p.fuel or info["fuel"]] * n,
        "seller_type": [p.seller_type or "Individual"] * n, "transmission": [p.transmission or info["transmission"]] * n,
        "age": ages, "log_km": [log_km] * n, "owners": [p.owners] * n,
        "engine_cc": [p.cc or info["cc"]] * n, "power_bhp": [p.power or info["power"]] * n,
        "mileage_kmpl": [info["mileage"]] * n, "seats": [p.seats or info["seats"]] * n})


LABELS = {
    "brand": ("Brand", "How the market rates this brand overall."),
    "model_key": ("Model", "Demand and original price of this exact model."),
    "state": ("State", "Local demand and road tax paid when it was bought."),
    "age": ("Age", "Older vehicles are worth less; usually the biggest factor."),
    "log_km": ("Kilometres", "More km means more wear."),
    "owners": ("Owners", "Each extra owner lowers buyer trust."),
    "power": ("Engine", "Bigger engines usually cost more new and used."),
    "engine_cc": ("Engine", "Bigger engines usually cost more new and used."),
    "power_bhp": ("Power", "More power signals a higher variant."),
    "mileage_kmpl": ("Mileage", "Fuel efficiency of the variant."),
    "seats": ("Seats", "Seating capacity."),
    "fuel": ("Fuel", "Diesel and CNG cars price differently from petrol."),
    "transmission": ("Gearbox", "Automatics usually ask more."),
    "seller_type": ("Seller", "Dealers usually ask more than individuals."),
    "body": ("Type", "Scooters and motorcycles hold value differently."),
    "new_price": ("Price new today", "What this model costs new in 2026 (BikeWale); used prices track it."),
}
FUEL_NOTE_2W = "Electric two-wheelers lose value faster than petrol ones (about 12% vs 6% a year)."


def _value_text(col, r, p):
    v = r[col]
    if col == "model_key":
        return p.model
    if col == "log_km":
        return f"{p.km:,.0f} km"
    if col == "owners":
        return OWNER_WORD[p.owners]
    if col == "new_price":
        return f"₹{inr(v)}" if isinstance(v, (int, float)) and v == v and v > 0 else "not sold now"
    unit = {"age": " yr", "power": " cc", "engine_cc": " cc", "power_bhp": " bhp", "mileage_kmpl": " kmpl", "seats": ""}
    return f"{float(v):g}{unit[col]}" if col in unit else str(v)


def predict_ml(kind, p: PredictIn, info):
    t0 = time.perf_counter()
    mk = "bike" if kind == "2w" else "car"
    ref = META[mk]["ref_year"]
    age = max(0, NOW - p.year)
    scale = GROWTH ** (NOW - ref)
    ages = list(range(0, 16))
    pipe = MODELS[mk]["pipe"]
    curve_prices = np.expm1(pipe.predict(_frame(mk, p, info, ages))) * scale * np.array([cal_factor(mk, a) for a in ages])
    row = _frame(mk, p, info, [age])
    Z = pipe.named_steps["pre"].transform(row)
    contrib = BOOSTERS[mk].predict(xgb.DMatrix(Z), pred_contribs=True)[0]
    cal = cal_factor(mk, age)
    price = float(math.expm1(contrib.sum()) * scale * cal)    # SHAP values sum exactly to the prediction
    ms = (time.perf_counter() - t0) * 1000

    bias = float(contrib[-1])
    by = pd.Series(contrib[:-1], index=MODELS[mk]["groups"]).groupby(level=0).sum()
    r = row.iloc[0]
    factors = [{"feature": LABELS[g][0], "value": _value_text(g, r, p), "effect": round((math.exp(c) - 1) * 100, 1),
                "note": LABELS[g][1]} for g, c in by.sort_values(key=np.abs, ascending=False).items()]
    lo, hi = MODELS[mk]["band"]
    top = factors[0]
    title = f"{p.year} {p.brand} {p.model}"
    sim26 = CAL["similar"].get(f"{p.brand} | {p.model}|{age}") if (CAL and mk == "bike") else None
    sim = sim26 or META[mk]["similar"].get(f"{p.brand} | {p.model}|{age}")
    out = {
        "title": title, "age": age, "price": round(price, -2), "low": round(price * lo, -2), "high": round(price * hi, -2),
        "base_price": round(math.expm1(bias) * scale * cal, -2), "factors": factors,
        "similar": ({"median": round(sim[0] * (1 if sim26 else scale), -2), "count": sim[1], "year": 2026 if sim26 else ref}
                    if sim else None),
        "curve": [{"age": a, "price": round(float(v), -2)} for a, v in zip(ages, curve_prices)],
        "new_price": info.get("new_price"),
        "summary": (f"A {title} with {p.km:,.0f} km and a {OWNER_WORD[p.owners]} owner"
                    f"{' in ' + p.state if kind == '2w' and p.state and p.state != 'Other' else ''} should list for about "
                    f"₹{inr(round(price, -2))}. The biggest single factor is {top['feature'].lower()} "
                    f"({'+' if top['effect'] > 0 else ''}{top['effect']}%)."),
        "engine": {"kind": "ml", "name": "XGBoost", "ms": round(ms, 1),
                   "detail": (f"Our XGBoost model, trained on {METRICS[mk]['train']:,} used-{'bike' if mk == 'bike' else 'car'} listings "
                              f"({info['train_rows']} of them this model), worked out this price just now from the details you entered"
                              + (", then adjusted to 2026 market prices." if cal != 1.0 else "."))},
        "method": (f"Factor sizes are exact Tree SHAP values from the same XGBoost run, relative to the typical listing. "
                   f"Kaggle prices (~{ref}) scaled to 2026 rupees at {round((GROWTH - 1) * 100)}% a year (×{scale:.2f})"
                   + (f", then calibrated ×{cal:.2f} against {CAL['test']['ads_tested']:,} live 2026 BikeWale ads of the same models."
                      if cal != 1.0 else ".")),
        "range_note": "Range = the middle 60% of the model's errors on unseen test listings.",
    }
    if kind == "2w" and p.state and p.state != "Other":
        exs = info.get("new_price") or float(curve_prices[0]) / 0.90
        cc = float(row.power.iat[0])
        rate = road_tax_rate(p.state, exs, cc)
        if rate is not None:
            cs, cr = cheapest_state(exs, cc)
            out["road_tax"] = {"state": p.state, "rate": rate, "cheapest_state": cs, "cheapest_rate": cr,
                               "new_price_est": round(exs, -2), "source": SOURCES.get(p.state)}
    return out


# ------------------------------------------------------------------ 2026 ML engine (two-wheelers)
def _frame26(p: PredictIn, info, ages):
    n = len(ages)
    fuel = {"electric": "Electric", "cng": "CNG"}.get(info.get("fuel"), "Petrol")
    body = "scooter" if info.get("type") in ("scooter", "moped") else "motorcycle"
    newp = p.new_price or info.get("new_price") or np.nan
    return pd.DataFrame({"brand": [p.brand] * n, "model_key": [f"{p.brand} | {p.model}"] * n, "fuel": [fuel] * n,
                         "body": [body] * n, "state": [p.state or "Other"] * n, "age": ages,
                         "log_km": [math.log1p(p.km)] * n, "owners": [p.owners] * n, "new_price": [float(newp)] * n})


def predict_ml26(p: PredictIn, info):
    t0 = time.perf_counter()
    m = MODELS["bike2026"]
    age = max(0, NOW - p.year)
    ages = list(range(0, 16))
    curve_prices = np.expm1(m["pipe"].predict(_frame26(p, info, ages)))
    row = _frame26(p, info, [age])
    Z = m["pipe"].named_steps["pre"].transform(row)
    contrib = m["pipe"].named_steps["m"].get_booster().predict(xgb.DMatrix(Z, missing=np.nan), pred_contribs=True)[0]
    price = float(math.expm1(contrib.sum()))
    ms = (time.perf_counter() - t0) * 1000
    by = pd.Series(contrib[:-1], index=m["groups"]).groupby(level=0).sum()
    r = row.iloc[0]
    labels = dict(LABELS)
    if info.get("fuel") == "electric":
        labels["fuel"] = ("Fuel", FUEL_NOTE_2W)
    else:
        labels["fuel"] = ("Fuel", "Petrol two-wheelers hold value better than electric ones.")
    factors = [{"feature": labels[g][0], "value": _value_text(g, r, p), "effect": round((math.exp(c) - 1) * 100, 1),
                "note": labels[g][1]} for g, c in by.sort_values(key=np.abs, ascending=False).items()]
    lo, hi = m["band"]
    top = factors[0]
    title = f"{p.year} {p.brand} {p.model}"
    sim = CAL["similar"].get(f"{p.brand} | {p.model}|{age}") if CAL else None
    n_ads = METRICS["bike2026"]["train"] + METRICS["bike2026"]["test"]
    out = {
        "title": title, "age": age, "price": round(price, -2), "low": round(price * lo, -2), "high": round(price * hi, -2),
        "base_price": round(math.expm1(float(contrib[-1])), -2), "factors": factors,
        "similar": {"median": round(sim[0], -2), "count": sim[1], "year": 2026} if sim else None,
        "curve": [{"age": a, "price": round(float(v), -2)} for a, v in zip(ages, curve_prices)],
        "new_price": info.get("new_price"),
        "summary": (f"A {title} with {p.km:,.0f} km and a {OWNER_WORD[p.owners]} owner"
                    f"{' in ' + p.state if p.state and p.state != 'Other' else ''} should list for about "
                    f"₹{inr(round(price, -2))}. The biggest single factor is {top['feature'].lower()} "
                    f"({'+' if top['effect'] > 0 else ''}{top['effect']}%)."),
        "engine": {"kind": "ml", "name": "XGBoost", "ms": round(ms, 1),
                   "detail": (f"Our XGBoost model, trained on {n_ads:,} real used-bike ads listed on BikeWale in November 2026 "
                              f"({info['train_rows']} of them this model), worked out this price just now from the details you entered. "
                              f"On ads it had never seen, it lands within 20% of the asking price {METRICS['bike2026']['xgb_within_20pct']:.0%} of the time.")},
        "method": "Each bar is that detail's push on the price (Tree SHAP), compared with the typical ad in the data.",
        "range_note": "The range covers the middle 60% of the model's misses on ads it had not seen.",
    }
    if info.get("fuel") == "electric":
        k = p.kwh or info.get("kwh_est") or 3.0
        out["battery_health"] = round(dep.battery_health(age, p.km, k), 1)
        out["value_kept"] = round(price / dep.on_road_2w(info["new_price"], None, p.state, True)[0] * 100, 1) if info.get("new_price") else None
    return out


# ------------------------------------------------------------------ depreciation engine
def predict_dep(kind, p: PredictIn, info):
    t0 = time.perf_counter()
    age = max(0, NOW - p.year)
    exs = p.new_price or info["new_price"]
    if kind == "2w":
        out = dep.predict_2w(p.brand, info, exs, age, p.km, p.owners, p.state, p.kwh)
        src = "BikeWale"
        f = dep_factor(info, age)
        if f != 1.0:
            for k in ("price", "low", "high"):
                out[k] = round(out[k] * f, -2)
            out["value_kept"] = round(out["price"] / out["new_price"] * 100, 1)
            for c in out["curve"]:
                fc = dep_factor(info, c["age"])
                c["price"] = round(c["price"] * fc, -2)
                c["kept"] = round(c["kept"] * fc, 1)
            out["factors"].append({"feature": "2026 market check", "value": f"x{f:.2f}", "effect": round((f - 1) * 100, 1),
                                   "note": (f"Measured correction: real 2026 BikeWale ads of this kind and age list at {f:.2f}x "
                                            f"what the formula alone gives (backtest on {sum(v['ads'] for v in DEP_CAL['test'].values()):,} ads).")})
        sim = CAL["similar"].get(f"{p.brand} | {p.model}|{age}") if CAL else None
        out["similar"] = {"median": round(sim[0], -2), "count": sim[1], "year": 2026} if sim else None
    else:
        out = dep.predict_car(META["car_curve"], exs, age, p.km, p.owners)
        src = "CarWale"
    ms = (time.perf_counter() - t0) * 1000
    title = f"{p.year} {p.brand} {p.model}"
    why = ("it is electric — the 2020-21 training data has almost no EVs" if info.get("fuel") == "electric"
           else "it is a scooter — the training data has motorcycles only" if info.get("type") in ("scooter", "moped")
           else "it isn't in the 2020-21 training data (launched or renamed later)")
    out.update({
        "title": title, "age": age,
        "summary": (f"A {title} with {p.km:,.0f} km should list for about ₹{inr(out['price'])}, "
                    f"{out['value_kept']}% of its ₹{inr(out['new_price'])} price new."
                    + (f" Estimated battery health is about {out['battery_health']:.0f}%." if "battery_health" in out else "")),
        "engine": {"kind": "dep", "name": "Depreciation model", "ms": round(ms, 2),
                   "detail": (f"Too few used ads of this model exist to train on, so we start from its official {src} price "
                              f"and apply how fast similar {'cars' if kind == 'car' else 'two-wheelers'} lose value, measured on real listings.")},
        "method": "Each factor multiplies the price in turn (shown as % change).",
    })
    if info.get("fuel") == "electric":
        out["brand_note"] = ("Used EVs lose about 12% a year against about 6-8% for petrol two-wheelers "
                             "(30,527 BikeWale ads, Nov 2026).")
    return out


@app.get("/api/meta")
def meta():
    def pack(kind, popular):
        return {b: sorted(INDEX[kind][b].values(), key=lambda m: m["model"].lower()) for b in _order(INDEX[kind], popular)}
    return {"2w": {"brands": pack("2w", POPULAR_2W), "states": META["bike"]["states"]},
            "car": {"brands": pack("car", POPULAR_CAR), "fuels": META["car"]["fuels"]},
            "sources": CATALOG["sources"], "now": NOW,
            "counts": {"2w": sum(len(v) for v in INDEX["2w"].values()), "car": sum(len(v) for v in INDEX["car"].values())}}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(p: PredictIn):
    info = _info(p.type, p.brand, p.model)
    if info["engine"] == "ml26":
        return predict_ml26(p, info)
    return predict_ml(p.type, p, info) if info["engine"] == "ml" else predict_dep(p.type, p, info)


app.mount("/", StaticFiles(directory=FRONT, html=True), name="site")
