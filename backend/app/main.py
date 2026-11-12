"""ResaleIQ API.

  uvicorn backend.app.main:app --reload     (from the repo root)

GET  /api/meta      dropdown data (brands, models, states, EV catalog)
GET  /api/metrics   test-set metrics of every model
POST /api/predict   {"type": "bike" | "ev" | "car", ...}  -> price, range, factors (SHAP), curve
Everything else serves the frontend.
"""
import json
import math
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

from . import ev as ev_model
from .road_tax import SOURCES, cheapest_state, road_tax_rate

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT.parent / "frontend"
NOW = 2026

META = json.loads((ROOT / "models" / "meta.json").read_text())
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())
MODELS = {k: joblib.load(ROOT / "models" / f"{k}_xgb.joblib") for k in ("bike", "car")}
BOOSTERS = {k: v["pipe"].named_steps["m"].get_booster() for k, v in MODELS.items()}
GROWTH = META["growth"]
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


OWNER_WORD = {1: "first", 2: "second", 3: "third", 4: "fourth-or-later"}

app = FastAPI(title="ResaleIQ", version="1.0")
app.add_middleware(GZipMiddleware, minimum_size=800)


class PredictIn(BaseModel):
    type: Literal["bike", "ev", "car"]
    brand: str
    model: str                     # model key "Brand | Family" for bike/car, variant name for EV
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
    ex_showroom: Optional[float] = None
    kwh: Optional[float] = None


def _model_info(kind, brand, key):
    for r in META[kind]["brands"].get(brand, []):
        if r["key"] == key:
            return r
    raise HTTPException(422, f"Unknown model {key!r} for brand {brand!r}")


def _frame(kind, p: PredictIn, ages):
    info = _model_info(kind, p.brand, p.model)
    n = len(ages)
    log_km = math.log1p(p.km)
    if kind == "bike":
        return pd.DataFrame({
            "brand": [p.brand] * n, "model_key": [p.model] * n, "state": [p.state or "Other"] * n,
            "age": ages, "log_km": [log_km] * n, "owners": [p.owners] * n, "power": [p.cc or info["cc"]] * n})
    return pd.DataFrame({
        "brand": [p.brand] * n, "model_key": [p.model] * n, "fuel": [p.fuel or info["fuel"]] * n,
        "seller_type": [p.seller_type or "Individual"] * n, "transmission": [p.transmission or info["transmission"]] * n,
        "age": ages, "log_km": [log_km] * n, "owners": [p.owners] * n,
        "engine_cc": [p.cc or info["cc"]] * n, "power_bhp": [p.power or info["power"]] * n,
        "mileage_kmpl": [info["mileage"]] * n, "seats": [p.seats or info["seats"]] * n})


def _explain(kind, row: pd.DataFrame, p: PredictIn, scale):
    m = MODELS[kind]
    Z = m["pipe"].named_steps["pre"].transform(row)
    contrib = BOOSTERS[kind].predict(xgb.DMatrix(Z), pred_contribs=True)[0]
    bias, contrib = float(contrib[-1]), contrib[:-1]
    by = pd.Series(contrib, index=m["groups"]).groupby(level=0).sum()
    base_price = math.expm1(bias) * scale

    r = row.iloc[0].to_dict()
    r = {k: (f"{v:g}" if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items()}
    r.setdefault("power", ""); r.setdefault("engine_cc", ""); r.setdefault("power_bhp", ""); r.setdefault("mileage_kmpl", ""); r.setdefault("seats", ""); r.setdefault("fuel", ""); r.setdefault("transmission", ""); r.setdefault("seller_type", "")
    family = p.model.split(" | ", 1)[-1]
    labels = {
        "brand": ("Brand", p.brand, "How the market rates this brand overall."),
        "model_key": ("Model", family, "Demand and original price of this exact model family."),
        "state": ("State", p.state or "Other", "Local demand and road tax paid when it was bought."),
        "age": ("Age", f"{r['age']} yr", "Older vehicles are worth less; this is usually the biggest factor."),
        "log_km": ("Kilometres", f"{p.km:,.0f} km", "More km means more wear."),
        "owners": ("Owners", OWNER_WORD[p.owners], "Each extra owner lowers buyer trust."),
        "power": ("Engine", f"{r['power']} cc", "Bigger engines usually cost more new and used."),
        "engine_cc": ("Engine", f"{r['engine_cc']} cc", "Bigger engines usually cost more new and used."),
        "power_bhp": ("Power", f"{r['power_bhp']} bhp", "More power signals a higher variant."),
        "mileage_kmpl": ("Mileage", f"{r['mileage_kmpl']} kmpl", "Fuel efficiency of the variant."),
        "seats": ("Seats", f"{r['seats']}", "Seating capacity."),
        "fuel": ("Fuel", r["fuel"], "Diesel and CNG cars price differently from petrol."),
        "transmission": ("Gearbox", r["transmission"], "Automatics usually ask more."),
        "seller_type": ("Seller", r["seller_type"], "Dealers usually ask more than individuals."),
    }
    factors = []
    for g, c in by.sort_values(key=np.abs, ascending=False).items():
        name, val, note = labels[g]
        factors.append({"feature": name, "value": str(val), "effect": round((math.exp(c) - 1) * 100, 1), "note": note})
    return base_price, factors


def _similar(kind, key, age, scale):
    s = META[kind]["similar"].get(f"{key}|{int(round(age))}")
    if not s:
        return None
    return {"median": round(s[0] * scale, -2), "count": s[1]}


def predict_ml(kind, p: PredictIn):
    ref = META[kind]["ref_year"]
    age = max(0, NOW - p.year)
    scale = GROWTH ** (NOW - ref)                 # Kaggle-era rupees -> 2026 rupees
    ages = list(range(0, 16))
    curve_df = _frame(kind, p, ages)
    curve_prices = np.expm1(MODELS[kind]["pipe"].predict(curve_df)) * scale
    row = _frame(kind, p, [age])
    price = float(np.expm1(MODELS[kind]["pipe"].predict(row))[0] * scale)
    lo, hi = MODELS[kind]["band"]
    base_price, factors = _explain(kind, row, p, scale)
    family = p.model.split(" | ", 1)[-1]
    title = f"{p.year} {p.brand} {family}"
    top = max(factors, key=lambda f: abs(f["effect"]))
    summary = (f"A {title} with {p.km:,.0f} km and a {OWNER_WORD[p.owners]} owner"
               f"{' in ' + p.state if kind == 'bike' and p.state and p.state != 'Other' else ''} should list for about "
               f"₹{inr(round(price, -2))}. The biggest single factor is {top['feature'].lower()} "
               f"({'+' if top['effect'] > 0 else ''}{top['effect']}%).")
    out = {
        "title": title, "price": round(price, -2), "low": round(price * lo, -2), "high": round(price * hi, -2),
        "base_price": round(base_price, -2), "factors": factors, "summary": summary,
        "similar": _similar(kind, p.model, age, scale),
        "curve": [{"age": a, "price": round(float(v), -2)} for a, v in zip(ages, curve_prices)],
        "age": age,
        "method": (f"XGBoost trained on {METRICS[kind]['train']:,} {kind} listings (Kaggle, ~{ref}); "
                   f"test R² {METRICS[kind]['models']['XGBoost']['R2']:.3f}. Factor sizes are Tree SHAP values. "
                   f"Prices scaled to 2026 rupees at {round((GROWTH - 1) * 100)}% a year (×{scale:.2f})."),
        "range_note": "Range = the middle 60% of the model's errors on unseen test listings.",
    }
    if kind == "bike":
        new_est = float(curve_prices[0]) / 0.90
        cc = float(row.power.iat[0])
        if p.state and p.state != "Other":
            rate = road_tax_rate(p.state, new_est, cc)
            if rate is not None:
                cs, cr = cheapest_state(new_est, cc)
                out["road_tax"] = {"state": p.state, "rate": rate, "cheapest_state": cs, "cheapest_rate": cr,
                                   "new_price_est": round(new_est, -2), "source": SOURCES.get(p.state)}
    return out


def predict_ev(p: PredictIn):
    catalog = ev_model.EV_CATALOG.get(p.brand)
    if not catalog:
        raise HTTPException(422, f"Unknown EV brand {p.brand!r}")
    variant = next((v for v in catalog if v[0] == p.model), None)
    if not variant:
        raise HTTPException(422, f"Unknown EV variant {p.model!r}")
    exs = p.ex_showroom or variant[1]
    kwh = p.kwh or variant[2]
    age = max(0, NOW - p.year)
    out = ev_model.predict_ev(p.brand, exs, kwh, age, p.km, p.owners)
    title = f"{p.year} {p.brand} {p.model}"
    out.update({
        "title": title, "age": age,
        "summary": (f"A {title} with {p.km:,.0f} km should list for about ₹{inr(out['price'])}, "
                    f"{out['value_kept']}% of its ₹{inr(out['new_price'])} on-road price new. "
                    f"Estimated battery health is about {out['battery_health']:.0f}%."),
        "range_note": "Range = ±10%; the EV sample is small (424 ads), so treat this as indicative.",
    })
    if p.brand == "Ola":
        out["brand_note"] = ("Ola scooters lose about 12% a year against 7% for other EVs and petrol — "
                             "that one brand drives the whole 'EVs lose value fast' story.")
    return out


@app.get("/api/meta")
def meta():
    strip = lambda d: {k: v for k, v in d.items() if k != "similar"}
    return {"bike": strip(META["bike"]), "car": strip(META["car"]),
            "ev": {"brands": {b: [{"model": m, "price": pr, "kwh": k} for m, pr, k in vs]
                              for b, vs in ev_model.EV_CATALOG.items()}},
            "now": NOW}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(p: PredictIn):
    if p.type == "ev":
        return predict_ev(p)
    return predict_ml(p.type, p)


app.mount("/", StaticFiles(directory=FRONT, html=True), name="site")
