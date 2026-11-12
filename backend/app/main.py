"""ResaleIQ API.

  uvicorn backend.app.main:app --reload     (from the repo root)

GET  /api/meta                    verified model list per brand (+ engine, years, new price), states, cities
GET  /api/variants?type=&brand=&model=   variants of one model (name, new price, fuel, gearbox, years)
GET  /api/metrics                 test-set metrics of every model
POST /api/predict                 {"type": "2w" | "car", brand, model, variant?, year, km, owners, state, city?, seller?}
Everything else serves the frontend.

Pricing engines, chosen per model (best first):
  "market"  XGBoost trained on live 2026 listings (BikeWale two-wheelers, CarWale cars), run on every request.
            Knows the exact variant and its new price, city, seller type. Factors = exact Tree SHAP values.
  "ml"      XGBoost trained on the Kaggle 2020-21 data, calibrated to 2026 (older models with no 2026 ads).
  "dep"     Official new price x measured yearly value loss (models with too few ads anywhere).
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
from .road_tax import CESS, SOURCES, base_road_tax_rate, cheapest_state, gst_two_wheeler, road_tax_rate

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT.parent / "frontend"
NOW = 2026
MIN_TRAIN = 3

META = json.loads((ROOT / "models" / "meta.json").read_text())
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())
CATALOG = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())
KAGGLE = {k: joblib.load(ROOT / "models" / f"{k}_xgb.joblib") for k in ("bike", "car")}
KAGGLE_BOOST = {k: v["pipe"].named_steps["m"].get_booster() for k, v in KAGGLE.items()}
MARKET = {k: joblib.load(ROOT / "models" / f"{k}_market.joblib") for k in ("bike", "car")
          if (ROOT / "models" / f"{k}_market.joblib").exists()}
GROWTH = META["growth"]
# full variant lists (official + every variant seen in an ad), built by build_variants.py
VARIANTS = {k: json.loads((ROOT / "models" / f"variants_{k}.json").read_text()) for k in ("bike", "car")
            if (ROOT / "models" / f"variants_{k}.json").exists()}
# verified years per model (build_variants.py + years.py): official ranges where stated, else typo-trimmed ad years
MODEL_YEARS = {k: json.loads((ROOT / "models" / f"model_years_{k}.json").read_text()) for k in ("bike", "car")
               if (ROOT / "models" / f"model_years_{k}.json").exists()}


def _json(name):
    p = ROOT / "models" / name
    return json.loads(p.read_text()) if p.exists() else None


PREMIUM = _json("state_premium.json") or {}   # same variant + year: state median / national median (state_premium.py)
CAL = _json("calibration_2026.json")          # Kaggle bike model vs 2026 ads (calibrate_2026.py)
DEP_CAL = _json("dep_calibration_2026.json")  # depreciation formula vs 2026 ads
CITY_STATE = pd.read_csv(ROOT / "data" / "city_state.csv").drop_duplicates("city").set_index("city").state.to_dict()
OWNER_WORD = dep.OWNER_WORD
POPULAR = {"2w": ["Hero", "Honda", "Bajaj", "TVS", "Royal Enfield", "Yamaha", "Suzuki", "KTM", "Ather", "OLA", "VIDA",
                  "Jawa", "Yezdi", "Kawasaki", "Harley-Davidson", "Triumph", "BMW", "Ducati"],
           "car": ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Toyota", "Kia", "Honda", "MG", "Skoda", "Volkswagen", "Renault"]}


def fin(*xs):
    """First value that is a real, positive number (skips None and NaN)."""
    for x in xs:
        if isinstance(x, (int, float)) and math.isfinite(x) and x > 0:
            return float(x)
    return None


def inr(x):
    """Indian digit grouping: 1234567 -> 12,34,567."""
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "?"
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


def band_factor(table, bands, age):
    for lo, hi in bands:
        if lo < age <= hi:
            return table.get(f"{lo + 1}-{hi}", 1.0)
    return 1.0


# ------------------------------------------------------------------ the model list
def _build_index():
    idx = {"2w": {}, "car": {}}
    for kind, mk in (("2w", "bike"), ("car", "car")):
        mkt = MARKET.get(mk, {}).get("models", {})
        entries = {}
        for brand, models in CATALOG[mk].items():
            for m in models:
                entries[f"{brand} | {m['model']}"] = {**m, "brand": brand}
        for key, mi in mkt.items():                      # models on the site's own ads, no longer on sale
            if key not in entries and mi["n"] >= 10:
                brand, model = key.split(" | ", 1)
                entries[key] = {"model": model, "brand": brand, "on_sale": False, "new_price": None}
        for key, e in entries.items():
            brand = e["brand"]
            mi = mkt.get(key)
            kag = META[mk]["trained"].get(key)
            fuel = (e.get("fuel") or (mi or {}).get("fuel") or "petrol").lower()
            if e.get("electric"):
                fuel = "electric"
            if mi and mi["n"] >= MIN_TRAIN:
                engine = "market"
            elif kag and kag["n"] >= MIN_TRAIN and fuel != "electric":
                engine = "ml"
            elif e.get("new_price"):
                engine = "dep"
            else:
                continue
            year_list = MODEL_YEARS.get(mk, {}).get(key)
            if not year_list:          # no ad anywhere: a new launch, sold new this year and last
                year_list = [NOW - 1, NOW] if e.get("on_sale") else [NOW - 8 + i for i in range(9)]
            years = [min(year_list), max(year_list)]
            typ = e.get("type") or ("scooter" if (mi or {}).get("body") == "scooter" else "motorcycle")
            entry = {"model": e["model"], "type": typ if kind == "2w" else None, "fuel": fuel,
                     "on_sale": bool(e.get("on_sale")), "new_price": e.get("new_price"),
                     "new_price_catalog": e.get("new_price"), "engine": engine,
                     "n": (mi or {}).get("n", 0) if engine == "market" else (kag or {}).get("n", 0),
                     "years": [int(years[0]), int(min(NOW, years[1]))], "year_list": [int(y) for y in year_list if y <= NOW],
                     "cc": e.get("cc") or (kag or {}).get("cc"), "kwh_est": e.get("kwh_est"),
                     "variants": len(VARIANTS.get(mk, {}).get(key, [])) if engine == "market" else 0}
            if engine == "market" and not entry["new_price"]:
                entry["new_price"] = mi.get("new_price")
            if kind == "car" and kag:
                entry.update({k: kag[k] for k in ("power", "seats", "mileage", "transmission") if k in kag})
                entry["kaggle_fuel"] = kag.get("fuel")
            if kind == "car" and mi:
                entry["transmission_mode"] = mi.get("transmission")
            # NaN (unknown) -> None, so the JSON is valid
            entry = {k: (None if isinstance(v, float) and v != v else v) for k, v in entry.items()}
            idx[kind].setdefault(brand, {})[e["model"]] = entry
    return idx


INDEX = _build_index()
STATES = sorted({s for s in CITY_STATE.values()})
CITIES = {}
for mk in MARKET:                                     # keep the order: most ads first
    for st, cs in MARKET[mk]["cities"].items():
        lst = CITIES.setdefault(st, [])
        lst.extend(c for c in cs if c not in lst)
CITIES.pop("Other", None)

app = FastAPI(title="ResaleIQ", version="3.0")
app.add_middleware(GZipMiddleware, minimum_size=800)


class PredictIn(BaseModel):
    type: Literal["2w", "car"]
    brand: str
    model: str
    variant: Optional[str] = None
    year: int = Field(ge=1980, le=NOW)     # Maruti 800 ads go back to the 1980s
    km: float = Field(ge=0, le=1_000_000)
    owners: int = Field(1, ge=1, le=4)
    state: Optional[str] = None
    city: Optional[str] = None
    seller: Optional[Literal["individual", "dealer"]] = "individual"
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    cc: Optional[float] = None
    power: Optional[float] = None
    seats: Optional[float] = None
    new_price: Optional[float] = Field(None, ge=10_000, le=200_000_000)
    kwh: Optional[float] = Field(None, gt=0, le=50)


def _info(kind, brand, model):
    info = INDEX[kind].get(brand, {}).get(model)
    if not info:
        raise HTTPException(422, f"{brand} {model} is not in the verified list")
    return info


LABELS = {
    "brand": ("Brand", "How the market rates this brand overall."),
    "model_key": ("Model", "Demand for this exact model."),
    "variant": ("Variant", "Higher trims ask more; this is the exact version you picked."),
    "fuel": ("Fuel", "Electric, diesel and CNG versions hold value differently from petrol."),
    "body": ("Body type", "Different body types hold value differently."),
    "transmission": ("Gearbox", "Automatics usually ask more."),
    "state": ("State", "Prices differ by state: local demand and the road tax paid when new."),
    "city": ("City", "Big-city asking prices differ from small towns."),
    "seller": ("Seller", "Dealers ask more than private sellers (they add a margin and often a warranty)."),
    "age": ("Age", "Older vehicles are worth less; usually the biggest factor."),
    "log_km": ("Kilometres", "More km means more wear."),
    "owners": ("Owners", "Each extra owner lowers buyer trust."),
    "log_new": ("Price new", "What this exact variant costs new; used prices track it closely."),
    "log_new_model": ("Model's price today", "The model's current starting price."),
    "power": ("Engine", "Bigger engines usually cost more new and used."),
    "engine_cc": ("Engine", "Bigger engines usually cost more new and used."),
    "power_bhp": ("Power", "More power signals a higher variant."),
    "mileage_kmpl": ("Mileage", "Fuel efficiency of the variant."),
    "seats": ("Seats", "Seating capacity."),
    "seller_type": ("Seller", "Dealers usually ask more than individuals."),
}


# ------------------------------------------------------------------ market engine (2026 listings)
def _variant(mk, key, vid):
    if not vid:
        return None
    return next((v for v in VARIANTS.get(mk, {}).get(key, []) if v["id"] == str(vid)), None)


def predict_market(kind, p: PredictIn, info):
    t0 = time.perf_counter()
    mk = "bike" if kind == "2w" else "car"
    M = MARKET[mk]
    key = f"{p.brand} | {p.model}"
    mi = M["models"][key]
    var = _variant(mk, key, p.variant)
    age = max(0, NOW - p.year)
    fy = mi.get("fuel_by_year") or {}
    fuel_year = fy.get(p.year) or fy.get(str(p.year))
    fuel = (var or {}).get("fuel") or ((p.fuel or "").title() or None) or fuel_year or mi["fuel"]
    if kind == "2w":
        fuel = "Electric" if info["fuel"] == "electric" else ("CNG" if info["fuel"] == "cng" else "Petrol")
    trans = (var or {}).get("transmission") or (p.transmission if kind == "car" else None) or mi["transmission"]
    by_year = mi.get("new_price_by_year") or {}
    year_price = None
    if by_year:
        # newest year with data at or before the asked year; the price never drops for a newer year
        # only a 3-year look-back: an old generation that shared the name must not leak in
        get = lambda y: by_year.get(y, by_year.get(str(y)))
        ys = sorted(int(y) for y in by_year)
        near = [y for y in ys if p.year - 3 <= y <= p.year]
        year_price = max(get(y) for y in near) if near else get(min(ys, key=lambda y: abs(y - p.year)))
    newp = fin(p.new_price, (var or {}).get("new_price"), year_price, mi.get("new_price"), info.get("new_price_catalog"))
    cat_price = fin(info.get("new_price_catalog"))
    city = p.city if p.city in M["cats"]["city"] else None
    var_id = (var or {}).get("id") if (var or {}).get("learned") else None
    row = {"brand": p.brand, "model_key": key, "variant": var_id, "fuel": fuel, "body": mi["body"],
           "transmission": trans, "state": p.state or "Other", "city": city, "seller": p.seller or "individual",
           "log_km": math.log1p(p.km), "owners": p.owners,
           "log_new": math.log(newp) if newp else np.nan,
           "log_new_model": math.log(cat_price) if cat_price else np.nan}
    ages = list(range(0, 16))
    df = pd.DataFrame([{**row, "age": a} for a in ages])[M["features"]]
    for c in M["cats"]:
        df[c] = pd.Categorical(df[c], categories=M["cats"][c])
    d = xgb.DMatrix(df, enable_categorical=True)
    boosters = M["boosters"]
    curve = np.exp(np.mean([b.predict(d) for b in boosters], axis=0))
    one = df.iloc[[min(age, 15)]].copy()
    one["age"] = age                    # score the real age (also beyond the 0-15 curve)
    d1 = xgb.DMatrix(one, enable_categorical=True)
    # SHAP values are additive, so the ensemble's contributions are the mean of each model's
    contrib = np.mean([b.predict(d1, pred_contribs=True)[0] for b in boosters], axis=0)
    price = float(math.exp(contrib.sum()))
    ms = (time.perf_counter() - t0) * 1000

    vals = {"brand": p.brand, "model_key": p.model, "variant": (var or {}).get("name") or "not specified",
            "fuel": fuel, "body": {"scooter": "scooter", "motorcycle": "motorcycle"}.get(mi["body"], "car"),
            "transmission": trans, "state": p.state or "not given", "city": city or "other / not given",
            "seller": p.seller or "individual", "age": f"{age} yr", "log_km": f"{p.km:,.0f} km",
            "owners": OWNER_WORD[p.owners], "log_new": f"₹{inr(newp)}" if newp else "unknown",
            "log_new_model": f"₹{inr(cat_price)}" if cat_price else "not sold now"}
    factors = []
    for f, c in sorted(zip(M["features"], contrib[:-1]), key=lambda t: -abs(t[1])):
        if kind == "2w" and f == "transmission":
            continue
        if abs(c) < 0.002:
            continue
        factors.append({"feature": LABELS[f][0], "value": vals[f], "effect": round((math.exp(c) - 1) * 100, 1), "note": LABELS[f][1]})
    lo, hi = M["band"]
    sim = (M["similar_variant"].get(f"{(var or {}).get('id')}|{age}") if var else None) or M["similar"].get(f"{key}|{age}")
    seg = M["segment_metrics"]["overall"]
    site = "BikeWale" if kind == "2w" else "CarWale"
    n_ads = METRICS[f"{mk}_market"]["train"] + METRICS[f"{mk}_market"]["test"]
    title = f"{p.year} {p.brand} {p.model}" + (f" {var['name']}" if var else "")
    out = {
        "title": title, "age": age, "price": round(price, -2), "low": round(price * lo, -2), "high": round(price * hi, -2),
        "base_price": round(math.exp(float(contrib[-1])), -2), "factors": factors[:9],
        "similar": {"median": round(sim[0], -2), "count": sim[1], "year": 2026, "scope": "variant" if (var and M["similar_variant"].get(f"{var['id']}|{age}")) else "model"} if sim else None,
        "curve": [{"age": a, "price": round(float(v), -2)} for a, v in zip(ages, curve)],
        "new_price": round(newp, -2) if newp else None,
        "summary": (f"A {title} with {p.km:,.0f} km and a {OWNER_WORD[p.owners]} owner"
                    f"{', sold by a dealer' if p.seller == 'dealer' else ''}"
                    f"{' in ' + (city + ', ' if city else '') + p.state if p.state and p.state != 'Other' else ''} should list for about "
                    f"₹{inr(round(price, -2))}."),
        "engine": {"kind": "ml", "name": "XGBoost", "ms": round(ms, 1),
                   "detail": (f"Our XGBoost model learned from {n_ads:,} real used-{'two-wheeler' if kind == '2w' else 'car'} ads listed on {site} "
                              f"in November 2026 ({info['n']:,} of them this model) and worked out this price just now from your details. "
                              f"On ads it had never seen, half its prices were within {seg['median_abs_pct_err']:.0%} of the asking price, "
                              f"and {seg['within_20pct']:.0%} within 20%.")},
        "method": "Each bar is how much that detail moves the price, compared with a typical ad (Tree SHAP, from the same calculation).",
        "range_note": "The range covers the middle 60% of the model's misses on ads it had not seen.",
    }
    if newp:
        on_road = newp if kind == "car" else dep.on_road_2w(newp, info.get("cc"), p.state, fuel == "Electric")[0]
        out["value_kept"] = round(price / on_road * 100, 1)
    if kind == "2w" and fuel == "Electric":
        k = p.kwh or info.get("kwh_est") or 3.0
        out["battery_health"] = round(dep.battery_health(age, p.km, k), 1)
    y0, y1 = info["years"]
    if var and var.get("n") and var.get("years"):          # the chosen variant's own years
        vy0, vy1 = var["years"]
        if p.year < vy0 or (not var.get("official") and p.year > vy1):
            out["warning"] = (f"The {var['name']} variant appears in ads from {vy0}"
                              f"{'' if var.get('official') else f' to {vy1}'}; check the year, this price is rough.")
    if "warning" not in out and p.year not in info.get("year_list", range(y0, y1 + 1)):
        out["warning"] = (f"Few or no {p.model} ads are from {p.year} (listings run {y0}-{y1}), so treat this price as rough."
                          if p.year < y0 else f"No {p.model} ads from {p.year} yet; treat this price as rough.")
    out["tax"] = tax_note(kind, p, info, newp, fuel)
    return out


def tax_note(kind, p, info, newp, fuel):
    """Taxes on this vehicle, kept apart from the price model: GST (inside the ex-showroom price),
    state road tax on top, and how much more or less this state's used ads ask for the same variant and year."""
    mk = "bike" if kind == "2w" else "car"
    t = {"state": p.state if p.state and p.state != "Other" else None}
    prem = PREMIUM.get(mk, {}).get(p.state) if t["state"] else None
    if prem:
        t["premium"] = round(prem["ratio"] - 1, 3)
        t["premium_ads"] = prem["ads"]
    if kind == "2w":
        cc = info.get("cc")
        ev = fuel == "Electric"
        g_now, g_then, g_note = gst_two_wheeler(cc, ev, p.year)
        if g_now is not None:
            t.update({"gst_now": g_now, "gst_then": g_then, "gst_note": g_note, "cc": cc})
        if t["state"] and newp:
            base = base_road_tax_rate(p.state, newp, cc or 125, ev)
            if base is not None:
                eff = road_tax_rate(p.state, newp, cc or 125, ev)
                cs, cr = cheapest_state(newp, cc or 125)
                t.update({"road_base": base, "road_cess": CESS.get(p.state, 0.0), "road_rate": eff,
                          "road_amount": round(eff * newp, -2), "ex_showroom": round(newp, -2),
                          "cheapest_state": cs, "cheapest_rate": cr, "source": SOURCES.get(p.state)})
    return t if len(t) > 1 else None


# ------------------------------------------------------------------ Kaggle engine (older models, no 2026 ads)
def _frame_kaggle(mk, p: PredictIn, info, ages):
    n, key = len(ages), f"{p.brand} | {p.model}"
    if mk == "bike":
        return pd.DataFrame({"brand": [p.brand] * n, "model_key": [key] * n, "state": [p.state or "Other"] * n,
                             "age": ages, "log_km": [math.log1p(p.km)] * n, "owners": [p.owners] * n,
                             "power": [p.cc or info.get("cc") or 125] * n})
    tr = META["car"]["trained"][key]
    return pd.DataFrame({
        "brand": [p.brand] * n, "model_key": [key] * n, "fuel": [p.fuel or tr["fuel"]] * n,
        "seller_type": ["Dealer" if p.seller == "dealer" else "Individual"] * n,
        "transmission": [p.transmission or tr["transmission"]] * n, "age": ages,
        "log_km": [math.log1p(p.km)] * n, "owners": [p.owners] * n, "engine_cc": [p.cc or tr["cc"]] * n,
        "power_bhp": [p.power or tr["power"]] * n, "mileage_kmpl": [tr["mileage"]] * n, "seats": [p.seats or tr["seats"]] * n})


def predict_kaggle(kind, p: PredictIn, info):
    t0 = time.perf_counter()
    mk = "bike" if kind == "2w" else "car"
    ref = META[mk]["ref_year"]
    age = max(0, NOW - p.year)
    scale = GROWTH ** (NOW - ref)
    cal = (lambda a: band_factor(CAL["factor_by_age"], CAL["bands"], a)) if (mk == "bike" and CAL) else (lambda a: 1.0)
    ages = list(range(0, 16))
    pipe = KAGGLE[mk]["pipe"]
    raw = np.expm1(pipe.predict(_frame_kaggle(mk, p, info, ages))) * scale * np.array([cal(a) for a in ages])
    curve = np.minimum.accumulate(raw)          # this model has no monotone rule, so enforce it here
    row = _frame_kaggle(mk, p, info, [age])
    contrib = KAGGLE_BOOST[mk].predict(xgb.DMatrix(pipe.named_steps["pre"].transform(row)), pred_contribs=True)[0]
    price = float(math.expm1(contrib.sum()) * scale * cal(age))
    cap = float(curve[min(age, 15)])
    if price > cap * 1.001:                     # scale the factors' baseline so they still multiply to the price
        contrib[-1] += math.log(cap / price) if price > 0 else 0
        price = cap
    ms = (time.perf_counter() - t0) * 1000
    by = pd.Series(contrib[:-1], index=KAGGLE[mk]["groups"]).groupby(level=0).sum()
    r = row.iloc[0]
    unit = {"age": " yr", "power": " cc", "engine_cc": " cc", "power_bhp": " bhp", "mileage_kmpl": " kmpl", "seats": ""}
    def vt(g):
        if g == "model_key": return p.model
        if g == "log_km": return f"{p.km:,.0f} km"
        if g == "owners": return OWNER_WORD[p.owners]
        return f"{float(r[g]):g}{unit[g]}" if g in unit else str(r[g])
    factors = [{"feature": LABELS[g][0], "value": vt(g), "effect": round((math.exp(c) - 1) * 100, 1), "note": LABELS[g][1]}
               for g, c in by.sort_values(key=np.abs, ascending=False).items()]
    lo, hi = KAGGLE[mk]["band"]
    title = f"{p.year} {p.brand} {p.model}"
    return {
        "title": title, "age": age, "price": round(price, -2), "low": round(price * lo, -2), "high": round(price * hi, -2),
        "base_price": round(math.expm1(float(contrib[-1])) * scale * cal(age), -2), "factors": factors,
        "similar": None, "curve": [{"age": a, "price": round(float(v), -2)} for a, v in zip(ages, curve)],
        "new_price": None,
        "summary": f"A {title} with {p.km:,.0f} km and a {OWNER_WORD[p.owners]} owner should list for about ₹{inr(round(price, -2))}.",
        "engine": {"kind": "ml", "name": "XGBoost (older data)", "ms": round(ms, 1),
                   "detail": (f"This model is no longer listed in 2026, so we used our XGBoost model trained on "
                              f"{METRICS[mk]['train']:,} older (2020-21) listings, {info['n']} of them this model"
                              + (", adjusted to 2026 market prices." if mk == "bike" and CAL else ", with prices brought to 2026 rupees.")
                              + " Treat it as a rough guide.")},
        "method": "Each bar is how much that detail moves the price, compared with a typical listing (Tree SHAP).",
        "range_note": "The range covers the middle 60% of the model's misses on listings it had not seen.",
    }


# ------------------------------------------------------------------ depreciation engine (too few ads anywhere)
def predict_dep(kind, p: PredictIn, info):
    t0 = time.perf_counter()
    age = max(0, NOW - p.year)
    exs = p.new_price or info["new_price"]
    if kind == "2w":
        out = dep.predict_2w(p.brand, info, exs, age, p.km, p.owners, p.state, p.kwh)
        if DEP_CAL:
            grp = "ev" if info["fuel"] == "electric" else ("scooter" if info.get("type") in ("scooter", "moped") else "motorcycle")
            f = lambda a: band_factor(DEP_CAL["factor"].get(grp, {}), DEP_CAL["bands"], a)
            for k in ("price", "low", "high"):
                out[k] = round(out[k] * f(age), -2)
            out["value_kept"] = round(out["price"] / out["new_price"] * 100, 1)
            for c in out["curve"]:
                c["price"] = round(c["price"] * f(c["age"]), -2)
                c["kept"] = round(c["kept"] * f(c["age"]), 1)
            out["factors"].append({"feature": "2026 market check", "value": f"x{f(age):.2f}", "effect": round((f(age) - 1) * 100, 1),
                                   "note": "Correction measured on real 2026 ads of the same kind and age."})
    else:
        out = dep.predict_car(META["car_curve"], exs, age, p.km, p.owners)
    run = float("inf")
    for c in out["curve"]:
        run = min(run, c["price"]); c["price"] = run
        if "kept" in c:
            c["kept"] = round(min(c["kept"], c["price"] / max(out["new_price"], 1) * 100 if out.get("new_price") else c["kept"]), 1)
    if age < len(out["curve"]) and out["price"] > out["curve"][age]["price"]:
        out["price"] = out["curve"][age]["price"]
        out["low"], out["high"] = round(out["price"] * 0.9, -2), round(out["price"] * 1.1, -2)
    ms = (time.perf_counter() - t0) * 1000
    title = f"{p.year} {p.brand} {p.model}"
    out.update({
        "title": title, "age": age, "similar": None,
        "summary": (f"A {title} with {p.km:,.0f} km should list for about ₹{inr(out['price'])}, "
                    f"{out['value_kept']}% of its ₹{inr(out['new_price'])} price new."),
        "engine": {"kind": "dep", "name": "Depreciation estimate", "ms": round(ms, 2),
                   "detail": (f"Fewer than {MIN_TRAIN} used ads of this model exist anywhere, so no model can learn its price yet. "
                              f"We start from its official {'BikeWale' if kind == '2w' else 'CarWale'} price and apply how fast "
                              f"similar {'two-wheelers' if kind == '2w' else 'cars'} lose value, measured on real listings. Treat it as a rough guide.")},
        "method": "Each factor multiplies the price in turn (shown as % change).",
    })
    return out


@app.get("/api/meta")
def meta():
    def pack(kind):
        return {b: sorted(INDEX[kind][b].values(), key=lambda m: m["model"].lower()) for b in
                sorted(INDEX[kind], key=lambda b: (POPULAR[kind].index(b) if b in POPULAR[kind] else 99, b.lower()))}
    return {"2w": {"brands": pack("2w")}, "car": {"brands": pack("car")}, "states": STATES, "cities": CITIES,
            "sources": CATALOG["sources"], "now": NOW,
            "counts": {"2w": sum(len(v) for v in INDEX["2w"].values()), "car": sum(len(v) for v in INDEX["car"].values())}}


@app.get("/api/variants")
def variants(type: Literal["2w", "car"], brand: str, model: str):
    info = _info(type, brand, model)
    mk = "bike" if type == "2w" else "car"
    if info["engine"] != "market":
        return []
    clean = lambda v: {k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in v.items()}
    vs = [clean({**v, "new_price": fin(v.get("new_price"))}) for v in VARIANTS.get(mk, {}).get(f"{brand} | {model}", [])]
    return sorted(vs, key=lambda v: (v["new_price"] or 0, v["name"], v["years"][0]))


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(p: PredictIn):
    info = _info(p.type, p.brand, p.model)
    if p.state and p.state not in STATES and p.state != "Other":
        raise HTTPException(422, f"Unknown state {p.state!r}")
    if info["engine"] == "market":
        return predict_market(p.type, p, info)
    if info["engine"] == "ml":
        return predict_kaggle(p.type, p, info)
    return predict_dep(p.type, p, info)


app.mount("/", StaticFiles(directory=FRONT, html=True), name="site")
