"""Depreciation model for vehicles that are NOT in the ML training data.

The Kaggle data (2020-21) has only motorcycles that existed then: no scooters, no EVs, nothing newer.
Those vehicles are priced from their official 2026 new price (BikeWale / CarWale) and a measured
yearly value loss:

  Two-wheelers - Module 6 same-variant regression, 5,694 BikeWale ads, Nov 2026
      petrol motorcycles  -6.4 %/yr      petrol scooters  -8.5 %/yr
      EVs (non-Ola)       -7.3 %/yr      Ola               -12.0 %/yr
      median value kept under 1 year: petrol 87 %, other EVs 88 %, Ola 78 %
  Cars - same-model fit on 2,720 CarDekho listings of 55 models still on sale (train.py: car_value_curve)

  used price = new on-road price x start x (1 - loss)^age x km factor x owner factor
"""
import math

from .road_tax import road_tax_rate

TWO_WHEELER = {
    "ola":        {"start": 0.83, "loss": 0.120, "label": "Ola scooters"},
    "ev":         {"start": 0.91, "loss": 0.073, "label": "EVs other than Ola"},
    "scooter":    {"start": 0.90, "loss": 0.085, "label": "petrol scooters"},
    "motorcycle": {"start": 0.90, "loss": 0.064, "label": "petrol motorcycles"},
}
OWNER_FACTOR = {1: 1.0, 2: 0.95, 3: 0.91, 4: 0.88}
OWNER_WORD = {1: "first", 2: "second", 3: "third", 4: "fourth-or-later"}


def insurance_2w(exs, cc, electric):
    if electric:
        return 3_400 + 0.017 * exs
    tp = 2_901 if cc and cc <= 75 else 3_851 if not cc or cc <= 150 else 7_365 if cc <= 350 else 15_117
    return tp + 0.028 * exs


def on_road_2w(exs, cc, state, electric):
    rate = road_tax_rate(state, exs, cc or 0, electric)
    rate_used = 0.10 if rate is None else rate
    parts = {"ex_showroom": exs, "road_tax": rate_used * exs, "insurance": insurance_2w(exs, cc, electric),
             "registration": 700 if electric else 1_400}
    return sum(parts.values()), parts, rate, rate_used


def battery_health(age, km, kwh):
    """Estimated battery health (%) - explainer only."""
    cycles = km / (kwh * 30)
    return max(60.0, min(100.0, 100 - 1.8 * age - 0.015 * cycles))


def _km_factor(km, age, per_year, per_10k):
    expected = max(1.0, age * per_year)
    return math.exp(-per_10k * max(-1.0, min(4.0, (km - expected) / 10_000))), expected


def predict_2w(brand, info, exs, age, km, owners, state, kwh=None):
    electric = info["fuel"] == "electric"
    key = ("ola" if brand.upper() == "OLA" else "ev") if electric else ("scooter" if info["type"] in ("scooter", "moped") else "motorcycle")
    r = TWO_WHEELER[key]
    new_on_road, parts, rate, rate_used = on_road_2w(exs, info.get("cc"), state, electric)
    age_f = (1 - r["loss"]) ** age
    km_f, expected = _km_factor(km, age, 8_000, 0.02)
    own_f = OWNER_FACTOR.get(owners, 0.88)
    price = new_on_road * r["start"] * age_f * km_f * own_f

    factors = [
        {"feature": "Price when new", "value": f"₹{new_on_road:,.0f} on-road", "effect": None,
         "note": (f"BikeWale 2026 ex-showroom ₹{exs:,.0f} + road tax "
                  f"{'0% (EVs are exempt)' if electric else f'{rate_used*100:.0f}%' + ('' if rate is not None else ' (typical; state not in our table)')}"
                  f" + insurance + registration.")},
        {"feature": "Drive-off drop", "value": f"keeps {r['start']*100:.0f}%", "effect": round((r["start"] - 1) * 100, 1),
         "note": "Median value kept by used ads under one year old (BikeWale, Nov 2026)."},
        {"feature": "Age", "value": f"{age:g} yr", "effect": round((age_f - 1) * 100, 1),
         "note": f"{r['label'].capitalize()} lose about {r['loss']*100:.1f}% a year (same-variant regression, Module 6)."},
        {"feature": "Kilometres", "value": f"{km:,.0f} km", "effect": round((km_f - 1) * 100, 1),
         "note": f"Compared with a typical {expected:,.0f} km at this age (assumption: 2% per extra 10,000 km)."},
        {"feature": "Owners", "value": OWNER_WORD.get(owners, str(owners)), "effect": round((own_f - 1) * 100, 1),
         "note": "Each extra owner lowers buyer trust (assumed 4-5% per owner)."},
    ]
    curve = []
    pet = TWO_WHEELER["scooter" if info["type"] != "motorcycle" else "motorcycle"]
    for y in range(0, 11):
        row = {"age": y, "price": round(new_on_road * r["start"] * (1 - r["loss"]) ** y * km_f * own_f, -2),
               "kept": round(r["start"] * (1 - r["loss"]) ** y * 100, 1)}
        if electric:
            row["petrol"] = round(pet["start"] * (1 - pet["loss"]) ** y * 100, 1)
        curve.append(row)
    out = {"price": round(price, -2), "low": round(price * 0.9, -2), "high": round(price * 1.1, -2),
           "new_price": round(new_on_road, -2), "value_kept": round(price / new_on_road * 100, 1),
           "factors": factors, "curve": curve, "rate_key": key,
           "range_note": "Range = ±10%. Depreciation rates come from 5,694 BikeWale ads; the EV sample is small (424 ads)."}
    if electric:
        k = kwh or info.get("kwh_est") or 3.0
        out["battery_health"] = round(battery_health(age, km, k), 1)
        out["kwh"] = k
    return out


def predict_car(curve_fit, exs, age, km, owners):
    start, loss = curve_fit["start"], curve_fit["yearly_loss"]
    age_f = (1 - loss) ** age
    km_f, expected = _km_factor(km, age, curve_fit["typical_km_per_year"], 0.015)
    own_f = OWNER_FACTOR.get(owners, 0.88)
    price = exs * start * age_f * km_f * own_f
    factors = [
        {"feature": "Price when new", "value": f"₹{exs:,.0f} ex-showroom", "effect": None,
         "note": "CarWale 2026 starting ex-showroom price (base variant)."},
        {"feature": "Drive-off drop", "value": f"keeps {start*100:.0f}%", "effect": round((start - 1) * 100, 1),
         "note": "Assumption: a car loses ~10% the day it is registered."},
        {"feature": "Age", "value": f"{age:g} yr", "effect": round((age_f - 1) * 100, 1),
         "note": f"The same car model loses about {loss*100:.1f}% a year (fitted on {curve_fit['rows']:,} CarDekho listings of {curve_fit['models']} models)."},
        {"feature": "Kilometres", "value": f"{km:,.0f} km", "effect": round((km_f - 1) * 100, 1),
         "note": f"Compared with a typical {expected:,.0f} km at this age."},
        {"feature": "Owners", "value": OWNER_WORD.get(owners, str(owners)), "effect": round((own_f - 1) * 100, 1),
         "note": "Each extra owner lowers buyer trust."},
    ]
    curve = [{"age": y, "price": round(exs * start * (1 - loss) ** y * km_f * own_f, -2)} for y in range(0, 16)]
    return {"price": round(price, -2), "low": round(price * 0.88, -2), "high": round(price * 1.12, -2),
            "new_price": round(exs, -2), "value_kept": round(price / exs * 100, 1), "factors": factors, "curve": curve,
            "range_note": "Range = ±12%. Based on the base-variant price; higher variants will be worth more."}
