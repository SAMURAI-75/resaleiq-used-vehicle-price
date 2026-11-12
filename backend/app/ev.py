"""EV two-wheeler resale model.

The Kaggle 2020-21 data has almost no EVs, so EV prices come from the
Module 6 same-variant depreciation results on 2026 BikeWale listings:

    log(price) ~ age + log(1+km) + EV x age + variant + state

  petrol: -6.9 % / year     non-Ola EV: -7.3 % / year     Ola: -12.0 % / year
  value kept under 1 year (median): petrol 87 %, other EVs 88 %, Ola 78 %

Used price = new on-road price x start_value x (1 - yearly_loss)^age x km_factor x owner_factor
New prices are approximate 2026 ex-showroom figures; the user can override them.
"""
import math

# brand -> list of (model/variant, approx 2026 ex-showroom Rs, battery kWh)
EV_CATALOG = {
    "Ather": [("450X 2.9 kWh", 147_000, 2.9), ("450X 3.7 kWh", 157_000, 3.7),
              ("450 Apex", 191_000, 3.7), ("Rizta S 2.9 kWh", 105_000, 2.9),
              ("Rizta Z 3.7 kWh", 145_000, 3.7), ("450S", 130_000, 2.9)],
    "Ola": [("S1 Pro", 130_000, 4.0), ("S1 Air", 107_000, 3.0),
            ("S1 X 2 kWh", 80_000, 2.0), ("S1 X 3 kWh", 90_000, 3.0),
            ("S1 X+ 4 kWh", 100_000, 4.0), ("Roadster X", 100_000, 3.5)],
    "TVS": [("iQube 2.2 kWh", 95_000, 2.2), ("iQube 3.4 kWh", 120_000, 3.4),
            ("iQube S", 130_000, 3.4), ("iQube ST", 160_000, 5.1), ("Orbiter", 100_000, 3.1)],
    "Bajaj": [("Chetak 2903", 100_000, 2.9), ("Chetak 3501", 128_000, 3.5),
              ("Chetak Premium", 135_000, 3.2)],
    "Hero Vida": [("V2 Lite", 75_000, 2.2), ("V2 Plus", 85_000, 3.4), ("V2 Pro", 120_000, 3.9),
                  ("VX2", 70_000, 2.2)],
    "Honda": [("Activa e:", 117_000, 3.0), ("QC1", 90_000, 1.5)],
    "Ampere": [("Nexus", 110_000, 3.0), ("Magnus EX", 85_000, 2.3)],
    "Simple": [("One", 145_000, 5.0)],
    "Revolt": [("RV400", 150_000, 3.24)],
    "Ultraviolette": [("F77 Mach 2", 299_000, 7.1)],
    "Okinawa": [("Praise Pro", 100_000, 2.0)],
}

# Module 6 results
YEARLY_LOSS = {"Ola": 0.120, "_other": 0.073, "_petrol": 0.069}
START_VALUE = {"Ola": 0.83, "_other": 0.91, "_petrol": 0.90}   # back-solved from <1-yr medians
KM_PER_YEAR_TYPICAL = 8_000


def on_road_ev(ex_showroom: float) -> float:
    """EV on-road = ex-showroom + 0 % road tax + insurance (Rs 3,400 + 1.7 %) + registration Rs 700."""
    return ex_showroom + 3_400 + 0.017 * ex_showroom + 700


def battery_health(age: float, km: float, kwh: float) -> float:
    """Estimated battery health (%) — chart/explainer only, not a model input."""
    cycles = km / (kwh * 30)
    return max(60.0, min(100.0, 100 - 1.8 * age - 0.015 * cycles))


def predict_ev(brand: str, ex_showroom: float, kwh: float, age: float, km: float, owners: int):
    key = "Ola" if brand == "Ola" else "_other"
    loss, start = YEARLY_LOSS[key], START_VALUE[key]
    new_on_road = on_road_ev(ex_showroom)

    age_factor = (1 - loss) ** age
    expected_km = max(1.0, age * KM_PER_YEAR_TYPICAL)
    # extra km beyond typical use costs ~2 % per extra 10,000 km (bounded)
    km_factor = math.exp(-0.02 * max(-1.0, min(4.0, (km - expected_km) / 10_000)))
    owner_factor = {1: 1.0, 2: 0.95, 3: 0.91}.get(owners, 0.88)

    price = new_on_road * start * age_factor * km_factor * owner_factor
    health = battery_health(age, km, kwh)

    # petrol twin: same new price, petrol curve, for comparison chart
    curve = []
    for y in range(0, 9):
        ev_v = start * (1 - loss) ** y
        pet_v = START_VALUE["_petrol"] * (1 - YEARLY_LOSS["_petrol"]) ** y
        curve.append({"age": y, "ev": round(ev_v * 100, 1), "petrol": round(pet_v * 100, 1)})

    factors = [
        {"feature": "Price when new", "value": f"₹{new_on_road:,.0f} on-road",
         "effect": None, "note": "Ex-showroom + insurance + registration. EVs pay 0% road tax."},
        {"feature": "Drive-off drop", "value": f"keeps {start*100:.0f}%",
         "effect": round((start - 1) * 100, 1),
         "note": "Median value kept by a used ad under one year old (BikeWale, Nov 2026)."},
        {"feature": "Age", "value": f"{age:g} yr",
         "effect": round((age_factor - 1) * 100, 1),
         "note": f"{'Ola scooters' if key == 'Ola' else 'Non-Ola EVs'} lose about {loss*100:.1f}% a year (same-variant regression)."},
        {"feature": "Kilometres", "value": f"{km:,.0f} km",
         "effect": round((km_factor - 1) * 100, 1),
         "note": f"Compared with a typical {expected_km:,.0f} km for this age."},
        {"feature": "Owners", "value": str(owners),
         "effect": round((owner_factor - 1) * 100, 1), "note": "Each extra owner lowers buyer trust."},
    ]
    return {
        "price": round(price, -2),
        "low": round(price * 0.9, -2),
        "high": round(price * 1.1, -2),
        "new_price": round(new_on_road, -2),
        "value_kept": round(price / new_on_road * 100, 1),
        "battery_health": round(health, 1),
        "factors": factors,
        "curve": curve,
        "method": "EV depreciation model (Module 6 same-variant regression on 2026 BikeWale ads)",
    }
