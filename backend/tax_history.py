"""Indian taxes on a NEW two-wheeler or car, 1990-2026 -- the history behind every used vehicle's original price.

  python backend/tax_history.py      -> data/tax_history.json            (the rules, with sources and confidence)
                                        data/variant_tax_history.csv     (every variant x every year it was sold)

Two layers:
1. National tax inside the ex-showroom price, as an effective rate on the price before tax
   - before 1 Jul 2017: central excise (+ the 2016 infrastructure cess) and state VAT / sales tax on top
   - 1 Jul 2017 - 21 Sep 2025: GST + compensation cess
   - from 22 Sep 2025: GST 2.0 (cess folded in)
2. State road tax, paid on top of the ex-showroom price (two-wheelers; 11 states)

Confidence: "high" = official notification or several consistent reports; "medium" = one good report, or a rule
applied to a class we infer; "low" = not documented, assumed (mostly 1990-2007). Flat specific duties (e.g. Rs 15,000
on 1500-1999cc cars in 2008) and state-by-state VAT differences are left out; a typical VAT rate is used.
"""
import csv
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOW = 2026

SOURCES = {
    "budget_2008": "Union Budget 2008-09 excise note (indiabudget.gov.in) - two-wheelers 16% -> 12%",
    "notif_2008_2010": "Central Excise notification 02/2008 and amendments (taxtmi): 14->10% 7 Dec 2008, 10->8% 24 Feb 2009, 8->10% 27 Feb 2010",
    "budget_2012": "Union Budget 2012: basic excise 10% -> 12% (bikeadvice.in; tariff 17 Mar 2012)",
    "interim_2014": "Interim Budget Feb 2014: cars/two-wheelers excise cut till Jun 2014, extended; reversed Jan 2015 (moneylife.in, business-standard.com)",
    "bt_chronology": "Business Today, 'A taxing drive' (Aug 2009): car excise 35% (1989-90), 60% (1991-92), 40% (1996-97), 32% (2001-02), 24% (2003-04), small cars 12% (2008-09)",
    "gst_2017": "GST Council 2017: motor vehicles 28% + compensation cess (small petrol 1%, small diesel 3%, others 15%; two-wheelers >350cc 3%)",
    "cess_sep2017": "GST cess hike Sep 2017 (Notification 5/2017; Tribune): mid-size +2, large +5, SUV +7 points",
    "gst_ev_2019": "36th GST Council (PIB, 27 Jul 2019): electric vehicles 12% -> 5% from 1 Aug 2019",
    "gst_2025": "56th GST Council (Autocar Pro, Sep 2025): from 22 Sep 2025 two-wheelers <=350cc 18%, >350cc 40%; small cars 18%, others 40%; EVs 5%",
    "budget_2016": "Union Budget 2016: infrastructure cess on cars 1% (small petrol), 2.5% (small diesel), 4% (others)",
    "vat": "State VAT from Apr 2005 (12.5% typical, 14.5% in many states from 2015); sales tax before -- typical rates, not state-specific",
    "road_tax_table": "Module 3 road-tax table (official transport sites, DriveSpark, Tribune, DT Next, Digit, Deccan Chronicle) -- see PROGRESS.md",
}


def _d(y, m=1, d=1):
    return date(y, m, d)


# ------------------------------------------------------------------ 1. national tax
# two-wheeler (petrol) central excise by period: (start, end_exclusive, rate, confidence, source)
EXCISE_2W = [
    (_d(1990), _d(2001, 3, 1), 0.24, "low", "assumed; no reliable record found for 1990-2000"),
    (_d(2001, 3, 1), _d(2008, 3, 1), 0.16, "medium", "general CENVAT rate 16% (2001-2008); budget 2008 cut two-wheelers from 16%"),
    (_d(2008, 3, 1), _d(2008, 12, 7), 0.12, "high", "budget_2008"),
    (_d(2008, 12, 7), _d(2009, 2, 24), 0.10, "high", "notif_2008_2010"),
    (_d(2009, 2, 24), _d(2010, 2, 27), 0.08, "high", "notif_2008_2010"),
    (_d(2010, 2, 27), _d(2012, 3, 17), 0.10, "high", "notif_2008_2010"),
    (_d(2012, 3, 17), _d(2014, 2, 17), 0.12, "high", "budget_2012"),
    (_d(2014, 2, 17), _d(2015, 1, 1), 0.08, "medium", "interim_2014"),
    (_d(2015, 1, 1), _d(2017, 7, 1), 0.12, "medium", "interim_2014"),
]
# car central excise by period and class: small / mid / large / suv
EXCISE_CAR = [
    (_d(1990), _d(1991, 7, 24), {"all": 0.35}, "medium", "bt_chronology"),
    (_d(1991, 7, 24), _d(1996, 7, 22), {"all": 0.60}, "medium", "bt_chronology (includes special excise)"),
    (_d(1996, 7, 22), _d(2001, 3, 1), {"all": 0.40}, "medium", "bt_chronology"),
    (_d(2001, 3, 1), _d(2003, 3, 1), {"all": 0.32}, "medium", "bt_chronology"),
    (_d(2003, 3, 1), _d(2006, 3, 1), {"all": 0.24}, "medium", "bt_chronology"),
    (_d(2006, 3, 1), _d(2008, 3, 1), {"small": 0.16, "mid": 0.24, "large": 0.24, "suv": 0.24}, "low", "small-car concession from 2006 (rate assumed at the general 16%)"),
    (_d(2008, 3, 1), _d(2008, 12, 7), {"small": 0.12, "mid": 0.24, "large": 0.24, "suv": 0.24}, "high", "bt_chronology"),
    (_d(2008, 12, 7), _d(2010, 2, 27), {"small": 0.08, "mid": 0.20, "large": 0.20, "suv": 0.20}, "medium", "bt_chronology ('8, 10 and 20 per cent')"),
    (_d(2010, 2, 27), _d(2012, 3, 17), {"small": 0.10, "mid": 0.22, "large": 0.22, "suv": 0.22}, "medium", "notif_2008_2010 (+2 points across)"),
    (_d(2012, 3, 17), _d(2013, 3, 1), {"small": 0.12, "mid": 0.24, "large": 0.27, "suv": 0.27}, "medium", "budget_2012"),
    (_d(2013, 3, 1), _d(2014, 2, 17), {"small": 0.12, "mid": 0.24, "large": 0.27, "suv": 0.30}, "medium", "interim_2014 (SUV 30% before the cut)"),
    (_d(2014, 2, 17), _d(2015, 1, 1), {"small": 0.08, "mid": 0.20, "large": 0.24, "suv": 0.24}, "high", "interim_2014"),
    (_d(2015, 1, 1), _d(2017, 7, 1), {"small": 0.125, "mid": 0.24, "large": 0.27, "suv": 0.30}, "high", "interim_2014 (reversal), companiesinn table"),
]
INFRA_CESS_2016 = (_d(2016, 3, 1), _d(2017, 7, 1), {"small_petrol": 0.01, "small_diesel": 0.025, "other": 0.04})


def vat_rate(d):
    if d < _d(2005, 4, 1):
        return 0.10, "low"          # state sales tax, typical
    if d < _d(2015, 1, 1):
        return 0.125, "medium"
    return 0.145, "medium"


def car_class(fuel, cc, body):
    """Small / mid / large / suv from fuel, engine size (litres) and body type (CarWale body code).
    Under 4 m: hatchback (3), compact sedan (10), compact SUV (12), van (4). Big SUVs: 11, 13, 14."""
    under4 = str(body) in {"3", "10", "12", "4"}
    big_suv = str(body) in {"11", "13", "14"}
    f = (fuel or "Petrol").lower()
    if cc is None:                                      # no engine size: decide by body
        return "small" if under4 else ("suv" if big_suv else "mid")
    small_cc = cc <= (1.5 if "diesel" in f else 1.2)
    if under4 and small_cc:
        return "small"
    if cc > 1.5:
        return "suv" if big_suv else "large"
    return "mid"


def national_tax(kind, d, cc=None, fuel="Petrol", body=None):
    """Effective national tax on the price before tax for a new vehicle bought on date d.
    kind: 'bike' (cc in cc) or 'car' (cc in litres). Returns dict(rate, regime, parts, confidence, source)."""
    f = (fuel or "Petrol").lower()
    ev = "electric" in f and "hybrid" not in f
    if d >= _d(2025, 9, 22):                            # GST 2.0
        if ev:
            return {"rate": 0.05, "regime": "GST 2.0", "parts": "GST 5%", "confidence": "high", "source": "gst_2025"}
        if kind == "bike":
            big = (cc or 0) > 350
            return {"rate": 0.40 if big else 0.18, "regime": "GST 2.0", "parts": f"GST {40 if big else 18}%",
                    "confidence": "high" if cc else "medium", "source": "gst_2025"}
        cl = car_class(fuel, cc, body)
        r = 0.18 if cl == "small" else 0.40
        return {"rate": r, "regime": "GST 2.0", "parts": f"GST {round(r * 100)}% ({cl} car)", "confidence": "medium", "source": "gst_2025"}
    if d >= _d(2017, 7, 1):                             # GST + compensation cess
        if ev:
            r = 0.12 if d < _d(2019, 8, 1) else 0.05
            return {"rate": r, "regime": "GST", "parts": f"GST {round(r * 100)}%", "confidence": "high", "source": "gst_ev_2019"}
        if kind == "bike":
            cess = 0.03 if (cc or 0) > 350 else 0.0
            return {"rate": round(0.28 + cess, 4), "regime": "GST", "parts": "GST 28%" + (" + 3% cess" if cess else ""),
                    "confidence": "high" if cc else "medium", "source": "gst_2017"}
        cl = car_class(fuel, cc, body)
        if "hybrid" in f:
            cess = 0.15
        elif cl == "small":
            cess = 0.03 if "diesel" in f else 0.01
        else:
            cess = 0.15 if d < _d(2017, 9, 11) else {"mid": 0.17, "large": 0.20, "suv": 0.22}[cl]
        return {"rate": round(0.28 + cess, 4), "regime": "GST", "parts": f"GST 28% + {round(cess * 100)}% cess ({cl} car)",
                "confidence": "medium", "source": "gst_2017; cess_sep2017"}
    # before GST: excise (+ infra cess 2016-17), then VAT / sales tax on the price after excise
    vat, vconf = vat_rate(d)
    if ev:                                              # EVs had a concessional 6% excise (assumed) before GST
        exc, conf, src, extra = 0.06, "low", "assumed concessional excise for EVs", 0.0
    elif kind == "bike":
        row = next(r for r in EXCISE_2W if r[0] <= d < r[1]) if d >= EXCISE_2W[0][0] else EXCISE_2W[0]
        exc, conf, src, extra = row[2], row[3], row[4], 0.0
    else:
        row = next(r for r in EXCISE_CAR if r[0] <= d < r[1]) if d >= EXCISE_CAR[0][0] else EXCISE_CAR[0]
        cl = car_class(fuel, cc, body)
        exc = row[2].get("all", row[2].get(cl, 0.24))
        conf, src = row[3], row[4]
        extra = 0.0
        if INFRA_CESS_2016[0] <= d < INFRA_CESS_2016[1]:
            key = ("small_diesel" if "diesel" in f else "small_petrol") if cl == "small" else "other"
            extra = INFRA_CESS_2016[2][key]
    rate = (1 + exc + extra) * (1 + vat) - 1
    conf = min([conf, vconf], key=["low", "medium", "high"].index)
    parts = f"excise {exc * 100:g}%" + (f" + infra cess {extra * 100:g}%" if extra else "") + f", then VAT/sales tax {vat * 100:g}%"
    return {"rate": round(rate, 4), "regime": "excise + VAT", "parts": parts, "confidence": conf, "source": src}


# ------------------------------------------------------------------ 2. state road tax on two-wheelers, by period
# (state, start, end_exclusive, basis, slabs) -- basis "price" (upper price limit, rate) or "cc" (upper cc, rate)
ROAD_2W = [
    ("Delhi", _d(1990), _d(2100), "price", [(25_000, .02), (40_000, .04), (60_000, .06), (None, .08)], "medium"),
    ("Karnataka", _d(1990), _d(2010, 4, 1), "price", [(50_000, .08), (None, .10)], "medium"),
    ("Karnataka", _d(2010, 4, 1), _d(2017, 4, 1), "price", [(50_000, .10), (None, .12)], "high"),
    ("Karnataka", _d(2017, 4, 1), _d(2100), "price", [(50_000, .10), (100_000, .12), (None, .18)], "high"),
    ("Haryana", _d(1990), _d(2017, 4, 1), "price", [(20_000, .02), (60_000, .04), (200_000, .06), (None, .08)], "medium"),
    ("Haryana", _d(2017, 4, 1), _d(2100), "price", [(75_000, .04), (200_000, .06), (None, .08)], "high"),
    ("Maharashtra", _d(1990), _d(2017, 7, 1), "cc", [(100, .08), (299, .09), (None, .10)], "medium"),
    ("Maharashtra", _d(2017, 7, 1), _d(2100), "cc", [(99, .10), (299, .11), (None, .12)], "high"),
    ("Uttar Pradesh", _d(1990), _d(2100), "price", [(40_000, .07), (None, .10)], "medium"),
    ("Gujarat", _d(1990), _d(2100), "price", [(None, .06)], "medium"),
    ("Tamil Nadu", _d(1990), _d(2023, 11, 9), "price", [(None, .08)], "medium"),
    ("Tamil Nadu", _d(2023, 11, 9), _d(2100), "price", [(100_000, .10), (None, .12)], "high"),
    ("Rajasthan", _d(1990), _d(2019, 7, 1), "cc", [(125, .04), (200, .06), (500, .08), (None, .10)], "medium"),
    ("Rajasthan", _d(2019, 7, 1), _d(2100), "cc", [(200, .08), (500, .13), (None, .15)], "high"),
    ("Telangana", _d(1990), _d(2025, 8, 14), "price", [(50_000, .09), (None, .12)], "medium"),
    ("Telangana", _d(2025, 8, 14), _d(2100), "price", [(50_000, .09), (100_000, .12), (200_000, .15), (None, .18)], "high"),
    ("Punjab", _d(1990), _d(2024, 8, 22), "price", [(100_000, .07), (None, .09)], "medium"),
    ("Punjab", _d(2024, 8, 22), _d(2100), "price", [(100_000, .075), (200_000, .10), (None, .11)], "high"),
    ("West Bengal", _d(1990), _d(2100), "price", [(20_000, .08), (60_000, .09), (200_000, .10), (None, .12)], "medium"),
]
ROAD_CESS = {"Karnataka": 0.11}     # cess on the tax (2026 sources; assumed for earlier years too)
ROAD_STATES = sorted({r[0] for r in ROAD_2W})


def road_tax_2w(state, d, ex_showroom, cc, electric=False):
    """Effective road-tax rate (cess included) a new two-wheeler paid in `state` on date d; None if unknown."""
    if electric:
        return {"rate": 0.0, "confidence": "medium", "note": "EVs exempt in these states over the data period"}
    rows = [r for r in ROAD_2W if r[0] == state and r[1] <= d < r[2]]
    if not rows:
        return None
    _, start, _, basis, slabs, conf = rows[0]
    v = ex_showroom if basis == "price" else (cc or 125)
    rate = next(rt for lim, rt in slabs if lim is None or v <= lim)
    if start == _d(1990) and d < _d(2010):
        conf = "low"                                   # earliest known slab carried back
    return {"rate": round(rate * (1 + ROAD_CESS.get(state, 0.0)), 4), "confidence": conf}


# ------------------------------------------------------------------ files
def export_rules():
    rules = {
        "about": __doc__, "generated": date.today().isoformat(), "sources": SOURCES,
        "excise_two_wheeler": [{"from": a.isoformat(), "to": b.isoformat(), "rate": r, "confidence": c, "source": s} for a, b, r, c, s in EXCISE_2W],
        "excise_car": [{"from": a.isoformat(), "to": b.isoformat(), "rates": r, "confidence": c, "source": s} for a, b, r, c, s in EXCISE_CAR],
        "infra_cess_2016": {"from": INFRA_CESS_2016[0].isoformat(), "to": INFRA_CESS_2016[1].isoformat(), "rates": INFRA_CESS_2016[2]},
        "gst": {"2017-07-01": "two-wheelers 28% (+3% cess >350cc); cars 28% + cess (small petrol 1%, small diesel 3%, others 15%; from 2017-09-11 mid 17%, large 20%, SUV 22%; hybrids 15%); EVs 12%",
                "2019-08-01": "EVs 5%", "2025-09-22": "two-wheelers <=350cc 18%, >350cc 40%; small cars 18%, other cars 40%; EVs 5%"},
        "road_tax_two_wheeler": [{"state": s, "from": a.isoformat(), "to": b.isoformat(), "basis": basis, "slabs": sl, "confidence": c}
                                 for s, a, b, basis, sl, c in ROAD_2W],
        "road_tax_cess": ROAD_CESS,
    }
    (ROOT / "data" / "tax_history.json").write_text(json.dumps(rules, indent=1, default=str))


def export_variants():
    """Every variant in the app x every year it was sold -> the national tax it paid that year (1 July of the year)."""
    import joblib
    rows = []
    for kind in ("bike", "car"):
        vpath = ROOT / "models" / f"variants_{kind}.json"
        if not vpath.exists():
            continue
        variants = json.loads(vpath.read_text())
        specs = {}
        if kind == "bike":
            mk = joblib.load(ROOT / "models" / "bike_market.joblib")
            specs = (mk.get("specs") or {}).get("model_cc", {})
            body = {k: v.get("body") for k, v in mk["models"].items()}
        else:
            mk = joblib.load(ROOT / "models" / "car_market.joblib")
            body = {k: v.get("body") for k, v in mk["models"].items()}
        for key, vs in variants.items():
            brand, model = key.split(" | ", 1)
            for v in vs:
                if kind == "bike":
                    cc = specs.get(key)
                else:
                    m = re.search(r"(?<![\d.])([0-4]\.\d)(?![\d])", v["name"])
                    cc = float(m.group(1)) if m else None
                for y in range(v["years"][0], v["years"][1] + 1):
                    t = national_tax(kind, _d(y, 7, 1), cc, v.get("fuel") or "Petrol", body.get(key))
                    rows.append({"kind": "two-wheeler" if kind == "bike" else "car", "brand": brand, "model": model,
                                 "variant_id": v["id"], "variant": v["name"], "year": y, "fuel": v.get("fuel") or "",
                                 "engine": cc if cc is not None else "", "national_tax_rate": t["rate"], "regime": t["regime"],
                                 "breakdown": t["parts"], "confidence": t["confidence"]})
    out = ROOT / "data" / "variant_tax_history.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    export_rules()
    n = export_variants()
    print(f"wrote data/tax_history.json and data/variant_tax_history.csv ({n:,} variant-years)")
