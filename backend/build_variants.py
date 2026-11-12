"""Complete variant lists for the web app -> backend/models/variants_{bike,car}.json

  python backend/build_variants.py      (after scrape_variants.py and train_market.py)

The trained models learn a separate effect only for variants with 3+ ads. Every other variant can still be priced
well through its own new price (the strongest feature), so the dropdown lists ALL variants:
  - every official variant on sale (BikeWale model pages; two-wheelers), and
  - every variant seen in any 2026 ad (both sites), with its new price and the years it appears in.
Display names: BikeWale splits colours into separate versions ("Dash - Shadow Ash - Smoke Silver"). Colours with the
same price merge into one entry ("Dash"); colours with different prices stay apart ("Apex (Red)", "Apex (Black and Green)").
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from years import NOW, final_range, official_years, trimmed_ad_years

BRACKET_RANGE = re.compile(r"\[\d{4}\s*-\s*\d{4}\]")

ROOT = Path(__file__).resolve().parent


def clean_model(s):
    return (s.astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())


def tidy(word):
    """APEX -> Apex, but keep short codes (ABS, STD, ZX, 4V) as they are."""
    return word.title() if word.isalpha() and word.isupper() and len(word) > 3 else word


def split_name(name, kind):
    name = re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", str(name))).strip()   # years are shown separately
    if kind == "bike" and " - " in name:
        base, colour = name.split(" - ", 1)
        return " ".join(tidy(w) for w in base.split()), colour.replace(" - ", ", ")
    return " ".join(tidy(w) for w in name.split()), None


def build(kind):
    csv = "bikewale_used_2026.csv" if kind == "bike" else "carwale_used_2026.csv"
    u = pd.read_csv(ROOT / "data" / "used_2026" / csv, low_memory=False)
    u = u.rename(columns={"makeName": "brand", "priceNumeric": "price"})
    u["model_key"] = u.brand + " | " + clean_model(u.modelName)
    u["vid"] = u.versionId.astype("Int64").astype(str)
    M = joblib.load(ROOT / "models" / f"{kind}_market.joblib")
    learned = set(M["cats"]["variant"])
    fuel_map = {"electric": "Electric", "diesel": "Diesel", "cng": "CNG", "lpg": "LPG", "hybrid": "Hybrid"}

    def fuel_of(s):
        s = str(s).lower()
        return next((v for k, v in fuel_map.items() if k in s and not (k == "electric" and "hybrid" in s)), "Petrol")

    rows = []
    u = u[u.makeYear.between(1980, NOW)]
    g = u.groupby(["model_key", "vid"])
    seen = g.agg(name=("versionName", "first"), mname=("modelName", "first"), n=("price", "size"),
                 fuel=("fuel", "first"), trans=("transmission", "first"),
                 onroad=("onRoadPrice", lambda s: float(np.nanmedian(s[s > 50_000])) if (s > 50_000).any() else np.nan))
    counts = {k: grp.makeYear.value_counts().to_dict() for k, grp in g}
    cat = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())["car" if kind == "car" else "bike"]
    on_sale_models = {f"{b} | {m['model']}" for b, ms in cat.items() for m in ms if m.get("on_sale")}
    # the listing site's launch date of each model on sale (its latest launch or update; see years.trimmed_ad_years)
    launched = {}
    for b, ms in cat.items():
        for m in ms:
            try:
                launched[f"{b} | {m['model']}"] = int(str(m.get("launched") or "")[6:10])
            except ValueError:
                pass
    for (key, vid), r in seen.iterrows():
        rows.append({"model_key": key, "id": vid, "raw": r["name"], "mname": r["mname"], "n": int(r["n"]),
                     "counts": {int(y): int(c) for y, c in counts[(key, vid)].items()},
                     "fuel": fuel_of(r.fuel), "transmission": ("Automatic" if re.search(r"auto|amt|cvt|dct", str(r.trans), re.I)
                                                               else "Manual") if kind == "car" else "NA",
                     "new_price": r.onroad if kind == "car" else np.nan})
    if kind == "bike":                                   # official variants (+ their ex-showroom price)
        v = pd.read_csv(ROOT / "data" / "catalog" / "bike_variants.csv")
        price = dict(zip(v.versionId.astype(str), v.ex_showroom))
        have = {(r["model_key"], r["id"]) for r in rows}
        for r in v.itertuples():
            key = f"{r.brand} | {r.model}"
            if (key, str(r.versionId)) not in have:
                rows.append({"model_key": key, "id": str(r.versionId), "raw": r.versionName, "mname": r.model, "n": 0,
                             "counts": {}, "fuel": None, "transmission": "NA", "new_price": np.nan})
        for r in rows:
            if r["id"] in price:
                r["new_price"] = float(price[r["id"]])
                r["official"] = True
    # years each variant can be from (years.py: official ranges where the site states them, else typo-trimmed ad years)
    for r in rows:
        off = official_years(r["raw"], r["mname"])
        if kind == "bike":
            on_sale = bool(r.get("official"))                   # in BikeWale's current variant list
        else:                                                   # current generation of a model on sale
            on_sale = r["model_key"] in on_sale_models and not BRACKET_RANGE.search(str(r["mname"])) and \
                not (off and off[1] < NOW - 1)
        fr = final_range(off, r["counts"], on_sale, kind, launched.get(r["model_key"]))
        r["years"] = [int(fr[0]), int(fr[1])] if fr else [NOW, NOW]
        r["years_source"] = fr[2] if fr else "new"
    # a model only lists variants of its own fuel: the electric Chetak must not inherit the 1972-2006 petrol Chetak
    model_fuel = {f"{b} | {m['model']}": m.get("fuel") for b, ms in cat.items() for m in ms}
    def same_fuel(r):
        mf = model_fuel.get(r["model_key"])
        if kind != "bike" or not mf or not r.get("fuel"):
            return True
        return (mf == "electric") == (r["fuel"] == "Electric")
    rows = [r for r in rows if same_fuel(r)]
    # the same thin-tail rule at model level: small variants can each hold a few mis-filed old ads
    # (R15 V4 "2019"); ad-based years before the model's real start are cut, official years are kept
    model_counts = {}
    for r in rows:
        mc = model_counts.setdefault(r["model_key"], {})
        for y, c in r["counts"].items():
            mc[y] = mc.get(y, 0) + c
    model_start = {k: trimmed_ad_years(c, launched.get(k))[0] for k, c in model_counts.items() if c}
    for r in rows:
        st = model_start.get(r["model_key"])
        if st and r.get("years_source") == "ads" and r["years"][0] < st:
            r["years"] = [st, max(st, r["years"][1])]
    out = {}
    for key, grp in pd.DataFrame(rows).groupby("model_key"):
        model_new = M["models"].get(key, {}).get("new_price")
        if not (isinstance(model_new, (int, float)) and model_new == model_new):
            model_new = None                             # NaN is not valid JSON
        opts = {}
        for r in grp.sort_values("n", ascending=False).to_dict("records"):
            base, colour = split_name(r["raw"], kind)
            p = r["new_price"] if r["new_price"] == r["new_price"] else None
            k = (base.lower(), round(p / 1000) if p else None)
            if k in opts:                                # same name and price: merge (keep the id with most ads)
                o = opts[k]
                o["n"] += r["n"]; o["years"] = [min(o["years"][0], r["years"][0]), max(o["years"][1], r["years"][1])]
                o["ids"].append(r["id"])
                continue
            opts[k] = {"id": r["id"], "ids": [r["id"]], "base": base, "colour": colour, "new_price": p, "n": r["n"],
                       "years": r["years"], "fuel": r["fuel"], "transmission": r["transmission"],
                       "official": bool(r.get("official")), "price_is_model": p is None}
        # an older listing id with no price of its own ("Dash") folds into the priced version of the same name
        priced = [o for o in opts.values() if not o["price_is_model"]]
        for k, o in list(opts.items()):
            if o["price_is_model"]:
                twin = [t for t in priced if t["base"].lower() == o["base"].lower()]
                if twin:
                    t = max(twin, key=lambda t: t["n"])
                    t["n"] += o["n"]; t["ids"] += o["ids"]
                    t["years"] = [min(t["years"][0], o["years"][0]), max(t["years"][1], o["years"][1])]
                    del opts[k]
        names = {}
        for o in opts.values():
            names.setdefault(o["base"].lower(), []).append(o)
        final = []
        for same in names.values():
            for o in same:
                o["name"] = o["base"] + (f" ({o['colour']})" if len(same) > 1 and o["colour"] else "")
                if o["new_price"] is None:
                    o["new_price"] = model_new
                # the id the model knows best: the merged id with a learned effect, else the first
                o["id"] = next((i for i in o["ids"] if i in learned), o["ids"][0])
                o["learned"] = o["id"] in learned
                rec = {k: o[k] for k in ("id", "name", "new_price", "n", "years", "fuel", "transmission", "learned", "official")}
                final.append({k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()})   # NaN -> null
        out[key] = sorted(final, key=lambda o: ((o["new_price"] or 0), o["name"]))
    (ROOT / "models" / f"variants_{kind}.json").write_text(json.dumps(out))
    # model-level years (no variant chosen): the span of its variants' years
    # every year at least one variant covers -- a list, so gaps between generations are left out
    # (Grand Vitara: 2003-2015 and 2022-2026, not 2003-2026)
    model_years = {k: sorted({y for o in vs for y in range(o["years"][0], o["years"][1] + 1)}) for k, vs in out.items() if vs}
    model_years.update({k: list(range(v[0], v[1] + 1)) for k, v in kaggle_years(kind).items() if k not in model_years})
    (ROOT / "models" / f"model_years_{kind}.json").write_text(json.dumps(model_years))
    n = sum(len(v) for v in out.values())
    print(f"{kind}: {n:,} variants across {len(out):,} models ({sum(o['learned'] for v in out.values() for o in v):,} with a learned effect)")


def kaggle_years(kind):
    """Years for the older models priced from the Kaggle 2020-21 data (no 2026 ads): from that data, typo-trimmed."""
    import glob
    home = Path.home() / ".cache" / "kagglehub" / "datasets"
    if kind == "bike":
        f = glob.glob(str(home / "saisaathvik" / "*" / "versions" / "*" / "*.csv"))
        if not f:
            return {}
        d = pd.read_csv(f[0]).drop_duplicates()
        d["bike_name"] = d.bike_name.str.replace("\u200e", "").str.replace(r"\s+", " ", regex=True).str.strip()
        nm = pd.read_csv(ROOT / "data" / "catalog" / "name_map_bike.csv")
        d = d.merge(nm, left_on=["brand", "bike_name"], right_on=["kaggle_brand", "kaggle_name"], how="inner")
        d["key"] = d.brand_y + " | " + d.official_model
        d["year"] = (2021 - d.age).astype(int)
    else:
        f = glob.glob(str(home / "nehalbirla" / "vehicle-dataset-from-cardekho" / "versions" / "*" / "Car details v3.csv"))
        if not f:
            return {}
        d = pd.read_csv(f[0]).drop_duplicates()
        nm = pd.read_csv(ROOT / "data" / "catalog" / "name_map_car.csv")
        d = d.merge(nm, left_on="name", right_on="kaggle_name", how="inner")
        d["key"] = d.brand + " | " + d.official_model
    out = {}
    for k, grp in d.dropna(subset=["key"]).groupby("key"):
        y1, y2 = trimmed_ad_years(grp.year.value_counts().to_dict())
        out[k] = [int(y1), int(min(NOW, y2 + 1))]
    return out


if __name__ == "__main__":
    for k in ("bike", "car"):
        build(k)
