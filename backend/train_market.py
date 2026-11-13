"""Train the main price models on live 2026 listings: two-wheelers (BikeWale) and cars (CarWale).

  python backend/train_market.py bike
  python backend/train_market.py car

Features
  categorical (XGBoost native categories): brand, model, variant (version id), fuel, body, transmission,
                                           state, city, seller type
  numeric: age, log(1 + km), owners, log(new price of the exact variant), log(new price of the model)
  - cars: the variant's new on-road price comes with each CarWale ad
  - two-wheelers: the variant's ex-showroom price from BikeWale model pages (scrape_variants.py), else the model's
Monotone constraints: price never rises with age, km or owners, never falls as the new price rises.
Settings picked by a small search on a validation split, then scored once on a held-out 20% test split.
Output: models/{bike,car}_market.joblib + metrics in models/metrics.json.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
NOW = 2026
CATS = ["brand", "model_key", "variant", "fuel", "body", "transmission", "state", "city", "seller"]
BASE_NUMS = ["age", "log_km", "owners", "log_new", "log_new_model"]
# two-wheelers also get specs: battery size (EVs; the variant's, else from its name, else the model's median),
# engine cc and power (the model's, from the BikeWale catalogue). Cars need none: their variant's new price says it.
# Tested 13 Nov: all two-wheelers 12.4% -> 12.3% typical miss, electric scooters 61.6% -> 63.0% within 20% but
# 14.1% -> 14.4% typical miss -- all within noise (138 EV scooters in the test set), so they stay switched off.
EXTRA_NUMS = {"bike": [], "car": []}            # to try again: {"bike": ["battery_kwh", "cc", "bhp"], ...}
NUMS = list(BASE_NUMS)
MONO = {"age": -1, "log_km": -1, "owners": -1, "log_new": 1, "log_new_model": 1}
CFG = {
    "bike": {"csv": "bikewale_used_2026.csv", "min_price": 5_000, "max_km": 150_000},
    "car": {"csv": "carwale_used_2026.csv", "min_price": 40_000, "max_km": 300_000},
}


def clean_model(s):
    return (s.astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())


def fuel_of(s):
    s = s.fillna("Petrol").astype(str).str.lower()
    # hybrids first: "Hybrid (Electric + Petrol)" must not count as electric
    return np.select([s.str.contains("hybrid"), s.str.contains("electric"), s.str.contains("diesel"), s.str.contains("cng"),
                      s.str.contains("lpg")], ["Hybrid", "Electric", "Diesel", "CNG", "LPG"], "Petrol")


def trans_of(s):
    s = s.fillna("").astype(str).str.lower()
    return np.where(s.str.contains("auto") | s.str.contains("amt") | s.str.contains("cvt") | s.str.contains("dct"),
                    "Automatic", np.where(s == "", "NA", "Manual"))


def load(kind):
    c = CFG[kind]
    u = pd.read_csv(ROOT / "data" / "used_2026" / c["csv"])
    n0 = len(u)
    u = u.drop_duplicates("stockId").rename(columns={"makeName": "brand", "priceNumeric": "price", "kmNumeric": "km",
                                                     "ownersId": "owners", "cityName": "city"})
    u["model"] = clean_model(u.modelName)
    u["model_key"] = u.brand + " | " + u.model
    u["variant"] = u.versionId.astype("Int64").astype(str)
    u["age"] = (NOW - u.makeYear).clip(lower=0)
    u["fuel"] = fuel_of(u.fuel)
    u["transmission"] = trans_of(u.transmission) if kind == "car" else "NA"
    u["body"] = u.bodyStyleId.astype(str)
    if kind == "bike":
        u["body"] = np.where(u.body == "5", "scooter", "motorcycle")
    cs = pd.read_csv(ROOT / "data" / "city_state.csv").drop_duplicates("city")
    u["state"] = u.city.map(dict(zip(cs.city, cs.state))).fillna("Other")
    u["seller"] = u.get("sellerType", pd.Series(np.nan, index=u.index)).map({1: "dealer", 2: "individual"}).fillna("unknown")
    u["owners"] = u.owners.fillna(1).clip(1, 4)

    if kind == "car":
        u["new_price"] = pd.to_numeric(u.onRoadPrice, errors="coerce").where(lambda x: x > 50_000)
        cat = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())["car"]
    else:
        v = pd.read_csv(ROOT / "data" / "catalog" / "bike_variants.csv")
        u["new_price"] = u.versionId.map(dict(zip(v.versionId, v.ex_showroom)))
        cat = json.loads((ROOT / "data" / "catalog" / "catalog.json").read_text())["bike"]
    model_price = {f"{b} | {m['model']}": m["new_price"] for b, ms in cat.items() for m in ms if m.get("new_price")}
    u["new_price_model"] = u.model_key.map(model_price)
    if kind == "bike":
        specs = {f"{b} | {m['model']}": m for b, ms in cat.items() for m in ms}
        u["cc"] = u.model_key.map(lambda k: (specs.get(k) or {}).get("cc"))
        u["bhp"] = u.model_key.map(lambda k: (specs.get(k) or {}).get("bhp"))
        kwh_v = dict(zip(v.versionId, v.battery_kwh))
        name_kwh = pd.to_numeric(u.versionName.astype(str).str.extract(r"(\d+(?:\.\d+)?)\s*kWh", expand=False), errors="coerce")
        u["battery_kwh"] = u.versionId.map(kwh_v).fillna(name_kwh)
        u["battery_kwh"] = u.battery_kwh.fillna(u.groupby("model_key").battery_kwh.transform("median"))
        u.loc[u.fuel != "Electric", "battery_kwh"] = np.nan
    # model-level fallback for the variant price: median variant price seen for that model
    u["new_price"] = u.new_price.fillna(u.groupby("model_key").new_price.transform("median")).fillna(u.new_price_model)

    u = u[(u.price >= c["min_price"]) & u.km.between(0, c["max_km"]) & (u.age <= 25)]
    # obvious typos: asking more than 1.6x the new price, or under 3% of it
    r = u.price / u.new_price
    u = u[~((r > 1.6) | (r < 0.03))]
    # near-duplicate re-posts: same model, variant, year, km, price, city
    u = u.drop_duplicates(["model_key", "variant", "makeYear", "km", "price", "city"])
    # outliers: 1.5 x IQR on log price within model + age group (groups of 10+)
    u["lp"] = np.log(u.price)
    u["ag"] = pd.cut(u.age, [-1, 1, 3, 5, 8, 12, 30])
    g = u.groupby(["model_key", "ag"], observed=True).lp
    q1, q3, n = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75)), g.transform("size")
    u = u[(n < 10) | u.lp.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]

    u["log_km"] = np.log1p(u.km)
    u["log_new"] = np.log(u.new_price)
    u["log_new_model"] = np.log(u.new_price_model)
    # rare categories -> missing (the model then leans on the price and other features)
    for col, k in [("variant", 3), ("city", 25), ("model_key", 3)]:
        vc = u[col].value_counts()
        u.loc[u[col].map(vc) < k, col] = np.nan
    u = u.reset_index(drop=True)
    log = {"ads_loaded": int(n0), "final_rows": int(len(u)), "models": int(u.model_key.nunique()),
           "ev_rows": int((u.fuel == "Electric").sum()), "with_variant_price": float(u.new_price.notna().mean())}
    return u, log


def to_frame(df, cats):
    X = df[CATS + NUMS].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def fit(Xtr, ytr, Xva, yva, p):
    params = {"objective": "reg:squarederror", "tree_method": "hist", "max_cat_to_onehot": 1, "max_cat_threshold": 64,
              "monotone_constraints": "(" + ",".join(str(MONO.get(f, 0)) for f in CATS + NUMS) + ")",
              "seed": 42, "nthread": -1, **p}
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    b = xgb.train(params, dtr, num_boost_round=4000, evals=[(dva, "va")], early_stopping_rounds=150, verbose_eval=False)
    return b


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = np.abs(p / y - 1)
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(p - y))), "median_abs_pct_err": float(np.median(e)),
            "within_10pct": float(np.mean(e <= .1)), "within_20pct": float(np.mean(e <= .2)),
            "R2_log": float(1 - np.sum((np.log(y) - np.log(p)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)),
            "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}


PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "max_cat_to_onehot": 1, "max_cat_threshold": 16,
          "eta": 0.03, "max_depth": 10, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "lambda": 3.0}
# (models averaged, tree depth) per kind. Cars: 3 x depth 8 scores within 0.1 pt of 5 x depth 10 (8.3% vs 8.2% typical
# miss) at a third of the size (45 MB vs 128 MB, under GitHub's 100 MB file limit). Seed averaging beat single-model tuning.
SIZE = {"bike": (5, 10), "car": (3, 8)}


def ens_predict(boosters, X):
    d = xgb.DMatrix(X, enable_categorical=True)
    return np.mean([b.predict(d) for b in boosters], axis=0)


def noise_floor(u):
    """How far apart asking prices are for the SAME variant, year and seller type: the best any model could do."""
    d = u.dropna(subset=["variant"])
    g = d.groupby(["variant", "makeYear", "seller"]).price
    med, n = g.transform("median"), g.transform("size")
    m = n >= 5
    return {"ads": int(m.sum()), "median_abs_pct_from_cell_median": float(np.median(np.abs(d.price[m] / med[m] - 1)))}


def main(kind):
    global NUMS
    NUMS = BASE_NUMS + EXTRA_NUMS[kind]
    u, log = load(kind)
    print(kind, log, flush=True)
    cats = {c: sorted(u[c].dropna().astype(str).unique().tolist()) for c in CATS}
    tr, te = train_test_split(u, test_size=0.2, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.15, random_state=7)
    mono = "(" + ",".join(str(MONO.get(f, 0)) for f in CATS + NUMS) + ")"
    N_SEEDS, depth = SIZE[kind]
    params = {**PARAMS, "max_depth": depth, "monotone_constraints": mono}
    dtr2 = xgb.DMatrix(to_frame(tr2, cats), tr2.lp, enable_categorical=True)
    dva = xgb.DMatrix(to_frame(va, cats), va.lp, enable_categorical=True)
    rounds, pva = [], []
    for seed in range(N_SEEDS):
        b = xgb.train({**params, "seed": seed}, dtr2, 6000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
        rounds.append(b.best_iteration + 1)
        pva.append(b.predict(dva, iteration_range=(0, b.best_iteration + 1)))
    # price range per segment: the middle 60% of the misses on validation ads (never seen in that training)
    ratio_va = va.price.values / np.exp(np.mean(pva, axis=0))
    seg_va = (va.fuel if kind == "car" else va.fuel + " " + va.body).values
    bands = {"_all": [float(np.quantile(ratio_va, .2)), float(np.quantile(ratio_va, .8))]}
    for s_ in np.unique(seg_va):
        m = seg_va == s_
        if m.sum() >= 40:
            bands[s_] = [float(np.quantile(ratio_va[m], .2)), float(np.quantile(ratio_va[m], .8))]
    print("  price range by segment:", {k: [round(x, 3) for x in v] for k, v in bands.items()}, flush=True)
    print("  rounds per seed:", rounds, flush=True)
    dtr = xgb.DMatrix(to_frame(tr, cats), tr.lp, enable_categorical=True)
    boosters = [xgb.train({**params, "seed": s}, dtr, num_boost_round=r) for s, r in zip(range(N_SEEDS), rounds)]
    pt = np.exp(ens_predict(boosters, to_frame(te, cats)))
    res = {"overall": metrics(te.price, pt)}
    seg = te.fuel if kind == "car" else te.fuel + " " + te.body
    for s_ in sorted(seg.unique()):
        mask = (seg == s_).values
        if mask.sum() >= 30:
            res[s_] = metrics(te.price[mask], pt[mask])
    kept = (tr.price / tr.new_price).groupby(tr.age).median()
    base = te.new_price * te.age.map(kept).fillna(kept.median())
    ok = base.notna().values
    res["baseline_rule"] = metrics(te.price[ok], base[ok])
    if kind == "car" and "valuationPrice" in te and (te.valuationPrice > 0).sum() >= 100:   # BikeWale's field is not a real valuation
        vm = (te.valuationPrice > 0).values
        res["site_valuation"] = metrics(te.price[vm], te.valuationPrice[vm])
        res["ours_same_ads"] = metrics(te.price[vm], pt[vm])
    floor = noise_floor(u)
    print("  noise floor:", floor, flush=True)
    for k, v in res.items():
        print(f"  {k:28s} n={v['n']:>6}  median err {v['median_abs_pct_err']:.3f}  within10 {v['within_10pct']:.3f}  "
              f"within20 {v['within_20pct']:.3f}  R2log {v['R2_log']:.3f}  MAE {v['MAE']:,.0f}", flush=True)
    p = PARAMS
    ratio = te.price.values / pt
    years = u.groupby("model_key").makeYear.agg(lambda s: [int(s.quantile(.02)), int(s.max())]).to_dict()
    variants = (u.dropna(subset=["variant"]).groupby(["model_key", "variant"])
                .agg(name=("versionName", "first"), new_price=("new_price", "median"), fuel=("fuel", "first"),
                     transmission=("transmission", "first"), n=("price", "size"), y0=("makeYear", "min"), y1=("makeYear", "max"))
                .reset_index())
    var_map = {}
    for r in variants.itertuples():
        var_map.setdefault(r.model_key, []).append({"id": r.variant, "name": str(r.name), "new_price": float(r.new_price) if r.new_price == r.new_price else None,
                                                    "fuel": r.fuel, "transmission": r.transmission, "n": int(r.n), "years": [int(r.y0), int(r.y1)]})
    model_info = (u.groupby("model_key").agg(n=("price", "size"), new_price=("new_price", "median"), fuel=("fuel", lambda s: s.mode().iat[0]),
                                             body=("body", lambda s: s.mode().iat[0]), transmission=("transmission", lambda s: s.mode().iat[0]),
                                             km_per_year=("km", lambda s: float(np.median(s)))).to_dict("index"))
    # new price of the variants actually sold in each model year (a 2010 Innova's variant cost far less than today's)
    by_year = u.dropna(subset=["new_price"]).groupby(["model_key", "makeYear"]).new_price.median()
    for (k, y), v in by_year.items():
        if k in model_info:
            model_info[k].setdefault("new_price_by_year", {})[int(y)] = float(v)
    fuel_by_year = u.groupby(["model_key", "makeYear"]).fuel.agg(lambda s: s.mode().iat[0])
    for (k, y), v in fuel_by_year.items():
        if k in model_info:
            model_info[k].setdefault("fuel_by_year", {})[int(y)] = v
    sim = u.groupby(["model_key", "age"]).price.agg(["median", "count"]).reset_index()
    similar = {f"{r.model_key}|{int(r.age)}": [float(r['median']), int(r['count'])] for _, r in sim.iterrows() if r["count"] >= 2}
    simv = u.dropna(subset=["variant"]).groupby(["variant", "age"]).price.agg(["median", "count"]).reset_index()
    similar_v = {f"{r.variant}|{int(r.age)}": [float(r['median']), int(r['count'])] for _, r in simv.iterrows() if r["count"] >= 2}
    cities = {}
    counts = u.city.value_counts()
    for c in counts.index:                           # biggest city first: it is the default in the form
        st = u.loc[u.city == c, "state"].mode()
        cities.setdefault(st.iat[0] if len(st) else "Other", []).append(c)
    spec_maps = {}
    if kind == "bike":
        spec_maps = {"variant_kwh": {str(k): float(x) for k, x in u.dropna(subset=["battery_kwh"]).groupby("variant").battery_kwh.median().items()},
                     "model_kwh": {k: float(x) for k, x in u.dropna(subset=["battery_kwh"]).groupby("model_key").battery_kwh.median().items()},
                     "model_cc": {k: float(x) for k, x in u.dropna(subset=["cc"]).groupby("model_key").cc.median().items()},
                     "model_bhp": {k: float(x) for k, x in u.dropna(subset=["bhp"]).groupby("model_key").bhp.median().items()}}
    joblib.dump({"boosters": boosters, "cats": cats, "features": CATS + NUMS, "params": params, "rounds": rounds,
                 "specs": spec_maps,
                 "noise_floor": floor,
                 "similar": similar, "similar_variant": similar_v, "cities": cities,
                 "segment_metrics": res,
                 "band": bands["_all"], "bands": bands, "years": years,
                 "variants": var_map, "models": model_info}, ROOT / "models" / f"{kind}_market.joblib", compress=3)
    mp = ROOT / "models" / "metrics.json"
    allm = json.loads(mp.read_text())
    allm[f"{kind}_market"] = {"segments": res, "cleaning": log, "train": int(len(tr)), "test": int(len(te)), "params": p,
                              "noise_floor": floor, "seeds": N_SEEDS, "depth": depth}
    mp.write_text(json.dumps(allm, indent=2))
    print(f"saved models/{kind}_market.joblib", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bike")
