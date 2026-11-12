"""Train the price models served by the website.

  python backend/train.py

Steps (mirrors Colab Modules 1, 2, 4, 7):
  1. Download Kaggle data with kagglehub (bikes: saisaathvik, cars: nehalbirla CarDekho v3)
  2. Clean: drop duplicates, odd km, price outliers (1.5x IQR per model family + age group, groups of 10+)
  3. Features: city -> state, model family, log1p(km); target = log(price)
  4. 80/20 split (random_state=42); Linear Regression, Random Forest, XGBoost
  5. Metrics on rupee scale (np.expm1 back-transform): MAE, RMSE, R2
  6. Save models/*.joblib, models/meta.json, models/metrics.json, docs/figures/*.png
"""
import json
import os
import re
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
FIGS = ROOT.parent / "docs" / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Kaggle prices are older listings; bring to 2026 rupees with the project's 5 %/yr new-price growth
GROWTH = 1.05
BIKE_REF_YEAR, CAR_REF_YEAR = 2021, 2020   # year the listings were collected

CITY_STATE = {
    "Delhi": "Delhi", "Bangalore": "Karnataka", "Mumbai": "Maharashtra", "Gurgaon": "Haryana", "Faridabad": "Haryana",
    "Pune": "Maharashtra", "Jaipur": "Rajasthan", "Chennai": "Tamil Nadu", "Ahmedabad": "Gujarat", "Ghaziabad": "Uttar Pradesh",
    "Hyderabad": "Telangana", "Noida": "Uttar Pradesh", "Kolkata": "West Bengal", "Ludhiana": "Punjab", "Thane": "Maharashtra",
    "Lucknow": "Uttar Pradesh", "Jhansi": "Uttar Pradesh", "Vadodara": "Gujarat", "Surat": "Gujarat", "Jalandhar": "Punjab",
    "Chandigarh": "Chandigarh", "Gautam Buddha Nagar": "Uttar Pradesh", "Rajkot": "Gujarat", "Bhopal": "Madhya Pradesh",
    "Dehradun": "Uttarakhand", "Kanchipuram": "Tamil Nadu", "Indore": "Madhya Pradesh", "Patna": "Bihar",
    "Navi Mumbai": "Maharashtra", "Nagpur": "Maharashtra", "Bhubaneshwar": "Odisha", "Coimbatore": "Tamil Nadu",
    "Guwahati": "Assam", "Kanpur": "Uttar Pradesh", "Tiruvallur": "Tamil Nadu", "Aurangabad": "Maharashtra", "Karnal": "Haryana",
    "Visakhapatnam": "Andhra Pradesh", "Howrah": "West Bengal", "Nashik": "Maharashtra", "Jodhpur": "Rajasthan",
    "Ranchi": "Jharkhand", "Alibag": "Maharashtra", "Kalyan": "Maharashtra", "Gorakhpur": "Uttar Pradesh", "Ernakulam": "Kerala",
    "24 Pargana": "West Bengal", "Cuttack": "Odisha", "Sirsa": "Haryana", "Gandhinagar": "Gujarat", "Agra": "Uttar Pradesh",
    "Patiala": "Punjab", "Meerut": "Uttar Pradesh", "Kota": "Rajasthan", "Rohtak": "Haryana", "Varanasi": "Uttar Pradesh",
    "Udaipur": "Rajasthan", "Alipore": "West Bengal", "Ambala": "Haryana", "Sonepat": "Haryana", "Nadiad": "Gujarat",
    "Ahmednagar": "Maharashtra", "Jabalpur": "Madhya Pradesh", "Mohali": "Punjab", "Bardhaman": "West Bengal",
    "Trivandrum": "Kerala", "Amritsar": "Punjab", "Kolhapur": "Maharashtra", "Gwalior": "Madhya Pradesh", "Durg": "Chhattisgarh",
    "Bhatinda": "Punjab", "Pondicherry": "Puducherry", "Hissar": "Haryana", "Hosur": "Tamil Nadu", "Mysore": "Karnataka",
    "Sonipat": "Haryana", "Durgapur": "West Bengal", "Kharar": "Punjab", "Saharanpur": "Uttar Pradesh", "Solapur": "Maharashtra",
    "Anand": "Gujarat", "Ajmer": "Rajasthan", "Moradabad": "Uttar Pradesh", "Raipur": "Chhattisgarh",
    "Dakshina Kannada": "Karnataka", "Adalaj": "Gujarat", "Bagalkot": "Karnataka", "Thrissur": "Kerala",
    "Ranga Reddy": "Telangana", "Siliguri": "West Bengal", "Jammu": "Jammu and Kashmir", "Unnao": "Uttar Pradesh",
    "Godhara": "Gujarat", "Satara": "Maharashtra", "Palakkad": "Kerala", "Panchkula": "Haryana", "Belgaum": "Karnataka",
    "Asansol": "West Bengal", "Chaksu": "Rajasthan", "Hooghly": "West Bengal", "Dhanbad": "Jharkhand", "Jhajjar": "Haryana",
    "Thiruvallur": "Tamil Nadu", "Alwar": "Rajasthan", "Kanpur Nagar": "Uttar Pradesh", "Bareilly": "Uttar Pradesh",
    "Salem": "Tamil Nadu", "Dharwar": "Karnataka", "Jamnagar": "Gujarat", "Allahabad": "Uttar Pradesh",
    "Baripara": "West Bengal", "Hamirpur(hp)": "Himachal Pradesh", "Kottayam": "Kerala", "Rewari": "Haryana",
    "Wardha": "Maharashtra", "Pali": "Rajasthan", "Bhiwani": "Haryana", "Jind": "Haryana", "Hubli": "Karnataka",
    "Vellore": "Tamil Nadu",
}
OWNER_MAP = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth Owner Or More": 4, "Fourth & Above Owner": 4}


def find_csv(dataset, name=None):
    path = kagglehub.dataset_download(dataset)
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    f = name if name else files[0]
    return pd.read_csv(os.path.join(path, f))


def iqr_filter(df, group_cols, col="price", min_group=10):
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby(group_cols):
        if len(g) < min_group:
            continue
        q1, q3 = g[col].quantile([.25, .75])
        iqr = q3 - q1
        bad = (g[col] < q1 - 1.5 * iqr) | (g[col] > q3 + 1.5 * iqr)
        keep[g.index[bad]] = False
    return df[keep]


def age_group(a):
    return pd.cut(a, [-1, 2, 5, 8, 12, 100], labels=["0-2", "3-5", "6-8", "9-12", "13+"])


# ---------------------------------------------------------------- bikes
def bike_family(name, brand):
    rest = name[len(brand):].strip() if name.startswith(brand) else name
    toks = [t for t in rest.split() if not re.fullmatch(r"\d+(\.\d+)?cc", t, flags=re.I)]
    toks = [t for t in toks if t.lower() not in {"abs", "bs6", "bs-vi", "bsvi", "std", "standard"}]
    return " ".join(toks[:2]) if toks else "Other"


def load_bikes():
    df = find_csv("saisaathvik/used-bikes-prices-in-india")
    n0 = len(df)
    df = df.drop_duplicates().dropna()
    n_dup = n0 - len(df)
    df["state"] = df["city"].map(CITY_STATE).fillna("Other")
    df["owners"] = df["owner"].map(OWNER_MAP).fillna(1).astype(int)
    df["family"] = [bike_family(n, b) for n, b in zip(df.bike_name, df.brand)]
    # rare families -> brand-level "Other"
    fc = df.family.value_counts()
    df.loc[df.family.map(fc) < 3, "family"] = "Other"
    df["model_key"] = df.brand + " | " + df.family
    df = df[(df.kms_driven >= 100) & (df.kms_driven <= 150_000)]
    n_km = len(df)
    df["age_grp"] = age_group(df.age)
    df = iqr_filter(df, ["model_key", "age_grp"], "price")
    df["log_km"] = np.log1p(df.kms_driven)
    log = {"rows_loaded": n0, "duplicates_removed": n_dup, "after_km_filter": n_km, "final_rows": len(df)}
    return df.reset_index(drop=True), log


BIKE_CAT = ["brand", "model_key", "state"]
BIKE_NUM = ["age", "log_km", "owners", "power"]

# ---------------------------------------------------------------- cars
def num(s):
    return pd.to_numeric(s.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")


def load_cars():
    df = find_csv("nehalbirla/vehicle-dataset-from-cardekho", "Car details v3.csv")
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    df = df[df.owner != "Test Drive Car"].copy()
    df["brand"] = df.name.str.split().str[0]
    df["model"] = df.name.str.split().str[1]
    df["model_key"] = df.brand + " | " + df.model
    df["engine_cc"] = num(df.engine)
    df["power_bhp"] = num(df.max_power)
    df["mileage_kmpl"] = num(df.mileage)
    df["owners"] = df.owner.map(OWNER_MAP).fillna(1).astype(int)
    df["age"] = CAR_REF_YEAR - df.year
    df["price"] = df.selling_price
    df = df.dropna(subset=["engine_cc", "power_bhp", "mileage_kmpl", "seats"])
    df = df[(df.km_driven >= 500) & (df.km_driven <= 400_000)]
    df["age_grp"] = age_group(df.age)
    df = iqr_filter(df, ["model_key", "age_grp"], "price")
    df["log_km"] = np.log1p(df.km_driven)
    log = {"rows_loaded": n0, "duplicates_removed": n_dup, "final_rows": len(df)}
    return df.reset_index(drop=True), log


CAR_CAT = ["brand", "model_key", "fuel", "seller_type", "transmission"]
CAR_NUM = ["age", "log_km", "owners", "engine_cc", "power_bhp", "mileage_kmpl", "seats"]


# ---------------------------------------------------------------- training
def make_pre(cat, numc):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), cat),
        ("num", "passthrough", numc),
    ])


def feature_groups(pre, cat, numc):
    """Original feature name for each transformed column (for grouping SHAP values)."""
    groups = []
    enc = pre.named_transformers_["cat"]
    names = enc.get_feature_names_out(cat)
    for n in names:
        groups.append(next(c for c in cat if n.startswith(c + "_")))
    groups += numc
    return groups


def scores(y_true_log, y_pred_log):
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {"MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": float(r2_score(yt, yp)),
            "R2_log": float(r2_score(y_true_log, y_pred_log))}


def train(df, cat, numc, tag):
    X, y = df[cat + numc], np.log1p(df.price)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=6, subsample=0.85,
                                colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, random_state=42, n_jobs=-1),
    }
    out, fitted = {}, {}
    for name, m in models.items():
        pipe = Pipeline([("pre", make_pre(cat, numc)), ("m", m)])
        pipe.fit(Xtr, ytr)
        out[name] = scores(yte, pipe.predict(Xte))
        fitted[name] = pipe
        print(f"  [{tag}] {name:18s} MAE {out[name]['MAE']:>10,.0f}  RMSE {out[name]['RMSE']:>10,.0f}  R2 {out[name]['R2']:.3f}")

    xgb = fitted["XGBoost"]
    pred = np.expm1(xgb.predict(Xte))
    ratio = np.expm1(yte.values) / pred
    band = [float(np.quantile(ratio, .2)), float(np.quantile(ratio, .8))]
    groups = feature_groups(xgb.named_steps["pre"], cat, numc)
    joblib.dump({"pipe": xgb, "groups": groups, "band": band, "cat": cat, "num": numc}, MODELS / f"{tag}_xgb.joblib", compress=3)
    plots(tag, xgb, Xte, yte, groups, fitted)
    return out, len(Xtr), len(Xte)


def plots(tag, xgb, Xte, yte, groups, fitted):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    yt, yp = np.expm1(yte), np.expm1(xgb.predict(Xte))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(yt, yp, s=6, alpha=.35, color="#0071e3")
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lim, lim, color="#1d1d1f", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (Rs)"); ax.set_ylabel("Predicted price (Rs)")
    ax.set_title(f"XGBoost: actual vs predicted ({tag}s, test set)")
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_actual_vs_pred.png", dpi=150); plt.close(fig)

    # mean |SHAP| by original feature (tree SHAP via XGBoost pred_contribs)
    import xgboost as xgb_lib
    Z = xgb.named_steps["pre"].transform(Xte)
    contrib = xgb.named_steps["m"].get_booster().predict(xgb_lib.DMatrix(Z), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=groups).groupby(level=0).sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#0071e3")
    ax.set_xlabel("mean |SHAP value| (log-price units)")
    ax.set_title(f"What drives {tag} prices (XGBoost, Tree SHAP)")
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_shap_importance.png", dpi=150); plt.close(fig)

    # model comparison
    names = list(fitted)
    r2 = [r2_score(yt, np.expm1(fitted[n].predict(Xte))) for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(names, r2, color=["#86868b", "#30b07a", "#0071e3"])
    for i, v in enumerate(r2):
        ax.text(i, v + .01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("R² (test set)")
    ax.set_title(f"Model comparison ({tag}s)")
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_model_comparison.png", dpi=150); plt.close(fig)


def mode(s):
    return s.mode().iat[0] if not s.mode().empty else None


def build_meta(bikes, cars):
    bike_brands = {}
    for (brand, fam), g in bikes.groupby(["brand", "family"]):
        bike_brands.setdefault(brand, []).append({
            "model": fam, "key": f"{brand} | {fam}", "cc": float(g.power.median()), "n": int(len(g))})
    for b in bike_brands:
        bike_brands[b].sort(key=lambda r: -r["n"])
    car_brands = {}
    for (brand, model), g in cars.groupby(["brand", "model"]):
        car_brands.setdefault(brand, []).append({
            "model": model, "key": f"{brand} | {model}", "n": int(len(g)),
            "cc": float(g.engine_cc.median()), "power": float(g.power_bhp.median()),
            "seats": float(g.seats.median()), "mileage": float(g.mileage_kmpl.median()),
            "fuel": mode(g.fuel), "transmission": mode(g.transmission)})
    for b in car_brands:
        car_brands[b].sort(key=lambda r: -r["n"])
    # median price per model key x age (for "similar listings")
    bike_sim = bikes.groupby(["model_key", "age"]).price.agg(["median", "count"]).reset_index()
    car_sim = cars.groupby(["model_key", "age"]).price.agg(["median", "count"]).reset_index()
    return {
        "bike": {"brands": dict(sorted(bike_brands.items(), key=lambda kv: -sum(r["n"] for r in kv[1]))),
                 "states": sorted(s for s in bikes.state.unique() if s != "Other") + ["Other"],
                 "ref_year": BIKE_REF_YEAR, "rows": int(len(bikes)),
                 "similar": {f"{r.model_key}|{int(r.age)}": [float(r["median"]), int(r["count"])] for _, r in bike_sim.iterrows()}},
        "car": {"brands": dict(sorted(car_brands.items(), key=lambda kv: -sum(r["n"] for r in kv[1]))),
                "fuels": sorted(cars.fuel.unique()), "sellers": sorted(cars.seller_type.unique()),
                "ref_year": CAR_REF_YEAR, "rows": int(len(cars)),
                "similar": {f"{r.model_key}|{int(r.age)}": [float(r["median"]), int(r["count"])] for _, r in car_sim.iterrows()}},
        "growth": GROWTH,
    }


def main():
    print("Loading bikes ..."); bikes, blog = load_bikes(); print("  ", blog)
    print("Loading cars ...");  cars, clog = load_cars();   print("  ", clog)
    print("Training bikes ..."); bm, btr, bte = train(bikes, BIKE_CAT, BIKE_NUM, "bike")
    print("Training cars ...");  cm, ctr, cte = train(cars, CAR_CAT, CAR_NUM, "car")
    metrics = {"bike": {"models": bm, "train": btr, "test": bte, "cleaning": blog},
               "car": {"models": cm, "train": ctr, "test": cte, "cleaning": clog}}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "meta.json").write_text(json.dumps(build_meta(bikes, cars)))
    print("Saved models, meta.json, metrics.json and figures.")


if __name__ == "__main__":
    main()
