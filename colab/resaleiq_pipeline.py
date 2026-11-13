# %% [markdown]
# # ResaleIQ — Used Two-Wheeler Resale Price Prediction with EV Depreciation Analysis
# AIML (CS52) project, MSRIT. **Runtime → Run all.** The first cell asks for `resaleiq_colab_data.zip` (in the repo's `colab/` folder). Total run time about 12 minutes.
#
# | Module | What it does |
# |---|---|
# | 1 | Load and inspect the Kaggle used-bike data |
# | 2 | Clean: duplicates, odd km, official model names, city → state, price outliers |
# | 3 | Features: log km, owners, state road-tax rate |
# | 4 | Models: base-paper replication (DT, SVR, KNN, SGD) vs Linear Regression, Random Forest, XGBoost |
# | 5 | Road-tax side check |
# | 6 | 2026 test: does a model trained on 2020-21 ads still work on 30k live BikeWale ads? |
# | 7 | EV vs petrol depreciation on 2026 ads (same model, same variant) |
# | 8 | SHAP explanations |
# | 9 | Main models on 30k bike + 92k car ads from 2026, benchmarked against CarWale's own valuation |
# | 10 | Save every figure and table for the report |

# %%
# ---------- Setup: libraries + data ----------
import os, sys, json, zipfile, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip -q install kagglehub xgboost shap statsmodels  # noqa
    if not os.path.exists("data/name_map_bike.csv"):
        from google.colab import files
        print("Upload resaleiq_colab_data.zip (from the repo's colab/ folder)")
        up = files.upload()
        zipfile.ZipFile(next(iter(up))).extractall("data")
DATA = "data"
os.makedirs("figures", exist_ok=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("ready")

# %% [markdown]
# ## Module 1 — Load and inspect

# %%
path = kagglehub.dataset_download("saisaathvik/used-bikes-prices-in-india")
raw = pd.read_csv(os.path.join(path, [f for f in os.listdir(path) if f.endswith(".csv")][0]))
print("shape:", raw.shape)
print("duplicate rows:", raw.duplicated().sum())
print(raw.isnull().sum().to_dict())
raw.head()

# %% [markdown]
# ## Module 2 — Clean
# * exact duplicates removed (the Kaggle file repeats most rows)
# * every messy listing name mapped to an **official BikeWale model name** (e.g. "Bajaj Pulsar 200 NS 200cc" → "Pulsar NS200"); map built and checked in `backend/build_catalog.py`
# * city → state; km kept between 100 and 1,50,000
# * price outliers: 1.5 × IQR within each model + age group (groups of 10+)

# %%
df = raw.drop_duplicates().dropna().copy()
n_dedup = len(df)
df["bike_name"] = df.bike_name.str.replace("‎", "").str.replace(r"\s+", " ", regex=True).str.strip()
nm = pd.read_csv(f"{DATA}/name_map_bike.csv")
df = df.merge(nm[["kaggle_brand", "kaggle_name", "official_model"]], left_on=["brand", "bike_name"],
              right_on=["kaggle_brand", "kaggle_name"], how="left")
print(f"matched to an official model: {df.official_model.notna().mean():.1%} of rows")
df = df[df.official_model.notna()].copy()
df["model"] = df.brand + " " + df.official_model

cs = pd.read_csv(f"{DATA}/city_state.csv")
df["state"] = df.city.map(dict(zip(cs.city, cs.state))).fillna("Other")
df["owners"] = df.owner.map({"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth Owner Or More": 4})
df = df[(df.kms_driven >= 100) & (df.kms_driven <= 150_000)]
df["age_grp"] = pd.cut(df.age, [-1, 2, 5, 8, 12, 100])

grp = df.groupby(["model", "age_grp"], observed=True).price
q1, q3, n = grp.transform(lambda s: s.quantile(.25)), grp.transform(lambda s: s.quantile(.75)), grp.transform("size")
iqr = q3 - q1
keep = (n < 10) | df.price.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
df = df[keep].reset_index(drop=True)
print(f"{len(raw):,} rows → {n_dedup:,} after duplicates → {len(df):,} clean bikes, {df.model.nunique()} models, {df.state.nunique()} states")
df.state.value_counts().head(12)

# %% [markdown]
# ## Module 3 — Features (incl. state road tax)
# Road-tax rate = the state's current two-wheeler slab applied to the model's 2026 new price (BikeWale).

# %%
SLABS = {  # (upper limit, rate) by price; Maharashtra & Rajasthan by cc
    "Delhi": [(25e3, .02), (40e3, .04), (60e3, .06), (None, .08)], "Karnataka": [(50e3, .10), (1e5, .12), (None, .18)],
    "Haryana": [(75e3, .04), (2e5, .06), (None, .08)], "Uttar Pradesh": [(40e3, .07), (None, .10)], "Gujarat": [(None, .06)],
    "Tamil Nadu": [(1e5, .10), (None, .12)], "Telangana": [(5e4, .09), (1e5, .12), (2e5, .15), (None, .18)],
    "Punjab": [(1e5, .075), (2e5, .10), (None, .11)], "West Bengal": [(2e4, .08), (6e4, .09), (2e5, .10), (None, .12)]}
CC_SLABS = {"Maharashtra": [(99, .10), (299, .11), (None, .12)], "Rajasthan": [(200, .08), (500, .13), (None, .15)]}

def slab(table, v):
    return next(r for lim, r in table if lim is None or v <= lim)

def tax_rate(state, price, cc):
    if state in SLABS: return slab(SLABS[state], price)
    if state in CC_SLABS: return slab(CC_SLABS[state], cc)
    return np.nan

cat = pd.read_csv(f"{DATA}/bike_catalog.csv")
newp = dict(zip(cat.brand + " " + cat.model, cat.new_price))
df["new_price_2026"] = df.model.map(newp)
# models no longer sold: estimate from engine size (median 2026 price of models with similar cc)
cc_price = cat[cat.fuel == "petrol"].dropna(subset=["cc", "new_price"])
df["new_price_2026"] = df.new_price_2026.fillna(df.power.map(lambda c: cc_price[(cc_price.cc - c).abs() <= 25].new_price.median()))
df["road_tax_rate"] = [tax_rate(s, p, c) for s, p, c in zip(df.state, df.new_price_2026, df.power)]
df["log_km"] = np.log1p(df.kms_driven)
print("bikes with a known road-tax rate:", df.road_tax_rate.notna().sum())
df[["model", "state", "age", "kms_driven", "owners", "power", "road_tax_rate", "price"]].head()

# %% [markdown]
# ## Module 4 — Models
# Target = log(price) (prices are skewed); every score is computed **after converting back to rupees** with `np.expm1`.
# The base paper reports "accuracy" for a regression task; we report MAE / RMSE / R², plus **% of bikes predicted within ±20%** as a fair stand-in for "accuracy".

# %%
CAT = ["brand", "model", "state"]
NUM = ["age", "log_km", "owners", "power"]
X, y = df[CAT + NUM], np.log1p(df.price)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def pre(scale=False):
    num = Pipeline([("s", StandardScaler())]) if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), CAT),
                              ("num", num, NUM)])

def score(name, yt_log, yp_log):
    yt, yp = np.expm1(yt_log), np.expm1(yp_log)
    return {"Model": name, "MAE (Rs)": mean_absolute_error(yt, yp), "RMSE (Rs)": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp), "Within ±20%": np.mean(np.abs(yp / yt - 1) <= .2)}

models = {
    "Decision Tree (base paper)": (DecisionTreeRegressor(random_state=42), False),
    "SVR (base paper)": (SVR(C=3.0, epsilon=0.05), True),
    "KNN (base paper)": (KNeighborsRegressor(n_neighbors=7, weights="distance"), True),
    "SGD (base paper)": (SGDRegressor(random_state=42, max_iter=3000, tol=1e-4), True),
    "Linear Regression": (LinearRegression(), False),
    "Random Forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42), False),
    "XGBoost": (XGBRegressor(n_estimators=900, learning_rate=.04, max_depth=6, subsample=.85, colsample_bytree=.8,
                             min_child_weight=2, random_state=42, n_jobs=-1), False),
}
rows, fitted = [], {}
for name, (m, scale) in models.items():
    p = Pipeline([("pre", pre(scale)), ("m", m)]).fit(Xtr, ytr)
    fitted[name] = p
    rows.append(score(name, yte, p.predict(Xte)))
results = pd.DataFrame(rows).set_index("Model")
results.style.format({"MAE (Rs)": "{:,.0f}", "RMSE (Rs)": "{:,.0f}", "R2": "{:.3f}", "Within ±20%": "{:.1%}"})

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
colors = ["#86868b"] * 4 + ["#a1a1a6", "#30b07a", "#0071e3"]
results["R2"].plot.barh(ax=ax[0], color=colors); ax[0].set_title("R² on unseen bikes (higher is better)"); ax[0].set_xlim(0, 1)
results["MAE (Rs)"].plot.barh(ax=ax[1], color=colors); ax[1].set_title("Average error in rupees (lower is better)"); ax[1].set_yticklabels([])
plt.tight_layout(); plt.savefig("figures/m4_model_comparison.png", dpi=150); plt.show()

xgb_pipe = fitted["XGBoost"]
yt, yp = np.expm1(yte), np.expm1(xgb_pipe.predict(Xte))
plt.figure(figsize=(4.8, 4.6)); plt.scatter(yt, yp, s=5, alpha=.35, color="#0071e3")
lim = [yt.min(), yt.max()]; plt.plot(lim, lim, "k-", lw=1); plt.xscale("log"); plt.yscale("log")
plt.xlabel("Actual price (Rs)"); plt.ylabel("Predicted (Rs)"); plt.title("XGBoost — test set")
plt.tight_layout(); plt.savefig("figures/m4_actual_vs_pred.png", dpi=150); plt.show()

# %% [markdown]
# ## Module 5 — Road-tax side check (kept small)
# Does knowing the state (and its road-tax rate) improve prices, and does the tax rate help on a state the model never saw?

# %%
def xgb_r2(cat, num, data=df):
    d = data.dropna(subset=num)
    Xa, ya = d[cat + num], np.log1p(d.price)
    a, b, c, e = train_test_split(Xa, ya, test_size=.2, random_state=42)
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), cat), ("num", "passthrough", num)])
    p = Pipeline([("pre", ct), ("m", XGBRegressor(n_estimators=500, learning_rate=.06, max_depth=6, random_state=42, n_jobs=-1))]).fit(a, c)
    return r2_score(np.expm1(e), np.expm1(p.predict(b))), mean_absolute_error(np.expm1(e), np.expm1(p.predict(b)))

taxed = df[df.road_tax_rate.notna()]
base = ["age", "log_km", "owners", "power"]
tax_tbl = pd.DataFrame([("No location", *xgb_r2(["brand", "model"], base, taxed)),
                        ("State", *xgb_r2(["brand", "model", "state"], base, taxed)),
                        ("State + road-tax rate", *xgb_r2(["brand", "model", "state"], base + ["road_tax_rate"], taxed))],
                       columns=["Setup", "R2", "MAE (Rs)"]).set_index("Setup")
display(tax_tbl.style.format({"R2": "{:.4f}", "MAE (Rs)": "{:,.0f}"})) if "display" in dir() else print(tax_tbl)

out = []
for s in taxed.state.value_counts().index:
    tr, te = taxed[taxed.state != s], taxed[taxed.state == s]
    for label, num in [("no tax", base), ("with tax", base + ["road_tax_rate"])]:
        ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["brand", "model"]), ("num", "passthrough", num)])
        p = Pipeline([("pre", ct), ("m", XGBRegressor(n_estimators=400, learning_rate=.06, max_depth=6, random_state=42, n_jobs=-1))])
        p.fit(tr[["brand", "model"] + num], np.log1p(tr.price))
        out.append([s, label, mean_absolute_error(te.price, np.expm1(p.predict(te[["brand", "model"] + num])))])
loso = pd.DataFrame(out, columns=["held-out state", "setup", "MAE"]).pivot(index="held-out state", columns="setup", values="MAE")
loso["tax helps?"] = np.where(loso["with tax"] < loso["no tax"], "yes", "no")
loso.round(0)

# %% [markdown]
# ## Module 6 — The 2026 test
# The model above learned from ads posted in 2020-21. Here it prices **every matching used-bike ad live on BikeWale in Nov 2026**
# (same official model names). Kaggle-era rupees are brought to 2026 with 5% a year. If the model "goes stale", the
# median of actual ÷ predicted will be far from 1 — and that ratio becomes the calibration factor the website uses.

# %%
u = pd.read_csv(f"{DATA}/bikewale_used_2026.csv")
u = u.rename(columns={"makeName": "brand", "priceNumeric": "price", "kmNumeric": "km", "ownersId": "owners", "cityName": "city"})
u["model_name"] = u.modelName.str.replace(r"\[.*?\]", "", regex=True).str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.strip()
u["model"] = u.brand + " " + u.model_name
u["age"] = 2026 - u.makeYear
u = u[(u.price > 5_000) & (u.km.between(100, 150_000)) & (u.age.between(0, 20))]
print(f"{len(u):,} usable 2026 ads, {u.model.nunique()} models, {u.city.nunique()} cities; EV share {np.mean(u.fuel == 'Electric'):.1%}")

trained_models = set(df.model)
t26 = u[(u.fuel == "Petrol") & u.model.isin(trained_models)].copy()
t26["state"] = t26.city.map(dict(zip(cs.city, cs.state))).fillna("Other")
t26["log_km"] = np.log1p(t26.km)
t26["power"] = t26.model.map(df.groupby("model").power.median())
t26["owners"] = t26.owners.clip(1, 4)
SCALE = 1.05 ** 5
t26["pred"] = np.expm1(xgb_pipe.predict(t26[CAT + NUM])) * SCALE
t26["ratio"] = t26.price / t26.pred
calib = float(t26.ratio.median())
by_age = t26.groupby(pd.cut(t26.age, [-1, 1, 3, 5, 8, 12, 20])).ratio.median()
print(f"{len(t26):,} 2026 petrol ads of models the model knows")
print(f"Before calibration: MAE Rs {mean_absolute_error(t26.price, t26.pred):,.0f}, R² {r2_score(t26.price, t26.pred):.3f}, median actual/pred = {calib:.3f}")
t26["pred_cal"] = t26.pred * calib
print(f"After  calibration: MAE Rs {mean_absolute_error(t26.price, t26.pred_cal):,.0f}, R² {r2_score(t26.price, t26.pred_cal):.3f}, "
      f"within ±20%: {np.mean(np.abs(t26.pred_cal / t26.price - 1) <= .2):.1%}")
by_age

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(t26.price, t26.pred_cal, s=4, alpha=.25, color="#0071e3"); l = [t26.price.min(), t26.price.max()]
ax[0].plot(l, l, "k-", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("2026 asking price (Rs)"); ax[0].set_ylabel("Predicted (calibrated)"); ax[0].set_title("2020-21 model on 2026 ads")
by_age.plot.bar(ax=ax[1], color="#30b07a"); ax[1].axhline(1, color="k", lw=1); ax[1].set_ylabel("median actual ÷ predicted")
ax[1].set_title("Is the old model too low? (by age)"); ax[1].set_xlabel("age (years)")
plt.tight_layout(); plt.savefig("figures/m6_2026_test.png", dpi=150); plt.show()
json.dump({"calibration": calib, "n": int(len(t26)), "by_age": {str(k): float(v) for k, v in by_age.items()}}, open("figures/m6_calibration.json", "w"), indent=1)

# %% [markdown]
# ## Module 7 — EV vs petrol depreciation (2026 ads, same variant)
# `log(price) ~ age + log(1+km) + EV×age + variant + city`, standard errors clustered by variant.
# The **variant** term (BikeWale version id) absorbs brand, engine, fuel and new price, so the age slope is how fast *the same variant* loses value.

# %%
import statsmodels.formula.api as smf
e = u[u.fuel.isin(["Petrol", "Electric"]) & (u.age <= 7)].copy()
e["ev"] = (e.fuel == "Electric").astype(int)
e["log_price"] = np.log(e.price); e["log_km"] = np.log1p(e.km)
e["ola"] = (e.brand.str.upper() == "OLA").astype(int)
e["scooter"] = (e.bodyStyleId.astype(str) == "5").astype(int)
vc = e.versionId.value_counts(); e = e[e.versionId.map(vc) >= 2]
cc_ = e.city.value_counts(); e["city2"] = np.where(e.city.map(cc_) >= 30, e.city, "other")

def ev_gap(d, label):
    m = smf.ols("log_price ~ age + log_km + ev:age + C(versionId) + C(city2)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.versionId)[0]})
    pet = (np.exp(m.params["age"]) - 1) * 100
    evr = (np.exp(m.params["age"] + m.params["ev:age"]) - 1) * 100
    return {"Check": label, "Ads": len(d), "EV ads": int(d.ev.sum()), "Petrol %/yr": round(pet, 1), "EV %/yr": round(evr, 1),
            "EV − petrol (pts)": round(evr - pet, 1), "p": round(m.pvalues["ev:age"], 3)}

ev_tbl = pd.DataFrame([ev_gap(e, "All two-wheelers, age ≤ 7"),
                       ev_gap(e[e.ola == 0], "Without Ola"),
                       ev_gap(e[(e.ev == 0) | (e.ola == 1)], "Ola only vs petrol"),
                       ev_gap(e[e.scooter == 1], "Scooters only")]).set_index("Check")
ev_tbl

# %%
kept = e.merge(cat[["brand", "model", "new_price"]].assign(model_name=cat.model), left_on=["brand", "model_name"], right_on=["brand", "model_name"], how="inner")
kept["kept"] = kept.price / kept.new_price
kept["group"] = np.where(kept.ev == 0, "Petrol", np.where(kept.ola == 1, "Ola", "EV (non-Ola)"))
curve = kept[kept.kept.between(.1, 1.5)].groupby(["group", "age"]).kept.median().unstack(0)
ax = curve.mul(100).plot(figsize=(6.5, 4), marker="o", color={"Petrol": "#0071e3", "EV (non-Ola)": "#30b07a", "Ola": "#ff7a1a"})
ax.set_ylabel("% of today's ex-showroom price"); ax.set_xlabel("age (years)"); ax.set_title("Value kept by age — 2026 BikeWale ads")
plt.tight_layout(); plt.savefig("figures/m7_value_kept.png", dpi=150); plt.show()

# %% [markdown]
# ## Module 8 — SHAP: what drives the price?

# %%
import shap
Z = xgb_pipe.named_steps["pre"].transform(Xte)
names = xgb_pipe.named_steps["pre"].get_feature_names_out()
explainer = shap.TreeExplainer(xgb_pipe.named_steps["m"])
sv = explainer.shap_values(Z)
groups = [next(c for c in CAT if n.startswith(f"cat__{c}_")) if n.startswith("cat__") else n.replace("num__", "") for n in names]
imp = pd.Series(np.abs(sv).mean(0), index=groups).groupby(level=0).sum().sort_values()
imp.rename({"power": "engine cc", "log_km": "kilometres", "model": "model", "brand": "brand"}).plot.barh(figsize=(6, 3.4), color="#0071e3")
plt.title("What drives used-bike prices (mean |SHAP|)"); plt.tight_layout(); plt.savefig("figures/m8_shap_importance.png", dpi=150); plt.show()

num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
shap.summary_plot(sv[:, num_idx], Z[:, num_idx], feature_names=[names[i].replace("num__", "") for i in num_idx], show=False)
plt.tight_layout(); plt.savefig("figures/m8_shap_beeswarm.png", dpi=150); plt.show()

# %% [markdown]
# ## Module 9 — Main models: trained on live 2026 listings (cars and two-wheelers)
# The models the website uses. Trained on 30,306 BikeWale two-wheeler ads and 91,748 CarWale car ads (Nov 2026).
# * **Features:** brand, model, exact variant, the variant's new price, fuel, body type, gearbox, state, city, seller type, age, km, owners
# * **Monotone rules:** the price never rises with age, km or owners, and never falls as the new price rises
# * **Model:** XGBoost with native categories; several boosted models with different random seeds, averaged
# * **Benchmarks:** a simple rule (new price × typical share kept at that age) and, for cars, **CarWale's own valuation** on the same ads
# * **Noise floor:** how far asking prices sit apart for the same variant, year and seller — roughly the best any model can do
#
# Takes about 10 minutes on Colab (the car data is large).

# %%
import xgboost as xgb
NOW = 2026
MCATS = ["brand", "model_key", "variant", "fuel", "body", "transmission", "state", "city", "seller"]
MNUMS = ["age", "log_km", "owners", "log_new", "log_new_model"]
MONO = {"age": -1, "log_km": -1, "owners": -1, "log_new": 1, "log_new_model": 1}
SIZE = {"bike": (5, 10), "car": (3, 8)}        # (models averaged, tree depth) -- same as the website
city_state_full = pd.read_csv(f"{DATA}/city_state_full.csv").drop_duplicates("city")
CITY2STATE = dict(zip(city_state_full.city, city_state_full.state))

def clean_model(s):
    return (s.astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())

def fuel_of(s):
    s = s.fillna("Petrol").astype(str).str.lower()   # hybrids first: "Hybrid (Electric + Petrol)" is not electric
    return np.select([s.str.contains("hybrid"), s.str.contains("electric"), s.str.contains("diesel"), s.str.contains("cng"),
                      s.str.contains("lpg")], ["Hybrid", "Electric", "Diesel", "CNG", "LPG"], "Petrol")

def load_market(kind):
    u = pd.read_csv(f"{DATA}/market_{kind}_ads.csv", low_memory=False)
    n0 = len(u)
    u = u.drop_duplicates("stockId").rename(columns={"makeName": "brand", "priceNumeric": "price", "kmNumeric": "km",
                                                     "ownersId": "owners", "cityName": "city"})
    u["model_key"] = u.brand + " | " + clean_model(u.modelName)
    u["variant"] = u.versionId.astype("Int64").astype(str)
    u["age"] = (NOW - u.makeYear).clip(lower=0)
    u["fuel"] = fuel_of(u.fuel)
    t = u.transmission.fillna("").astype(str).str.lower()
    u["transmission"] = np.where(t.str.contains("auto|amt|cvt|dct"), "Automatic", np.where(t == "", "NA", "Manual")) if kind == "car" else "NA"
    u["body"] = u.bodyStyleId.astype(str)
    if kind == "bike":
        u["body"] = np.where(u.body == "5", "scooter", "motorcycle")
    u["state"] = u.city.map(CITY2STATE).fillna("Other")
    u["seller"] = u.sellerType.map({1: "dealer", 2: "individual"}).fillna("unknown")
    u["owners"] = u.owners.fillna(1).clip(1, 4)
    if kind == "car":
        u["new_price"] = pd.to_numeric(u.onRoadPrice, errors="coerce").where(lambda x: x > 50_000)   # variant's new on-road price
        catm = pd.read_csv(f"{DATA}/car_catalog.csv")
    else:
        v = pd.read_csv(f"{DATA}/bike_variants.csv")
        u["new_price"] = u.versionId.map(dict(zip(v.versionId, v.ex_showroom)))                      # variant's ex-showroom price
        catm = pd.read_csv(f"{DATA}/bike_catalog.csv")
    u["new_price_model"] = u.model_key.map(dict(zip(catm.brand + " | " + catm.model, catm.new_price)))
    u["new_price"] = u.new_price.fillna(u.groupby("model_key").new_price.transform("median")).fillna(u.new_price_model)
    u = u[(u.price >= (40_000 if kind == "car" else 5_000)) & u.km.between(0, 300_000 if kind == "car" else 150_000) & (u.age <= 25)]
    r = u.price / u.new_price
    u = u[~((r > 1.6) | (r < 0.03))]                                                   # typos vs the new price
    u = u.drop_duplicates(["model_key", "variant", "makeYear", "km", "price", "city"])  # re-posted ads
    u["lp"] = np.log(u.price)
    u["ag"] = pd.cut(u.age, [-1, 1, 3, 5, 8, 12, 30])
    g = u.groupby(["model_key", "ag"], observed=True).lp
    q1, q3, n = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75)), g.transform("size")
    u = u[(n < 10) | u.lp.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]       # outliers per model + age group
    u["log_km"] = np.log1p(u.km); u["log_new"] = np.log(u.new_price); u["log_new_model"] = np.log(u.new_price_model)
    for col, k in [("variant", 3), ("city", 25), ("model_key", 3)]:                   # rare categories -> missing
        vc = u[col].value_counts(); u.loc[u[col].map(vc) < k, col] = np.nan
    print(f"{kind}: {n0:,} ads loaded -> {len(u):,} after cleaning, {u.model_key.nunique()} models, {int((u.fuel == 'Electric').sum()):,} electric")
    return u.reset_index(drop=True)

def mframe(df, cats):
    X = df[MCATS + MNUMS].copy()
    for c in MCATS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X

def mmetrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = np.abs(p / y - 1)
    return {"Ads": len(y), "Typical miss": np.median(e), "Within ±10%": np.mean(e <= .1), "Within ±20%": np.mean(e <= .2),
            "R2 (log)": 1 - np.sum((np.log(y) - np.log(p)) ** 2) / np.sum((np.log(y) - np.log(y).mean()) ** 2)}

def train_market(kind):
    u = load_market(kind)
    cats = {c: sorted(u[c].dropna().astype(str).unique().tolist()) for c in MCATS}
    tr, te = train_test_split(u, test_size=0.2, random_state=42)
    tr2, va = train_test_split(tr, test_size=0.15, random_state=7)
    seeds, depth = SIZE[kind]
    params = {"objective": "reg:squarederror", "tree_method": "hist", "max_cat_to_onehot": 1, "max_cat_threshold": 16,
              "eta": 0.03, "max_depth": depth, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "lambda": 3.0,
              "monotone_constraints": "(" + ",".join(str(MONO.get(f, 0)) for f in MCATS + MNUMS) + ")"}
    dtr2 = xgb.DMatrix(mframe(tr2, cats), tr2.lp, enable_categorical=True)
    dva = xgb.DMatrix(mframe(va, cats), va.lp, enable_categorical=True)
    rounds = [xgb.train({**params, "seed": s}, dtr2, 6000, evals=[(dva, "va")], early_stopping_rounds=200,
                        verbose_eval=False).best_iteration + 1 for s in range(seeds)]
    dtr = xgb.DMatrix(mframe(tr, cats), tr.lp, enable_categorical=True)
    boosters = [xgb.train({**params, "seed": s}, dtr, num_boost_round=r) for s, r in zip(range(seeds), rounds)]
    dte = xgb.DMatrix(mframe(te, cats), enable_categorical=True)
    pt = np.exp(np.mean([b.predict(dte) for b in boosters], axis=0))
    rows = {"All test ads": mmetrics(te.price, pt)}
    seg = te.fuel if kind == "car" else te.fuel + " " + te.body
    for s_ in sorted(seg.unique()):
        m = (seg == s_).values
        if m.sum() >= 30:
            rows[f"  {s_}"] = mmetrics(te.price[m], pt[m])
    kept = (tr.price / tr.new_price).groupby(tr.age).median()
    base = te.new_price * te.age.map(kept).fillna(kept.median()); ok = base.notna().values
    rows["Simple rule: new price × typical share kept"] = mmetrics(te.price[ok], base[ok])
    if kind == "car":
        vm = (te.valuationPrice > 0).values
        rows["CarWale's own valuation (same ads)"] = mmetrics(te.price[vm], te.valuationPrice[vm])
        rows["Our model on those same ads"] = mmetrics(te.price[vm], pt[vm])
    d = u.dropna(subset=["variant"]); g = d.groupby(["variant", "makeYear", "seller"]).price
    cell, n = g.transform("median"), g.transform("size")
    floor = float(np.median(np.abs(d.price[n >= 5] / cell[n >= 5] - 1)))
    out = pd.DataFrame(rows).T
    print(f"noise floor ({kind}): asking prices for the same variant, year and seller sit {floor:.1%} apart (median)")
    return out, floor, boosters, cats, te

market_results, floors = {}, {}
for kind in ("car", "bike"):
    market_results[kind], floors[kind], *_ = train_market(kind)
    display(market_results[kind].style.format({"Ads": "{:,.0f}", "Typical miss": "{:.1%}", "Within ±10%": "{:.1%}",
                                               "Within ±20%": "{:.1%}", "R2 (log)": "{:.3f}"}))

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"wspace": 0.9})
for a, kind in zip(ax, ("car", "bike")):
    r = market_results[kind]["Typical miss"].iloc[::-1] * 100
    colors = ["#0071e3" if i.startswith("All") or i.startswith("Our") else "#86868b" for i in r.index]
    r.plot.barh(ax=a, color=colors)
    a.axvline(floors[kind] * 100, color="#30b07a", ls="--", lw=1.2)
    a.text(floors[kind] * 100, -0.7, " noise floor", color="#30b07a", fontsize=9)
    a.set_title(("Cars" if kind == "car" else "Two-wheelers") + ": typical miss (lower is better)", fontsize=11)
    a.set_xlabel("% gap to the real asking price (median)")
plt.savefig("figures/m9_market_models.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Module 10 — Save tables for the report

# %%
results.round(4).to_csv("figures/m4_results.csv"); tax_tbl.round(4).to_csv("figures/m5_road_tax.csv")
loso.round(0).to_csv("figures/m5_loso.csv"); ev_tbl.to_csv("figures/m7_ev_vs_petrol.csv")
for kind in market_results:
    market_results[kind].round(4).to_csv(f"figures/m9_market_{kind}.csv")
print("Saved:", sorted(os.listdir("figures")))
if IN_COLAB:
    import shutil; shutil.make_archive("resaleiq_figures", "zip", "figures")
    from google.colab import files; files.download("resaleiq_figures.zip")
