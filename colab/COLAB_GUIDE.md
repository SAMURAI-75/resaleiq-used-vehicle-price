# Colab Guide — what's left in `bike_price_project`

Done already: Modules 1 (load), 2 (clean + city→state), 3 (road-tax table), 6 (EV v2, in a browser tab).
Left: **4 baselines → 5 road-tax check → 6 import → 7 SHAP → 8 report figures.** Run each block as its own cell.
Column names below assume `bikes_clean_with_prices_and_tax.csv`; if yours differ (e.g. the tax column), rename at the top.

---
## Module 4 — Baseline models + metrics (≈20 min)
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("bikes_clean_with_prices_and_tax.csv")
df["owners"] = df["owner"].map({"First Owner":1,"Second Owner":2,"Third Owner":3,"Fourth Owner Or More":4})
df["log_km"] = np.log1p(df["kms_driven"])
TAX = "road_tax_rate"            # <- rename if your column is called something else

CAT = ["brand", "family", "state"]
NUM = ["age", "log_km", "owners", "power", TAX]
X, y = df[CAT + NUM], np.log1p(df["price"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def pre():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), CAT),
                              ("num", "passthrough", NUM)])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=6, subsample=0.85,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1),
}
rows, fitted = [], {}
for name, m in models.items():
    p = Pipeline([("pre", pre()), ("m", m)]).fit(Xtr, ytr)
    yt, yp = np.expm1(yte), np.expm1(p.predict(Xte))          # back to rupees before scoring
    rows.append([name, mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp)), r2_score(yt, yp)])
    fitted[name] = p
results = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).round(3)
results
```
Expect R² ≈ 0.87–0.93 (website copy got LR 0.865, RF 0.894, XGB 0.932).
**Viva line:** "We report MAE/RMSE/R², not 'accuracy' like the base paper, because price is continuous."

Optional — base-paper replication row (shows we beat it on the same data):
```python
from sklearn.tree import DecisionTreeRegressor
p = Pipeline([("pre", pre()), ("m", DecisionTreeRegressor(random_state=42))]).fit(Xtr, ytr)
yt, yp = np.expm1(yte), np.expm1(p.predict(Xte))
print("Decision Tree (base paper) R2:", r2_score(yt, yp))
```

---
## Module 5 — Road-tax side check (≈20 min, keep it small)
```python
def r2_for(cat, num):
    Xa = df[cat + num]; Xtr_, Xte_, ytr_, yte_ = train_test_split(Xa, y, test_size=0.2, random_state=42)
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False), cat),
                            ("num", "passthrough", num)])
    p = Pipeline([("pre", ct), ("m", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42))]).fit(Xtr_, ytr_)
    return r2_score(np.expm1(yte_), np.expm1(p.predict(Xte_)))

base_num = ["age", "log_km", "owners", "power"]
table = pd.DataFrame({
    "Setup": ["No location", "State", "State + road-tax rate"],
    "R2": [r2_for(["brand", "family"], base_num),
           r2_for(["brand", "family", "state"], base_num),
           r2_for(["brand", "family", "state"], base_num + [TAX])]}).round(4)
table
```
Leave-one-state-out (does the tax rate help on a state the model never saw?):
```python
out = []
for s in df.state.value_counts().index[:11]:
    tr, te = df[df.state != s], df[df.state == s]
    for label, num in [("no tax", base_num), ("with tax", base_num + [TAX])]:
        ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["brand", "family"]),
                                ("num", "passthrough", num)])
        p = Pipeline([("pre", ct), ("m", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42))])
        p.fit(tr[["brand","family"] + num], np.log1p(tr.price))
        out.append([s, label, mean_absolute_error(te.price, np.expm1(p.predict(te[["brand","family"] + num])))])
loso = pd.DataFrame(out, columns=["held_out_state", "setup", "MAE"]).pivot(index="held_out_state", columns="setup", values="MAE").round(0)
loso
```
Report: the Classic 350 table (Spearman 0.78, ≈₹1,427 per +1% tax) + these two small tables. One paragraph. Done.

---
## Module 6 — Bring in the 2026 EV data (≈10 min)
1. In the BikeWale browser tab, run the download step of `module6_v2_browser.js` → save `two_wheelers_2026_same_variant.csv`.
2. Upload to Colab (Files pane) and re-run the regression so the notebook has it:
```python
import statsmodels.formula.api as smf
ev = pd.read_csv("two_wheelers_2026_same_variant.csv")
d = ev[ev.age <= 7].copy()
d["log_price"] = np.log(d.price); d["log_km"] = np.log1p(d.km); d["is_ev"] = (d.fuel == "Electric").astype(int)
m = smf.ols("log_price ~ age + log_km + is_ev:age + C(variant_id) + C(state)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.variant_id})
print(m.params[["age", "is_ev:age"]], m.pvalues[["age", "is_ev:age"]])
# per-year % change = (exp(coef) - 1) * 100 ; repeat with d[d.brand != "Ola"]
```
(Adjust `price, km, fuel, variant_id, state, brand` to your CSV's column names.)

---
## Module 7 — SHAP (≈10 min)
```python
import shap
xgb_pipe = fitted["XGBoost"]
Z = xgb_pipe.named_steps["pre"].transform(Xte)
names = xgb_pipe.named_steps["pre"].get_feature_names_out()
explainer = shap.TreeExplainer(xgb_pipe.named_steps["m"])
sv = explainer.shap_values(Z)
shap.summary_plot(sv, Z, feature_names=names, max_display=15)       # beeswarm for the report
shap.plots.waterfall(shap.Explanation(sv[0], explainer.expected_value, Z[0], feature_names=names))  # one bike
```
Grouped importance (one bar per original feature — cleaner for slides):
```python
groups = [next(c for c in CAT if n.startswith(f"cat__{c}_")) if n.startswith("cat__") else n.replace("num__", "") for n in names]
pd.Series(np.abs(sv).mean(0), index=groups).groupby(level=0).sum().sort_values().plot.barh()
```

---
## Module 8 — Report figures (≈20 min)
```python
import matplotlib.pyplot as plt
# 1. Actual vs predicted
yp = np.expm1(fitted["XGBoost"].predict(Xte)); yt = np.expm1(yte)
plt.figure(figsize=(5,5)); plt.scatter(yt, yp, s=5, alpha=.4); plt.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "k-")
plt.xscale("log"); plt.yscale("log"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("XGBoost, test set"); plt.show()
# 2. Model comparison bar
results.plot.bar(x="Model", y="R2", legend=False, ylim=(0,1)); plt.ylabel("R²"); plt.show()
# 3. Residuals by age
pd.DataFrame({"age": Xte.age, "err%": (yp - yt) / yt * 100}).groupby("age")["err%"].median().plot(); plt.axhline(0, c="k"); plt.show()
```
Ready-made versions of figures 1, 2 and the SHAP bar are already in `docs/figures/` (from the website pipeline).
Flowchart for the report: use the pipeline diagram in `docs/ROADMAP.md` (draw it in draw.io / Excalidraw in 5 minutes).
