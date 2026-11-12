"""How much more (or less) used ads ask in each state for the SAME variant and model year.

  python backend/state_premium.py   -> backend/models/state_premium.json

For every variant + year with 10+ ads, each ad's price is divided by that cell's all-India median.
The state's median of that ratio is its premium (Karnataka cars ~1.15 = 15% above the national median).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    cs = pd.read_csv(ROOT / "data" / "city_state.csv").drop_duplicates("city").set_index("city").state
    out = {}
    for kind, f, minp in (("bike", "bikewale_used_2026.csv", 5_000), ("car", "carwale_used_2026.csv", 40_000)):
        d = pd.read_csv(ROOT / "data" / "used_2026" / f, low_memory=False)
        d = d[d.priceNumeric >= minp].copy()
        d["state"] = d.cityName.map(cs)
        d["cell"] = d.versionId.astype(str) + "|" + d.makeYear.astype(str)
        g = d.groupby("cell").priceNumeric
        d = d[g.transform("size") >= 10]
        d["rel"] = d.priceNumeric / d.groupby("cell").priceNumeric.transform("median")
        st = d.groupby("state").rel.agg(["median", "count"])
        out[kind] = {s: {"ratio": float(r["median"]), "ads": int(r["count"])} for s, r in st.iterrows() if r["count"] >= 100}
        print(kind, {s: round(v["ratio"], 3) for s, v in sorted(out[kind].items(), key=lambda kv: kv[1]["ratio"])})
    (ROOT / "models" / "state_premium.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
