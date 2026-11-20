"""Bundle the small data files the Colab notebook needs -> colab/resaleiq_colab_data.zip

  python colab/make_colab_data.py
"""
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from train import CITY_STATE  # noqa: E402

CAT = ROOT / "backend" / "data" / "catalog"
OUT = ROOT / "colab" / "data"
OUT.mkdir(exist_ok=True)

pd.DataFrame(sorted(CITY_STATE.items()), columns=["city", "state"]).to_csv(OUT / "city_state.csv", index=False)
pd.read_csv(CAT / "name_map_bike.csv").to_csv(OUT / "name_map_bike.csv", index=False)
cat = json.loads((CAT / "catalog.json").read_text())["bike"]
rows = [{"brand": b, "model": m["model"], "type": m.get("type"), "fuel": m.get("fuel"), "cc": m.get("cc"),
         "new_price": m.get("new_price"), "on_sale": m.get("on_sale")} for b, ms in cat.items() for m in ms]
pd.DataFrame(rows).to_csv(OUT / "bike_catalog.csv", index=False)
# Modules 6-7 (2026 test, EV study): the 12 Nov BikeWale collection the report's EV numbers come from
used_v1 = ROOT / "backend" / "data" / "used_2026" / "bikewale_used_2026_v1_nov12.csv"
pd.read_csv(used_v1).to_csv(OUT / "bikewale_used_2026.csv", index=False)
# Module 9 (main models): the 13 Nov collections with seller type, + catalogues and the full city -> state map
pd.read_csv(ROOT / "backend" / "data" / "used_2026" / "bikewale_used_2026.csv").to_csv(OUT / "market_bike_ads.csv", index=False)
pd.read_csv(ROOT / "backend" / "data" / "used_2026" / "carwale_used_2026.csv", low_memory=False).to_csv(OUT / "market_car_ads.csv", index=False)
pd.read_csv(ROOT / "backend" / "data" / "catalog" / "bike_variants.csv").to_csv(OUT / "bike_variants.csv", index=False)
pd.read_csv(ROOT / "backend" / "data" / "city_state.csv").to_csv(OUT / "city_state_full.csv", index=False)
car = json.loads((CAT / "catalog.json").read_text())["car"]
pd.DataFrame([{"brand": b, "model": x["model"], "new_price": x.get("new_price")} for b, xs in car.items() for x in xs]).to_csv(OUT / "car_catalog.csv", index=False)

with zipfile.ZipFile(ROOT / "colab" / "resaleiq_colab_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob("*.csv")):
        z.write(f, f.name)
    z.write(ROOT / "backend" / "tax_history.py", "tax_history.py")          # Module 9 / 9b: India tax history rules
    z.write(ROOT / "backend" / "data" / "tax_history.json", "tax_history.json")
print("wrote", ROOT / "colab" / "resaleiq_colab_data.zip", [f.name for f in OUT.glob("*.csv")])
