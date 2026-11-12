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
used = ROOT / "backend" / "data" / "used_2026" / "bikewale_used_2026.csv"
pd.read_csv(used).to_csv(OUT / "bikewale_used_2026.csv", index=False)

with zipfile.ZipFile(ROOT / "colab" / "resaleiq_colab_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob("*.csv")):
        z.write(f, f.name)
print("wrote", ROOT / "colab" / "resaleiq_colab_data.zip", [f.name for f in OUT.glob("*.csv")])
