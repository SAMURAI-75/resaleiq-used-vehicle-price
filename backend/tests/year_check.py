"""Year test for every vehicle (server running on localhost:8000).

  python backend/tests/year_check.py

1. Coverage   -- for every 2026 ad (both sites), is its year selectable for its model? for its variant?
2. Consistency -- each variant's years sit inside its model's list; every list is non-empty and within 1980-2026.
3. Spot check -- the first year offered vs the Indian launch year of well-known models. These launch years are
   general knowledge written down by hand (not scraped), so this is a sanity check, not a source of truth.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8000"
NOW = 2026
LAUNCH = {  # (kind, brand, model): Indian launch year of this name/generation line
    ("2w", "Honda", "Activa"): 2001, ("2w", "Bajaj", "Pulsar NS200"): 2012, ("2w", "Royal Enfield", "Classic 350"): 2009,
    ("2w", "Royal Enfield", "Himalayan"): 2016, ("2w", "Royal Enfield", "Meteor 350"): 2020, ("2w", "Royal Enfield", "Hunter 350"): 2022,
    ("2w", "Royal Enfield", "Guerrilla 450"): 2024, ("2w", "Bajaj", "Pulsar NS400Z"): 2024, ("2w", "Ather", "450X"): 2020,
    ("2w", "OLA", "S1 Pro"): 2021, ("2w", "TVS", "iQube"): 2020, ("2w", "Bajaj", "Chetak"): 2020, ("2w", "TVS", "Raider 125"): 2021,
    ("2w", "Yamaha", "R15 V4"): 2021, ("2w", "KTM", "390 Duke"): 2013,
    ("car", "Hyundai", "Creta"): 2015, ("car", "Kia", "Seltos"): 2019, ("car", "Tata", "Nexon"): 2017, ("car", "Tata", "Nexon EV"): 2020,
    ("car", "Maruti Suzuki", "Brezza"): 2022, ("car", "Maruti Suzuki", "Fronx"): 2023, ("car", "Tata", "Punch"): 2021,
    ("car", "Mahindra", "XUV700"): 2021, ("car", "Kia", "Sonet"): 2020, ("car", "MG", "Hector"): 2019, ("car", "Hyundai", "Venue"): 2019,
    ("car", "Mahindra", "Thar"): 2010,
}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def clean(s):
    return (s.astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.replace(r"^(Old Generation|New|Old)\s+", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())


def main():
    meta = get("/api/meta")
    idx = {(k, b, m["model"]): m for k in ("2w", "car") for b, ms in meta[k]["brands"].items() for m in ms}
    report, fails = {}, []

    # 2. consistency
    bad_lists = [k for k, m in idx.items() if not m.get("year_list") or min(m["year_list"]) < 1980 or max(m["year_list"]) > NOW]
    var_outside, n_var = [], 0
    variants = {}
    for (k, b, mo), m in idx.items():
        if m["engine"] != "market":
            continue
        vs = get("/api/variants?" + urllib.parse.urlencode({"type": k, "brand": b, "model": mo}))
        variants[(k, b, mo)] = vs
        ys = set(m["year_list"])
        for v in vs:
            n_var += 1
            if not set(range(v["years"][0], v["years"][1] + 1)) <= ys:
                var_outside.append((k, b, mo, v["name"], v["years"]))
    report["models"] = len(idx)
    report["models_with_bad_year_list"] = len(bad_lists)
    report["variants"] = n_var
    report["variants_outside_model_years"] = len(var_outside)

    # 1. coverage of real 2026 ads
    for kind, site in (("2w", "bikewale"), ("car", "carwale")):
        d = pd.read_csv(ROOT / "data" / "used_2026" / f"{site}_used_2026.csv", low_memory=False)
        d["model"] = clean(d.modelName)
        d = d[d.makeYear.between(1980, NOW)]
        model_ok = var_ok = var_n = 0
        known = 0
        vmap = {(b, mo): {i: v for v in vs for i in [v["id"]]} for (k, b, mo), vs in variants.items() if k == kind}
        for r in d.itertuples():
            m = idx.get((kind, r.makeName, r.model))
            if not m:
                continue
            known += 1
            model_ok += r.makeYear in m["year_list"]
            v = vmap.get((r.makeName, r.model), {}).get(str(r.versionId))
            if v:
                var_n += 1
                var_ok += v["years"][0] <= r.makeYear <= v["years"][1]
        report[f"{kind}_ads_checked"] = known
        report[f"{kind}_ads_year_offered_for_model"] = f"{model_ok / known:.2%}"
        report[f"{kind}_ads_year_offered_for_variant"] = f"{var_ok / max(var_n, 1):.2%} of {var_n:,}"

    # 3. spot check against known launch years
    spots = []
    for key, launch in LAUNCH.items():
        m = idx.get(key)
        if not m:
            spots.append((key, launch, None, "model not in list")); continue
        first = min(m["year_list"])
        verdict = "ok" if abs(first - launch) <= 1 else ("EARLIER than launch" if first < launch else "later than launch")
        spots.append((key, launch, first, verdict))
    report["spot_checks_ok"] = f"{sum(s[3] == 'ok' for s in spots)}/{len(spots)}"
    print(json.dumps(report, indent=1))
    for s in spots:
        if s[3] != "ok":
            print("  spot:", s)
    for v in var_outside[:10]:
        print("  variant outside model years:", v)
    (Path(__file__).parent / "year_check.json").write_text(json.dumps({"report": report, "spots": spots,
                                                                       "variants_outside": var_outside[:200]}, indent=1, default=str))


if __name__ == "__main__":
    main()
