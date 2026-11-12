"""Audit: do the years in used ads agree with the official production years? Every variant, both sites.

  python backend/tests/year_audit.py    -> backend/tests/year_audit.json + a summary by vehicle type

For each variant whose official years are known (from the site's own [YYYY-YYYY] names):
  - ads before the first year            -> impossible (typo or mis-filed)
  - ads after the last year + 1          -> impossible
  - official years with no ad at all     -> an ads-only range would have hidden them
For variants without official years, how many ads the typo rule drops.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from years import NOW, official_years, trimmed_ad_years  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def kind_of(site, r):
    f = str(r.fuel).lower()
    fuel = "electric" if ("electric" in f and "hybrid" not in f) else "hybrid" if "hybrid" in f else \
        "diesel" if "diesel" in f else "cng" if "cng" in f else "petrol"
    if site == "bikewale":
        return f"two-wheeler {fuel} {'scooter' if str(r.bodyStyleId) == '5' else 'motorcycle'}"
    return f"car {fuel}"


def audit(site):
    d = pd.read_csv(ROOT / "data" / "used_2026" / f"{site}_used_2026.csv", low_memory=False)
    d = d[d.makeYear.between(1980, NOW)]
    d["type"] = [kind_of(site, r) for r in d.itertuples()]
    rows = []
    for (vid, vname, mname), g in d.groupby(["versionId", "versionName", "modelName"]):
        counts = g.makeYear.value_counts().to_dict()
        off = official_years(vname, mname)
        r = {"site": site, "type": g["type"].mode().iat[0], "brand": g.makeName.iat[0], "model": mname, "variant": vname,
             "ads": int(len(g)), "ad_years": [int(min(counts)), int(max(counts))]}
        if off:
            y1, y2, src = off
            early = int(sum(n for y, n in counts.items() if y < y1))
            late = int(sum(n for y, n in counts.items() if y > y2 + 1))
            missing = [y for y in range(y1, min(y2, NOW) + 1) if y not in counts]
            r.update({"official": [y1, y2], "source": src, "ads_before": early, "ads_after": late, "years_without_ads": len(missing)})
        else:
            t1, t2 = trimmed_ad_years(counts)
            r.update({"official": None, "trimmed": [t1, t2], "typo_ads_dropped": int(sum(n for y, n in counts.items() if y < t1 or y > t2))})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = pd.concat([audit("bikewale"), audit("carwale")], ignore_index=True)
    off = df[df.official.notna()]
    summary = []
    for t, g in df.groupby("type"):
        o = g[g.official.notna()]
        summary.append({
            "type": t, "variants": int(len(g)), "ads": int(g.ads.sum()),
            "variants_with_official_years": f"{len(o) / len(g):.0%}",
            "ads_outside_official_years": f"{(o.ads_before.sum() + o.ads_after.sum()) / max(o.ads.sum(), 1):.2%}",
            "variants_with_an_impossible_ad": f"{((o.ads_before + o.ads_after) > 0).mean():.1%}" if len(o) else "-",
            "official_years_missing_from_ads": f"{o.years_without_ads.sum() / max((o.official.str[1] - o.official.str[0] + 1).clip(upper=99).sum(), 1):.0%}" if len(o) else "-",
            "typo_ads_dropped_(no_official_years)": int(g.typo_ads_dropped.fillna(0).sum()),
        })
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    worst = off.assign(bad=off.ads_before + off.ads_after).sort_values("bad", ascending=False).head(15)
    print("\nMost impossible ads (official years vs ad years):")
    for r in worst.itertuples():
        print(f"  {r.site:8s} {r.brand} {r.model} | {r.variant} | official {r.official} | ads {r.ad_years} | before {r.ads_before}, after {r.ads_after} of {r.ads}")
    (Path(__file__).parent / "year_audit.json").write_text(json.dumps({"summary": summary, "worst": worst.drop(columns="bad").to_dict("records")}, indent=1, default=str))


if __name__ == "__main__":
    main()
