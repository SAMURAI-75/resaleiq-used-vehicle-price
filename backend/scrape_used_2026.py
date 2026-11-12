"""Collect every used two-wheeler ad on BikeWale (all India) -> backend/data/used_2026/bikewale_used_2026.csv

  python backend/scrape_used_2026.py

Uses the same JSON endpoint the BikeWale used-bike search page calls (vehicleType=1 = two-wheelers).
Fields kept: stock id, make, model, variant, version id, year, asking price, km, owners, fuel, city, body style.
Polite: one request every 0.4 s. Seller contact details are not stored.
"""
import csv
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "used_2026"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
API = ("https://www.bikewale.com/api/stocks/?vehicleType=1&pn={pn}&ps=24&sc=-1&so=-1"
       "&lcr={off}&stockfetched={off}&shouldfetchnearbycars=False")   # the site pages by offset, 24 at a time
FIELDS = ["stockId", "makeName", "modelName", "rootName", "versionName", "versionId", "makeYear", "priceNumeric",
          "kmNumeric", "ownersId", "fuel", "cityName", "bodyStyleId", "transmission"]


def get(pn):
    req = urllib.request.Request(API.format(pn=pn, off=24 * (pn - 1)), headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    seen, rows, pn, total = set(), [], 1, None
    empty = 0
    while True:
        try:
            d = get(pn)
        except Exception as e:  # noqa: BLE001
            print("  retry page", pn, e); time.sleep(3)
            try:
                d = get(pn)
            except Exception:
                pn += 1; continue
        total = total or d.get("totalCount")
        stocks = d.get("stocks") or []
        new = 0
        for s in stocks:
            if s.get("stockId") in seen:
                continue
            seen.add(s.get("stockId")); new += 1
            rows.append({k: s.get(k) for k in FIELDS})
        if pn % 100 == 0:
            print(f"  page {pn}: {len(rows):,} / {total:,}")
        empty = empty + 1 if not stocks or new == 0 else 0
        if empty >= 3 or (total and len(rows) >= total):
            break
        pn += 1
        time.sleep(0.4)
    path = OUT / "bikewale_used_2026.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["scraped_on"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "scraped_on": date.today().isoformat()})
    print(f"saved {len(rows):,} ads (site total {total:,}) -> {path}")


if __name__ == "__main__":
    main()
