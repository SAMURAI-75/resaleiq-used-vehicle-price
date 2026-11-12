"""Collect every used-vehicle ad on BikeWale (two-wheelers) or CarWale (cars), all India
-> backend/data/used_2026/{site}_used_2026.csv

  python backend/scrape_used_2026.py bikewale
  python backend/scrape_used_2026.py carwale

Both sites serve their used-search page from a JSON endpoint, 24 ads per request, paged by offset.
The all-India list stops after ~1,450 pages, so we crawl city by city (CarWale's city list, 1,568 cities;
the two sites share city ids). 6 requests in flight, each worker pausing 0.3 s. Seller contact details
are not stored. Ads repeated across cities ("nearby" listings) are de-duplicated by stock id.
"""
import csv
import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "used_2026"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
SITES = {
    "bikewale": "https://www.bikewale.com/api/stocks/?vehicleType=1&city={city}&pn={pn}&ps=24&sc=-1&so=-1&lcr={off}&stockfetched={off}&shouldfetchnearbycars=False",
    "carwale": "https://www.carwale.com/api/stocks/?city={city}&pn={pn}&ps=24&sc=-1&so=-1&lcr={off}&stockfetched={off}&shouldfetchnearbycars=False",
}
FIELDS = ["stockId", "makeName", "modelName", "rootName", "versionName", "versionId", "makeYear", "priceNumeric",
          "kmNumeric", "ownersId", "fuel", "cityName", "cityId", "bodyStyleId", "transmission",
          # extras: variant's new on-road price (CarWale), the site's own valuation (benchmark only, never a feature),
          # seller type (1 dealer, 2 individual), trim, condition, certification, seats
          "onRoadPrice", "valuationPrice", "sellerType", "trimName", "isTopVariant", "overAllCondition",
          "certificationScore", "seatingCapacity", "isNearbyCityListing"]
CAP_PAGES = 1400
lock = threading.Lock()


def get(site, city, pn):
    url = SITES[site].format(city=city, pn=pn, off=24 * (pn - 1))
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            time.sleep(0.3)
            return d
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * attempt)
    return {"stocks": [], "totalCount": 0}


def main(site):
    cities = [c["CityId"] for c in json.loads((ROOT / "data" / "carwale_cities.json").read_text())]
    rows, seen = [], set()

    def keep(stocks):
        with lock:
            for s in stocks:
                if s.get("isNearbyCityListing") or s.get("isSimilarStockListing"):
                    continue
                sid = s.get("stockId")
                if sid in seen:
                    continue
                seen.add(sid)
                rows.append({k: s.get(k) for k in FIELDS})

    def first(city):
        d = get(site, city, 1)
        keep(d.get("stocks") or [])
        return city, int(d.get("totalCount") or 0)

    with ThreadPoolExecutor(6) as ex:
        totals = [t for t in ex.map(first, cities) if t[1] > 24]
    print(f"  {site}: {len(totals)} cities with >24 ads, {len(rows):,} ads from first pages", flush=True)
    jobs = [(c, pn) for c, n in totals for pn in range(2, min(CAP_PAGES, math.ceil(n / 24)) + 1)]
    print(f"  {len(jobs):,} more pages to fetch", flush=True)
    done = 0

    def page(job):
        nonlocal done
        keep(get(site, *job).get("stocks") or [])
        with lock:
            done += 1
            if done % 500 == 0:
                print(f"  {done:,}/{len(jobs):,} pages, {len(rows):,} ads", flush=True)

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(page, jobs))
    path = OUT / f"{site}_used_2026.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["scraped_on"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "scraped_on": date.today().isoformat()})
    print(f"saved {len(rows):,} ads -> {path}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "bikewale")
