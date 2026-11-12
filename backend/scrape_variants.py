"""Variant-level new prices for every two-wheeler model on sale -> backend/data/catalog/bike_variants.csv

  python backend/scrape_variants.py

Used-bike ads carry a version id but no new price; the brand page gives only a model's "onwards" price.
Each BikeWale model page embeds its versions (id, name, ex-showroom price), so an ad's exact variant can be
priced new. One request per model, 0.5 s apart.
"""
import csv
import html
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}


def versions(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        s = r.read().decode("utf-8", "ignore")
    k = s.find('"versions":[{"versionId"')
    if k < 0:
        return []
    arr, _ = json.JSONDecoder().raw_decode(html.unescape(s[k + len('"versions":'):]))
    return arr


def main():
    bw = json.loads((ROOT / "data" / "catalog" / "bikewale_models.json").read_text())
    rows = []
    for brand, e in bw["makes"].items():
        for m in e["models"]:
            if m.get("status") != 2 or not m.get("slug"):
                continue
            url = f"https://www.bikewale.com/{e['slug']}-bikes/{m['slug']}/"
            try:
                for v in versions(url):
                    po = v.get("priceOverview") or {}
                    p = po.get("averageExShowRoomPrice") or po.get("price") or po.get("exShowRoomPrice")
                    if p:
                        rows.append({"versionId": v["versionId"], "brand": brand, "model": m["model"],
                                     "versionName": v.get("versionName"), "ex_showroom": p})
            except Exception as ex:  # noqa: BLE001
                print("  skip", brand, m["model"], ex)
            time.sleep(0.5)
        print(f"  {brand}: {len(rows)} variants so far")
    out = ROOT / "data" / "catalog" / "bike_variants.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["versionId", "brand", "model", "versionName", "ex_showroom"])
        w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} variants -> {out}")


if __name__ == "__main__":
    main()
