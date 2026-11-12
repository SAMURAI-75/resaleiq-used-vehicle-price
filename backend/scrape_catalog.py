"""Build the verified vehicle catalog from BikeWale and CarWale brand pages.

  python backend/scrape_catalog.py

Each brand page embeds its model list as JSON (<script id="initial-state">).
For every brand we keep:
  - models on sale: official name, ex-showroom price (avg, "onwards"), cc, bhp, body style, fuel, launch date
  - discontinued model names (no price)
Output: backend/data/catalog/{bikewale,carwale}_models.json  (+ the scrape date)
"""
import html
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "catalog"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}

SITES = {
    "bikewale": ("https://www.bikewale.com", "{}-bikes/", "bajaj"),
    "carwale": ("https://www.carwale.com", "{}-cars/", "maruti-suzuki"),
}


CAR_MAKES = ["Maruti Suzuki", "Hyundai", "Mahindra", "Tata", "Toyota", "Honda", "Ford", "Chevrolet", "Renault",
             "Volkswagen", "Skoda", "Nissan", "BMW", "Mercedes-Benz", "Audi", "Datsun", "Fiat", "Jeep", "Kia", "MG",
             "Land Rover", "Volvo", "Jaguar", "Mitsubishi", "Lexus", "Isuzu", "Force Motors", "Citroen", "Mini"]


def get_state(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        s = r.read().decode("utf-8", "ignore")
    m = re.search(r'id="initial-state"[^>]*>(.*?)</script>', s, re.S)
    return json.loads(html.unescape(m.group(1))) if m else None


def specs(m):
    return {x.get("itemName"): x.get("value") for x in (m.get("specsSummary") or [])}


def num(x):
    try:
        return float(re.sub(r"[^\d.]", "", str(x)))
    except ValueError:
        return None


def scrape(site):
    base, pattern, seed = SITES[site]
    first = get_state(f"{base}/{pattern.format(seed)}")
    if "allMakes" in first["makePage"]:
        makes = [(m["makeName"], m.get("maskingName") or m.get("makeMaskingName")) for m in first["makePage"]["allMakes"]]
    else:  # CarWale: brands that appear in the CarDekho training data (+ current big sellers)
        makes = [(n, n.lower().replace(" ", "-")) for n in CAR_MAKES]
    out = {"source": base, "scraped_on": date.today().isoformat(), "makes": {}}
    for name, mask in makes:
        try:
            st = first if mask == seed else get_state(f"{base}/{pattern.format(mask)}")
        except Exception as e:  # noqa: BLE001
            print("  skip", name, e)
            continue
        mp = (st or {}).get("makePage") or {}
        models = []
        for m in mp.get("models") or []:
            po = m.get("priceOverview") or {}
            sp = specs(m)
            models.append({
                "model": m.get("modelName"), "slug": m.get("modelMaskingName"),
                "price": po.get("averageExShowRoomPrice") or po.get("price") or None,
                "cc": num(sp.get("Displacement") or sp.get("Engine")),
                "bhp": num(sp.get("Max Power(bhp)") or sp.get("Max Power")),
                "kwh": num(sp.get("Battery Capacity")),
                "range_km": num(sp.get("Range") or sp.get("Riding Range")),
                "specs": sp, "body": m.get("bodyStyleId"), "fuel": m.get("fuelTypeId"),
                "electric": bool(m.get("isElectricVehicle")), "status": m.get("status"),
                "launched": m.get("launchedOn"), "root": m.get("rootName"),
            })
        disc = [d.get("modelName") for d in (mp.get("discontinuedModels") or [])]
        disc_root = sorted({d.get("rootName") for d in (mp.get("discontinuedModels") or []) if d.get("rootName")})
        out["makes"][name] = {"slug": mask, "models": models, "discontinued": disc, "discontinued_roots": disc_root}
        print(f"  {site} {name:22s} on sale {len(models):3d}  discontinued {len(disc):3d}")
        time.sleep(0.5)
    (OUT / f"{site}_models.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or SITES):
        scrape(s)
