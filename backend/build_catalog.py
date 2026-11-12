"""Build the verified vehicle catalog the website shows.

  python backend/build_catalog.py        (after scrape_catalog.py)

1. Official model names come from BikeWale / CarWale (on-sale + discontinued).
2. Every Kaggle listing name is matched to one official name
   ("Bajaj Pulsar 200 NS 200cc" -> "Pulsar NS200"). Hand overrides fix the rest.
3. Output backend/data/catalog/catalog.json + name_map_{bike,car}.csv (for review / the report).

A model appears in the website's dropdown only if it is
  (a) on sale today (official 2026 ex-showroom price known), or
  (b) discontinued but present in the training data (>= MIN_ROWS listings),
so every choice can actually be priced.
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "data" / "catalog"
MIN_ROWS = 3

# Kaggle brand -> BikeWale/CarWale brand
BIKE_BRAND = {"Royal Enfield": "Royal Enfield", "Bajaj": "Bajaj", "Hero": "Hero", "Yamaha": "Yamaha", "Honda": "Honda",
              "Suzuki": "Suzuki", "TVS": "TVS", "KTM": "KTM", "Harley-Davidson": "Harley-Davidson", "Kawasaki": "Kawasaki",
              "Benelli": "Benelli", "Triumph": "Triumph", "Ducati": "Ducati", "BMW": "BMW", "Jawa": "Jawa",
              "Indian": "Indian", "Yezdi": "Yezdi"}
CAR_BRAND = {"Maruti": "Maruti Suzuki", "Hyundai": "Hyundai", "Mahindra": "Mahindra", "Tata": "Tata", "Toyota": "Toyota",
             "Honda": "Honda", "Ford": "Ford", "Chevrolet": "Chevrolet", "Renault": "Renault", "Volkswagen": "Volkswagen",
             "Skoda": "Skoda", "Nissan": "Nissan", "BMW": "BMW", "Mercedes-Benz": "Mercedes-Benz", "Audi": "Audi",
             "Datsun": "Datsun", "Fiat": "Fiat", "Jeep": "Jeep", "Kia": "Kia", "MG": "MG", "Land": "Land Rover",
             "Volvo": "Volvo", "Jaguar": "Jaguar", "Mitsubishi": "Mitsubishi", "Lexus": "Lexus", "Isuzu": "Isuzu",
             "Force": "Force Motors"}

# hand overrides: (brand, regex on Kaggle name) -> official name. Checked against name_map_*.csv.
BIKE_OVERRIDES = [
    ("Bajaj", r"pulsar\s*200\s*ns|pulsar ns\s*200", "Pulsar NS200"),
    ("Bajaj", r"pulsar rs\s*200", "Pulsar RS 200"),
    ("Bajaj", r"pulsar 135", "Pulsar 135 LS"),
    ("Bajaj", r"pulsar 220", "Pulsar 220F"),
    ("Bajaj", r"avenger cruise", "Avenger Cruise 220"),
    ("Bajaj", r"avenger street 150", "Avenger Street 150"),
    ("Bajaj", r"avenger street 160", "Avenger Street 160"),
    ("Bajaj", r"avenger street 180", "Avenger Street 180"),
    ("Bajaj", r"avenger street 220", "Avenger Street 220"),
    ("Bajaj", r"avenger 2[02]0", "Avenger"),
    ("Bajaj", r"ct\s*100", "CT 100"),
    ("Bajaj", r"platina", "Platina 100"),
    ("Bajaj", r"discover 100", "Discover 100"),
    ("Bajaj", r"discover 110", "Discover 110"),
    ("Bajaj", r"discover 125", "Discover 125"),
    ("Bajaj", r"discover 135", "Discover 135"),
    ("Bajaj", r"discover 150", "Discover 150"),
    ("Bajaj", r"dominar 400", "Dominar 400"),
    ("Royal Enfield", r"classic.*500", "Classic 500"),
    ("Royal Enfield", r"classic.*350", "Classic 350"),
    ("Royal Enfield", r"thunderbird x.*500", "Thunderbird 500X"),
    ("Royal Enfield", r"thunderbird x.*350", "Thunderbird 350X"),
    ("Royal Enfield", r"thunderbird.*500", "Thunderbird 500"),
    ("Royal Enfield", r"thunderbird.*350", "Thunderbird 350"),
    ("Royal Enfield", r"(bullet )?electra", "Electra"),
    ("Royal Enfield", r"(bullet|standard|machismo).*500", "Bullet 500"),
    ("Royal Enfield", r"(bullet|standard|machismo).*350", "Bullet 350"),
    ("Royal Enfield", r"himalayan", "Himalayan"),
    ("Royal Enfield", r"continental gt 535", "Continental GT"),
    ("Royal Enfield", r"continental gt 650", "Continental GT 650"),
    ("Royal Enfield", r"interceptor", "Interceptor 650"),
    ("Royal Enfield", r"meteor", "Meteor 350"),
    ("Hero", r"splendor.*pro", "Splendor PRO"),
    ("Hero", r"super splendor", "Super Splendor"),
    ("Hero", r"splendor.*ismart", "Splendor iSmart"),
    ("Hero", r"splendor.*nxg", "Splendor NXG"),
    ("Hero", r"splendor", "Splendor Plus"),
    ("Hero", r"passion\s*x\s*pro", "Passion XPro"),
    ("Hero", r"passion pro", "Passion Pro"),
    ("Hero", r"passion", "Passion Plus"),
    ("Hero", r"hf deluxe|hf dawn", "HF Deluxe"),
    ("Hero", r"cd deluxe", "CD Deluxe"),
    ("Hero", r"cd dawn", "CD Dawn"),
    ("Hero", r"karizma zmr", "Karizma ZMR"),
    ("Hero", r"karizma", "Karizma"),
    ("Hero", r"cbz xtreme|xtreme 150", "Xtreme"),
    ("Hero", r"xtreme 200r", "Xtreme 200R"),
    ("Hero", r"xtreme 200s", "Xtreme 200S"),
    ("Hero", r"xpulse 200t", "Xpulse 200T"),
    ("Hero", r"xpulse", "Xpulse 200"),
    ("Hero", r"hunk", "Hunk"),
    ("Hero", r"ignitor", "Ignitor"),
    ("Hero", r"achiever", "Achiever 150"),
    ("Hero", r"glamour", "Glamour"),
    ("Hero", r"cbz", "CBZ"),
    ("Honda", r"shine\s*sp|sp\s*125|sp125", "SP 125"),
    ("Honda", r"shine", "Shine"),
    ("Honda", r"unicorn 160|unicorn.*160", "CB Unicorn 160"),
    ("Honda", r"unicorn", "CB Unicorn 150"),
    ("Honda", r"hornet 160", "CB Hornet 160R"),
    ("Honda", r"cbr 150|cbr150", "CBR150 R"),
    ("Honda", r"cbr 250|cbr250", "CBR250R"),
    ("Honda", r"stunner", "CBF Stunner"),
    ("Honda", r"trigger", "CB Trigger"),
    ("Honda", r"twister", "CB Twister"),
    ("Honda", r"dream yuga", "Dream Yuga"),
    ("Honda", r"dream neo", "Dream Neo"),
    ("Honda", r"cd 110|cd110", "CD 110 Dream"),
    ("Honda", r"livo", "Livo"),
    ("Honda", r"x-blade|xblade", "X-Blade"),
    ("TVS", r"apache rtr 160 4v", "Apache RTR 160 4V"),
    ("TVS", r"apache rtr 200", "Apache RTR 200 4V"),
    ("TVS", r"apache rtr 180", "Apache RTR 180"),
    ("TVS", r"apache rtr 160|apache rtr160", "Apache RTR 160"),
    ("TVS", r"apache rr ?310", "Apache RR 310"),
    ("TVS", r"apache 150", "Apache"),
    ("TVS", r"star city", "Star City Plus"),
    ("TVS", r"star sport|^tvs sport", "Sport"),
    ("TVS", r"radeon", "Radeon"),
    ("TVS", r"victor", "Victor"),
    ("TVS", r"phoenix", "Phoenix"),
    ("TVS", r"flame", "Flame"),
    ("TVS", r"jive", "Jive"),
    ("Yamaha", r"fz ?25|fz 250", "FZ25"),
    ("Yamaha", r"fazer ?25", "Fazer 25"),
    ("Yamaha", r"fz ?x|fzx", "FZ X"),
    ("Yamaha", r"fz ?s|fzs", "FZ S"),
    ("Yamaha", r"fz.*(v ?3|\bfi\b)", "FZ FI"),
    ("Yamaha", r"fz.*v ?2", "FZ V 2.0"),
    ("Yamaha", r"\bfz", "FZ16"),
    ("Yamaha", r"fazer", "Fazer"),
    ("Yamaha", r"yzf ?r15 v3|r15 v3", "YZF R15 V3"),
    ("Yamaha", r"yzf ?r15|r15", "YZF R15"),
    ("Yamaha", r"yzf ?r3|\br3\b", "YZF R3"),
    ("Yamaha", r"mt ?15|mt-15", "MT 15"),
    ("Yamaha", r"sz", "SZ-RR"),
    ("Yamaha", r"saluto", "Saluto"),
    ("Yamaha", r"sz|ss 125", "SS 125"),
    ("Yamaha", r"ybr", "YBR 125"),
    ("Yamaha", r"crux", "Crux"),
    ("Yamaha", r"gladiator", "Gladiator"),
    ("Yamaha", r"rx ?100", "RX100"),
    ("Yamaha", r"rx ?135", "RX135"),
    ("Yamaha", r"fz ?x|fzx", "FZ-X"),
    ("Suzuki", r"gixxer sf 250|gixxer 250 sf", "Gixxer SF 250"),
    ("Suzuki", r"gixxer 250", "Gixxer 250"),
    ("Suzuki", r"gixxer sf", "Gixxer SF"),
    ("Suzuki", r"gixxer", "Gixxer"),
    ("Suzuki", r"intruder", "Intruder 150"),
    ("Suzuki", r"hayabusa", "Hayabusa"),
    ("Suzuki", r"gs ?150", "GS150R"),
    ("Suzuki", r"slingshot", "SlingShot"),
    ("Suzuki", r"hayate", "Hayate"),
    ("KTM", r"rc ?390", "RC 390"),
    ("KTM", r"rc ?200", "RC 200"),
    ("KTM", r"rc ?125", "RC 125"),
    ("KTM", r"390 adventure", "390 Adventure"),
    ("KTM", r"duke ?390|390 duke", "390 Duke"),
    ("KTM", r"duke ?250|250 duke", "250 Duke"),
    ("KTM", r"duke ?200|200 duke", "200 Duke"),
    ("KTM", r"duke ?125|125 duke", "125 Duke"),
    ("Kawasaki", r"ninja 300", "Ninja 300"),
    ("Kawasaki", r"ninja 650", "Ninja 650"),
    ("Kawasaki", r"ninja 250", "Ninja 250R"),
    ("Kawasaki", r"z ?900", "Z900"),
    ("Kawasaki", r"z ?800", "Z800"),
    ("Kawasaki", r"z ?650", "Z650"),
    ("Kawasaki", r"versys 650", "Versys 650"),
    ("Kawasaki", r"zx[- ]?10", "Ninja ZX-10R"),
    ("Harley-Davidson", r"street 750", "Street 750"),
    ("Harley-Davidson", r"street rod", "Street Rod"),
    ("Harley-Davidson", r"iron 883|883 iron", "Iron 883"),
    ("Harley-Davidson", r"forty[- ]eight|48", "Forty Eight"),
    ("Harley-Davidson", r"fat boy", "Fat Boy"),
    ("Harley-Davidson", r"fat bob", "Fat Bob"),
    ("Harley-Davidson", r"superlow|super low", "SuperLow"),
    ("Harley-Davidson", r"heritage", "Heritage Classic"),
    ("Harley-Davidson", r"street glide", "Street Glide"),
    ("Harley-Davidson", r"road king", "Road King"),
    ("Harley-Davidson", r"street bob", "Street Bob"),
    ("Harley-Davidson", r"low rider", "Low Rider"),
    ("Harley-Davidson", r"breakout", "Breakout"),
    ("Harley-Davidson", r"roadster", "Roadster"),
    ("Benelli", r"tnt 600|600i", "TNT600i"),
    ("Benelli", r"tnt 300|302r|tnt ?25", "TNT 300"),
    ("Benelli", r"imperiale", "Imperiale 400"),
    ("Jawa", r"perak", "Perak"),
    ("Jawa", r"forty ?two|42", "42"),
]


def clean_official(name):
    n = re.sub(r"\[.*?\]", "", name or "")
    n = re.sub(r"^(old generation|new|old)\s+", "", n.strip(), flags=re.I)
    n = re.sub(r"\s+\d{4}$", "", n)
    return re.sub(r"\s+", " ", n).strip()


def norm(s):
    s = s.lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"(\d+)\s*cc\b", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s):
    return re.sub(r"[^a-z0-9]", "", norm(s))


def auto_match(kname, candidates):
    """Pick the longest official name whose every token appears in the Kaggle name."""
    k = compact(kname)
    best, best_len = None, 0
    for c in candidates:
        toks = [re.sub(r"[^a-z0-9]", "", t) for t in norm(c).split()]
        toks = [t for t in toks if t]
        if not toks:
            continue
        ok = True
        for t in toks:
            if len(t) == 1:          # single letters ("R", "S") must be their own word
                hit = re.search(r"\b" + t + r"\b", norm(kname))
            else:
                pat = (r"(?<!\d)" + t + r"(?!\d)") if t.isdigit() else re.escape(t)
                hit = re.search(pat, k)
            if not hit:
                ok = False
                break
        if ok and sum(map(len, toks)) > best_len:
            best, best_len = c, sum(map(len, toks))
    return best


def official_names(entry, use_roots=False):
    names = {clean_official(m["model"]) for m in entry["models"] if m.get("status") == 2}
    names |= {clean_official(n) for n in entry["discontinued"]}
    if use_roots:
        names |= {m["root"] for m in entry["models"] if m.get("root")}
        names |= set(entry.get("discontinued_roots") or [])
    return sorted(n for n in names if n)


def map_names(df, name_col, brand_col, brand_map, site, overrides, use_roots=False):
    rows = []
    for (kb, kname), g in df.groupby([brand_col, name_col]):
        brand = brand_map.get(kb)
        official, how = None, "unmatched"
        if brand and brand in site["makes"]:
            low = norm(kname)
            for ob, pat, target in overrides:
                if ob == brand and re.search(pat, low):
                    official, how = target, "override"
                    break
            if not official:
                cands = official_names(site["makes"][brand], use_roots)
                official = auto_match(kname[len(kb):] if kname.startswith(kb) else kname, cands)
                how = "auto" if official else "unmatched"
        rows.append({"kaggle_brand": kb, "kaggle_name": kname, "rows": len(g), "brand": brand,
                     "official_model": official, "how": how})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


BODY = {5: "scooter"}
FUEL = {1: "petrol", 5: "electric", 3: "cng"}


def bike_type(m):
    n = m["model"].lower()
    if re.search(r"\bxl\b|luna|moped", n):
        return "moped"
    return BODY.get(m["body"], "motorcycle")


def main():
    bw = json.loads((CAT / "bikewale_models.json").read_text())
    cw = json.loads((CAT / "carwale_models.json").read_text())

    bikes = pd.read_csv(glob.glob(str(Path.home() / ".cache/kagglehub/datasets/saisaathvik/*/versions/*/*.csv"))[0])
    bikes = bikes.drop_duplicates()
    bikes["bike_name"] = bikes.bike_name.str.replace("‎", "").str.replace(r"\s+", " ", regex=True).str.strip()
    bmap = map_names(bikes, "bike_name", "brand", BIKE_BRAND, bw, BIKE_OVERRIDES)
    bmap.to_csv(CAT / "name_map_bike.csv", index=False)

    cars = pd.read_csv(glob.glob(str(Path.home() / ".cache/kagglehub/datasets/nehalbirla/vehicle-dataset-from-cardekho/versions/*/Car details v3.csv"))[0])
    cars = cars.drop_duplicates()
    cars["kb"] = cars.name.str.split().str[0]
    cmap = map_names(cars, "name", "kb", CAR_BRAND, cw, CAR_OVERRIDES, use_roots=True)
    cmap.to_csv(CAT / "name_map_car.csv", index=False)

    for tag, m in (("bike", bmap), ("car", cmap)):
        tot = m.rows.sum()
        print(f"{tag}: {len(m)} names, matched rows {m[m.official_model.notna()].rows.sum()}/{tot} "
              f"({m[m.official_model.notna()].rows.sum()/tot:.1%}); auto {sum(m.how=='auto')}, override {sum(m.how=='override')}")

    trained_bike = bmap.dropna(subset=["official_model"]).groupby(["brand", "official_model"]).rows.sum()
    trained_car = cmap.dropna(subset=["official_model"]).groupby(["brand", "official_model"]).rows.sum()

    # ---- bike catalog
    cat_bike = {}
    for brand, e in bw["makes"].items():
        out = {}
        for m in e["models"]:
            if m.get("status") != 2 or not m.get("price"):
                continue
            fuel = FUEL.get(m["fuel"])
            if fuel is None:
                continue
            name = clean_official(m["model"])
            rng = m.get("range_km")
            out[name] = {"model": name, "type": bike_type(m), "fuel": fuel, "cc": m.get("cc"), "bhp": m.get("bhp"),
                         "new_price": int(m["price"]), "on_sale": True, "launched": (m.get("launched") or "")[:10],
                         "range_km": rng, "kwh_est": round(rng / 35, 1) if (fuel == "electric" and rng) else None,
                         "train_rows": int(trained_bike.get((brand, name), 0))}
        for (b, name), n in trained_bike.items():
            if b != brand or n < MIN_ROWS:
                continue
            if name in out:
                continue
            out[name] = {"model": name, "type": "motorcycle", "fuel": "petrol", "cc": None, "bhp": None,
                         "new_price": None, "on_sale": False, "train_rows": int(n)}
        if out:
            cat_bike[brand] = sorted(out.values(), key=lambda r: (not r["on_sale"], r["model"]))

    # ---- car catalog
    cat_car = {}
    for brand, e in cw["makes"].items():
        out = {}
        for m in e["models"]:
            if m.get("status") != 2 or not m.get("price"):
                continue
            name = clean_official(m["model"])
            out[name] = {"model": name, "new_price": int(m["price"]), "on_sale": True,
                         "electric": bool(m.get("electric")), "launched": (m.get("launched") or "")[:10],
                         "train_rows": int(trained_car.get((brand, name), 0))}
        for (b, name), n in trained_car.items():
            if b != brand or n < MIN_ROWS or name in out:
                continue
            out[name] = {"model": name, "new_price": None, "on_sale": False, "electric": False, "train_rows": int(n)}
        if out:
            cat_car[brand] = sorted(out.values(), key=lambda r: (not r["on_sale"], r["model"]))

    catalog = {"bike": cat_bike, "car": cat_car,
               "sources": {"bike": f"BikeWale brand pages, scraped {bw['scraped_on']}",
                           "car": f"CarWale brand pages, scraped {cw['scraped_on']}"}}
    (CAT / "catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False))
    nb = sum(len(v) for v in cat_bike.values()); nc = sum(len(v) for v in cat_car.values())
    print(f"catalog: {nb} two-wheeler models across {len(cat_bike)} brands, {nc} car models across {len(cat_car)} brands")


CAR_OVERRIDES = [
    ("Maruti Suzuki", r"swift dzire|dzire", "Dzire"),
    ("Maruti Suzuki", r"wagon r", "Wagon R"),
    ("Maruti Suzuki", r"alto k10", "Alto K10"),
    ("Maruti Suzuki", r"alto 800", "Alto 800"),
    ("Maruti Suzuki", r"\balto\b", "Alto"),
    ("Maruti Suzuki", r"vitara brezza", "Vitara Brezza"),
    ("Maruti Suzuki", r"s[- ]cross", "S-Cross"),
    ("Maruti Suzuki", r"s[- ]presso", "S-Presso"),
    ("Maruti Suzuki", r"zen estilo", "Estilo"),
    ("Maruti Suzuki", r"omni", "Omni"),
    ("Maruti Suzuki", r"gypsy", "Gypsy"),
    ("Maruti Suzuki", r"\b800\b", "800"),
    ("Hyundai", r"grand i10 nios", "Grand i10 Nios"),
    ("Hyundai", r"grand i10", "Grand i10"),
    ("Hyundai", r"i20 active", "i20 Active"),
    ("Hyundai", r"elite i20|i20", "i20"),
    ("Hyundai", r"\bi10\b", "i10"),
    ("Hyundai", r"santro xing", "Santro"),
    ("Hyundai", r"santro", "Santro"),
    ("Hyundai", r"\bxcent\b", "Xcent"),
    ("Hyundai", r"verna", "Verna"),
    ("Hyundai", r"creta", "Creta"),
    ("Hyundai", r"\beon\b", "Eon"),
    ("Hyundai", r"elantra", "Elantra"),
    ("Hyundai", r"tucson", "Tucson"),
    ("Hyundai", r"getz", "Getz"),
    ("Hyundai", r"accent", "Accent"),
    ("Hyundai", r"venue", "Venue"),
    ("Hyundai", r"santa fe", "Santa Fe"),
    ("Mahindra", r"xuv ?500", "XUV500"),
    ("Mahindra", r"xuv ?300", "XUV300"),
    ("Mahindra", r"scorpio", "Scorpio"),
    ("Mahindra", r"bolero", "Bolero"),
    ("Mahindra", r"kuv ?100", "KUV100"),
    ("Mahindra", r"tuv ?300", "TUV300"),
    ("Mahindra", r"marazzo", "Marazzo"),
    ("Mahindra", r"thar", "Thar"),
    ("Mahindra", r"xylo", "Xylo"),
    ("Mahindra", r"quanto", "Quanto"),
    ("Mahindra", r"verito", "Logan/Verito"),
    ("Mahindra", r"logan", "Logan/Verito"),
    ("Mahindra", r"kuv|nuvosport", "NuvoSport"),
    ("Tata", r"nexon", "Nexon"),
    ("Tata", r"tiago", "Tiago"),
    ("Tata", r"tigor", "Tigor"),
    ("Tata", r"indica", "Indica"),
    ("Tata", r"indigo", "Indigo"),
    ("Tata", r"hexa", "Hexa"),
    ("Tata", r"harrier", "Harrier"),
    ("Tata", r"safari", "Safari"),
    ("Tata", r"sumo", "Sumo"),
    ("Tata", r"zest", "Zest"),
    ("Tata", r"bolt", "Bolt"),
    ("Tata", r"nano", "Nano"),
    ("Tata", r"altroz", "Altroz"),
    ("Toyota", r"innova crysta", "Innova Crysta"),
    ("Toyota", r"innova", "Innova"),
    ("Toyota", r"fortuner", "Fortuner"),
    ("Toyota", r"corolla altis", "Corolla Altis"),
    ("Toyota", r"etios liva", "Etios Liva"),
    ("Toyota", r"etios cross", "Etios"),
    ("Toyota", r"etios", "Etios"),
    ("Toyota", r"camry", "Camry"),
    ("Toyota", r"yaris", "Yaris"),
    ("Toyota", r"qualis", "Qualis"),
    ("Toyota", r"glanza", "Glanza"),
    ("Honda", r"\bcity\b", "City"),
    ("Honda", r"amaze", "Amaze"),
    ("Honda", r"jazz", "Jazz"),
    ("Honda", r"wr[- ]?v", "WR-V"),
    ("Honda", r"br[- ]?v", "BR-V"),
    ("Honda", r"brio", "Brio"),
    ("Honda", r"civic", "Civic"),
    ("Honda", r"accord", "Accord"),
    ("Honda", r"mobilio", "Mobilio"),
    ("Honda", r"cr[- ]?v", "CR-V"),
    ("Ford", r"ecosport", "EcoSport"),
    ("Ford", r"figo aspire|aspire", "Aspire"),
    ("Ford", r"freestyle", "Freestyle"),
    ("Ford", r"figo", "Figo"),
    ("Ford", r"fiesta", "Fiesta"),
    ("Ford", r"endeavour", "Endeavour"),
    ("Ford", r"ikon", "Ikon"),
    ("Chevrolet", r"beat", "Beat"),
    ("Chevrolet", r"spark", "Spark"),
    ("Chevrolet", r"cruze", "Cruze"),
    ("Chevrolet", r"enjoy", "Enjoy"),
    ("Chevrolet", r"sail", "Sail"),
    ("Chevrolet", r"tavera", "Tavera"),
    ("Chevrolet", r"aveo", "Aveo"),
    ("Chevrolet", r"optra", "Optra"),
    ("Chevrolet", r"captiva", "Captiva"),
    ("Renault", r"kwid", "Kwid"),
    ("Renault", r"duster", "Duster"),
    ("Renault", r"triber", "Triber"),
    ("Renault", r"lodgy", "Lodgy"),
    ("Renault", r"pulse", "Pulse"),
    ("Renault", r"scala", "Scala"),
    ("Renault", r"captur", "Captur"),
    ("Volkswagen", r"polo", "Polo"),
    ("Volkswagen", r"vento", "Vento"),
    ("Volkswagen", r"ameo", "Ameo"),
    ("Volkswagen", r"jetta", "Jetta"),
    ("Volkswagen", r"passat", "Passat"),
    ("Skoda", r"rapid", "Rapid"),
    ("Skoda", r"octavia", "Octavia"),
    ("Skoda", r"superb", "Superb"),
    ("Skoda", r"laura", "Laura"),
    ("Skoda", r"fabia", "Fabia"),
    ("Nissan", r"micra", "Micra"),
    ("Nissan", r"sunny", "Sunny"),
    ("Nissan", r"terrano", "Terrano"),
    ("Nissan", r"kicks", "Kicks"),
    ("Datsun", r"redi[- ]?go", "redi-GO"),
    ("Datsun", r"go plus|go\+", "GO+"),
    ("Datsun", r"\bgo\b", "GO"),
    ("Fiat", r"punto", "Punto"),
    ("Fiat", r"linea", "Linea"),
    ("Jeep", r"compass", "Compass"),
    ("Jeep", r"wrangler", "Wrangler"),
    ("Kia", r"seltos", "Seltos"),
    ("Kia", r"carnival", "Carnival"),
    ("MG", r"hector", "Hector"),
    ("Land Rover", r"discovery sport", "Discovery Sport"),
    ("Land Rover", r"range rover evoque", "Range Rover Evoque"),
    ("Land Rover", r"freelander", "Freelander 2"),
    ("BMW", r"\bx1\b", "X1"), ("BMW", r"\bx3\b", "X3"), ("BMW", r"\bx4\b", "X4"), ("BMW", r"\bx5\b", "X5"),
    ("BMW", r"3 series", "3-Series"), ("BMW", r"5 series", "5 Series"), ("BMW", r"7 series", "7 Series"),
    ("Mercedes-Benz", r"gla", "GLA"), ("Mercedes-Benz", r"gl[- ]?class|gls", "GLS"),
    ("Mercedes-Benz", r"e[- ]class", "E-Class"), ("Mercedes-Benz", r"c[- ]class", "C-Class"),
    ("Mercedes-Benz", r"b[- ]class", "B-Class"), ("Mercedes-Benz", r"m[- ]class|gle", "GLE"),
    ("Mercedes-Benz", r"s[- ]class", "S-Class"), ("Mercedes-Benz", r"cla", "CLA"),
    ("Audi", r"\ba4\b", "A4"), ("Audi", r"\ba6\b", "A6"), ("Audi", r"\bq3\b", "Q3"), ("Audi", r"\bq5\b", "Q5"),
    ("Audi", r"\bq7\b", "Q7"), ("Audi", r"\ba3\b", "A3"),
    ("Volvo", r"xc ?90", "XC90"), ("Volvo", r"xc ?40", "XC40"), ("Volvo", r"xc ?60", "XC60"),
    ("Volvo", r"v40", "V40"), ("Volvo", r"s60", "S60"),
    ("Jaguar", r"xf", "XF"), ("Jaguar", r"xe", "XE"), ("Jaguar", r"xj", "XJ"),
    ("Mitsubishi", r"pajero", "Pajero Sport"), ("Mitsubishi", r"outlander", "Outlander"), ("Mitsubishi", r"lancer", "Lancer"),
    ("Lexus", r"es ?300", "ES"),
    ("Isuzu", r"d[- ]max", "D-Max"), ("Isuzu", r"mu[- ]?x", "MU-X"),
    ("Force Motors", r"gurkha", "Gurkha"), ("Force Motors", r"one", "One"),
]

if __name__ == "__main__":
    main()
