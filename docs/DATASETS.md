# Datasets

| # | Dataset | Link | Rows | Years | Used for | Status |
|---|---|---|---|---|---|---|
| 1 | **Used Bikes Prices in India** (saisaathvik) | kaggle.com/datasets/saisaathvik/used-bikes-prices-in-india | 32,648 (7,324 unique) | ~2020-21 | Main two-wheeler training set (Colab + website) | ✅ downloaded |
| 2 | **Vehicle dataset from CarDekho** — `Car details v3.csv` (nehalbirla) | kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho | 8,128 (6,926 unique) | up to 2020 | Car model on the website | ✅ downloaded |
| 3 | **Motorcycle Dataset** — `BIKE DETAILS.csv` (nehalbirla) | kaggle.com/datasets/nehalbirla/motorcycle-dataset | ~1,061 | ~2020 | Ex-showroom prices → price when new / value kept (Module 3) | ✅ used in Colab |
| 4 | **BikeWale used EVs, all India** (self-collected) | bikewale.com/used/ (84 city pages) + ev.sale | 888 → 867 clean | Nov 2026 | `ev_used_india_2026.csv` — battery kWh, est. health | ✅ in Colab |
| 5 | **BikeWale same-variant 2026 set** (self-collected) | bikewale.com used listings + new-variant ex-showroom (685 variants) | 5,694 (5,270 petrol, 424 EV) | Nov 2026 | Module 6 EV vs petrol regression; EV pricing in the app | ⚠️ still in browser tab — download `two_wheelers_2026_same_variant.csv` |
| 6 | **State road-tax table** (self-built) | Official transport sites, DriveSpark, Tribune, DT Next, Digit, etc. | 11 states × periods | 2010-2026 | `road_tax_table.csv`, `backend/app/road_tax.py` | ✅ |
| 8 | **Verified model catalog** (self-built) | bikewale.com/{brand}-bikes/, carwale.com/{brand}-cars/ (embedded JSON) | 541 2W + 345 car models | 12 Nov 2026 | Website dropdowns, official names, 2026 ex-showroom prices; `backend/data/catalog/` | ✅ |
| 7 | **New prices by family** (self-built) | Dataset 3 + MaxAbout / 91wheels / BikeDekho + BikeWale 2026 | 206 families | 2019-20 & 2026 | `new_prices_by_family.csv` | ✅ in Colab |

## Column cheat-sheet
- **Bikes (1):** `bike_name, price, city, kms_driven, owner, age, power (cc), brand`
- **Cars (2, v3):** `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats`
- **Motorcycles (3):** `name, selling_price, year, seller_type, owner, km_driven, ex_showroom_price`

## Cleaning applied (website pipeline, `backend/train.py`)
| Step | Bikes | Cars |
|---|---|---|
| Loaded | 32,648 | 8,128 |
| Exact duplicates removed | 25,324 | 1,202 |
| km filter | 100 – 1,50,000 | 500 – 4,00,000 |
| Price outliers | 1.5×IQR per model family + age group (groups ≥ 10) | same, per brand+model + age group |
| Final | **7,167** | **6,559** |

> The Colab notebook's final bike count is 6,733 (it also dropped unmapped cities and used the full 443-city map). The website keeps unmapped cities as state = "Other". Both are fine; quote the Colab number in the report and mention the website retrains on its own copy.

## Datasets considered and rejected
- **Kaggle "CarDekho car details v4"** (2,059 rows, 2023) — different schema and reference year; could extend the car model later.
- **Ready-made state road-tax datasets** — none trustworthy found; built our own from official/news sources.
- **Battery health from listings** — BikeWale/BikeDekho/Bikes4Sale don't publish it; estimated with a formula instead (chart only).

## Name matching (Kaggle → official), `backend/build_catalog.py`
- 351 hand-written rules + automatic token matching; every target checked to be an official BikeWale/CarWale name.
- Bikes: 467 Kaggle names, 98.4% of rows matched (unmatched: brands no longer sold in India — Hyosung, Mahindra 2W, MV Agusta, LML…). Cars: 2,058 names, 99.7% matched.
- Review files: `backend/data/catalog/name_map_bike.csv`, `name_map_car.csv` (good appendix material).
- Finding: the Kaggle bike data has **no scooters and nothing launched after 2021** → those are priced by the depreciation model.
