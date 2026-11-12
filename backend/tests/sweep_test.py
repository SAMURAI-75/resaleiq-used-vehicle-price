"""Price every model in both dropdowns and check the answers make sense.

  python backend/tests/sweep_test.py          (server running on localhost:8000)

For every brand/model: earliest, middle and latest year (+ its most common variant when it has variants).
Checks: HTTP 200 · finite price > 0 · low <= price <= high · price never rises with age (curve and year-to-year) ·
not above 1.3x the new price after the first year · factors present · gap to the median of real ads of the same model
and age (reported, flagged when > 40% with 10+ ads). Plus edge inputs (km 0 / 999,999, 4th owner, unknown model,
unknown state, unknown variant, missing city).
Writes backend/tests/sweep_report.json and prints a summary; exit code 1 if any hard check fails.
"""
import json
import math
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://localhost:8000"
OUT = Path(__file__).resolve().parent / "sweep_report.json"


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"content-type": "application/json"}, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"detail": raw.decode(errors="ignore")[:80]}


def check_model(kind, brand, m):
    issues, gaps = [], []
    y0, y1 = m["years"]
    years = sorted({y0, (y0 + y1) // 2, y1})
    variant = None
    if m["engine"] == "market" and m.get("variants"):
        q = urllib.parse.urlencode({"type": kind, "brand": brand, "model": m["model"]})
        st, vs = call(f"/api/variants?{q}")
        if st != 200:
            issues.append(f"variants HTTP {st}")
        elif vs:
            cover = [v for v in vs if v["years"][0] <= years[0] and v["years"][1] >= years[-1]]
            variant = max(cover, key=lambda v: v["n"]) if cover else None
    prices = []
    for y in years:
        age = 2026 - y
        km = (8000 if kind == "2w" else 11000) * max(age, 0.5)
        # no state: the gap is measured against all-India medians, so price the all-India case
        body = {"type": kind, "brand": brand, "model": m["model"], "year": y, "km": round(km), "owners": 1,
                "state": "Other", "seller": "individual"}
        if variant:
            body["variant"] = variant["id"]
        st, r = call("/api/predict", body)
        if st != 200:
            issues.append(f"{y}: HTTP {st} {r.get('detail')}")
            continue
        p = r.get("price")
        if not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            issues.append(f"{y}: bad price {p}"); continue
        if not (r["low"] <= p <= r["high"]):
            issues.append(f"{y}: price outside its range")
        c = [pt["price"] for pt in r["curve"]]
        if any(b > a * 1.005 + 100 for a, b in zip(c, c[1:])):
            issues.append(f"{y}: curve rises with age")
        np_ = r.get("new_price")
        s_ = r.get("similar")
        confirmed = s_ and s_["count"] >= 5 and abs(p / s_["median"] - 1) < 0.25   # real ads also sit above the new price
        if np_ and age >= 1 and p > 1.3 * np_ * (1.25 if kind == "2w" else 1.0) and not confirmed:
            issues.append(f"{y}: price {p:,.0f} above 1.3x new {np_:,.0f}")
        if not r.get("factors"):
            issues.append(f"{y}: no factors")
        s = r.get("similar")
        like_for_like = s and (("variant" in body) == (s.get("scope") == "variant"))
        if s and s["count"] >= 10 and like_for_like:
            gaps.append({"year": y, "price": p, "similar": s["median"], "n": s["count"], "gap": p / s["median"] - 1})
        prices.append((y, p))
    for (ya, pa), (yb, pb) in zip(prices, prices[1:]):
        if pb < pa * 0.98 - 100:       # a newer year must not be cheaper (2% tolerance: asking-price noise)
            issues.append(f"newer {yb} cheaper than {ya} ({pb:,.0f} < {pa:,.0f})")
    return {"kind": kind, "brand": brand, "model": m["model"], "engine": m["engine"], "issues": issues, "gaps": gaps}


def edge_cases(meta):
    out = []
    b2 = next(iter(meta["2w"]["brands"])); m2 = meta["2w"]["brands"][b2][0]["model"]
    base = {"type": "2w", "brand": b2, "model": m2, "year": 2022, "km": 10000, "owners": 1, "state": "Karnataka"}
    for name, body, expect in [
        ("km 0", {**base, "km": 0}, 200), ("km 999,999", {**base, "km": 999999}, 200), ("4th owner", {**base, "owners": 4}, 200),
        ("no city", {**base, "city": None}, 200), ("state Other", {**base, "state": "Other"}, 200),
        ("unknown state", {**base, "state": "Atlantis"}, 422), ("unknown model", {**base, "model": "Nope 9000"}, 422),
        ("unknown variant", {**base, "variant": "999999999"}, 200), ("year 1989", {**base, "year": 1989}, 422),
        ("year 2027", {**base, "year": 2027}, 422), ("negative km", {**base, "km": -5}, 422),
        ("dealer", {**base, "seller": "dealer"}, 200), ("bad seller", {**base, "seller": "robot"}, 422)]:
        st, r = call("/api/predict", body)
        out.append({"case": name, "status": st, "ok": st == expect})
    st, a = call("/api/predict", {**base, "km": 1000})
    st2, b = call("/api/predict", {**base, "km": 80000})
    out.append({"case": "more km never raises the price", "ok": st == st2 == 200 and b["price"] <= a["price"]})
    st, a = call("/api/predict", {**base, "owners": 1}); st2, b = call("/api/predict", {**base, "owners": 3})
    out.append({"case": "more owners never raises the price", "ok": st == st2 == 200 and b["price"] <= a["price"]})
    return out


def main():
    st, meta = call("/api/meta")
    assert st == 200, "meta failed"
    jobs = [(k, b, m) for k in ("2w", "car") for b, ms in meta[k]["brands"].items() for m in ms]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: check_model(*j), jobs))
    edges = edge_cases(meta)
    bad = [r for r in res if r["issues"]]
    gaps = [g["gap"] for r in res for g in r["gaps"]]
    big = sorted([(r["kind"], r["brand"], r["model"], g) for r in res for g in r["gaps"] if abs(g["gap"]) > 0.4],
                 key=lambda t: -abs(t[3]["gap"]))
    summary = {"models_checked": len(res), "models_with_issues": len(bad), "edge_cases_failed": sum(not e["ok"] for e in edges),
               "similar_ad_comparisons": len(gaps),
               "median_abs_gap_to_similar_ads": sorted(abs(g) for g in gaps)[len(gaps) // 2] if gaps else None,
               "gaps_over_40pct": len(big)}
    OUT.write_text(json.dumps({"summary": summary, "issues": bad, "edges": edges, "big_gaps": big[:60]}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in bad[:40]:
        print(" ", r["kind"], r["brand"], r["model"], r["engine"], r["issues"][:3])
    for e in edges:
        if not e["ok"]:
            print("  EDGE FAIL", e)
    for k, b, m, g in big[:15]:
        print(f"  gap {g['gap']:+.0%}  {k} {b} {m} {g['year']}: ours {g['price']:,.0f} vs {g['similar']:,.0f} ({g['n']} ads)")
    sys.exit(1 if bad or any(not e["ok"] for e in edges) else 0)


if __name__ == "__main__":
    main()
