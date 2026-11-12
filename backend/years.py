"""Which years can each model and variant be from?  (shared by build_variants.py and tests/year_audit.py)

Official production years, when the listing site states them:
  - in the variant name:  "ZXi Plus AMT [2021-2023]", "Single Disc [2020-2024]", "Model S (O) 2.0 Diesel [2021]"
  - else in the model (generation) name: "Venue [2019-2022]", "Classic 350 [2020]"
  These are the site's own catalogue years. One extra year is allowed at the end: a bike made in December is often
  bought and registered in January ("Year bought" is the registration year).
Otherwise (mostly vehicles still on sale): the years seen in ads, after dropping stray years — a year with a single ad
that is 2+ years away from every other ad year of that variant is treated as a typo.
"""
import re

NOW = 2026
BRACKET = re.compile(r"\[(\d{4})(?:\s*-\s*(\d{4}))?\]")


def bracket_years(name):
    """Last [Y1-Y2] range in a name -> (y1, y2) or None.
    Single-year labels ("Activa Standard [2022]") are ignored: the audit (tests/year_audit.py) found 18-56% of ads
    outside them -- they mark a spec or colour update, not the only year the vehicle was made. Year ranges hold:
    1.3-1.7% of car ads and 9-11% of two-wheeler ads fall outside them."""
    found = [f for f in BRACKET.findall(str(name)) if f[1]]
    if not found:
        return None
    y1, y2 = found[-1]
    y1 = int(y1); y2 = int(y2)
    if not (1980 <= y1 <= NOW + 1 and y1 <= y2):
        return None
    return y1, min(y2, NOW)


def official_years(version_name, model_name):
    """(y1, y2, source) from the variant's own bracket, else the generation's bracket, else None."""
    v = bracket_years(version_name)
    if v:
        return v[0], v[1], "variant name"
    m = bracket_years(model_name)
    if m:
        return m[0], m[1], "model name"
    return None


def trimmed_ad_years(year_counts, launched=None):
    """year_counts: {year: n ads}. Drops
    1. isolated single-ad years (2+ years from any other ad year): likely typos;
    2. a thin leading tail (together <= 2% of 30+ ads) -- but ONLY before the listing site's launch date, and only when
       the model's real volume starts at that date (so the date is a first launch, not a later update). This cuts
       mis-filed older bikes (8 of 447 "R15 V4" ads from 2014-2020; launched Sep 2021) while keeping genuine
       low-volume early years (TVS iQube 2021, launched Jan 2020) and old Activas (its 2020 date is a BS6 update)."""
    ys = sorted(year_counts)
    keep = []
    for y in ys:
        others = [o for o in ys if o != y]
        isolated = all(abs(o - y) >= 2 for o in others) if others else False
        if year_counts[y] == 1 and isolated and len(ys) > 1:
            continue
        keep.append(y)
    keep = keep or ys
    total = sum(year_counts[y] for y in keep)
    if launched and total >= 30:
        first_volume = next((y for y in keep if year_counts[y] / total >= 0.02), keep[0])
        tail = [y for y in keep if y < launched]
        if tail and first_volume >= launched - 1 and sum(year_counts[y] for y in tail) <= 0.02 * total:
            keep = [y for y in keep if y >= launched] or keep
    return min(keep), max(keep)


def final_range(official, ad_counts, on_sale, kind="car", launched=None):
    """Years offered in the form. official = (y1, y2, src) or None; ad_counts = {year: n}; on_sale = still sold new.
    Cars: the official range is trusted as is. Two-wheelers: their labels are less reliable, so the official range is
    widened to cover the (typo-trimmed) ad years too."""
    if official:
        y1, y2, _ = official
        end = NOW if on_sale and y2 >= NOW - 1 else min(NOW, y2 + 1)   # +1: bought / registered the next January
        if kind == "bike" and ad_counts:
            a1, a2 = trimmed_ad_years(ad_counts, launched)
            y1, end = min(y1, a1), max(end, min(NOW, a2))
        return y1, end, "official"
    if ad_counts:
        y1, y2 = trimmed_ad_years(ad_counts, launched)
        return y1, (NOW if on_sale else y2), "ads"
    return (NOW - 1, NOW, "new") if on_sale else None
