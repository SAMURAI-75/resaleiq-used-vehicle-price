"""Current (2026) two-wheeler road-tax slabs by state.

Rates are the latest period from the Module 3 road-tax table (see PROGRESS.md).
Each slab list is [(upper_limit_rupees_or_None, rate), ...] checked in order.
EVs are exempt (0%) in all listed states for the data period. Karnataka adds an 11% cess on the tax.
GST (in the ex-showroom price, not road tax): see gst_two_wheeler().
"""

# price-based slabs: (upper price limit in Rs, rate)
PRICE_SLABS = {
    "Delhi":         [(25_000, 0.02), (40_000, 0.04), (60_000, 0.06), (None, 0.08)],
    "Karnataka":     [(50_000, 0.10), (100_000, 0.12), (None, 0.18)],
    "Haryana":       [(75_000, 0.04), (200_000, 0.06), (None, 0.08)],
    "Uttar Pradesh": [(40_000, 0.07), (None, 0.10)],
    "Gujarat":       [(None, 0.06)],
    "Tamil Nadu":    [(100_000, 0.10), (None, 0.12)],
    "Telangana":     [(50_000, 0.09), (100_000, 0.12), (200_000, 0.15), (None, 0.18)],
    "Punjab":        [(100_000, 0.075), (200_000, 0.10), (None, 0.11)],
    "West Bengal":   [(20_000, 0.08), (60_000, 0.09), (200_000, 0.10), (None, 0.12)],
}

# engine-size based slabs: (upper cc limit, rate)
CC_SLABS = {
    "Maharashtra": [(99, 0.10), (299, 0.11), (None, 0.12)],
    "Rajasthan":   [(200, 0.08), (500, 0.13), (None, 0.15)],
}

SOURCES = {
    "Delhi": "Delhi MV Taxation Act (official)",
    "Karnataka": "DriveSpark (slabs, Apr 2017 on); 11% cess: PolicyBazaar, BankBazaar (2026)",
    "Haryana": "Haryana Transport (official)",
    "Uttar Pradesh": "Dainik Jagran",
    "Gujarat": "Policybazaar",
    "Tamil Nadu": "DT Next (from 9 Nov 2023)",
    "Telangana": "Deccan Chronicle (from 14 Aug 2025)",
    "Punjab": "Tribune (from 22 Aug 2024)",
    "West Bengal": "Digit; matches ZigWheels on-road RTO",
    "Maharashtra": "ICICI Lombard (Jul 2017 on)",
    "Rajasthan": "Rajasthan Transport (official)",
}

# cess charged ON the road tax (share of the tax). Karnataka: 11% infrastructure cess (PolicyBazaar, BankBazaar, 2026).
CESS = {"Karnataka": 0.11}

STATES = sorted(set(PRICE_SLABS) | set(CC_SLABS))


def _pick(slabs, value):
    for limit, rate in slabs:
        if limit is None or value <= limit:
            return rate
    return slabs[-1][1]


def base_road_tax_rate(state: str, ex_showroom: float, cc: float, is_ev: bool = False) -> float | None:
    """Slab rate (0-1) before cess, or None if the state is not in our table."""
    if is_ev:
        return 0.0
    if state in PRICE_SLABS:
        return _pick(PRICE_SLABS[state], ex_showroom)
    if state in CC_SLABS:
        return _pick(CC_SLABS[state], cc)
    return None


def road_tax_rate(state: str, ex_showroom: float, cc: float, is_ev: bool = False) -> float | None:
    """Effective road-tax rate (0-1) for a new two-wheeler, cess included; None if the state is unknown."""
    b = base_road_tax_rate(state, ex_showroom, cc, is_ev)
    return None if b is None else b * (1 + CESS.get(state, 0.0))


def gst_two_wheeler(cc, electric: bool, year: int):
    """GST built into the ex-showroom price. GST 2.0 (22 Sep 2025): up to 350cc 18%, above 350cc 40%, EVs 5%.
    Before: 28% (+3% cess above 350cc). Returns (rate now, rate when this bike was bought or None if unclear, note)."""
    if electric:
        return 0.05, 0.05, "EVs pay 5% GST (unchanged)."
    if not cc:
        return None, None, None
    big = cc > 350
    now = 0.40 if big else 0.18
    then = (0.31 if big else 0.28) if year <= 2024 else (now if year >= 2026 else None)
    note = ("Bikes above 350cc moved from 31% (28% + 3% cess) to 40% GST on 22 Sep 2025."
            if big else "Bikes up to 350cc moved from 28% to 18% GST on 22 Sep 2025.")
    return now, then, note


def cheapest_state(ex_showroom: float, cc: float):
    rates = {s: road_tax_rate(s, ex_showroom, cc) for s in STATES}
    s = min(rates, key=rates.get)
    return s, rates[s]
