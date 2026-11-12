"""Current (2026) two-wheeler road-tax slabs by state.

Rates are the latest period from the Module 3 road-tax table (see PROGRESS.md).
Each slab list is [(upper_limit_rupees_or_None, rate), ...] checked in order.
EVs are exempt (0%) in all listed states for the data period.
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
    "Karnataka": "DriveSpark (Apr 2017 on)",
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

STATES = sorted(set(PRICE_SLABS) | set(CC_SLABS))


def _pick(slabs, value):
    for limit, rate in slabs:
        if limit is None or value <= limit:
            return rate
    return slabs[-1][1]


def road_tax_rate(state: str, ex_showroom: float, cc: float, is_ev: bool = False) -> float | None:
    """Return the road-tax rate (0-1) for a new two-wheeler, or None if state unknown."""
    if is_ev:
        return 0.0
    if state in PRICE_SLABS:
        return _pick(PRICE_SLABS[state], ex_showroom)
    if state in CC_SLABS:
        return _pick(CC_SLABS[state], cc)
    return None


def cheapest_state(ex_showroom: float, cc: float):
    rates = {s: road_tax_rate(s, ex_showroom, cc) for s in STATES}
    s = min(rates, key=rates.get)
    return s, rates[s]
