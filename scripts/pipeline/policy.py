"""Policy-period helpers aligned with the dissertation report.

Policy period boundaries
------------------------
Pre-GreenDeal  : 2008–2012  — baseline, no major national retrofit scheme
GreenDeal-ECO1 : 2013–2015  — Green Deal finance + ECO1 obligations
                              Note: ECO2 formally launched April 2015, but
                              annual EPC data cannot resolve within-year
                              transitions. 2015 is assigned to GreenDeal-ECO1
                              (ECO1 delivery was still the dominant scheme for
                              the majority of that lodgement year).
ECO2           : 2016–2018  — ECO2 insulation + fuel-poverty targeting
MEES           : 2019–2020  — Minimum Energy Efficiency Standards for PRS
                              (enforcement began April 2018; 2019–2020 captures
                              the first full compliance cycle)
Post-Strategy  : 2021–2025  — Net Zero Strategy, low-carbon heat policies
"""

from __future__ import annotations

POLICY_ORDER: list[str] = [
    "Pre-GreenDeal",
    "GreenDeal-ECO1",
    "ECO2",
    "MEES",
    "Post-Strategy",
]

# Inclusive year boundaries for each policy period.
# 2015 → GreenDeal-ECO1 (see module docstring for rationale).
_BOUNDARIES: list[tuple[int, int, str]] = [
    (2008, 2012, "Pre-GreenDeal"),
    (2013, 2015, "GreenDeal-ECO1"),
    (2016, 2018, "ECO2"),
    (2019, 2020, "MEES"),
    (2021, 2025, "Post-Strategy"),
]


def policy_period(year: int | float | None) -> str:
    """Return the policy-period label for a given lodgement year.

    Years outside 2008–2025, or non-numeric values, return 'Other'.
    2015 is assigned to 'GreenDeal-ECO1' — see module docstring.
    """
    if year is None:
        return "Other"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "Other"

    for lo, hi, label in _BOUNDARIES:
        if lo <= y <= hi:
            return label
    return "Other"
