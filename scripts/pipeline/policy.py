"""Policy-period helpers aligned with the dissertation report."""

from __future__ import annotations

POLICY_ORDER: list[str] = [
    "Pre-GreenDeal",
    "GreenDeal-ECO1",
    "ECO2",
    "MEES",
    "Post-Strategy",
]


def policy_period(year: int | float | None) -> str:
    if year is None:
        return "Other"
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "Other"

    if 2008 <= y <= 2012:
        return "Pre-GreenDeal"
    if 2013 <= y <= 2015:
        return "GreenDeal-ECO1"
    if 2016 <= y <= 2018:
        return "ECO2"
    if 2019 <= y <= 2020:
        return "MEES"
    if 2021 <= y <= 2025:
        return "Post-Strategy"
    return "Other"
