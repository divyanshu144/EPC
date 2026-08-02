"""Threshold-based retraining recommendation logic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def recommend_retraining(
    drift_report: dict[str, Any],
    *,
    max_feature_alerts: int = 3,
    min_new_rows: int = 1_000,
    model_metrics: dict[str, float] | None = None,
    max_metric_degradation: float = 0.10,
) -> dict[str, Any]:
    """Return a human-readable retraining recommendation.

    This is operational decision support only. It does not claim causal or
    statistical significance and it never triggers retraining automatically.
    """

    summary = drift_report.get("summary", {})
    coverage = drift_report.get("coverage", {})
    reasons: list[str] = []
    triggered_checks: list[str] = []

    feature_alert_count = int(summary.get("numeric_alert_count", 0)) + int(
        summary.get("categorical_alert_count", 0)
    )
    if feature_alert_count >= max_feature_alerts:
        reasons.append(
            f"{feature_alert_count} feature drift alerts exceeded threshold {max_feature_alerts}."
        )
        triggered_checks.append("multiple_feature_drift_alerts")

    target_alert_count = int(summary.get("target_alert_count", 0))
    if target_alert_count:
        reasons.append(f"{target_alert_count} target drift alert(s) detected.")
        triggered_checks.append("target_drift")

    class_balance_shift = coverage.get("below_c_class_balance_shift")
    missing_balance_alert = class_balance_shift is not None and class_balance_shift > 0.10
    if missing_balance_alert:
        reasons.append(f"BELOW_C class balance shifted by {class_balance_shift:.1%}.")
        triggered_checks.append("below_c_class_balance_shift")

    if coverage.get("alert"):
        reasons.append(
            "Coverage drift detected in certificate count, year range, or authority coverage."
        )
        triggered_checks.extend(coverage.get("triggered_checks", []))

    current_count = int(coverage.get("current_certificate_count", 0) or 0)
    reference_count = int(coverage.get("reference_certificate_count", 0) or 0)
    new_volume = max(current_count - reference_count, 0)
    if new_volume >= min_new_rows:
        reasons.append(f"Current dataset contains {new_volume:,} more rows than reference.")
        triggered_checks.append("new_data_volume")

    if model_metrics:
        for metric_name, degradation in model_metrics.items():
            if degradation > max_metric_degradation:
                reasons.append(f"Model metric degradation for {metric_name} is {degradation:.1%}.")
                triggered_checks.append(f"metric_degradation:{metric_name}")

    unique_checks = list(dict.fromkeys(triggered_checks))
    should_retrain = bool(unique_checks)
    if any(check in unique_checks for check in ("target_drift", "below_c_class_balance_shift")):
        severity = "high"
    elif coverage.get("alert") or feature_alert_count >= max_feature_alerts:
        severity = "medium"
    else:
        severity = "low"

    if not reasons:
        reasons.append("No material drift threshold was exceeded. Continue monitoring.")

    return {
        "should_retrain": should_retrain,
        "severity": severity,
        "reasons": reasons,
        "triggered_checks": unique_checks,
        "checked_at": _now_iso(),
    }
