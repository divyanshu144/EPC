"""Query helpers for the optional analytical serving database."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ew_housing_energy_impact.db.models import (
    ArtifactRegistry,
    AuthorityYearSummary,
    ModelMetric,
    PolicyPeriodSummary,
    ValidationRun,
)


def _row_to_dict(obj) -> dict:
    return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}


def get_authorities(session: Session) -> list[str]:
    """Return available authority names in sorted order."""

    rows = session.execute(
        select(AuthorityYearSummary.authority).distinct().order_by(AuthorityYearSummary.authority)
    )
    return [row[0] for row in rows]


def get_trends(
    session: Session,
    authority: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[dict]:
    """Return authority/year trend rows with optional filters."""

    stmt = select(AuthorityYearSummary)
    if authority:
        stmt = stmt.where(AuthorityYearSummary.authority == authority)
    if year_min is not None:
        stmt = stmt.where(AuthorityYearSummary.year >= year_min)
    if year_max is not None:
        stmt = stmt.where(AuthorityYearSummary.year <= year_max)
    stmt = stmt.order_by(AuthorityYearSummary.year, AuthorityYearSummary.authority)
    return [_row_to_dict(row) for row in session.scalars(stmt)]


def get_policy_summary(session: Session, authority: str | None = None) -> list[dict]:
    """Return policy-period summary rows, optionally for one authority."""

    stmt = select(PolicyPeriodSummary)
    if authority:
        stmt = stmt.where(PolicyPeriodSummary.authority == authority)
    else:
        stmt = stmt.where(PolicyPeriodSummary.authority.is_(None))
    stmt = stmt.order_by(PolicyPeriodSummary.policy_period)
    return [_row_to_dict(row) for row in session.scalars(stmt)]


def get_latest_validation(session: Session) -> dict | None:
    """Return the latest validation run, if one exists."""

    stmt = select(ValidationRun).order_by(ValidationRun.checked_at.desc()).limit(1)
    row = session.scalars(stmt).first()
    return _row_to_dict(row) if row else None


def get_artifacts(session: Session, limit: int = 100) -> list[dict]:
    """Return recent artifact registry rows."""

    stmt = select(ArtifactRegistry).order_by(ArtifactRegistry.created_at.desc()).limit(limit)
    return [_row_to_dict(row) for row in session.scalars(stmt)]


def get_model_metrics(session: Session, task_type: str | None = None) -> list[dict]:
    """Return model metric rows, optionally filtered by task type."""

    stmt = select(ModelMetric)
    if task_type:
        stmt = stmt.where(ModelMetric.task_type == task_type)
    stmt = stmt.order_by(ModelMetric.created_at.desc(), ModelMetric.model_name)
    return [_row_to_dict(row) for row in session.scalars(stmt)]
