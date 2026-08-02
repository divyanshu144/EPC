"""SQLAlchemy models for optional analytical serving tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AuthorityYearSummary(Base):
    __tablename__ = "authority_year_summary"
    __table_args__ = (UniqueConstraint("authority", "year", name="uq_authority_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    authority: Mapped[str] = mapped_column(String(255), index=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    mean_epc_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_current_energy_efficiency: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_co2: Mapped[float | None] = mapped_column(Float, nullable=True)
    below_c_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    certificate_count: Mapped[int] = mapped_column(Integer, default=0)
    gas_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_gas_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_build_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class PolicyPeriodSummary(Base):
    __tablename__ = "policy_period_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_period: Mapped[str] = mapped_column(String(100), index=True)
    authority: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    mean_epc_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_current_energy_efficiency: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_co2: Mapped[float | None] = mapped_column(Float, nullable=True)
    below_c_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    certificate_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ValidationRun(Base):
    __tablename__ = "validation_runs"
    __table_args__ = (UniqueConstraint("run_id", name="uq_validation_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(50), index=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    year_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    year_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    authority_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, default=0)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ArtifactRegistry(Base):
    __tablename__ = "artifact_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    artifact_name: Mapped[str] = mapped_column(String(255), index=True)
    artifact_type: Mapped[str] = mapped_column(String(100), index=True)
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[str] = mapped_column(String(255), index=True)
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String(50), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
