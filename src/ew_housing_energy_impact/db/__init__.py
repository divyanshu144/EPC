"""Optional SQLAlchemy-backed analytical serving layer."""

from ew_housing_energy_impact.db.engine import create_tables, get_engine, session_scope

__all__ = ["create_tables", "get_engine", "session_scope"]
