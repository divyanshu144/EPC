"""Prefect orchestration entry points for the EPC pipeline."""

from ew_housing_energy_impact.orchestration.flows import epc_pipeline_flow

__all__ = ["epc_pipeline_flow"]
