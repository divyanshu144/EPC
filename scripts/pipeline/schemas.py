"""Shared schema definitions for EPC pipeline steps."""

from __future__ import annotations

# Core columns kept from raw EPC certificates (download/merge step).
CORE_COLS: list[str] = [
    # --- IDs & location ---
    "LMK_KEY",
    "BUILDING_REFERENCE_NUMBER",
    "POSTCODE",
    "POSTTOWN",
    "COUNTY",
    "LOCAL_AUTHORITY",
    "LOCAL_AUTHORITY_LABEL",
    # --- Dates ---
    "LODGEMENT_DATE",
    # --- Ratings ---
    "CURRENT_ENERGY_RATING",
    "POTENTIAL_ENERGY_RATING",
    "CURRENT_ENERGY_EFFICIENCY",
    "POTENTIAL_ENERGY_EFFICIENCY",
    # --- Energy & emissions ---
    "ENERGY_CONSUMPTION_CURRENT",
    "ENERGY_CONSUMPTION_POTENTIAL",
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISSIONS_POTENTIAL",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "TOTAL_FLOOR_AREA",
    # --- Property characteristics ---
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "CONSTRUCTION_AGE_BAND",
    "NUMBER_HABITABLE_ROOMS",
    "NUMBER_HEATED_ROOMS",
    "FLOOR_HEIGHT",
    "MECHANICAL_VENTILATION",
    # --- Heating & lighting ---
    "MAIN_FUEL",
    "MAIN_HEATING_CONTROLS",
    "LOW_ENERGY_LIGHTING",
    "MAINS_GAS_FLAG",
    # --- Socioeconomic ---
    "TENURE",
    "TRANSACTION_TYPE",
]

# Light dtypes for large CSV reads/writes.
CORE_DTYPES: dict[str, str] = {
    "LMK_KEY": "string",
    "BUILDING_REFERENCE_NUMBER": "string",
    "POSTCODE": "string",
    "POSTTOWN": "string",
    "COUNTY": "string",
    "LOCAL_AUTHORITY": "string",
    "LOCAL_AUTHORITY_LABEL": "string",
    "LODGEMENT_DATE": "string",
    "CURRENT_ENERGY_RATING": "string",
    "POTENTIAL_ENERGY_RATING": "string",
    "PROPERTY_TYPE": "string",
    "BUILT_FORM": "string",
    "CONSTRUCTION_AGE_BAND": "string",
    "MAIN_FUEL": "string",
    "MAIN_HEATING_CONTROLS": "string",
    "MECHANICAL_VENTILATION": "string",
    "MAINS_GAS_FLAG": "string",
    "TENURE": "string",
    "TRANSACTION_TYPE": "string",
}

# Subset used in the South West cleaning step (plus engineered fields).
CLEAN_KEEP_COLS: list[str] = [
    "LMK_KEY",
    "POSTCODE",
    "LOCAL_AUTHORITY",
    "LOCAL_AUTHORITY_LABEL",
    "LODGEMENT_DATE",
    "CURRENT_ENERGY_RATING",
    "CURRENT_ENERGY_EFFICIENCY",
    "POTENTIAL_ENERGY_RATING",
    "POTENTIAL_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    "CO2_EMISSIONS_CURRENT",
    "TOTAL_FLOOR_AREA",
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "CONSTRUCTION_AGE_BAND",
    "MAIN_FUEL",
    "MAINS_GAS_FLAG",
    "TENURE",
    "TRANSACTION_TYPE",
    # optional extras:
    "NUMBER_HABITABLE_ROOMS",
    "NUMBER_HEATED_ROOMS",
    "FLOOR_HEIGHT",
    "MECHANICAL_VENTILATION",
    "MAIN_HEATING_CONTROLS",
]

# Dtypes used in cleaning (keep aligned with CORE_DTYPES).
CLEAN_DTYPES: dict[str, str] = {
    "LMK_KEY": "string",
    "POSTCODE": "string",
    "LOCAL_AUTHORITY": "string",
    "LOCAL_AUTHORITY_LABEL": "string",
    "LODGEMENT_DATE": "string",
    "CURRENT_ENERGY_RATING": "string",
    "POTENTIAL_ENERGY_RATING": "string",
    "PROPERTY_TYPE": "string",
    "BUILT_FORM": "string",
    "CONSTRUCTION_AGE_BAND": "string",
    "MAIN_FUEL": "string",
    "MAINS_GAS_FLAG": "string",
    "TRANSACTION_TYPE": "string",
    "TENURE": "string",
}
