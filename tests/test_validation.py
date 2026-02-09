import pandas as pd

from ew_housing_energy_impact.validation import validate_clean_data


def test_validation_ok_minimal():
    df = pd.DataFrame(
        {
            "LMK_KEY": ["a"],
            "POSTCODE": ["AB1"],
            "LOCAL_AUTHORITY": ["X"],
            "LOCAL_AUTHORITY_LABEL": ["Test LA"],
            "LODGEMENT_DATE": ["2020-01-01"],
            "CURRENT_ENERGY_RATING": ["D"],
            "CURRENT_ENERGY_EFFICIENCY": [60],
            "POTENTIAL_ENERGY_RATING": ["C"],
            "POTENTIAL_ENERGY_EFFICIENCY": [75],
            "ENERGY_CONSUMPTION_CURRENT": [200],
            "CO2_EMISSIONS_CURRENT": [5],
            "TOTAL_FLOOR_AREA": [80],
            "PROPERTY_TYPE": ["House"],
            "BUILT_FORM": ["Detached"],
            "CONSTRUCTION_AGE_BAND": ["2007-2011"],
            "MAIN_FUEL": ["mains gas"],
            "MAINS_GAS_FLAG": ["Y"],
            "TENURE": ["Owner-occupied"],
            "TRANSACTION_TYPE": ["marketed sale"],
            "NUMBER_HABITABLE_ROOMS": [4],
            "NUMBER_HEATED_ROOMS": [4],
            "FLOOR_HEIGHT": [2.4],
            "MECHANICAL_VENTILATION": ["Natural"],
            "MAIN_HEATING_CONTROLS": ["Programmer"],
            "YEAR": [2020],
            "POLICY_PERIOD": ["MEES"],
            "BELOW_C": [True],
        }
    )

    res = validate_clean_data(df)
    assert res.ok is True
