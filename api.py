from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"

app = FastAPI(title="EPC South West API", version="0.1.0")


class TrendResponse(BaseModel):
    year: int
    mean_epc: float
    mean_co2: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trends", response_model=list[TrendResponse])
def trends():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Data not found. Run pipeline first.")
    df = pd.read_csv(DATA_PATH, usecols=["YEAR", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT"])
    yearly = (
        df.groupby("YEAR", as_index=False)
        .agg(mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
             mean_co2=("CO2_EMISSIONS_CURRENT", "mean"))
        .sort_values("YEAR")
    )
    return yearly.to_dict(orient="records")


@app.get("/policy-summary")
def policy_summary():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Data not found. Run pipeline first.")
    df = pd.read_csv(DATA_PATH, usecols=["POLICY_PERIOD", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT", "BELOW_C"])
    summary = (
        df.groupby("POLICY_PERIOD", as_index=False)
        .agg(mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
             mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
             below_c_share=("BELOW_C", "mean"))
    )
    return summary.to_dict(orient="records")
