"""Research prediction API for trusted, locally configured benchmark artifacts.

Run with EPC_MODEL_RUN and EPC_MODEL_VERIFICATION set, then:
uvicorn ew_housing_energy_impact.ml.serving:create_app --factory
Never accept a model path or serialized model from an API caller.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ew_housing_energy_impact.ml.evaluation import CATEGORICAL, FEATURES, predictive_inputs
from ew_housing_energy_impact.registry import sha256_file


class PropertyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    property_type: str | None = Field(default=None, min_length=1, max_length=100)
    age_band: str | None = Field(default=None, min_length=1, max_length=100)
    tenure: str | None = Field(default=None, min_length=1, max_length=100)
    main_fuel: str | None = Field(default=None, min_length=1, max_length=100)
    floor_area: float | None = Field(default=None, ge=20, le=1000)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    properties: list[PropertyInput] = Field(min_length=1, max_length=100)


def create_app(run: Path | None = None, verification: Path | None = None) -> FastAPI:
    """Fail at startup if completion, verification or artifact integrity is missing."""
    if run is None:
        run = Path(os.environ["EPC_MODEL_RUN"])
    if verification is None:
        verification = Path(os.environ["EPC_MODEL_VERIFICATION"])
    run, verification = Path(run), Path(verification)
    result = json.loads((run / "result.json").read_text())
    audit = json.loads(verification.read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if (run / "INCOMPLETE").exists() or result["status"] != "complete":
        raise ValueError("Cannot serve an incomplete benchmark")
    if (
        not audit.get("ok")
        or not audit.get("models_reloaded")
        or audit["run_id"] != result["run_id"]
    ):
        raise ValueError("Successful model verification for this run is required")
    if sha256_file(run / "protocol.json") != result["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    if protocol["features"] != FEATURES:
        raise ValueError("Unsupported model feature schema")
    models = {}
    for task in ("regression", "classification"):
        name = f"selected_{task}.joblib"
        if sha256_file(run / name) != result["artifacts"][name]:
            raise ValueError(f"Model hash mismatch: {task}")
        # joblib deserialisation is only for operator-controlled local artifacts.
        models[task] = joblib.load(run / name)
    encoder = (
        models["regression"]
        .named_steps["preprocess"]
        .named_transformers_["cat"]
        .named_steps["encode"]
    )
    categories = {
        name: [str(v) for v in values]
        for name, values in zip(CATEGORICAL, encoder.categories_, strict=True)
    }
    app = FastAPI(title="EPC research predictions", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "run_id": result["run_id"]}

    @app.get("/model")
    def metadata():
        return {
            "run_id": result["run_id"],
            "selected": result["selected"],
            "dataset_sha256": protocol["dataset_sha256"],
            "categories": categories,
            "floor_area_unit": "square metres",
            "intended_use": "Research only; not validated for eligibility or energy savings.",
            "unknown_categories": "Accepted using the fitted encoder; reported in each prediction.",
        }

    @app.post("/predict")
    def predict(request: PredictionRequest):
        rows, warnings = [], []
        for item in request.properties:
            row = dict(
                zip(
                    CATEGORICAL,
                    [item.property_type, item.age_band, item.tenure, item.main_fuel],
                    strict=True,
                )
            )
            unknown = [
                name
                for name, value in row.items()
                if value is not None and value not in categories[name]
            ]
            missing = [name for name, value in row.items() if value is None]
            row["LOG_FLOOR_AREA"] = (
                np.log(item.floor_area) if item.floor_area is not None else np.nan
            )
            if item.floor_area is None:
                missing.append("floor_area")
            rows.append(row)
            warnings.append({"unknown_categories": unknown, "missing_inputs": missing})
        inputs = predictive_inputs(pd.DataFrame(rows))
        scores = models["regression"].predict(inputs)
        probabilities = models["classification"].predict_proba(inputs)[:, 1]
        if not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
            raise HTTPException(status_code=500, detail="Model produced non-finite predictions")
        return {
            "run_id": result["run_id"],
            "predictions": [
                {
                    "epc_score": float(score),
                    "below_c_probability": float(probability),
                    "below_c": bool(probability > 0.5),
                    **warning,
                }
                for score, probability, warning in zip(scores, probabilities, warnings, strict=True)
            ],
        }

    return app
