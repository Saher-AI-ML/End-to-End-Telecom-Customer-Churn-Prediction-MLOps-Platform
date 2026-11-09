"""FastAPI service that serves the saved churn model.

Run from the project root:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model = joblib.load(MODELS_DIR / "churn_model.joblib")
meta = json.loads((MODELS_DIR / "metadata.json").read_text())
metrics = json.loads((MODELS_DIR / "metrics.json").read_text())  

THRESHOLD = meta["threshold"]
NUM_COLS = meta["features"]["numeric"]
CAT_COLS = meta["features"]["categorical"]
CATEGORIES = meta["categories"] 

app = FastAPI(title="WE Churn API", description="Predicts the chance that a customer will churn.")


class Customer(BaseModel):
    governorate: Optional[str] = None
    customer_type: Optional[str] = None
    subscription_plan: Optional[str] = None
    payment_method: Optional[str] = None
    monthly_bill_egp: Optional[float] = Field(None, ge=0)
    tenure_months: Optional[float] = Field(None, ge=0)
    complaint_count: Optional[int] = Field(None, ge=0)
    downtime_hours_monthly: Optional[float] = Field(None, ge=0)

    model_config = ConfigDict(json_schema_extra={"example": {
        "governorate": "Cairo", "customer_type": "Individual", "subscription_plan": "WE Plus 40Mbps",
        "payment_method": "Credit Card", "monthly_bill_egp": 400.0, "tenure_months": 24,
        "complaint_count": 2, "downtime_hours_monthly": 3.5}})

    @field_validator("governorate", "customer_type", "subscription_plan", "payment_method")
    @classmethod
    def check_category(cls, value, info):
        allowed = CATEGORIES[info.field_name]
        if value is not None and value not in allowed:
            raise ValueError(f"'{value}' is not valid. Allowed values: {allowed}")
        return value


# 3) The OUTPUT schema.
class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float


def to_frame(customer: Customer) -> pd.DataFrame:
    """Turn the request into a one-row DataFrame with the raw column names (None -> NaN),
    exactly like the handoff test in train.py."""
    df = pd.DataFrame([customer.model_dump()], columns=NUM_COLS + CAT_COLS)
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


# 4) The endpoints.
@app.get("/health")
def health():
    """Is the API running and is the model loaded?"""
    return {"status": "ok", "model": meta["model_name"]}


@app.get("/model-info")
def model_info():
    """Threshold, feature names and allowed categories (the dashboard will use this for its dropdowns)."""
    return {"model": meta["model_name"], "threshold": THRESHOLD, "features": meta["features"],
            "categories": CATEGORIES, "numeric_ranges": meta["numeric_ranges"],
            "trained_at_utc": meta["trained_at_utc"]}


@app.get("/metrics")
def get_metrics():
    """The model's CV and test scores from metrics.json (the dashboard's performance page shows them)."""
    return metrics


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    """Probability that this customer churns, plus a yes/no using the saved threshold."""
    probability = float(model.predict_proba(to_frame(customer))[0, 1])
    return Prediction(churn_probability=probability, will_churn=probability >= THRESHOLD, threshold=THRESHOLD)
