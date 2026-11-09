"""Shared test helpers (pytest loads this file automatically)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """A fake browser that calls the API in memory, without starting a real server."""
    return TestClient(app)


@pytest.fixture(scope="session")
def valid_customer(client):
    """A valid request. The categories come from the model's own /model-info,
    so this test data never goes out of date when you retrain."""
    categories = client.get("/model-info").json()["categories"]
    return {
        "governorate": categories["governorate"][0],
        "customer_type": categories["customer_type"][0],
        "subscription_plan": categories["subscription_plan"][0],
        "payment_method": categories["payment_method"][0],
        "monthly_bill_egp": 400.0,
        "tenure_months": 24,
        "complaint_count": 1,
        "downtime_hours_monthly": 2.0,
    }
