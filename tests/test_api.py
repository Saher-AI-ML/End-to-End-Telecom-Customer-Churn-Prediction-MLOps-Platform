"""Tests for the FastAPI service. Each function starting with test_ is one check."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_has_threshold_and_categories(client):
    info = client.get("/model-info").json()
    assert 0 < info["threshold"] < 1
    for column in ["governorate", "customer_type", "subscription_plan", "payment_method"]:
        assert len(info["categories"][column]) > 0


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "roc_auc" in response.json()["test"]


def test_predict_valid_customer(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    # the yes/no answer must follow the saved threshold
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])


def test_predict_works_with_missing_fields(client):
    # The pipeline fills in missing values, so a partial request is still accepted.
    response = client.post("/predict", json={"tenure_months": 12, "complaint_count": 1})
    assert response.status_code == 200


def test_predict_rejects_unknown_category(client, valid_customer):
    bad = dict(valid_customer, governorate="Atlantis")
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_negative_tenure(client, valid_customer):
    bad = dict(valid_customer, tenure_months=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_text_in_number_field(client, valid_customer):
    bad = dict(valid_customer, monthly_bill_egp="abc")
    assert client.post("/predict", json=bad).status_code == 422


def test_risky_customer_scores_higher_than_safe_customer(client, valid_customer):
    safe = dict(valid_customer, complaint_count=0, downtime_hours_monthly=0.0, tenure_months=60)
    risky = dict(valid_customer, complaint_count=8, downtime_hours_monthly=10.0, tenure_months=20)
    safe_p = client.post("/predict", json=safe).json()["churn_probability"]
    risky_p = client.post("/predict", json=risky).json()["churn_probability"]
    assert risky_p > safe_p
