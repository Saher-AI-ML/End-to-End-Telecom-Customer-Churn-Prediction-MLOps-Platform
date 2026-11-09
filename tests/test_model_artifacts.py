"""Tests for the saved model files in models/ (the handoff between training and the API)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
import xgboost

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def meta():
    return json.loads((MODELS_DIR / "metadata.json").read_text())


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS_DIR / "churn_model.joblib")


def one_row(meta, **overrides):
    """A one-row DataFrame with the raw column names, like the API builds."""
    numeric = meta["features"]["numeric"]
    categorical = meta["features"]["categorical"]
    row = {column: 1.0 for column in numeric}
    row.update({column: meta["categories"][column][0] for column in categorical})
    row.update(overrides)
    df = pd.DataFrame([row])
    for column in categorical:
        df[column] = df[column].astype(object)
    return df


def test_metadata_has_required_keys(meta):
    for key in ["features", "categories", "threshold", "library_versions", "numeric_ranges", "model_name"]:
        assert key in meta
    assert 0 < meta["threshold"] < 1


def test_model_predicts_one_row(model, meta):
    probabilities = model.predict_proba(one_row(meta))
    assert probabilities.shape == (1, 2)
    assert 0.0 <= probabilities[0, 1] <= 1.0


def test_model_handles_missing_values(model, meta):
    row = one_row(meta, tenure_months=np.nan, payment_method=np.nan)
    probabilities = model.predict_proba(row)
    assert 0.0 <= probabilities[0, 1] <= 1.0


def test_model_handles_unseen_category(model, meta):
    probabilities = model.predict_proba(one_row(meta, governorate="Atlantis"))
    assert 0.0 <= probabilities[0, 1] <= 1.0


def test_installed_libraries_match_training_versions(meta):
    versions = meta["library_versions"]
    assert versions["scikit-learn"] == sklearn.__version__
    assert versions["xgboost"] == xgboost.__version__
