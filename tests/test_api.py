"""Parity and API tests. Need artifacts/: run `python -m src.train_export` first."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import build, clean, engineer, load_raw
from src.record import engineer_record

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.joblib").exists(), reason="run src.train_export first")


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(ART / "sample_requests.csv")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:
        yield c


def _payload(row):
    d = row.drop("offline_probability").to_dict()
    return {k: (int(v) if k == "EDUCATION" or k.startswith("PAY_") and not k.startswith("PAY_AMT") else float(v))
            for k, v in d.items()}


def test_fast_features_match_pandas_path(sample):
    raw = pd.read_csv(ART / "sample_requests.csv").drop(columns=["offline_probability"])
    full = raw.copy()
    for c, v in (("SEX", 1), ("AGE", 30), ("MARRIAGE", 1), ("default", 0)):
        full[c] = v
    ref = engineer(clean(full))
    for i in range(len(raw)):
        got = engineer_record(raw.iloc[i].to_dict())
        np.testing.assert_allclose([got[c] for c in ref.columns], ref.iloc[i].to_numpy(float),
                                   rtol=1e-9, atol=1e-9, err_msg=f"row {i}")


def test_api_matches_offline_probabilities(client, sample):
    for _, row in sample.head(60).iterrows():
        r = client.post("/predict", json=_payload(row))
        assert r.status_code == 200
        assert abs(r.json()["probability_of_default"] - row["offline_probability"]) < 1e-5


def test_response_shape_and_reasons(client, sample):
    r = client.post("/predict?explain=true", json=_payload(sample.iloc[0])).json()
    assert r["risk_band"] in {"low", "medium", "high", "very_high"}
    assert len(r["reasons"]) == 3 and r["model_version"] and r["request_id"]
    assert "reasons" not in client.post("/predict", json=_payload(sample.iloc[0])).json()


def test_rejects_bad_input(client, sample):
    p = _payload(sample.iloc[0])
    assert client.post("/predict", json={**p, "LIMIT_BAL": -5}).status_code == 422
    assert client.post("/predict", json={**p, "SEX": 1}).status_code == 422  # protected field refused
    assert client.post("/predict", json={k: v for k, v in p.items() if k != "PAY_1"}).status_code == 422


def test_health_and_card(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Demonstration" in client.get("/model-card").json()["intended_use"]
