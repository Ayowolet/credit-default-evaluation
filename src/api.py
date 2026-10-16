"""Credit default scoring API (FastAPI).

    uvicorn src.api:app --port 8000

POST /predict   raw account fields in, probability of default + risk band out (optional reason codes)
GET  /health    liveness + model version
GET  /model-card  what the served model is, how it was evaluated, and its intended use

Demonstration service on public data. Not validated for real credit decisions.
Inputs are never logged: each request logs only an id, timing, and the risk band.
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import xgboost
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.record import engineer_record

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
log = logging.getLogger("credit-api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _build_input_model():
    fields = {"LIMIT_BAL": (float, Field(gt=0, le=1e8, description="Credit limit"))}
    fields["EDUCATION"] = (int, Field(ge=0, le=6, description="1 grad school, 2 university, 3 high school, 4+ other"))
    for k in range(1, 7):
        fields[f"PAY_{k}"] = (int, Field(ge=-2, le=9, description=f"Repayment status {k} months ago: -2 no consumption, -1 paid in full, 0 revolving, 1+ months late"))
    for k in range(1, 7):
        fields[f"BILL_AMT{k}"] = (float, Field(ge=-1e7, le=1e8, description=f"Statement amount {k} months ago"))
    for k in range(1, 7):
        fields[f"PAY_AMT{k}"] = (float, Field(ge=0, le=1e8, description=f"Payment made {k} months ago"))
    return create_model("CreditRecord", __config__=ConfigDict(extra="forbid"), **fields)


CreditRecord = _build_input_model()


class Reason(BaseModel):
    feature: str
    contribution: float = Field(description="Contribution to the model output in log-odds")
    direction: str


class Prediction(BaseModel):
    probability_of_default: float
    risk_band: str
    model_version: str
    request_id: str
    reasons: Optional[List[Reason]] = None


class Model:
    """Loads the artefact once and scores single records without pandas."""

    def __init__(self, model_dir: Path):
        pipe = joblib.load(model_dir / "model.joblib")
        self.card = json.loads((model_dir / "model_card.json").read_text())
        self.input_features = self.card["input_features"]
        mask = pipe.named_steps["select"].get_support()
        self.idx = np.flatnonzero(mask)
        self.names = [self.input_features[i] for i in self.idx]
        self.clf = pipe.named_steps["clf"]
        self.clf.set_params(n_jobs=1)  # request-level serving: no thread fan-out
        self.bands = self.card["risk_bands"]
        self.version = self.card["version"]

    def _vector(self, rec: dict) -> np.ndarray:
        feats = engineer_record(rec)
        full = np.fromiter((feats[n] for n in self.input_features), dtype=np.float64,
                           count=len(self.input_features))
        return full[self.idx].reshape(1, -1)

    def band(self, p: float) -> str:
        return next(name for upper, name in self.bands if p < upper)

    def predict(self, rec: dict, explain: bool = False):
        x = self._vector(rec)
        p = float(self.clf.predict_proba(x)[0, 1])
        reasons = None
        if explain:  # exact TreeSHAP values from the booster itself (last column is the bias)
            contrib = self.clf.get_booster().predict(xgboost.DMatrix(x, feature_names=self.names), pred_contribs=True)[0][:-1]
            order = np.argsort(-np.abs(contrib))[:3]
            reasons = [Reason(feature=self.names[i], contribution=round(float(contrib[i]), 4),
                              direction="increases_risk" if contrib[i] > 0 else "decreases_risk") for i in order]
        return p, self.band(p), reasons


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Model(MODEL_DIR)
    log.info(json.dumps({"event": "model_loaded", "version": app.state.model.version}))
    yield


app = FastAPI(title="Credit default scoring API", version="1.0.0", lifespan=lifespan,
              description="Demonstration service on public data. Not validated for real credit decisions.")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model.version}


@app.get("/model-card")
def model_card(request: Request):
    return request.app.state.model.card


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(record: CreditRecord, request: Request, explain: bool = False):
    model: Model = request.app.state.model
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        p, band, reasons = model.predict(record.model_dump(), explain=explain)
    except Exception:  # never leak internals or inputs
        log.exception(json.dumps({"event": "predict_error", "request_id": rid}))
        raise HTTPException(status_code=500, detail="scoring failed")
    log.info(json.dumps({"event": "predict", "request_id": rid, "risk_band": band,
                         "explain": explain, "ms": round((time.perf_counter() - t0) * 1000, 3)}))
    return Prediction(probability_of_default=round(p, 6), risk_band=band, model_version=model.version,
                      request_id=rid, reasons=reasons)
