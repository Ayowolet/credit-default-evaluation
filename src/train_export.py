"""Train the serving model and write a versioned artefact + model card.

    python -m src.train_export

Uses the same split, seed and pipeline as the evaluation (src/run.py), so the served model is
exactly the one whose held-out metrics are reported. Writes to artifacts/ (git-ignored).
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

from src.evaluate import SEED, make_models, score
from src.features import DATA_PATH, build

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_NAME = "xgboost_selected"
# Illustrative risk bands on predicted default probability. NOT a credit policy.
RISK_BANDS = [(0.10, "low"), (0.25, "medium"), (0.45, "high"), (1.01, "very_high")]


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    X, y, _ = build()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = make_models()[MODEL_NAME].fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    support = pipe.named_steps["select"].get_support()

    joblib.dump(pipe, ARTIFACTS / "model.joblib")
    card = {
        "model": MODEL_NAME,
        "version": datetime.now(timezone.utc).strftime("%Y.%m.%d") + "-" + hashlib.sha256(
            (ARTIFACTS / "model.joblib").read_bytes()).hexdigest()[:8],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(X_tr)),
        "input_features": list(X.columns),
        "selected_features": list(X.columns[support]),
        "test_metrics": score(y_te, p),
        "risk_bands": RISK_BANDS,
        "data_source": "Default of Credit Card Clients (Yeh & Lien, 2009), public dataset",
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__},
        "intended_use": "Demonstration only. Not validated for real credit decisions.",
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2))

    # Small fixture of raw records + offline probabilities for parity tests and the load test.
    raw = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    fixture = raw.loc[X_te.index[:200]].drop(columns=["default", "SEX", "AGE", "MARRIAGE"])
    fixture["offline_probability"] = p[:200]
    fixture.to_csv(ARTIFACTS / "sample_requests.csv", index=False)
    print("version", card["version"], "| test AUC", round(card["test_metrics"]["roc_auc"], 3),
          "| selected", len(card["selected_features"]), "of", len(card["input_features"]))


if __name__ == "__main__":
    main()
