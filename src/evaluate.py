"""Metrics, model factories, and leakage/stability checks."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def score(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def xgb(seed=SEED, **kw):
    params = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                  eval_metric="logloss", n_jobs=2, random_state=seed)
    params.update(kw)
    return XGBClassifier(**params)


def make_models(seed=SEED) -> dict:
    """All models are pipelines so any preprocessing/selection is fitted inside each CV fold."""
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=2, random_state=seed))]),
        "xgboost": Pipeline([("clf", xgb(seed))]),
        "xgboost_selected": Pipeline([
            ("select", SelectFromModel(xgb(seed, n_estimators=100), threshold="median")),
            ("clf", xgb(seed))]),
    }


def single_feature_auc_screen(X: pd.DataFrame, y, flag_above=0.85) -> pd.DataFrame:
    """Cheap leakage screen: any single input that nearly separates the classes alone is suspect."""
    rows = []
    for c in X.columns:
        a = roc_auc_score(y, X[c])
        rows.append((c, max(a, 1 - a)))
    res = pd.DataFrame(rows, columns=["feature", "auc"]).sort_values("auc", ascending=False)
    res["flag"] = res["auc"] > flag_above
    return res.reset_index(drop=True)


def top_k_jaccard(importances: list, k=10) -> float:
    """Mean pairwise Jaccard overlap of the top-k features across repeated fits (stability)."""
    tops = [set(np.argsort(-np.asarray(v))[:k]) for v in importances]
    vals = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    return float(np.mean(vals))
