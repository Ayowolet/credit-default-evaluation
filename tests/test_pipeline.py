import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.evaluate import make_models, single_feature_auc_screen
from src.features import PROTECTED, TARGET, build

X, y, sens = build()


def test_no_target_or_protected_attributes_in_inputs():
    assert TARGET not in X.columns
    for col in PROTECTED:
        assert col not in X.columns


def test_inputs_are_finite_and_aligned():
    assert np.isfinite(X.to_numpy()).all()
    assert X.index.equals(y.index) and X.index.equals(sens.index)


def test_feature_selection_lives_inside_the_pipeline():
    steps = [name for name, _ in make_models()["xgboost_selected"].steps]
    assert steps == ["select", "clf"]


def test_selection_inside_cv_gives_chance_level_on_noise():
    rng = np.random.default_rng(0)
    Xn, yn = rng.normal(size=(200, 500)), rng.integers(0, 2, 200)
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=1000))])
    assert cross_val_score(pipe, Xn, yn, cv=5, scoring="roc_auc").mean() < 0.65


def test_fit_is_reproducible():
    Xs, ys = X.iloc[:3000], y.iloc[:3000]
    p1 = make_models()["xgboost"].fit(Xs, ys).predict_proba(Xs)[:, 1]
    p2 = make_models()["xgboost"].fit(Xs, ys).predict_proba(Xs)[:, 1]
    assert np.allclose(p1, p2)


def test_leakage_screen_flags_a_label_copy_and_not_clean_data():
    assert single_feature_auc_screen(X, y)["flag"].sum() == 0
    leaky = X.copy()
    leaky["leak"] = y.values + np.random.default_rng(0).normal(0, 0.5, len(y))
    assert single_feature_auc_screen(leaky, y).query("flag")["feature"].tolist() == ["leak"]
