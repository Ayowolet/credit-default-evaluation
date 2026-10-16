"""Run the full study: python -m src.run   (from the repository root).

Everything is seeded. Outputs go to results/ (metrics.json, results.md, *.png).
"""
import copy
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import SEED, make_models, score, single_feature_auc_screen, top_k_jaccard, xgb
from src.features import build

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    R = {}  # everything that goes to metrics.json
    X, y, sens = build()
    R["data"] = {"rows": int(len(X)), "features": int(X.shape[1]), "default_rate": float(y.mean())}
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sens, test_size=0.2, stratify=y, random_state=SEED)
    R["data"].update(train=int(len(X_tr)), test=int(len(X_te)))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print("data", R["data"])

    # ---- 1. Model comparison: 5-fold CV on train only, then ONE evaluation on the held-out test set
    models = make_models()
    cv_rows, test_rows, fitted, test_prob = [], [], {}, {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring={
            "roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"})
        cv_rows.append({"model": name,
                        "cv_roc_auc": res["test_roc_auc"].mean(), "cv_roc_auc_sd": res["test_roc_auc"].std(),
                        "cv_pr_auc": res["test_pr_auc"].mean(), "cv_brier": -res["test_brier"].mean()})
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        fitted[name], test_prob[name] = pipe, p
        test_rows.append({"model": name, **score(y_te, p)})
        print("done", name, round(time.time() - t0), "s")
    cv_df, test_df = pd.DataFrame(cv_rows), pd.DataFrame(test_rows)
    R["cv"], R["test"] = cv_rows, test_rows

    # ---- 2. What feature selection actually did
    sel = fitted["xgboost_selected"].named_steps["select"]
    kept = list(X_tr.columns[sel.get_support()])
    R["selection"] = {"before": int(X_tr.shape[1]), "after": len(kept), "kept": kept}
    print("selection", X_tr.shape[1], "->", len(kept))

    # ---- 3. Leakage experiment A: selecting features BEFORE cross-validation (pure noise data)
    wrong, right = [], []
    for s in range(10):
        rng = np.random.default_rng(s)
        Xn, yn = rng.normal(size=(200, 2000)), rng.integers(0, 2, 200)
        Xsel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)  # selection sees ALL labels
        wrong.append(cross_val_score(LogisticRegression(max_iter=2000), Xsel, yn, cv=5, scoring="roc_auc").mean())
        pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=2000))])
        right.append(cross_val_score(pipe, Xn, yn, cv=5, scoring="roc_auc").mean())
    R["leak_selection_noise"] = {"auc_selection_outside_cv": float(np.mean(wrong)), "sd_outside": float(np.std(wrong)),
                                 "auc_selection_inside_cv": float(np.mean(right)), "sd_inside": float(np.std(right))}
    print("leak A", R["leak_selection_noise"])

    # ---- 4. Leakage experiment B: a feature that encodes the label
    rng = np.random.default_rng(SEED)
    Xl_tr, Xl_te = X_tr.copy(), X_te.copy()
    Xl_tr["leaky_feature"] = y_tr.values + rng.normal(0, 0.5, len(y_tr))
    Xl_te["leaky_feature"] = y_te.values + rng.normal(0, 0.5, len(y_te))
    m_leak = xgb().fit(Xl_tr, y_tr)
    auc_leak = float(roc_auc_score(y_te, m_leak.predict_proba(Xl_te)[:, 1]))
    gain = pd.Series(m_leak.get_booster().get_score(importance_type="total_gain"))
    leak_share = float(gain["leaky_feature"] / gain.sum())
    scr_clean = single_feature_auc_screen(X_tr, y_tr)
    scr_leak = single_feature_auc_screen(Xl_tr, y_tr)
    row_order_auc = float(roc_auc_score(y, np.arange(len(y))))
    R["leak_label_feature"] = {
        "test_auc_clean": float(test_df.loc[test_df.model == "xgboost", "roc_auc"].iloc[0]),
        "test_auc_with_leak": auc_leak, "leak_share_of_gain": leak_share,
        "max_single_feature_auc_clean": float(scr_clean.auc.iloc[0]), "top_clean_feature": scr_clean.feature.iloc[0],
        "flagged_clean": int(scr_clean.flag.sum()),
        "max_single_feature_auc_with_leak": float(scr_leak.auc.iloc[0]), "flagged_with_leak": scr_leak.feature[scr_leak.flag].tolist(),
        "row_order_auc": row_order_auc}
    print("leak B", R["leak_label_feature"])

    # ---- 5. Explanations: SHAP (checked), compared with permutation importance, and stability across seeds
    m = fitted["xgboost"].named_steps["clf"]
    Xs = X_te.sample(1500, random_state=SEED)
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(Xs)
    margin = m.predict(Xs, output_margin=True)
    base = float(np.ravel(expl.expected_value)[0])
    additivity_err = float(np.max(np.abs(sv.sum(axis=1) + base - margin)))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    perm = permutation_importance(fitted["xgboost"], X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=SEED, n_jobs=1)
    perm_s = pd.Series(perm.importances_mean, index=X.columns)
    rho = float(spearmanr(mean_abs.reindex(X.columns), perm_s).correlation)
    top_shap, top_perm = set(mean_abs.index[:10]), set(perm_s.sort_values(ascending=False).index[:10])
    imps = []
    for s in range(5):
        ms = xgb(seed=100 + s).fit(X_tr, y_tr)
        imps.append(np.abs(shap.TreeExplainer(ms).shap_values(Xs.iloc[:800])).mean(axis=0))
    R["explain"] = {"shap_additivity_max_abs_error": additivity_err, "spearman_shap_vs_permutation": rho,
                    "top10_overlap_shap_vs_permutation": len(top_shap & top_perm),
                    "top10_jaccard_across_5_seeds": top_k_jaccard(imps, 10),
                    "top10_shap": mean_abs.index[:10].tolist()}
    print("explain", {k: v for k, v in R["explain"].items() if k != "top10_shap"})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    top = mean_abs.head(15)[::-1]
    ax.barh(top.index, top.values, color="#333")
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("Global drivers of predicted default (XGBoost, test sample)")
    fig.tight_layout(); fig.savefig(OUT / "shap_global.png"); plt.close(fig)

    p_all = m.predict_proba(Xs)[:, 1]
    i = int(np.argmax(p_all))
    shap.plots.waterfall(shap.Explanation(values=sv[i], base_values=base, data=Xs.iloc[i].values,
                                          feature_names=list(X.columns)), max_display=10, show=False)
    plt.title(f"One applicant: predicted default probability {p_all[i]:.0%}")
    plt.tight_layout(); plt.savefig(OUT / "shap_local.png"); plt.close()

    # ---- 6. ROC + calibration
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in test_prob.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].legend(fontsize=7); ax[0].set_title("ROC, held-out test")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    for name in ["logistic_regression", "xgboost"]:
        frac, mean_p = calibration_curve(y_te, test_prob[name], n_bins=10, strategy="quantile")
        ax[1].plot(mean_p, frac, "o-", label=name)
    ax[1].plot([0, 0.8], [0, 0.8], "k--", lw=0.8); ax[1].legend(fontsize=7)
    ax[1].set_title("Calibration, held-out test"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("observed default rate")
    fig.tight_layout(); fig.savefig(OUT / "roc_calibration.png"); plt.close(fig)

    # ---- 7. Slice check on attributes that were NOT model inputs (not a fairness guarantee)
    p = test_prob["xgboost"]
    cutoff = np.quantile(p, 0.80)  # approve the lowest-risk 80%
    tmp = pd.DataFrame({"y": y_te.values, "p": p, "SEX": s_te["SEX"].map({1: "male", 2: "female"}).values,
                        "AGE_BAND": s_te["AGE_BAND"].astype(str).values})
    slice_rows = []
    for col in ["SEX", "AGE_BAND"]:
        for g, d in tmp.groupby(col):
            slice_rows.append({"slice": f"{col}={g}", "n": len(d), "observed_default_rate": d.y.mean(),
                               "mean_predicted": d.p.mean(), "auc": roc_auc_score(d.y, d.p),
                               "approval_rate": float((d.p <= cutoff).mean())})
    slice_df = pd.DataFrame(slice_rows)
    R["slices"] = slice_rows

    # ---- 8. Single-row inference latency (what an API call would pay), artefacts reloaded from disk
    lat_rows = []
    rows = [X_te.iloc[[k]] for k in range(300)]
    for name, pipe in fitted.items():
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "m.joblib"
            joblib.dump(pipe, f)
            size_kb = f.stat().st_size / 1024
            mdl = joblib.load(f)
        clf = mdl.named_steps["clf"]
        if hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)  # single-request serving: no thread fan-out
        for r in rows[:30]:
            mdl.predict_proba(r)  # warm-up
        times = []
        for _ in range(3):
            for r in rows:
                a = time.perf_counter(); mdl.predict_proba(r); times.append((time.perf_counter() - a) * 1000)
        t = np.array(times)
        lat_rows.append({"model": name, "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95),
                         "p99_ms": np.percentile(t, 99), "artifact_kb": size_kb})
    lat_df = pd.DataFrame(lat_rows)
    R["latency"] = lat_rows
    R["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
                        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(lat_df.model, lat_df.p95_ms, color="#555")
    ax.set_xlabel("p95 single-row latency (ms, scoring in-process)")
    fig.tight_layout(); fig.savefig(OUT / "latency.png"); plt.close(fig)

    # ---- write tables
    (OUT / "metrics.json").write_text(json.dumps(R, indent=2, default=float))
    parts = [
        "## Cross-validation (5-fold, training split only)\n" + md(cv_df),
        "## Held-out test (evaluated once)\n" + md(test_df),
        "## Feature selection\n" + f"{X_tr.shape[1]} features in, {len(kept)} kept (SelectFromModel, median threshold, fitted inside every CV fold).",
        "## Leakage check A: selecting features before cross-validation, on pure noise\n" + md(pd.DataFrame([{
            "selection": "outside CV (wrong)", "mean_auc": R["leak_selection_noise"]["auc_selection_outside_cv"], "sd": R["leak_selection_noise"]["sd_outside"]},
            {"selection": "inside CV pipeline (right)", "mean_auc": R["leak_selection_noise"]["auc_selection_inside_cv"], "sd": R["leak_selection_noise"]["sd_inside"]}])),
        "## Leakage check B\n" + json.dumps(R["leak_label_feature"], indent=2),
        "## Explanation checks\n" + json.dumps({k: v for k, v in R["explain"].items()}, indent=2),
        "## Slices (attributes not used as inputs)\n" + md(slice_df),
        "## Latency\n" + md(lat_df, "{:.2f}"),
        "## Environment\n" + json.dumps(R["environment"], indent=2),
    ]
    (OUT / "results.md").write_text("\n\n".join(parts) + "\n")
    print("finished in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
