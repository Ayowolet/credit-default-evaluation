## Cross-validation (5-fold, training split only)
| model | cv_roc_auc | cv_roc_auc_sd | cv_pr_auc | cv_brier |
|---|---|---|---|---|
| logistic_regression | 0.755 | 0.012 | 0.505 | 0.141 |
| random_forest | 0.781 | 0.008 | 0.551 | 0.135 |
| xgboost | 0.783 | 0.009 | 0.552 | 0.134 |
| xgboost_selected | 0.784 | 0.008 | 0.552 | 0.134 |

## Held-out test (evaluated once)
| model | roc_auc | pr_auc | ks | brier |
|---|---|---|---|---|
| logistic_regression | 0.763 | 0.518 | 0.415 | 0.140 |
| random_forest | 0.788 | 0.565 | 0.440 | 0.133 |
| xgboost | 0.791 | 0.560 | 0.447 | 0.133 |
| xgboost_selected | 0.790 | 0.564 | 0.449 | 0.132 |

## Feature selection
42 features in, 21 kept (SelectFromModel, median threshold, fitted inside every CV fold).

## Leakage check A: selecting features before cross-validation, on pure noise
| selection | mean_auc | sd |
|---|---|---|
| outside CV (wrong) | 0.854 | 0.032 |
| inside CV pipeline (right) | 0.505 | 0.047 |

## Leakage check B
{
  "test_auc_clean": 0.790997333441073,
  "test_auc_with_leak": 0.9503305422003151,
  "leak_share_of_gain": 0.7574594776705253,
  "max_single_feature_auc_clean": 0.7215720176457908,
  "top_clean_feature": "n_late",
  "flagged_clean": 0,
  "max_single_feature_auc_with_leak": 0.9188102038599195,
  "flagged_with_leak": [
    "leaky_feature"
  ],
  "row_order_auc": 0.49115707753355115
}

## Explanation checks
{
  "shap_additivity_max_abs_error": 2.6226043701171875e-06,
  "spearman_shap_vs_permutation": 0.8860708208410988,
  "top10_overlap_shap_vs_permutation": 8,
  "top10_jaccard_across_5_seeds": 0.8272727272727274,
  "top10_shap": [
    "PAY_1",
    "max_delay",
    "n_late",
    "BILL_AMT1",
    "LIMIT_BAL",
    "PAY_AMT2",
    "mean_delay",
    "bill_trend",
    "PAY_AMT1",
    "PAY_AMT3"
  ]
}

## Slices (attributes not used as inputs)
| slice | n | observed_default_rate | mean_predicted | auc | approval_rate |
|---|---|---|---|---|---|
| SEX=female | 2898 | 0.205 | 0.215 | 0.780 | 0.817 |
| SEX=male | 1898 | 0.245 | 0.233 | 0.804 | 0.773 |
| AGE_BAND=30-44 | 2393 | 0.199 | 0.207 | 0.789 | 0.826 |
| AGE_BAND=45+ | 807 | 0.263 | 0.238 | 0.801 | 0.765 |
| AGE_BAND=<30 | 1596 | 0.232 | 0.236 | 0.784 | 0.779 |

## Latency
| model | p50_ms | p95_ms | p99_ms | artifact_kb |
|---|---|---|---|---|
| logistic_regression | 0.87 | 1.25 | 1.90 | 3.30 |
| random_forest | 15.03 | 18.89 | 21.69 | 19232.08 |
| xgboost | 2.61 | 3.52 | 4.50 | 347.24 |
| xgboost_selected | 2.02 | 2.78 | 3.31 | 473.12 |

## Environment
{
  "python": "3.13.16",
  "platform": "Linux-6.18.44-fc-v139-x86_64-with-glibc2.39",
  "cpus": 2,
  "sklearn": "1.9.1",
  "xgboost": "3.4.1",
  "shap": "0.54.1"
}
