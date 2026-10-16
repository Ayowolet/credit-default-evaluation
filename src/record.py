"""Single-record feature engineering for serving: no pandas, ~100x faster than the batch path.

`engineer_record` must produce exactly the same values, in the same order, as
`src.features.engineer` for one row. tests/test_record_parity.py enforces this on hundreds of rows.
"""
import math

EDUCATION_MAP = {0: 4, 5: 4, 6: 4}  # undocumented codes merged into 'other', as in training


def _clean_nan(v: float) -> float:
    return 0.0 if (math.isnan(v) or math.isinf(v)) else float(v)


def engineer_record(rec: dict) -> dict:
    """rec holds the raw fields (LIMIT_BAL, EDUCATION, PAY_1..6, BILL_AMT1..6, PAY_AMT1..6)."""
    limit = float(rec["LIMIT_BAL"])
    edu = EDUCATION_MAP.get(int(rec["EDUCATION"]), int(rec["EDUCATION"]))
    pay = [float(rec[f"PAY_{k}"]) for k in range(1, 7)]
    bill = [float(rec[f"BILL_AMT{k}"]) for k in range(1, 7)]
    paid = [float(rec[f"PAY_AMT{k}"]) for k in range(1, 7)]
    util = [b / limit for b in bill]

    out = {"LIMIT_BAL": limit}
    for e in (1, 2, 3):
        out[f"EDU_{e}"] = float(edu == e)
    for k in range(6):
        out[f"PAY_{k + 1}"] = pay[k]
        out[f"BILL_AMT{k + 1}"] = bill[k]
        out[f"PAY_AMT{k + 1}"] = paid[k]
        out[f"util_{k + 1}"] = util[k]
    for k in range(5):
        prev_bill = max(bill[k + 1], 1.0)
        out[f"pay_ratio_{k + 1}"] = min(max(paid[k] / prev_bill, 0.0), 5.0)

    out["n_late"] = float(sum(p >= 1 for p in pay))
    out["max_delay"] = max(pay)
    out["mean_delay"] = sum(pay) / 6
    out["delay_trend"] = pay[0] - pay[5]
    out["mean_util"] = sum(util) / 6
    out["max_util"] = max(util)
    out["bill_trend"] = (bill[0] - bill[5]) / limit
    out["n_zero_payments"] = float(sum(a == 0 for a in paid))
    out["pay_to_bill"] = sum(paid) / max(sum(max(b, 0.0) for b in bill), 1.0)
    return {k: _clean_nan(v) for k, v in out.items()}
