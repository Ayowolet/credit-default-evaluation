"""Data loading, cleaning and feature engineering for the credit default study.

Data: "Default of Credit Card Clients" (Yeh & Lien, 2009; UCI Machine Learning Repository),
as mirrored in the PyCaret datasets folder. Target = `default` (observed default next month),
NOT a vendor-supplied score.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "credit.csv"
TARGET = "default"

# Protected / sensitive attributes: never used as model inputs, only for slice checks.
PROTECTED = ["SEX", "AGE", "MARRIAGE"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light, documented cleaning. Returns a new frame."""
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    # Undocumented category codes are merged into 'other' (EDUCATION 4) / (MARRIAGE 3).
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df.attrs["dropped_duplicates"] = n0 - len(df)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs. Month 1 is the most recent month, month 6 the oldest."""
    out = pd.DataFrame(index=df.index)
    out["LIMIT_BAL"] = df["LIMIT_BAL"]
    for e in (1, 2, 3):  # one-hot, education 4 = baseline
        out[f"EDU_{e}"] = (df["EDUCATION"] == e).astype(int)

    pay = df[[f"PAY_{k}" for k in range(1, 7)]]
    bill = df[[f"BILL_AMT{k}" for k in range(1, 7)]]
    paid = df[[f"PAY_AMT{k}" for k in range(1, 7)]]

    for k in range(1, 7):
        out[f"PAY_{k}"] = df[f"PAY_{k}"]
        out[f"BILL_AMT{k}"] = df[f"BILL_AMT{k}"]
        out[f"PAY_AMT{k}"] = df[f"PAY_AMT{k}"]
        out[f"util_{k}"] = df[f"BILL_AMT{k}"] / df["LIMIT_BAL"]

    # Payment in month k relative to the bill of the previous month (k+1).
    for k in range(1, 6):
        prev_bill = df[f"BILL_AMT{k + 1}"].clip(lower=1)
        out[f"pay_ratio_{k}"] = (df[f"PAY_AMT{k}"] / prev_bill).clip(0, 5)

    out["n_late"] = (pay >= 1).sum(axis=1)
    out["max_delay"] = pay.max(axis=1)
    out["mean_delay"] = pay.mean(axis=1)
    out["delay_trend"] = df["PAY_1"] - df["PAY_6"]
    out["mean_util"] = out[[f"util_{k}" for k in range(1, 7)]].mean(axis=1)
    out["max_util"] = out[[f"util_{k}" for k in range(1, 7)]].max(axis=1)
    out["bill_trend"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"]
    out["n_zero_payments"] = (paid == 0).sum(axis=1)
    out["pay_to_bill"] = paid.sum(axis=1) / bill.clip(lower=0).sum(axis=1).clip(lower=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build(path: Path = DATA_PATH):
    """Return (X, y, sensitive) with aligned indices."""
    df = clean(load_raw(path))
    X = engineer(df)
    y = df[TARGET].astype(int)
    sens = df[PROTECTED].copy()
    sens["AGE_BAND"] = pd.cut(sens["AGE"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"])
    return X, y, sens
