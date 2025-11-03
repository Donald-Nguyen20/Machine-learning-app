# ML_TAB/Steps/Step3/outlier_tools.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ---------- utils ----------
def _infer_timestamp_col(df: pd.DataFrame, user_col: Optional[str] = None) -> Optional[str]:
    if user_col and user_col in df.columns:
        return user_col
    # ưu tiên dtype datetime
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    # fallback theo tên
    pat = re.compile(r"(time|date|timestamp|datetime)", re.I)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None

def _numeric_columns(df: pd.DataFrame, prefer: Optional[List[str]] = None) -> List[str]:
    if prefer:
        return [c for c in prefer if c in df.columns]
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _mk_result_df(
    rows: List[Tuple[int, Optional[object], str, object, float, str]]
) -> pd.DataFrame:
    # rows: (row_index, timestamp, column, value, score, method)
    if not rows:
        return pd.DataFrame(columns=["row_index","timestamp","column","value","score","method"])
    return pd.DataFrame(rows, columns=["row_index","timestamp","column","value","score","method"])

# ---------- Detector 1: IQR (điểm/column-level) ----------
def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    cols = _numeric_columns(df, columns)
    ts_col = _infer_timestamp_col(df, timestamp_col)
    rows: List[Tuple[int, Optional[object], str, object, float, str]] = []

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr):
            continue
        lower = q1 if iqr == 0 else q1 - factor * iqr
        upper = q3 if iqr == 0 else q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        if mask.any():
            for idx in df.index[mask]:
                ts = df.loc[idx, ts_col] if (ts_col and ts_col in df.columns) else None
                val = df.loc[idx, col]
                # score = độ lệch biên (xa biên -> lớn hơn)
                if val < lower:
                    score = float(lower - val)
                elif val > upper:
                    score = float(val - upper)
                else:
                    score = 0.0
                rows.append((int(idx), ts, col, val, score, "IQR"))
    return _mk_result_df(rows)

# ---------- Detector 2: Z-score (điểm/column-level) ----------
def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    z: float = 3.0,
    timestamp_col: Optional[str] = None,
    ddof: int = 0,
) -> pd.DataFrame:
    cols = _numeric_columns(df, columns)
    ts_col = _infer_timestamp_col(df, timestamp_col)
    rows: List[Tuple[int, Optional[object], str, object, float, str]] = []

    for col in cols:
        series = df[col].astype(float)
        mu, sigma = series.mean(), series.std(ddof=ddof)
        if sigma == 0 or np.isnan(sigma):
            continue
        zscores = (series - mu) / sigma
        mask = zscores.abs() > z
        if mask.any():
            for idx in df.index[mask]:
                ts = df.loc[idx, ts_col] if (ts_col and ts_col in df.columns) else None
                val = df.loc[idx, col]
                score = float(zscores.loc[idx])
                rows.append((int(idx), ts, col, val, score, "Z-SCORE"))
    return _mk_result_df(rows)

# ---------- Detector 3: IsolationForest (hàng/row-level) ----------
def detect_outliers_isoforest(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    cols = _numeric_columns(df, columns)
    if not cols:
        return _mk_result_df([])
    ts_col = _infer_timestamp_col(df, timestamp_col)

    X = df[cols].astype(float)
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1
    )
    pred = iso.fit_predict(X)            # -1 là outlier
    scores = iso.decision_function(X)    # càng nhỏ càng bất thường

    rows: List[Tuple[int, Optional[object], str, object, float, str]] = []
    mask = (pred == -1)
    if mask.any():
        for idx in df.index[mask]:
            ts = df.loc[idx, ts_col] if (ts_col and ts_col in df.columns) else None
            rows.append((int(idx), ts, "<row>", None, float(scores[df.index.get_loc(idx)]), "ISOFOR"))
    return _mk_result_df(rows)

# ---------- Detector 4: LOF ----------
def detect_outliers_lof(
    df,
    columns=None,
    n_neighbors=20,
    contamination=0.05,
    timestamp_col=None
):
    cols = columns or df.select_dtypes(include='number').columns
    if not len(cols):
        return pd.DataFrame(columns=["row_index","timestamp","column","value","score","method"])

    ts_col = _infer_timestamp_col(df, timestamp_col)
    X = df[cols].astype(float)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = lof.fit_predict(X)   # -1 là outlier
    scores = lof.negative_outlier_factor_

    rows = []
    for idx, label in zip(df.index, pred):
        if label == -1:  # chỉ lấy outlier
            ts = df.loc[idx, ts_col] if ts_col and ts_col in df.columns else None
            score = float(scores[df.index.get_loc(idx)])
            rows.append((int(idx), ts, "<row>", None, score, "LOF"))
    return _mk_result_df(rows)