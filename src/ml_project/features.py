from typing import Tuple
import pandas as pd

def make_features(df: pd.DataFrame, target_col: str = "label_id") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data (have: {list(df.columns)[:10]}...)")
    y = df[target_col]
    X = df.drop(columns=[target_col, "label_name"], errors="ignore")

    # Drop non-informative columns
    nunique = X.nunique(dropna=False)
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Fill NA and ensure numeric
    X = X.fillna(0)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X, y
