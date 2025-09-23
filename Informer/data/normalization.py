# Informer/data/normalization.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, List, Callable, Optional
import joblib
import os

# --- Normalizers atomiques (par série) ---
def _standardize(series: pd.Series, scaler: Optional[StandardScaler]=None):
    scaler = scaler or StandardScaler()
    vals = scaler.fit_transform(series.to_numpy().reshape(-1,1)) if not hasattr(scaler, "mean_") \
           else scaler.transform(series.to_numpy().reshape(-1,1))
    return vals.ravel(), scaler

def _minmax(series: pd.Series, scaler: Optional[MinMaxScaler]=None):
    scaler = scaler or MinMaxScaler()
    vals = scaler.fit_transform(series.to_numpy().reshape(-1,1)) if not hasattr(scaler, "data_min_") \
           else scaler.transform(series.to_numpy().reshape(-1,1))
    return vals.ravel(), scaler

def _log_standard(series: pd.Series, scaler: Optional[StandardScaler]=None):
    log_series = np.log1p(series)
    return _standardize(log_series, scaler)

# Registre des méthodes par nom
METHODS: Dict[str, Callable] = {
    "standard": _standardize,
    "minmax": _minmax,
    "log_standard": _log_standard,
}

def fit_scalers_on_train(
    train_dfs: List[pd.DataFrame],
    feature_config: Dict[str, str],
) -> Dict[str, object]:
    """
    Fit one scaler per feature on l'ENSEMBLE du train (concat des train_dfs).
    """
    scalers: Dict[str, object] = {}
    train_concat = pd.concat(train_dfs, axis=0, copy=False)
    for feat, method in feature_config.items():
        if method not in METHODS:
            raise ValueError(f"Unknown normalization method '{method}' for feature '{feat}'")
        # Fit une fois sur train concat
        _, scaler = METHODS[method](train_concat[feat], scaler=None)
        scalers[feat] = scaler
    return scalers

def transform_with_scalers(
    dfs: List[pd.DataFrame],
    feature_config: Dict[str, str],
    scalers: Dict[str, object],
) -> List[pd.DataFrame]:
    """
    Applique les scalers fournis à chaque DataFrame (sans refit).
    """
    out = []
    for df in dfs:
        df2 = df.copy()
        for feat, method in feature_config.items():
            func = METHODS[method]
            vals, _ = func(df2[feat], scaler=scalers[feat])
            df2[feat] = vals
        out.append(df2)
    return out

def save_scalers(scalers: Dict[str, object], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scalers, path)

def load_scalers(path: str) -> Dict[str, object]:
    return joblib.load(path)
