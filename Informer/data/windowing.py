# Informer/data/windowing.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

def df_list_to_sequences(
    dfs: List[pd.DataFrame],
    features: List[str],
    target: str,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit des séquences par DataFrame (sans jamais les croiser) puis concatène.
    X shape = (N, sequence_length, F), y shape = (N,)
    """
    X_all, y_all = [], []
    for df in dfs:
        if len(df) < sequence_length:
            continue  # trop court, on saute
        X_mat = df[features].to_numpy()
        y_vec = df[target].astype(int).to_numpy()

        # fenêtres glissantes dans CE fichier uniquement
        L = len(df) - sequence_length + 1
        for i in range(L):
            X_all.append(X_mat[i:i+sequence_length])
            y_all.append(y_vec[i+sequence_length-1])

    if not X_all:
        return np.empty((0, sequence_length, len(features))), np.empty((0,), dtype=int)

    return np.stack(X_all), np.asarray(y_all, dtype=int)
