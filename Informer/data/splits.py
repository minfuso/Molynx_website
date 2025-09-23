# Informer/data/splits.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple

def load_split_dfs(
    files_by_split: Dict[str, List[str]],
    features: List[str],
    target: str,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Charge les CSV selon le split (train/val/test) et retourne des listes de DataFrames,
    NON normalisés, triés par index, sans NaN, colonnes restreintes à features+target.
    Conserve la séparation par fichier.
    """
    def _load_files(file_list: List[str]) -> List[pd.DataFrame]:
        dfs = []
        for f in file_list:
            df = pd.read_csv(f, parse_dates=["open_time"], index_col="open_time")
            df = df.sort_index()
            df = df[features + [target]]
            if df.isna().any().any():
                raise ValueError(f"NaN values in {f}. Clean before processing.")
            dfs.append(df)
        return dfs

    train_dfs = _load_files(files_by_split.get("train", []))
    val_dfs   = _load_files(files_by_split.get("val", []))
    test_dfs  = _load_files(files_by_split.get("test", []))
    return train_dfs, val_dfs, test_dfs
