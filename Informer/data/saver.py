# Informer/data/saver.py
from __future__ import annotations
import numpy as np
import os
from typing import Tuple

def save_splits(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    out_dir: str,
    name: str
) -> str:
    """
    Save train/val/test splits in a compressed .npz file.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray): arrays to save
        out_dir (str): Directory to save the file
        name (str): Base name for the file (without extension)

    Returns:
        str: Path of the saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")

    np.savez_compressed(
        path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    return path


def load_splits(path: str) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]
]:
    """
    Load train/val/test splits from a compressed .npz file.

    Args:
        path (str): Path to the .npz file

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    data = np.load(path)
    return (
        (data["X_train"], data["y_train"]),
        (data["X_val"], data["y_val"]),
        (data["X_test"], data["y_test"]),
    )
