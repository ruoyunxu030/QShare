"""Packet Fisher matrices F_j = X_j^T X_j."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def packet_fishers(Xs: Sequence[np.ndarray]) -> np.ndarray:
    """Stack F_j = X_j^T X_j into array (N, d, d)."""
    Fs = []
    d = None
    for Xj in Xs:
        Xj = np.asarray(Xj, float)
        if Xj.ndim == 1:
            Xj = Xj.reshape(1, -1)
        if d is None:
            d = Xj.shape[1]
        elif Xj.shape[1] != d:
            raise ValueError(f"inconsistent d: {Xj.shape[1]} vs {d}")
        Fs.append(Xj.T @ Xj)
    return np.stack(Fs, axis=0)
