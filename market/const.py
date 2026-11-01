"""Shared market constants and data_real path."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    env = os.environ.get("QSHARE_DATA")
    if env:
        return Path(env)
    return ROOT / "data_real"


PKT_ROWS = 40
PKTS_PER_SELLER = 5
N_SELLERS = 8
Q_LEVELS = [0.0, 0.1, 0.3, 0.5]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
S1_FIXED = 5
