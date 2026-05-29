import argparse
import logging
import os
import sqlite3
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)


def compute_spread(wide: pd.DataFrame) -> pd.Series:
    """Return log(KC=F) - log(RM=F) as a monthly Series."""
    return np.log(wide["KC=F"]) - np.log(wide["RM=F"])


def compute_zscore(s: pd.Series) -> pd.Series:
    """Expanding z-score: (s - expanding_mean) / expanding_std.

    Index 0 is NaN (need >= 2 points for std).
    """
    exp = s.expanding(min_periods=2)
    return (s - exp.mean()) / exp.std()


def fit_ar1(s: pd.Series) -> tuple[float, float]:
    """OLS AR(1) fit on spread series s. Returns (rho, half_life_months).

    half_life is nan when |rho| == 0 or |rho| >= 1 (non-stationary / explosive).
    """
    y = s.values[1:]
    x = s.values[:-1]
    X = np.column_stack([np.ones_like(x), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = float(coefs[1])
    if 0.0 < abs(rho) < 1.0:
        half_life = -np.log(2) / np.log(abs(rho))
    else:
        half_life = float("nan")
    return rho, half_life


def generate_signal(
    z: pd.Series, entry: float = 2.0, exit_thresh: float = 0.5
) -> pd.Series:
    """Stateful trading signal from z-score series.

    +1 = long spread, -1 = short spread, 0 = flat.
    NaN z-scores leave the current position unchanged.
    """
    signals: list[int] = []
    current = 0
    for zi in z:
        if not np.isnan(zi):
            if zi > entry:
                current = -1
            elif zi < -entry:
                current = 1
            elif abs(zi) < exit_thresh:
                current = 0
        signals.append(current)
    return pd.Series(signals, index=z.index, dtype=int)
