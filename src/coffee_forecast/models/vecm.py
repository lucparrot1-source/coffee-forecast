import argparse  # noqa: F401
import json  # noqa: F401
import logging
import os  # noqa: F401
import sqlite3
import traceback  # noqa: F401
from datetime import datetime, timezone  # noqa: F401

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    VECMResults,
    select_order,
)

from coffee_forecast.alerts import send_pipeline_alert  # noqa: F401
from coffee_forecast.db import get_connection  # noqa: F401
from coffee_forecast.db.migrations import ensure_schema  # noqa: F401
from coffee_forecast.logging_config import configure_logging  # noqa: F401

log = logging.getLogger(__name__)

ENDOG_SYMBOLS = ["KC=F", "RM=F"]
EXOG_SYMBOLS = ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]


def load_aligned_data(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices_monthly, inner-join on common dates, log-transform.

    Returns (endog_df, exog_df) both on log scale, or (empty, empty) if no data.
    """
    all_symbols = ENDOG_SYMBOLS + EXOG_SYMBOLS
    df = pd.read_sql(
        "SELECT date, symbol, adj_close FROM prices_monthly"
        f" WHERE symbol IN ({','.join('?' * len(all_symbols))})"
        " ORDER BY date",
        conn,
        params=tuple(all_symbols),
    )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna().sort_index()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    log_wide = wide.apply(np.log)
    endog_out = log_wide[ENDOG_SYMBOLS]
    exog_out = log_wide[EXOG_SYMBOLS]
    return endog_out, exog_out


def select_lag_order(endog: pd.DataFrame, exog: pd.DataFrame, maxlags: int = 12) -> int:
    """Return AIC-optimal VAR lag order (minimum 1) for VECM pre-selection."""
    res = select_order(endog.values, maxlags=maxlags, deterministic="co", exog=exog.values)
    return max(1, int(res.aic))


def fit_vecm(endog: pd.DataFrame, exog: pd.DataFrame, lag_order: int) -> VECMResults:
    """Fit a VECM with coint_rank=1 and exogenous drivers.

    k_ar_diff = lag_order - 1: number of lagged-difference terms.
    deterministic='co': constant restricted to cointegration relation.
    """
    model = VECM(
        endog.values,
        k_ar_diff=lag_order - 1,
        coint_rank=1,
        exog=exog.values,
        deterministic="co",
    )
    return model.fit()
