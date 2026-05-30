import argparse
import json
import logging
import os
import sqlite3
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

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
