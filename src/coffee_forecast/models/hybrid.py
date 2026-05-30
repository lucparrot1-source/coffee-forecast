import argparse
import json
import logging
import os
import sqlite3
import traceback
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging
from coffee_forecast.models.gamlss import compute_regime_labels

log = logging.getLogger(__name__)


def load_vecm_forecasts(conn: sqlite3.Connection, vecm_run_id: int) -> pd.DataFrame:
    """Load point forecasts from a VECM run.

    Returns DataFrame with columns: horizon, symbol, point_forecast.
    """
    return pd.read_sql(
        "SELECT horizon, symbol, point_forecast FROM forecasts WHERE run_id = ?",
        conn,
        params=(vecm_run_id,),
    )


def load_gamlss_quantiles(conn: sqlite3.Connection, gamlss_run_id: int) -> pd.DataFrame:
    """Load SHASH residual quantiles from a GAMLSS run.

    Returns DataFrame with columns: symbol, regime, q10, q25, q50, q75, q90.
    Quantiles are in log space (residual scale, not price scale).
    """
    return pd.read_sql(
        "SELECT symbol, regime, q10, q25, q50, q75, q90"
        " FROM gamlss_params WHERE run_id = ?",
        conn,
        params=(gamlss_run_id,),
    )


def get_current_regime(conn: sqlite3.Connection) -> str:
    raise NotImplementedError


def combine_forecasts(vecm_df: pd.DataFrame, gamlss_df: pd.DataFrame, regime: str) -> pd.DataFrame:
    raise NotImplementedError


def write_hybrid_forecasts(conn: sqlite3.Connection, run_id: int, combined_df: pd.DataFrame, forecast_date: str) -> None:
    raise NotImplementedError


def run_hybrid_model(conn: sqlite3.Connection, vecm_run_id: int, gamlss_run_id: int) -> int:
    raise NotImplementedError
