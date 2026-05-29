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
