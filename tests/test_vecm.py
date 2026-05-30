import sqlite3

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema


@pytest.fixture
def mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def test_vecm_residuals_table_exists(mem_conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in mem_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "vecm_residuals" in tables
