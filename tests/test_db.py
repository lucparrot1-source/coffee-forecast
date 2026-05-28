import sqlite3

import pytest

from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
    import importlib
    import coffee_forecast.db as db_module
    importlib.reload(db_module)
    c = db_module.get_connection()
    ensure_schema(c)
    yield c
    c.close()


def test_all_tables_created(conn):
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {"prices", "prices_monthly", "model_runs", "forecasts", "backtest_results", "accuracy_log"}
    assert expected.issubset(tables)


def test_prices_unique_constraint(conn):
    conn.execute(
        "INSERT INTO prices (date, symbol, close) VALUES ('2024-01-01', 'KC=F', 180.0)"
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO prices (date, symbol, close) VALUES ('2024-01-01', 'KC=F', 185.0)"
        )
