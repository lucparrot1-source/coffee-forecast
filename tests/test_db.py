import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema


@pytest.fixture  # type: ignore[misc]
def conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[sqlite3.Connection, None, None]:
    monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
    c = get_connection()
    ensure_schema(c)
    yield c
    c.close()


def test_all_tables_created(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    expected = {
        "prices",
        "prices_monthly",
        "model_runs",
        "forecasts",
        "backtest_results",
        "accuracy_log",
    }
    assert expected.issubset(tables)


def test_prices_unique_constraint(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO prices (date, symbol, close) VALUES ('2024-01-01', 'KC=F', 180.0)")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO prices (date, symbol, close) VALUES ('2024-01-01', 'KC=F', 185.0)"
        )
