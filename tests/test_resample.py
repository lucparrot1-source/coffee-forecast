from datetime import date

import pytest

from coffee_forecast.data.resample import resample
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
    c = get_connection()
    ensure_schema(c)
    yield c
    c.close()


def _insert(conn, dt: str, symbol: str, adj_close: float) -> None:
    conn.execute(
        "INSERT INTO prices (date, symbol, adj_close) VALUES (?, ?, ?)",
        (dt, symbol, adj_close),
    )


def test_resample_computes_mean(conn):
    _insert(conn, "2020-01-02", "KC=F", 100.0)
    _insert(conn, "2020-01-15", "KC=F", 120.0)
    _insert(conn, "2020-01-30", "KC=F", 140.0)
    conn.commit()
    resample(conn, as_of=date(2020, 2, 1))  # Feb 1 → Jan is a complete month
    row = conn.execute(
        "SELECT adj_close FROM prices_monthly WHERE date = '2020-01-01' AND symbol = 'KC=F'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 120.0) < 1e-9  # mean(100, 120, 140) = 120


def test_resample_excludes_current_month(conn):
    _insert(conn, "2020-01-15", "KC=F", 100.0)
    conn.commit()
    resample(conn, as_of=date(2020, 1, 31))  # Jan 31 → January is still current
    row = conn.execute(
        "SELECT adj_close FROM prices_monthly WHERE date = '2020-01-01' AND symbol = 'KC=F'"
    ).fetchone()
    assert row is None


def test_resample_multiple_symbols(conn):
    _insert(conn, "2020-01-02", "KC=F", 100.0)
    _insert(conn, "2020-01-02", "RM=F", 50.0)
    conn.commit()
    resample(conn, as_of=date(2020, 2, 1))
    kc = conn.execute(
        "SELECT adj_close FROM prices_monthly WHERE date = '2020-01-01' AND symbol = 'KC=F'"
    ).fetchone()
    rm = conn.execute(
        "SELECT adj_close FROM prices_monthly WHERE date = '2020-01-01' AND symbol = 'RM=F'"
    ).fetchone()
    assert kc is not None and abs(kc[0] - 100.0) < 1e-9
    assert rm is not None and abs(rm[0] - 50.0) < 1e-9


def test_resample_upsert_idempotent(conn):
    _insert(conn, "2020-01-02", "KC=F", 100.0)
    conn.commit()
    resample(conn, as_of=date(2020, 2, 1))
    resample(conn, as_of=date(2020, 2, 1))  # second run — should not duplicate
    count = conn.execute(
        "SELECT COUNT(*) FROM prices_monthly WHERE date = '2020-01-01' AND symbol = 'KC=F'"
    ).fetchone()[0]
    assert count == 1


def test_resample_empty_table_no_error(conn):
    resample(conn, as_of=date(2020, 2, 1))  # no rows in prices — should not raise
    count = conn.execute("SELECT COUNT(*) FROM prices_monthly").fetchone()[0]
    assert count == 0
