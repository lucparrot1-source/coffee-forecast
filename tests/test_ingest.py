from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from coffee_forecast.data.ingest import _latest_date, ingest
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
    c = get_connection()
    ensure_schema(c)
    yield c
    c.close()


def _make_df(symbol: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "adj_close": 100.5,
        }
    )


def test_latest_date_empty_table(conn):
    assert _latest_date(conn, "KC=F") is None


def test_latest_date_with_row(conn):
    conn.execute("INSERT INTO prices (date, symbol, close) VALUES ('2020-06-15', 'KC=F', 100.0)")
    conn.commit()
    assert _latest_date(conn, "KC=F") == date(2020, 6, 15)


def test_ingest_inserts_rows(conn):
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-02", "2020-01-03"])
    ingest(conn, start_override=date(2020, 1, 1), tickers=["KC=F"], provider=provider)
    count = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol = 'KC=F'").fetchone()[0]
    assert count == 2


def test_ingest_no_duplicates(conn):
    conn.execute(
        "INSERT INTO prices (date, symbol, close, adj_close)"
        " VALUES ('2020-01-02', 'KC=F', 100.0, 100.0)"
    )
    conn.commit()
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-02", "2020-01-03"])
    ingest(conn, start_override=date(2020, 1, 1), tickers=["KC=F"], provider=provider)
    count = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol = 'KC=F'").fetchone()[0]
    assert count == 2  # duplicate was ignored, not doubled


def test_ingest_empty_df_skipped(conn):
    provider = MagicMock()
    provider.fetch.return_value = pd.DataFrame(
        columns=["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
    )
    ingest(conn, start_override=date(2020, 1, 1), tickers=["KC=F"], provider=provider)
    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert count == 0


def test_ingest_increments_from_latest(conn):
    conn.execute("INSERT INTO prices (date, symbol, close) VALUES ('2020-01-05', 'KC=F', 100.0)")
    conn.commit()
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-06"])
    ingest(conn, tickers=["KC=F"], provider=provider)
    # fetch was called with start = 2020-01-06 (latest + 1 day)
    call_args = provider.fetch.call_args
    assert call_args[0][1] == date(2020, 1, 6)
