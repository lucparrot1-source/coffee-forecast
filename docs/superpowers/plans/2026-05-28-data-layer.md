# Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `data/` sub-package: a swappable `PriceProvider` interface, a Yahoo Finance implementation with retries, an incremental daily ingestion CLI, and a monthly rollup CLI.

**Architecture:** A `PriceProvider` ABC with one `fetch()` method lives in `providers.py`. `YahooProvider` implements it, wrapping `yfinance.download()` with tenacity retries. `ingest.py` and `resample.py` are runnable CLIs that call these components and write to the existing SQLite schema.

**Tech Stack:** Python 3.11, yfinance 0.2.52, pandas 2.2.3, tenacity 9.1.2, sqlite3 (stdlib)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/coffee_forecast/data/__init__.py` | Create | Empty package marker |
| `src/coffee_forecast/data/providers.py` | Create | `PriceProvider` ABC + `YahooProvider` |
| `src/coffee_forecast/data/ingest.py` | Create | CLI: incremental daily fetch → `prices` table |
| `src/coffee_forecast/data/resample.py` | Create | CLI: daily → monthly rollup → `prices_monthly` table |
| `tests/test_providers.py` | Create | Tests for `YahooProvider` |
| `tests/test_ingest.py` | Create | Tests for `ingest()` and `_latest_date()` |
| `tests/test_resample.py` | Create | Tests for `resample()` |

---

## Task 1: `providers.py` — `PriceProvider` ABC + `YahooProvider`

**Files:**
- Create: `src/coffee_forecast/data/__init__.py`
- Create: `src/coffee_forecast/data/providers.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers.py`:

```python
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from coffee_forecast.data.providers import PriceProvider, YahooProvider, _COLUMNS


def _make_multi_index_df(ticker: str) -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    idx.name = "Date"
    cols = pd.MultiIndex.from_tuples([
        ("Open", ticker), ("High", ticker), ("Low", ticker),
        ("Close", ticker), ("Adj Close", ticker), ("Volume", ticker),
    ])
    return pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.5, 1000.0],
         [101.0, 102.0, 100.0, 101.5, 101.5, 1100.0]],
        index=idx, columns=cols,
    )


def test_price_provider_is_abstract():
    with pytest.raises(TypeError):
        PriceProvider()  # type: ignore[abstract]


def test_fetch_returns_expected_columns():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert list(df.columns) == _COLUMNS


def test_fetch_returns_correct_symbol():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert (df["symbol"] == "KC=F").all()


def test_fetch_correct_row_count():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert len(df) == 2


def test_fetch_dates_are_strings():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert df["date"].iloc[0] == "2020-01-02"


def test_fetch_empty_returns_empty_dataframe():
    with patch("coffee_forecast.data.providers.yf.download", return_value=pd.DataFrame()):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert df.empty
    assert list(df.columns) == _COLUMNS


def test_fetch_single_ticker_flat_columns():
    """yfinance returns flat columns for a single ticker — YahooProvider must normalise."""
    idx = pd.to_datetime(["2020-01-02"])
    idx.name = "Date"
    flat_df = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.5, 1000.0]],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    )
    with patch("coffee_forecast.data.providers.yf.download", return_value=flat_df):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 2))
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "KC=F"
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_providers.py -v
```

Expected: `ImportError: No module named 'coffee_forecast.data'`

- [ ] **Step 3: Create package marker**

Create `src/coffee_forecast/data/__init__.py` (empty file).

- [ ] **Step 4: Create `providers.py`**

Create `src/coffee_forecast/data/providers.py`:

```python
import logging
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

TICKERS: list[str] = ["KC=F", "RM=F", "BRL=X", "VND=X", "DX-Y.NYB"]
_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


class PriceProvider(ABC):
    @abstractmethod
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return DataFrame with columns: date, symbol, open, high, low, close, volume, adj_close."""


class YahooProvider(PriceProvider):
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        raw = self._download(symbols, start, end)
        if raw.empty:
            return pd.DataFrame(columns=_COLUMNS)
        # yfinance returns flat columns for a single ticker; normalise to MultiIndex
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, symbols[0]) for col in raw.columns]
            )
        rows: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                sym_df = raw.xs(sym, level=1, axis=1).copy().reset_index()
            except KeyError:
                log.warning("No data returned for %s", sym)
                continue
            sym_df = sym_df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            sym_df["symbol"] = sym
            sym_df["date"] = pd.to_datetime(sym_df["date"]).dt.strftime("%Y-%m-%d")
            rows.append(sym_df[_COLUMNS])
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        return yf.download(  # type: ignore[no-any-return]
            symbols, start=start, end=end, auto_adjust=False, progress=False
        )
```

- [ ] **Step 5: Run tests — expect PASS**

```
pytest tests/test_providers.py -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```
git add src/coffee_forecast/data/ tests/test_providers.py
git commit -m "feat: add PriceProvider ABC and YahooProvider"
```

---

## Task 2: `ingest.py` — incremental daily fetch CLI

**Files:**
- Create: `src/coffee_forecast/data/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest.py`:

```python
import sqlite3
from datetime import date, timedelta
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
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "adj_close": 100.5,
    })


def test_latest_date_empty_table(conn):
    assert _latest_date(conn, "KC=F") is None


def test_latest_date_with_row(conn):
    conn.execute(
        "INSERT INTO prices (date, symbol, close) VALUES ('2020-06-15', 'KC=F', 100.0)"
    )
    conn.commit()
    assert _latest_date(conn, "KC=F") == date(2020, 6, 15)


def test_ingest_inserts_rows(conn):
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-02", "2020-01-03"])
    ingest(conn, start_override=date(2020, 1, 1), tickers=["KC=F"], provider=provider)
    count = conn.execute(
        "SELECT COUNT(*) FROM prices WHERE symbol = 'KC=F'"
    ).fetchone()[0]
    assert count == 2


def test_ingest_no_duplicates(conn):
    conn.execute(
        "INSERT INTO prices (date, symbol, close, adj_close) VALUES ('2020-01-02', 'KC=F', 100.0, 100.0)"
    )
    conn.commit()
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-02", "2020-01-03"])
    ingest(conn, start_override=date(2020, 1, 1), tickers=["KC=F"], provider=provider)
    count = conn.execute(
        "SELECT COUNT(*) FROM prices WHERE symbol = 'KC=F'"
    ).fetchone()[0]
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
    conn.execute(
        "INSERT INTO prices (date, symbol, close) VALUES ('2020-01-05', 'KC=F', 100.0)"
    )
    conn.commit()
    provider = MagicMock()
    provider.fetch.return_value = _make_df("KC=F", ["2020-01-06"])
    ingest(conn, tickers=["KC=F"], provider=provider)
    # fetch was called with start = 2020-01-06 (latest + 1 day)
    call_args = provider.fetch.call_args
    assert call_args[0][1] == date(2020, 1, 6)
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_ingest.py -v
```

Expected: `ImportError: cannot import name '_latest_date' from 'coffee_forecast.data.ingest'`

- [ ] **Step 3: Create `ingest.py`**

Create `src/coffee_forecast/data/ingest.py`:

```python
import argparse
import logging
import os
import sqlite3
import traceback
from datetime import date, timedelta

import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.data.providers import PriceProvider, TICKERS, YahooProvider
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)

DEFAULT_START = date(2000, 1, 1)
_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


def _latest_date(conn: sqlite3.Connection, symbol: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(date) FROM prices WHERE symbol = ?", (symbol,)
    ).fetchone()
    return date.fromisoformat(row[0]) if row[0] else None


def ingest(
    conn: sqlite3.Connection,
    start_override: date | None = None,
    tickers: list[str] | None = None,
    provider: PriceProvider | None = None,
) -> None:
    _tickers = tickers if tickers is not None else TICKERS
    _provider: PriceProvider = provider if provider is not None else YahooProvider()
    today = date.today()

    for sym in _tickers:
        latest = _latest_date(conn, sym)
        if start_override:
            fetch_start = start_override
        elif latest:
            fetch_start = latest + timedelta(days=1)
        else:
            fetch_start = DEFAULT_START

        if fetch_start > today:
            log.info("%s: already up to date", sym)
            continue

        log.info("%s: fetching %s → %s", sym, fetch_start, today)
        df = _provider.fetch([sym], fetch_start, today)

        if df.empty:
            log.info("%s: no data returned", sym)
            continue

        records = (
            df.astype(object).where(pd.notna(df), other=None)[_COLS].values.tolist()
        )
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO prices"
            " (date, symbol, open, high, low, close, volume, adj_close)"
            " VALUES (?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
        log.info("%s: inserted %d rows", sym, conn.total_changes - before)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest daily prices into SQLite")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Override fetch start date YYYY-MM-DD")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)
    ingest(conn, start_override=args.start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_ingest.py -v
```

Expected: 5 passed

- [ ] **Step 5: Run full test suite — expect no regressions**

```
pytest -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```
git add src/coffee_forecast/data/ingest.py tests/test_ingest.py
git commit -m "feat: add incremental ingest CLI"
```

---

## Task 3: `resample.py` — monthly rollup CLI

**Files:**
- Create: `src/coffee_forecast/data/resample.py`
- Create: `tests/test_resample.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resample.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_resample.py -v
```

Expected: `ImportError: cannot import name 'resample' from 'coffee_forecast.data.resample'`

- [ ] **Step 3: Create `resample.py`**

Create `src/coffee_forecast/data/resample.py`:

```python
import argparse
import logging
import os
import sqlite3
import traceback
from datetime import date

import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)


def resample(conn: sqlite3.Connection, as_of: date | None = None) -> None:
    cutoff = (as_of or date.today()).replace(day=1).isoformat()[:7]  # 'YYYY-MM'

    rows = conn.execute(
        "SELECT date, symbol, adj_close FROM prices WHERE adj_close IS NOT NULL"
    ).fetchall()

    if not rows:
        log.info("No daily prices found — skipping resample")
        return

    df = pd.DataFrame(rows, columns=["date", "symbol", "adj_close"])
    df["year_month"] = df["date"].str[:7]
    df = df[df["year_month"] < cutoff]

    if df.empty:
        log.info("No complete months to resample")
        return

    monthly = df.groupby(["year_month", "symbol"], as_index=False)["adj_close"].mean()
    monthly["date"] = monthly["year_month"] + "-01"

    records = monthly[["date", "symbol", "adj_close"]].values.tolist()
    conn.executemany(
        "INSERT OR REPLACE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        records,
    )
    conn.commit()

    for sym, count in monthly.groupby("symbol").size().items():
        log.info("%s: %d monthly rows upserted", sym, count)
    log.info("Total: %d monthly rows upserted", len(records))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Resample daily prices to monthly")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)
    resample(conn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_resample.py -v
```

Expected: 5 passed

- [ ] **Step 5: Run full test suite — expect no regressions**

```
pytest -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```
git add src/coffee_forecast/data/resample.py tests/test_resample.py
git commit -m "feat: add monthly resample CLI"
```

---

## Post-implementation verification

- [ ] **Smoke test ingest CLI** (requires network + real Yahoo data)

```
python -m coffee_forecast.data.ingest --start 2024-01-01 --db /tmp/smoke.db
```

Expected: log output showing rows inserted per ticker, no exceptions.

- [ ] **Smoke test resample CLI**

```
python -m coffee_forecast.data.resample --db /tmp/smoke.db
```

Expected: log output showing monthly rows upserted per ticker.

- [ ] **Final commit tag**

```
git tag step-2-data-layer
```
