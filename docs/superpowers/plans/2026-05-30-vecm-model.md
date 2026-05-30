# VECM Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a VECM point-forecast pipeline (`src/coffee_forecast/models/vecm.py`) that fits a cointegrated model on log Arabica/Robusta prices with 4 FX exogenous drivers and writes 1/2/3-month forecasts and in-sample residuals to SQLite.

**Architecture:** Plain-function module mirroring `spread.py`. `run_vecm_model()` orchestrates load → lag-select → fit → extract residuals → forecast → write DB. `main()` wraps it with argparse and Resend alert.

**Tech Stack:** Python 3.11, statsmodels 0.14.4 (`VECM`, `select_order`), pandas, numpy, SQLite (existing schema + new `vecm_residuals` table).

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/coffee_forecast/db/schema.sql` | Add `vecm_residuals` table |
| Create | `src/coffee_forecast/models/vecm.py` | All VECM logic |
| Create | `tests/test_vecm.py` | All tests |

---

## Shared test helper (add once, used in every task)

Add this at the top of `tests/test_vecm.py`. It produces synthetic cointegrated log-price data — both series share a random walk "common factor" with small noise. The exog series are stationary noise (irrelevant for cointegration but structurally required).

```python
import sqlite3
import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.vecm import (
    load_aligned_data,
    select_lag_order,
    fit_vecm,
    extract_residuals,
    generate_forecasts,
    write_run,
    write_forecasts,
    write_residuals,
    run_vecm_model,
)


def _make_cointegrated(n: int = 80, seed: int = 42):
    """Return (endog_df, exog_df) of log-scale synthetic cointegrated series."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(n))
    kc = common + rng.standard_normal(n) * 0.05
    rm = common + rng.standard_normal(n) * 0.05
    brl = rng.standard_normal(n) * 0.1
    vnd = rng.standard_normal(n) * 0.1
    idr = rng.standard_normal(n) * 0.1
    dxy = rng.standard_normal(n) * 0.1
    dates = pd.date_range("2014-01-01", periods=n, freq="MS")
    endog = pd.DataFrame({"KC=F": kc, "RM=F": rm}, index=dates)
    exog = pd.DataFrame({"BRL=X": brl, "VND=X": vnd, "IDR=X": idr, "DX-Y.NYB": dxy}, index=dates)
    return endog, exog


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _populate_prices_monthly(conn, endog, exog):
    """Insert exp(log-price) rows into prices_monthly for smoke tests."""
    rows = []
    for dt in endog.index:
        date_str = dt.strftime("%Y-%m-%d")
        for sym in ["KC=F", "RM=F"]:
            rows.append((date_str, sym, float(np.exp(endog.loc[dt, sym]))))
        for sym in ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]:
            rows.append((date_str, sym, float(np.exp(exog.loc[dt, sym]))))
    conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
```

---

## Task 1: Add `vecm_residuals` table to schema

**Files:**
- Modify: `src/coffee_forecast/db/schema.sql`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_vecm_residuals_table_exists(mem_conn):
    tables = {r[0] for r in mem_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "vecm_residuals" in tables
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_vecm.py::test_vecm_residuals_table_exists -v
```
Expected: FAIL — `vecm_residuals` not yet in schema.

- [ ] **Step 3: Add table to schema**

Append to `src/coffee_forecast/db/schema.sql` (before the final `CREATE INDEX` block):

```sql
CREATE TABLE IF NOT EXISTS vecm_residuals (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES model_runs(id),
    date     TEXT    NOT NULL,   -- YYYY-MM-01
    symbol   TEXT    NOT NULL,   -- KC=F or RM=F
    residual REAL    NOT NULL,
    UNIQUE (run_id, date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_vecm_residuals_run ON vecm_residuals (run_id, date);
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_vecm.py::test_vecm_residuals_table_exists -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/db/schema.sql tests/test_vecm.py
git commit -m "feat: add vecm_residuals table to schema"
```

---

## Task 2: Implement `load_aligned_data`

**Files:**
- Create: `src/coffee_forecast/models/vecm.py` (skeleton + this function)
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_aligned_data_shape(mem_conn):
    endog, exog = _make_cointegrated(n=80)
    _populate_prices_monthly(mem_conn, endog, exog)
    endog_out, exog_out = load_aligned_data(mem_conn)
    assert endog_out.shape == (80, 2)
    assert exog_out.shape == (80, 4)
    assert list(endog_out.columns) == ["KC=F", "RM=F"]
    assert list(exog_out.columns) == ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]


def test_load_aligned_data_log_transform(mem_conn):
    endog, exog = _make_cointegrated(n=20)
    _populate_prices_monthly(mem_conn, endog, exog)
    endog_out, _ = load_aligned_data(mem_conn)
    # Values should match the original log-scale data (within float precision)
    np.testing.assert_allclose(endog_out["KC=F"].values, endog["KC=F"].values, atol=1e-10)


def test_load_aligned_data_empty_db(mem_conn):
    endog_out, exog_out = load_aligned_data(mem_conn)
    assert endog_out.empty
    assert exog_out.empty
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_vecm.py::test_load_aligned_data_shape tests/test_vecm.py::test_load_aligned_data_log_transform tests/test_vecm.py::test_load_aligned_data_empty_db -v
```
Expected: FAIL — `vecm` module not found.

- [ ] **Step 3: Create `src/coffee_forecast/models/vecm.py` with this content**

```python
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
        params=all_symbols,
    )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna().sort_index()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    log_wide = np.log(wide)
    return log_wide[ENDOG_SYMBOLS], log_wide[EXOG_SYMBOLS]
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_vecm.py::test_load_aligned_data_shape tests/test_vecm.py::test_load_aligned_data_log_transform tests/test_vecm.py::test_load_aligned_data_empty_db -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement load_aligned_data"
```

---

## Task 3: Implement `select_lag_order`

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_select_lag_order_returns_int_in_range():
    endog, exog = _make_cointegrated(n=80)
    lag = select_lag_order(endog, exog, maxlags=6)
    assert isinstance(lag, int)
    assert 1 <= lag <= 6
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_vecm.py::test_select_lag_order_returns_int_in_range -v
```
Expected: FAIL — `select_lag_order` not defined.

- [ ] **Step 3: Add function to `vecm.py`** (after `load_aligned_data`)

```python
def select_lag_order(endog: pd.DataFrame, exog: pd.DataFrame, maxlags: int = 12) -> int:
    """Return AIC-optimal VAR lag order (minimum 1) for VECM pre-selection."""
    res = select_order(endog.values, maxlags=maxlags, deterministic="co", exog=exog.values)
    return max(1, int(res.aic))
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_vecm.py::test_select_lag_order_returns_int_in_range -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement select_lag_order"
```

---

## Task 4: Implement `fit_vecm`

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fit_vecm_returns_result_with_resid_and_llf():
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    assert hasattr(result, "resid")
    assert hasattr(result, "predict")
    assert hasattr(result, "llf")
    assert result.resid.shape[1] == 2  # one residual column per endogenous variable
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_vecm.py::test_fit_vecm_returns_result_with_resid_and_llf -v
```
Expected: FAIL — `fit_vecm` not defined.

- [ ] **Step 3: Add function to `vecm.py`** (after `select_lag_order`)

```python
def fit_vecm(endog: pd.DataFrame, exog: pd.DataFrame, lag_order: int):
    """Fit a VECM with coint_rank=1 and exogenous drivers.

    k_ar_diff = lag_order - 1: number of lagged-difference terms in the VECM equation.
    deterministic='co': constant restricted to the cointegration relation (standard for commodity prices).
    """
    model = VECM(
        endog.values,
        k_ar_diff=lag_order - 1,
        coint_rank=1,
        exog=exog.values,
        deterministic="co",
    )
    return model.fit()
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_vecm.py::test_fit_vecm_returns_result_with_resid_and_llf -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement fit_vecm"
```

---

## Task 5: Implement `extract_residuals`

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_residuals_shape_and_columns():
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = extract_residuals(result, endog)
    assert set(df.columns) == {"date", "symbol", "residual"}
    assert set(df["symbol"].unique()) == {"KC=F", "RM=F"}
    assert df["residual"].notna().all()
    # Each date has 2 rows (one per endogenous symbol)
    assert len(df) == result.resid.shape[0] * 2
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_vecm.py::test_extract_residuals_shape_and_columns -v
```
Expected: FAIL — `extract_residuals` not defined.

- [ ] **Step 3: Add function to `vecm.py`** (after `fit_vecm`)

```python
def extract_residuals(result, endog: pd.DataFrame) -> pd.DataFrame:
    """Return long-format DataFrame of in-sample residuals (log scale).

    VECM residuals have fewer rows than endog: the first k_ar_diff rows are consumed
    by lagged differences. Dates are aligned from the tail of endog.index.
    """
    n_resid = result.resid.shape[0]
    dates = endog.index[-n_resid:]
    rows = []
    for i, dt in enumerate(dates):
        for j, sym in enumerate(ENDOG_SYMBOLS):
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": sym,
                "residual": float(result.resid[i, j]),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_vecm.py::test_extract_residuals_shape_and_columns -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement extract_residuals"
```

---

## Task 6: Implement `generate_forecasts`

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_forecasts_horizons():
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = generate_forecasts(result, endog.columns.tolist(), exog.shape[1])
    assert len(df) == 6  # 3 horizons × 2 symbols
    assert set(df["horizon"].unique()) == {1, 2, 3}
    assert set(df["symbol"].unique()) == {"KC=F", "RM=F"}


def test_generate_forecasts_back_transform():
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = generate_forecasts(result, endog.columns.tolist(), exog.shape[1])
    for _, row in df.iterrows():
        assert abs(np.exp(row["log_forecast"]) - row["point_forecast"]) < 1e-10
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_vecm.py::test_generate_forecasts_horizons tests/test_vecm.py::test_generate_forecasts_back_transform -v
```
Expected: FAIL — `generate_forecasts` not defined.

- [ ] **Step 3: Add function to `vecm.py`** (after `extract_residuals`)

```python
def generate_forecasts(result, endog_cols: list[str], n_exog: int, steps: int = 3) -> pd.DataFrame:
    """Produce point forecasts at horizons 1..steps.

    Naïve exog assumption: Δexog = 0 for all forecast steps (exchange rates unchanged).
    Forecasts are on log scale; point_forecast = exp(log_forecast).
    """
    exog_fc = np.zeros((steps, n_exog))
    fc = result.predict(steps=steps, exog_fc=exog_fc)  # shape (steps, n_endog)
    rows = []
    for h in range(steps):
        for j, sym in enumerate(endog_cols):
            log_fc = float(fc[h, j])
            rows.append({
                "horizon": h + 1,
                "symbol": sym,
                "log_forecast": log_fc,
                "point_forecast": float(np.exp(log_fc)),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_vecm.py::test_generate_forecasts_horizons tests/test_vecm.py::test_generate_forecasts_back_transform -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement generate_forecasts"
```

---

## Task 7: Implement DB write functions

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_write_run_inserts_model_run(mem_conn):
    params = {
        "lag_order": 2, "coint_rank": 1, "exog_symbols": [],
        "train_start": "2014-01-01", "train_end": "2024-01-01", "n_obs": 100,
    }
    run_id = write_run(mem_conn, params, {"log_likelihood": 300.0})
    row = mem_conn.execute(
        "SELECT model_type, status FROM model_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row == ("vecm", "success")


def test_write_forecasts_inserts_six_rows(mem_conn):
    params = {
        "lag_order": 2, "coint_rank": 1, "exog_symbols": [],
        "train_start": "2014-01-01", "train_end": "2024-01-01", "n_obs": 100,
    }
    run_id = write_run(mem_conn, params, {})
    forecasts_df = pd.DataFrame([
        {"horizon": h, "symbol": sym, "log_forecast": 5.0, "point_forecast": np.exp(5.0)}
        for h in [1, 2, 3] for sym in ["KC=F", "RM=F"]
    ])
    write_forecasts(mem_conn, run_id, forecasts_df, "2024-01-01")
    count = mem_conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert count == 6


def test_write_residuals_inserts_rows(mem_conn):
    params = {
        "lag_order": 1, "coint_rank": 1, "exog_symbols": [],
        "train_start": "2014-01-01", "train_end": "2024-01-01", "n_obs": 10,
    }
    run_id = write_run(mem_conn, params, {})
    residuals_df = pd.DataFrame([
        {"date": "2014-02-01", "symbol": "KC=F", "residual": 0.01},
        {"date": "2014-02-01", "symbol": "RM=F", "residual": -0.02},
    ])
    write_residuals(mem_conn, run_id, residuals_df)
    count = mem_conn.execute(
        "SELECT COUNT(*) FROM vecm_residuals WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert count == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_vecm.py::test_write_run_inserts_model_run tests/test_vecm.py::test_write_forecasts_inserts_six_rows tests/test_vecm.py::test_write_residuals_inserts_rows -v
```
Expected: FAIL — `write_run`, `write_forecasts`, `write_residuals` not defined.

- [ ] **Step 3: Add functions to `vecm.py`** (after `generate_forecasts`)

```python
def write_run(conn: sqlite3.Connection, params: dict, metrics: dict) -> int:
    cur = conn.execute(
        "INSERT INTO model_runs (run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            "vecm",
            params["train_start"],
            params["train_end"],
            json.dumps(params),
            json.dumps(metrics),
            "success",
        ),
    )
    conn.commit()
    return cur.lastrowid


def write_forecasts(conn: sqlite3.Connection, run_id: int, forecasts_df: pd.DataFrame, forecast_date: str) -> None:
    records = []
    for _, row in forecasts_df.iterrows():
        h = int(row["horizon"])
        target_date = (pd.Timestamp(forecast_date) + pd.DateOffset(months=h)).strftime("%Y-%m-%d")
        records.append((run_id, forecast_date, target_date, h, row["symbol"], row["point_forecast"]))
    conn.executemany(
        "INSERT OR REPLACE INTO forecasts"
        " (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()


def write_residuals(conn: sqlite3.Connection, run_id: int, residuals_df: pd.DataFrame) -> None:
    records = [
        (run_id, r["date"], r["symbol"], r["residual"])
        for _, r in residuals_df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO vecm_residuals (run_id, date, symbol, residual)"
        " VALUES (?, ?, ?, ?)",
        records,
    )
    conn.commit()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_vecm.py::test_write_run_inserts_model_run tests/test_vecm.py::test_write_forecasts_inserts_six_rows tests/test_vecm.py::test_write_residuals_inserts_rows -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement write_run, write_forecasts, write_residuals"
```

---

## Task 8: Implement `run_vecm_model`, `main()`, and smoke test

**Files:**
- Modify: `src/coffee_forecast/models/vecm.py`
- Test: `tests/test_vecm.py`

- [ ] **Step 1: Write the failing smoke test**

```python
def test_run_vecm_model_smoke(mem_conn):
    endog, exog = _make_cointegrated(n=80)
    _populate_prices_monthly(mem_conn, endog, exog)

    run_id = run_vecm_model(mem_conn)

    assert run_id > 0

    fc_count = mem_conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert fc_count == 6  # 3 horizons × 2 symbols

    res_count = mem_conn.execute(
        "SELECT COUNT(*) FROM vecm_residuals WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    assert res_count > 0

    run_row = mem_conn.execute(
        "SELECT status, model_type FROM model_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert run_row == ("success", "vecm")


def test_run_vecm_model_empty_db_returns_minus_one(mem_conn):
    result = run_vecm_model(mem_conn)
    assert result == -1
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_vecm.py::test_run_vecm_model_smoke tests/test_vecm.py::test_run_vecm_model_empty_db_returns_minus_one -v
```
Expected: FAIL — `run_vecm_model` not defined.

- [ ] **Step 3: Add `run_vecm_model` and `main()` to `vecm.py`** (append to end of file)

```python
def run_vecm_model(conn: sqlite3.Connection) -> int:
    """Orchestrate load → lag-select → fit → residuals → forecasts → write DB.

    Returns the model_runs.id of the completed run, or -1 if no data found.
    """
    endog, exog = load_aligned_data(conn)
    if endog.empty:
        log.warning("No aligned monthly data found for all 6 symbols — skipping VECM")
        return -1

    lag_order = select_lag_order(endog, exog)
    log.info("Selected lag order: %d (AIC)", lag_order)

    result = fit_vecm(endog, exog, lag_order)
    log.info("VECM fitted: llf=%.4f, n_obs=%d", result.llf, len(endog))

    residuals_df = extract_residuals(result, endog)
    forecasts_df = generate_forecasts(result, endog.columns.tolist(), exog.shape[1])

    train_start = endog.index[0].strftime("%Y-%m-%d")
    train_end = endog.index[-1].strftime("%Y-%m-%d")

    params = {
        "lag_order": lag_order,
        "coint_rank": 1,
        "exog_symbols": exog.columns.tolist(),
        "train_start": train_start,
        "train_end": train_end,
        "n_obs": len(endog),
    }
    metrics = {"log_likelihood": float(result.llf)}

    run_id = write_run(conn, params, metrics)
    write_forecasts(conn, run_id, forecasts_df, train_end)
    write_residuals(conn, run_id, residuals_df)

    log.info(
        "VECM run complete: run_id=%d, forecasts=%d, residuals=%d",
        run_id, len(forecasts_df), len(residuals_df),
    )
    return run_id


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fit VECM and write forecasts to DB")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()
    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db
    conn = get_connection()
    ensure_schema(conn)
    run_vecm_model(conn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
```

- [ ] **Step 4: Run all tests to confirm they pass**

```
pytest tests/test_vecm.py -v
```
Expected: all PASS (11 tests total).

- [ ] **Step 5: Run mypy**

```
mypy src/coffee_forecast/models/vecm.py
```
Expected: no errors. If any type issues appear, fix them before committing.

- [ ] **Step 6: Commit**

```
git add src/coffee_forecast/models/vecm.py tests/test_vecm.py
git commit -m "feat: implement run_vecm_model and main (Step 5 complete)"
```

---

## Final check

- [ ] Run the full test suite to confirm no regressions

```
pytest --tb=short -q
```
Expected: all green.

- [ ] Tick off Step 5 in `CLAUDE.md` and commit

```
git add CLAUDE.md
git commit -m "docs: tick off Step 5 in CLAUDE.md"
```
