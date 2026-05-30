import sqlite3

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.vecm import (
    fit_vecm,
    load_aligned_data,
    select_lag_order,
)


@pytest.fixture
def mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _make_cointegrated(n: int = 80, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    exog = pd.DataFrame(
        {"BRL=X": brl, "VND=X": vnd, "IDR=X": idr, "DX-Y.NYB": dxy}, index=dates
    )
    return endog, exog


def _populate_prices_monthly(
    conn: sqlite3.Connection, endog: pd.DataFrame, exog: pd.DataFrame
) -> None:
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


def test_vecm_residuals_table_exists(mem_conn: sqlite3.Connection) -> None:
    tables = {
        r[0]
        for r in mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "vecm_residuals" in tables


def test_load_aligned_data_shape(mem_conn: sqlite3.Connection) -> None:
    endog, exog = _make_cointegrated(n=80)
    _populate_prices_monthly(mem_conn, endog, exog)
    endog_out, exog_out = load_aligned_data(mem_conn)
    assert endog_out.shape == (80, 2)
    assert exog_out.shape == (80, 4)
    assert list(endog_out.columns) == ["KC=F", "RM=F"]
    assert list(exog_out.columns) == ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]


def test_load_aligned_data_log_transform(mem_conn: sqlite3.Connection) -> None:
    endog, exog = _make_cointegrated(n=20)
    _populate_prices_monthly(mem_conn, endog, exog)
    endog_out, _ = load_aligned_data(mem_conn)
    # _populate_prices_monthly stores exp(log-scale data), so load should recover
    # original log values
    np.testing.assert_allclose(
        endog_out["KC=F"].values, endog["KC=F"].values, atol=1e-10
    )


def test_load_aligned_data_empty_db(mem_conn: sqlite3.Connection) -> None:
    endog_out, exog_out = load_aligned_data(mem_conn)
    assert endog_out.empty
    assert exog_out.empty


def test_select_lag_order_returns_int_in_range() -> None:
    endog, exog = _make_cointegrated(n=80)
    lag = select_lag_order(endog, exog, maxlags=6)
    assert isinstance(lag, int)
    assert 1 <= lag <= 6


def test_fit_vecm_returns_result_with_resid_and_llf() -> None:
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    assert hasattr(result, "resid")
    assert hasattr(result, "predict")
    assert hasattr(result, "llf")
    assert result.resid.shape[1] == 2  # one residual column per endogenous variable
