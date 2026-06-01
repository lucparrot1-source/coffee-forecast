import sqlite3

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.vecm import (
    extract_residuals,
    fit_vecm,
    generate_forecasts,
    load_aligned_data,
    run_vecm_model,
    select_lag_order,
    write_forecasts,
    write_residuals,
    write_run,
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


def test_extract_residuals_shape_and_columns() -> None:
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = extract_residuals(result, endog)
    assert set(df.columns) == {"date", "symbol", "residual"}
    assert set(df["symbol"].unique()) == {"KC=F", "RM=F"}
    assert df["residual"].notna().all()
    # Each date has 2 rows (one per endogenous symbol)
    assert len(df) == result.resid.shape[0] * 2


def test_generate_forecasts_horizons() -> None:
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = generate_forecasts(result, endog.columns.tolist(), exog.values[-1])
    assert len(df) == 6  # 3 horizons × 2 symbols
    assert set(df["horizon"].unique()) == {1, 2, 3}
    assert set(df["symbol"].unique()) == {"KC=F", "RM=F"}


def test_generate_forecasts_back_transform() -> None:
    endog, exog = _make_cointegrated(n=80)
    result = fit_vecm(endog, exog, lag_order=2)
    df = generate_forecasts(result, endog.columns.tolist(), exog.values[-1])
    for _, row in df.iterrows():
        assert abs(np.exp(row["log_forecast"]) - row["point_forecast"]) < 1e-10


def test_write_run_inserts_model_run(mem_conn: sqlite3.Connection) -> None:
    params = {
        "lag_order": 2, "coint_rank": 1, "exog_symbols": [],
        "train_start": "2014-01-01", "train_end": "2024-01-01", "n_obs": 100,
    }
    run_id = write_run(mem_conn, params, {"log_likelihood": 300.0})
    row = mem_conn.execute(
        "SELECT model_type, status FROM model_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row == ("vecm", "success")


def test_write_forecasts_inserts_six_rows(mem_conn: sqlite3.Connection) -> None:
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


def test_write_residuals_inserts_rows(mem_conn: sqlite3.Connection) -> None:
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


def test_run_vecm_model_smoke(mem_conn: sqlite3.Connection) -> None:
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


def test_run_vecm_model_empty_db_returns_minus_one(mem_conn: sqlite3.Connection) -> None:
    result = run_vecm_model(mem_conn)
    assert result == -1


def test_load_aligned_data_max_date_excludes_later_rows(mem_conn: sqlite3.Connection) -> None:
    """load_aligned_data with max_date must not include prices after that date."""
    symbols = ["KC=F", "RM=F", "BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]
    dates_all = pd.date_range("2018-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(0)
    for sym in symbols:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 48)))
        mem_conn.executemany(
            "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
            [(d.strftime("%Y-%m-%d"), sym, float(p)) for d, p in zip(dates_all, prices)],
        )
    mem_conn.commit()

    endog, exog = load_aligned_data(mem_conn, max_date="2019-12-01")

    assert not endog.empty
    assert endog.index.max() <= pd.Timestamp("2019-12-01")
    assert exog.index.max() <= pd.Timestamp("2019-12-01")


def test_load_aligned_data_max_date_none_loads_all(mem_conn: sqlite3.Connection) -> None:
    symbols = ["KC=F", "RM=F", "BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]
    dates_all = pd.date_range("2018-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(1)
    for sym in symbols:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 48)))
        mem_conn.executemany(
            "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
            [(d.strftime("%Y-%m-%d"), sym, float(p)) for d, p in zip(dates_all, prices)],
        )
    mem_conn.commit()

    endog_all, _ = load_aligned_data(mem_conn, max_date=None)
    endog_cut, _ = load_aligned_data(mem_conn, max_date="2019-12-01")

    assert len(endog_all) > len(endog_cut)
