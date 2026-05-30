import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.hybrid import (
    combine_forecasts,
    get_current_regime,
    load_gamlss_quantiles,
    load_vecm_forecasts,
    run_hybrid_model,
    write_hybrid_forecasts,
)


@pytest.fixture
def mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _seed_model_runs(conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert VECM and GAMLSS model run rows, return (vecm_run_id, gamlss_run_id)."""
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (1, '2024-01-01T00:00:00', 'vecm', '2014-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (2, '2024-01-01T00:00:00', 'gamlss', '2014-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    conn.commit()
    return 1, 2


def _seed_vecm_forecasts(conn: sqlite3.Connection, vecm_run_id: int) -> None:
    conn.executemany(
        "INSERT INTO forecasts (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (vecm_run_id, "2024-01-01", "2024-02-01", 1, "KC=F", 180.0),
            (vecm_run_id, "2024-01-01", "2024-03-01", 2, "KC=F", 185.0),
            (vecm_run_id, "2024-01-01", "2024-04-01", 3, "KC=F", 190.0),
            (vecm_run_id, "2024-01-01", "2024-02-01", 1, "RM=F", 100.0),
            (vecm_run_id, "2024-01-01", "2024-03-01", 2, "RM=F", 102.0),
            (vecm_run_id, "2024-01-01", "2024-04-01", 3, "RM=F", 104.0),
        ],
    )
    conn.commit()


def _seed_gamlss_params(conn: sqlite3.Connection, gamlss_run_id: int) -> None:
    conn.executemany(
        "INSERT INTO gamlss_params"
        " (run_id, symbol, regime, mu, sigma, nu, tau, q10, q25, q50, q75, q90, n_obs)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (gamlss_run_id, "KC=F", "Low",    0.01, 0.05, -0.1, 2.0, -0.07, -0.03, 0.01, 0.05, 0.09, 45),
            (gamlss_run_id, "RM=F", "Low",    0.00, 0.06, -0.1, 2.0, -0.08, -0.04, 0.00, 0.04, 0.08, 45),
            (gamlss_run_id, "KC=F", "Medium", 0.00, 0.08, -0.05, 1.8, -0.11, -0.05, 0.00, 0.05, 0.11, 42),
            (gamlss_run_id, "RM=F", "Medium", 0.00, 0.09, -0.05, 1.8, -0.12, -0.06, 0.00, 0.06, 0.12, 42),
            (gamlss_run_id, "KC=F", "High",   0.00, 0.12, -0.02, 1.5, -0.15, -0.08, 0.00, 0.08, 0.15, 30),
            (gamlss_run_id, "RM=F", "High",   0.00, 0.13, -0.02, 1.5, -0.16, -0.09, 0.00, 0.09, 0.16, 30),
        ],
    )
    conn.commit()


def _seed_kc_prices(conn: sqlite3.Connection, n: int = 48) -> None:
    """Insert synthetic KC=F monthly prices into prices_monthly."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    conn.commit()


# --- Tests for Task 1 ---

def test_load_vecm_forecasts_returns_six_rows(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, _ = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    df = load_vecm_forecasts(mem_conn, vecm_run_id)
    assert len(df) == 6
    assert set(df.columns) >= {"horizon", "symbol", "point_forecast"}


def test_load_vecm_forecasts_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    df = load_vecm_forecasts(mem_conn, vecm_run_id=999)
    assert df.empty


def test_load_gamlss_quantiles_returns_six_rows(mem_conn: sqlite3.Connection) -> None:
    _, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    df = load_gamlss_quantiles(mem_conn, gamlss_run_id)
    assert len(df) == 6
    assert set(df.columns) >= {"symbol", "regime", "q10", "q25", "q50", "q75", "q90"}


def test_load_gamlss_quantiles_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    df = load_gamlss_quantiles(mem_conn, gamlss_run_id=999)
    assert df.empty


# --- Tests for Task 2 ---

def test_get_current_regime_returns_valid_label(mem_conn: sqlite3.Connection) -> None:
    _seed_kc_prices(mem_conn, n=48)
    regime = get_current_regime(mem_conn)
    assert regime in {"Low", "Medium", "High"}


def test_get_current_regime_raises_when_no_data(mem_conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="no KC=F price data"):
        get_current_regime(mem_conn)


# --- Tests for Task 3 ---

def _make_vecm_df() -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [1, 2, 1, 2],
        "symbol": ["KC=F", "KC=F", "RM=F", "RM=F"],
        "point_forecast": [180.0, 185.0, 100.0, 102.0],
    })


def _make_gamlss_df() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["KC=F", "RM=F", "KC=F", "RM=F"],
        "regime": ["Low", "Low", "Medium", "Medium"],
        "q10": [-0.07, -0.08, -0.11, -0.12],
        "q25": [-0.03, -0.04, -0.05, -0.06],
        "q50": [0.01, 0.00, 0.00, 0.00],
        "q75": [0.05, 0.04, 0.05, 0.06],
        "q90": [0.09, 0.08, 0.11, 0.12],
    })


def test_combine_forecasts_output_shape() -> None:
    result = combine_forecasts(_make_vecm_df(), _make_gamlss_df(), "Low")
    assert result.shape == (4, 8)
    assert set(result.columns) == {"horizon", "symbol", "point_forecast", "p10", "p25", "p50", "p75", "p90"}


def test_combine_forecasts_quantile_math() -> None:
    result = combine_forecasts(_make_vecm_df(), _make_gamlss_df(), "Low")
    kc_h1 = result[(result["symbol"] == "KC=F") & (result["horizon"] == 1)].iloc[0]
    assert kc_h1["point_forecast"] == pytest.approx(180.0)
    assert kc_h1["p10"] == pytest.approx(180.0 * np.exp(-0.07), rel=1e-6)
    assert kc_h1["p50"] == pytest.approx(180.0 * np.exp(0.01), rel=1e-6)
    assert kc_h1["p90"] == pytest.approx(180.0 * np.exp(0.09), rel=1e-6)


def test_combine_forecasts_interval_ordering() -> None:
    result = combine_forecasts(_make_vecm_df(), _make_gamlss_df(), "Low")
    for _, row in result.iterrows():
        assert row["p10"] < row["p25"] < row["p50"] < row["p75"] < row["p90"]


def test_combine_forecasts_missing_regime_raises() -> None:
    with pytest.raises(ValueError, match="No GAMLSS params for regime 'High'"):
        combine_forecasts(_make_vecm_df(), _make_gamlss_df(), "High")


def test_combine_forecasts_missing_symbol_raises() -> None:
    gamlss_no_rm = _make_gamlss_df()[_make_gamlss_df()["symbol"] == "KC=F"].reset_index(drop=True)
    with pytest.raises(ValueError, match="No GAMLSS params for symbol 'RM=F'"):
        combine_forecasts(_make_vecm_df(), gamlss_no_rm, "Low")


# --- Tests for Task 4 ---

def _make_combined_df() -> pd.DataFrame:
    return pd.DataFrame({
        "horizon": [1, 2, 1, 2],
        "symbol": ["KC=F", "KC=F", "RM=F", "RM=F"],
        "point_forecast": [180.0, 185.0, 100.0, 102.0],
        "p10": [165.0, 169.0, 92.0, 93.0],
        "p25": [173.0, 178.0, 96.0, 98.0],
        "p50": [181.0, 186.0, 100.0, 102.0],
        "p75": [189.0, 194.0, 104.0, 106.0],
        "p90": [197.0, 202.0, 108.0, 110.0],
    })


def test_write_hybrid_forecasts_inserts_correct_row_count(mem_conn: sqlite3.Connection) -> None:
    _seed_model_runs(mem_conn)
    mem_conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (3, '2024-01-01T00:00:00', 'hybrid', '2024-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    mem_conn.commit()
    write_hybrid_forecasts(mem_conn, run_id=3, combined_df=_make_combined_df(), forecast_date="2024-01-01")
    rows = mem_conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id = 3").fetchone()[0]
    assert rows == 4


def test_write_hybrid_forecasts_quantiles_stored(mem_conn: sqlite3.Connection) -> None:
    _seed_model_runs(mem_conn)
    mem_conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (3, '2024-01-01T00:00:00', 'hybrid', '2024-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    mem_conn.commit()
    write_hybrid_forecasts(mem_conn, run_id=3, combined_df=_make_combined_df(), forecast_date="2024-01-01")
    row = mem_conn.execute(
        "SELECT point_forecast, p10, p50, p90 FROM forecasts"
        " WHERE run_id = 3 AND symbol = 'KC=F' AND horizon = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(180.0)
    assert row[1] == pytest.approx(165.0)
    assert row[2] == pytest.approx(181.0)
    assert row[3] == pytest.approx(197.0)


def test_write_hybrid_forecasts_target_dates(mem_conn: sqlite3.Connection) -> None:
    _seed_model_runs(mem_conn)
    mem_conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (3, '2024-01-01T00:00:00', 'hybrid', '2024-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    mem_conn.commit()
    write_hybrid_forecasts(mem_conn, run_id=3, combined_df=_make_combined_df(), forecast_date="2024-01-01")
    targets = {
        r[0]
        for r in mem_conn.execute(
            "SELECT target_date FROM forecasts WHERE run_id = 3 AND symbol = 'KC=F'"
        ).fetchall()
    }
    assert targets == {"2024-02-01", "2024-03-01"}


# --- Tests for Task 5 ---

def test_run_hybrid_model_returns_positive_run_id(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    run_id = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    assert run_id > 0


def test_run_hybrid_model_writes_model_run_record(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    run_id = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    row = mem_conn.execute(
        "SELECT model_type, status FROM model_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row == ("hybrid", "success")


def test_run_hybrid_model_writes_six_forecast_rows(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    run_id = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    count = mem_conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    assert count == 6


def test_run_hybrid_model_forecasts_have_ordered_quantiles(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    run_id = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    rows = mem_conn.execute(
        "SELECT p10, p25, p50, p75, p90 FROM forecasts WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert len(rows) == 6
    for p10, p25, p50, p75, p90 in rows:
        assert p10 < p25 < p50 < p75 < p90, (
            f"Quantile ordering violated: {p10=} {p25=} {p50=} {p75=} {p90=}"
        )


def test_run_hybrid_model_records_regime_and_run_ids_in_params(
    mem_conn: sqlite3.Connection,
) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    run_id = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    params_json = mem_conn.execute(
        "SELECT params FROM model_runs WHERE id = ?", (run_id,)
    ).fetchone()[0]
    params = json.loads(params_json)
    assert params["regime"] in {"Low", "Medium", "High"}
    assert params["vecm_run_id"] == vecm_run_id
    assert params["gamlss_run_id"] == gamlss_run_id


def test_run_hybrid_model_returns_minus_one_when_no_vecm_forecasts(
    mem_conn: sqlite3.Connection,
) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    # No vecm forecasts seeded — load_vecm_forecasts will return empty DataFrame
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    result = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    assert result == -1


def test_run_hybrid_model_returns_minus_one_when_no_gamlss_params(
    mem_conn: sqlite3.Connection,
) -> None:
    vecm_run_id, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    # No gamlss params seeded — load_gamlss_quantiles will return empty DataFrame
    _seed_kc_prices(mem_conn)
    result = run_hybrid_model(mem_conn, vecm_run_id, gamlss_run_id)
    assert result == -1


def test_run_hybrid_model_returns_minus_one_for_unknown_vecm_run_id(
    mem_conn: sqlite3.Connection,
) -> None:
    _, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    _seed_kc_prices(mem_conn)
    # vecm_run_id=999 has no model_runs entry and no forecasts
    result = run_hybrid_model(mem_conn, vecm_run_id=999, gamlss_run_id=gamlss_run_id)
    assert result == -1
