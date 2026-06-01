import json
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.backtest import (
    collect_backtest_rows,
    compute_summary_metrics,
    get_validation_dates,
    load_actuals,
    run_backtest,
    write_accuracy_log_entries,
    write_backtest_results,
)


@pytest.fixture
def mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _seed_kc_prices(conn: sqlite3.Connection, n: int) -> list[str]:
    """Seed n months of KC=F into prices_monthly. Returns list of date strings."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    conn.executemany(
        "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    conn.commit()
    return [d.strftime("%Y-%m-%d") for d in dates]


def test_get_validation_dates_returns_correct_count(mem_conn: sqlite3.Connection) -> None:
    """With 60 KC=F months and min_train_months=36, expect 60-36=24 validation dates."""
    _seed_kc_prices(mem_conn, n=60)
    dates = get_validation_dates(mem_conn, min_train_months=36)
    assert len(dates) == 24


def test_get_validation_dates_returns_empty_when_insufficient(mem_conn: sqlite3.Connection) -> None:
    _seed_kc_prices(mem_conn, n=30)
    dates = get_validation_dates(mem_conn, min_train_months=36)
    assert dates == []


def test_get_validation_dates_empty_when_no_data(mem_conn: sqlite3.Connection) -> None:
    dates = get_validation_dates(mem_conn, min_train_months=36)
    assert dates == []


def test_get_validation_dates_are_sorted(mem_conn: sqlite3.Connection) -> None:
    _seed_kc_prices(mem_conn, n=60)
    dates = get_validation_dates(mem_conn, min_train_months=36)
    assert dates == sorted(dates)


def test_get_validation_dates_last_date_has_h1_actual(mem_conn: sqlite3.Connection) -> None:
    """The last validation date must have at least one month of future KC=F data."""
    all_dates = _seed_kc_prices(mem_conn, n=60)
    val_dates = get_validation_dates(mem_conn, min_train_months=36)
    last_val = val_dates[-1]
    # There must be a KC=F price after last_val
    last_price_date = all_dates[-1]
    assert last_price_date > last_val


def test_load_actuals_returns_matching_rows(mem_conn: sqlite3.Connection) -> None:
    all_dates = _seed_kc_prices(mem_conn, n=12)
    result = load_actuals(mem_conn, ["KC=F"], all_dates[:3])
    assert len(result) == 3
    assert set(result.columns) >= {"date", "symbol", "actual"}


def test_load_actuals_returns_empty_for_unknown_dates(mem_conn: sqlite3.Connection) -> None:
    _seed_kc_prices(mem_conn, n=12)
    result = load_actuals(mem_conn, ["KC=F"], ["2099-01-01"])
    assert result.empty


def test_load_actuals_returns_empty_for_empty_dates_list(mem_conn: sqlite3.Connection) -> None:
    result = load_actuals(mem_conn, ["KC=F"], [])
    assert result.empty


def _seed_model_run(conn: sqlite3.Connection, model_type: str, run_id: int) -> None:
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, '2024-01-01T00:00:00', ?, '2018-01-01', '2024-01-01', '{}', '{}', 'success')",
        (run_id, model_type),
    )
    conn.commit()


def _seed_vecm_forecasts_for_collect(conn: sqlite3.Connection, run_id: int) -> None:
    """Seed three h=1,2,3 forecasts for KC=F from train_end 2024-01-01."""
    conn.executemany(
        "INSERT INTO forecasts (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (run_id, "2024-01-01", "2024-02-01", 1, "KC=F", 200.0),
            (run_id, "2024-01-01", "2024-03-01", 2, "KC=F", 202.0),
            (run_id, "2024-01-01", "2024-04-01", 3, "KC=F", 204.0),
        ],
    )
    conn.commit()


def _seed_actuals(conn: sqlite3.Connection) -> None:
    """Seed actuals for h=1 and h=2 target dates (h=3 has no actual — future)."""
    conn.executemany(
        "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [
            ("2024-02-01", "KC=F", 198.0),
            ("2024-03-01", "KC=F", 205.0),
            # 2024-04-01 not seeded — simulates a future month with no actual yet
        ],
    )
    conn.commit()


def test_collect_backtest_rows_returns_three_rows(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "vecm", 10)
    _seed_vecm_forecasts_for_collect(mem_conn, 10)
    _seed_actuals(mem_conn)

    rows = collect_backtest_rows(mem_conn, run_id=10, train_end="2024-01-01")
    assert len(rows) == 3


def test_collect_backtest_rows_actual_populated_when_available(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "vecm", 10)
    _seed_vecm_forecasts_for_collect(mem_conn, 10)
    _seed_actuals(mem_conn)

    rows = collect_backtest_rows(mem_conn, run_id=10, train_end="2024-01-01")
    h1_row = next(r for r in rows if r["horizon"] == 1)
    assert h1_row["actual"] == pytest.approx(198.0)


def test_collect_backtest_rows_actual_none_when_not_in_db(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "vecm", 10)
    _seed_vecm_forecasts_for_collect(mem_conn, 10)
    _seed_actuals(mem_conn)

    rows = collect_backtest_rows(mem_conn, run_id=10, train_end="2024-01-01")
    h3_row = next(r for r in rows if r["horizon"] == 3)
    assert h3_row["actual"] is None


def test_collect_backtest_rows_contains_train_end(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "vecm", 10)
    _seed_vecm_forecasts_for_collect(mem_conn, 10)
    _seed_actuals(mem_conn)

    rows = collect_backtest_rows(mem_conn, run_id=10, train_end="2024-01-01")
    assert all(r["train_end"] == "2024-01-01" for r in rows)


def test_collect_backtest_rows_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    rows = collect_backtest_rows(mem_conn, run_id=999, train_end="2024-01-01")
    assert rows == []


def test_collect_backtest_rows_actual_is_per_symbol(mem_conn: sqlite3.Connection) -> None:
    """Each symbol must receive its own actual, not another symbol's."""
    _seed_model_run(mem_conn, "vecm", 11)
    mem_conn.executemany(
        "INSERT INTO forecasts (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (11, "2024-01-01", "2024-02-01", 1, "KC=F", 200.0),
            (11, "2024-01-01", "2024-02-01", 1, "RM=F", 90.0),
        ],
    )
    mem_conn.executemany(
        "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [
            ("2024-02-01", "KC=F", 198.0),
            ("2024-02-01", "RM=F", 88.0),
        ],
    )
    mem_conn.commit()

    rows = collect_backtest_rows(mem_conn, run_id=11, train_end="2024-01-01")
    kc_row = next(r for r in rows if r["symbol"] == "KC=F")
    rm_row = next(r for r in rows if r["symbol"] == "RM=F")
    assert kc_row["actual"] == pytest.approx(198.0)
    assert rm_row["actual"] == pytest.approx(88.0)


def _make_rows_with_actuals() -> list[dict]:
    return [
        {
            "forecast_id": 1,
            "train_end": "2024-01-01",
            "target_date": "2024-02-01",
            "horizon": 1,
            "symbol": "KC=F",
            "actual": 198.0,
            "point_forecast": 200.0,
            "p10": 180.0,
            "p25": 190.0,
            "p50": 200.0,
            "p75": 210.0,
            "p90": 220.0,
        },
        {
            "forecast_id": 2,
            "train_end": "2024-01-01",
            "target_date": "2024-03-01",
            "horizon": 2,
            "symbol": "KC=F",
            "actual": None,  # no actual yet
            "point_forecast": 202.0,
            "p10": 182.0,
            "p25": 192.0,
            "p50": 202.0,
            "p75": 212.0,
            "p90": 222.0,
        },
    ]


def test_write_backtest_results_inserts_both_rows(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "backtest", 20)
    write_backtest_results(mem_conn, run_id=20, backtest_date="2024-05-01", rows=_make_rows_with_actuals())
    count = mem_conn.execute("SELECT COUNT(*) FROM backtest_results WHERE run_id = 20").fetchone()[0]
    assert count == 2


def test_write_backtest_results_stores_actual_and_forecast(mem_conn: sqlite3.Connection) -> None:
    _seed_model_run(mem_conn, "backtest", 20)
    write_backtest_results(mem_conn, run_id=20, backtest_date="2024-05-01", rows=_make_rows_with_actuals())
    row = mem_conn.execute(
        "SELECT actual, point_forecast, p10, p90 FROM backtest_results WHERE run_id = 20 AND horizon = 1"
    ).fetchone()
    assert row[0] == pytest.approx(198.0)
    assert row[1] == pytest.approx(200.0)
    assert row[2] == pytest.approx(180.0)
    assert row[3] == pytest.approx(220.0)


def test_write_accuracy_log_only_writes_rows_with_actual(mem_conn: sqlite3.Connection) -> None:
    """Rows with actual=None must be skipped."""
    write_accuracy_log_entries(mem_conn, _make_rows_with_actuals())
    count = mem_conn.execute("SELECT COUNT(*) FROM accuracy_log").fetchone()[0]
    assert count == 1  # only h=1 row has an actual


def test_write_accuracy_log_computes_mae(mem_conn: sqlite3.Connection) -> None:
    write_accuracy_log_entries(mem_conn, _make_rows_with_actuals())
    row = mem_conn.execute("SELECT mae FROM accuracy_log").fetchone()
    assert row is not None
    # actual=198, point_forecast=200 → mae = 2.0
    assert row[0] == pytest.approx(2.0)


def test_write_accuracy_log_coverage_80_is_one_when_inside_band(mem_conn: sqlite3.Connection) -> None:
    # actual=198 is between p10=180 and p90=220 → coverage_80 = 1
    write_accuracy_log_entries(mem_conn, _make_rows_with_actuals())
    row = mem_conn.execute("SELECT coverage_80 FROM accuracy_log").fetchone()
    assert row[0] == 1


def test_write_accuracy_log_coverage_80_is_zero_when_outside_band(mem_conn: sqlite3.Connection) -> None:
    outside_rows = [
        {
            "forecast_id": 3,
            "train_end": "2024-01-01",
            "target_date": "2024-02-01",
            "horizon": 1,
            "symbol": "KC=F",
            "actual": 250.0,  # above p90=220
            "point_forecast": 200.0,
            "p10": 180.0,
            "p25": 190.0,
            "p50": 200.0,
            "p75": 210.0,
            "p90": 220.0,
        }
    ]
    write_accuracy_log_entries(mem_conn, outside_rows)
    row = mem_conn.execute("SELECT coverage_80 FROM accuracy_log").fetchone()
    assert row[0] == 0


def test_write_accuracy_log_coverage_80_is_none_when_quantiles_missing(mem_conn: sqlite3.Connection) -> None:
    """coverage_80 must be NULL (not 0) when p10/p90 quantiles are missing."""
    missing_quantile_rows = [
        {
            "forecast_id": 4,
            "train_end": "2024-01-01",
            "target_date": "2024-02-01",
            "horizon": 1,
            "symbol": "KC=F",
            "actual": 198.0,
            "point_forecast": 200.0,
            "p10": None,  # missing — GAMLSS convergence failure
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }
    ]
    write_accuracy_log_entries(mem_conn, missing_quantile_rows)
    row = mem_conn.execute("SELECT coverage_80 FROM accuracy_log").fetchone()
    assert row[0] is None


def _seed_backtest_results(conn: sqlite3.Connection, run_id: int) -> None:
    """Seed 6 backtest_results rows: h=1,2,3 × KC=F,RM=F. Actuals all present."""
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, '2024-01-01T00:00:00', 'backtest', '2018-01-01', '2022-01-01', '{}', '{}', 'success')",
        (run_id,),
    )
    conn.executemany(
        "INSERT INTO backtest_results"
        " (run_id, backtest_date, train_end, target_date, horizon, symbol, actual,"
        "  point_forecast, p10, p25, p50, p75, p90)"
        " VALUES (?, '2024-05-01', '2022-01-01', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # horizon=1, KC=F: actual inside band
            (run_id, "2022-02-01", 1, "KC=F", 200.0, 195.0, 180.0, 190.0, 195.0, 200.0, 210.0),
            # horizon=2, KC=F: actual inside band
            (run_id, "2022-03-01", 2, "KC=F", 205.0, 200.0, 185.0, 195.0, 200.0, 205.0, 215.0),
            # horizon=3, KC=F: actual outside band
            (run_id, "2022-04-01", 3, "KC=F", 230.0, 200.0, 185.0, 195.0, 200.0, 205.0, 215.0),
            # horizon=1, RM=F: actual inside band
            (run_id, "2022-02-01", 1, "RM=F", 100.0, 98.0, 90.0, 95.0, 98.0, 101.0, 106.0),
            # horizon=2, RM=F: actual inside band
            (run_id, "2022-03-01", 2, "RM=F", 102.0, 100.0, 92.0, 97.0, 100.0, 103.0, 108.0),
            # horizon=3, RM=F: actual outside band
            (run_id, "2022-04-01", 3, "RM=F", 120.0, 100.0, 92.0, 97.0, 100.0, 103.0, 108.0),
        ],
    )
    conn.commit()


def test_compute_summary_metrics_returns_non_empty_dict(mem_conn: sqlite3.Connection) -> None:
    _seed_backtest_results(mem_conn, run_id=30)
    metrics = compute_summary_metrics(mem_conn, run_id=30)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_compute_summary_metrics_contains_expected_keys(mem_conn: sqlite3.Connection) -> None:
    _seed_backtest_results(mem_conn, run_id=30)
    metrics = compute_summary_metrics(mem_conn, run_id=30)
    assert "h1_KC=F_mae" in metrics
    assert "h1_KC=F_coverage_80" in metrics
    assert "h1_KC=F_coverage_50" in metrics
    assert "h1_RM=F_mae" in metrics


def test_compute_summary_metrics_coverage_80_value(mem_conn: sqlite3.Connection) -> None:
    """h1 KC=F has 1 window: actual=200 inside [180,210] → coverage_80=1.0."""
    _seed_backtest_results(mem_conn, run_id=30)
    metrics = compute_summary_metrics(mem_conn, run_id=30)
    assert metrics["h1_KC=F_coverage_80"] == pytest.approx(1.0)


def test_compute_summary_metrics_mae_value(mem_conn: sqlite3.Connection) -> None:
    """h1 KC=F: actual=200, point_forecast=195 → mae=5.0."""
    _seed_backtest_results(mem_conn, run_id=30)
    metrics = compute_summary_metrics(mem_conn, run_id=30)
    assert metrics["h1_KC=F_mae"] == pytest.approx(5.0)


def test_compute_summary_metrics_returns_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    metrics = compute_summary_metrics(mem_conn, run_id=999)
    assert metrics == {}


def test_compute_summary_metrics_coverage_80_is_none_when_all_quantiles_null(mem_conn: sqlite3.Connection) -> None:
    """coverage_80 must be None (not 0) when all rows have NULL quantiles."""
    conn = mem_conn
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (40, '2024-01-01T00:00:00', 'backtest', '2018-01-01', '2022-01-01', '{}', '{}', 'success')"
    )
    conn.execute(
        "INSERT INTO backtest_results"
        " (run_id, backtest_date, train_end, target_date, horizon, symbol, actual,"
        "  point_forecast, p10, p25, p50, p75, p90)"
        " VALUES (40, '2024-05-01', '2022-01-01', '2022-02-01', 1, 'KC=F', 200.0, 195.0,"
        "  NULL, NULL, NULL, NULL, NULL)"
    )
    conn.commit()

    metrics = compute_summary_metrics(conn, run_id=40)
    assert metrics["h1_KC=F_coverage_80"] is None
    assert metrics["h1_KC=F_coverage_50"] is None
    assert metrics["h1_KC=F_pinball_50"] is None
    assert metrics["h1_KC=F_mae"] == pytest.approx(5.0)  # point_forecast still valid


_ALL_SYMBOLS = ["KC=F", "RM=F", "BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]


def _seed_all_symbols(conn: sqlite3.Connection, n: int) -> None:
    """Seed n months of all 6 VECM symbols into prices_monthly."""
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    for seed, sym in enumerate(_ALL_SYMBOLS):
        rng = np.random.default_rng(seed + 100)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
        conn.executemany(
            "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
            [(d.strftime("%Y-%m-%d"), sym, float(p)) for d, p in zip(dates, prices)],
        )
    conn.commit()


def test_run_backtest_returns_minus_one_when_insufficient_data(mem_conn: sqlite3.Connection) -> None:
    _seed_kc_prices(mem_conn, n=30)
    result = run_backtest(mem_conn, min_train_months=36, skip_gamlss=True)
    assert result == -1


def test_run_backtest_skip_gamlss_returns_positive_run_id(mem_conn: sqlite3.Connection) -> None:
    """End-to-end: 42 months of data, min_train_months=36, max_windows=2."""
    _seed_all_symbols(mem_conn, n=42)
    result = run_backtest(mem_conn, min_train_months=36, max_windows=2, skip_gamlss=True)
    assert result > 0


def test_run_backtest_writes_success_model_run(mem_conn: sqlite3.Connection) -> None:
    _seed_all_symbols(mem_conn, n=42)
    run_id = run_backtest(mem_conn, min_train_months=36, max_windows=2, skip_gamlss=True)
    row = mem_conn.execute(
        "SELECT model_type, status FROM model_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row == ("backtest", "success")


def test_run_backtest_writes_backtest_results_rows(mem_conn: sqlite3.Connection) -> None:
    _seed_all_symbols(mem_conn, n=42)
    run_id = run_backtest(mem_conn, min_train_months=36, max_windows=2, skip_gamlss=True)
    count = mem_conn.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    # 2 windows × 3 horizons × 2 symbols = 12 rows
    assert count == 12
    accuracy_count = mem_conn.execute("SELECT COUNT(*) FROM accuracy_log").fetchone()[0]
    assert accuracy_count > 0


def test_run_backtest_respects_max_windows(mem_conn: sqlite3.Connection) -> None:
    _seed_all_symbols(mem_conn, n=45)
    run_id_2 = run_backtest(mem_conn, min_train_months=36, max_windows=2, skip_gamlss=True)
    run_id_3 = run_backtest(mem_conn, min_train_months=36, max_windows=3, skip_gamlss=True)
    count_2 = mem_conn.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id_2,)
    ).fetchone()[0]
    count_3 = mem_conn.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id_3,)
    ).fetchone()[0]
    assert count_3 > count_2


def test_run_backtest_stores_metrics_in_model_run(mem_conn: sqlite3.Connection) -> None:
    _seed_all_symbols(mem_conn, n=42)
    run_id = run_backtest(mem_conn, min_train_months=36, max_windows=2, skip_gamlss=True)
    metrics_json = mem_conn.execute(
        "SELECT metrics FROM model_runs WHERE id = ?", (run_id,)
    ).fetchone()[0]
    metrics = json.loads(metrics_json)
    assert isinstance(metrics, dict)


def test_main_exits_zero_with_skip_gamlss(tmp_path) -> None:
    """CLI with --skip-gamlss and a temp DB should exit 0."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    # Seed 42 months of all 6 symbols
    dates = pd.date_range("2018-01-01", periods=42, freq="MS")
    for seed, sym in enumerate(_ALL_SYMBOLS):
        rng = np.random.default_rng(seed + 200)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 42)))
        conn.executemany(
            "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
            [(d.strftime("%Y-%m-%d"), sym, float(p)) for d, p in zip(dates, prices)],
        )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable, "-m", "coffee_forecast.models.backtest",
            "--db", db_path,
            "--min-train-months", "36",
            "--max-windows", "1",
            "--skip-gamlss",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
