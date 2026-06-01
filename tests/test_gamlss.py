"""Tests for coffee_forecast.models.gamlss.

R-dependent tests are marked with @pytest.mark.r_required and are skipped
when Rscript is not installed. All other tests run fully in Python.
"""
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.models.gamlss import (
    _float_or_none,
    build_gamlss_input,
    call_rscript,
    compute_regime_labels,
    load_residuals,
    parse_gamlss_output,
    run_gamlss_model,
    write_gamlss_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_INLINE_SCHEMA = """
CREATE TABLE prices_monthly (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    adj_close REAL,
    UNIQUE (date, symbol)
);
CREATE TABLE model_runs (
    id INTEGER PRIMARY KEY,
    run_at TEXT, model_type TEXT,
    train_start TEXT, train_end TEXT,
    params TEXT, metrics TEXT, status TEXT
);
CREATE TABLE vecm_residuals (
    id INTEGER PRIMARY KEY,
    run_id INTEGER, date TEXT, symbol TEXT, residual REAL,
    UNIQUE (run_id, date, symbol)
);
CREATE TABLE gamlss_params (
    id INTEGER PRIMARY KEY,
    run_id INTEGER, symbol TEXT, regime TEXT,
    mu REAL, sigma REAL, nu REAL, tau REAL,
    q10 REAL, q25 REAL, q50 REAL, q75 REAL, q90 REAL,
    n_obs INTEGER,
    UNIQUE (run_id, symbol, regime)
);
"""

_VALID_OUTPUT_CSV = (
    "symbol,regime,mu,sigma,nu,tau,q10,q25,q50,q75,q90,n_obs\n"
    "KC=F,Low,0.01,0.05,-0.10,2.00,-0.07,-0.03,0.01,0.05,0.09,45\n"
    "KC=F,Medium,0.00,0.08,-0.05,1.80,-0.11,-0.05,0.00,0.05,0.11,42\n"
    "KC=F,High,,,,,,,,,,12\n"  # sparse regime: NAs
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_INLINE_SCHEMA)
    return conn


def _insert_kc_prices(conn: sqlite3.Connection, n_months: int = 48, seed: int = 0) -> None:
    """Populate prices_monthly with synthetic KC=F data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.05, n_months)))
    conn.executemany(
        "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?,?,?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    conn.commit()


def _insert_vecm_run(conn: sqlite3.Connection) -> int:
    """Insert a fake successful VECM run and return its id."""
    conn.execute(
        "INSERT INTO model_runs (run_at,model_type,train_start,train_end,params,metrics,status)"
        " VALUES ('2024-01-01','vecm','2016-01-01','2019-12-01','{}','{}','success')"
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_residuals(
    conn: sqlite3.Connection,
    run_id: int,
    symbols: list[str] | None = None,
    n: int = 36,
    seed: int = 1,
) -> None:
    """Insert fake VECM residuals for given symbols."""
    if symbols is None:
        symbols = ["KC=F"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="MS")
    for sym in symbols:
        residuals = rng.normal(0, 0.03, n)
        conn.executemany(
            "INSERT OR IGNORE INTO vecm_residuals (run_id,date,symbol,residual) VALUES (?,?,?,?)",
            [(run_id, d.strftime("%Y-%m-%d"), sym, float(r)) for d, r in zip(dates, residuals)],
        )
    conn.commit()


# ---------------------------------------------------------------------------
# compute_regime_labels
# ---------------------------------------------------------------------------


def test_compute_regime_labels_no_data_returns_empty():
    conn = _make_conn()
    result = compute_regime_labels(conn)
    assert result.empty


def test_compute_regime_labels_returns_series_with_correct_name():
    conn = _make_conn()
    _insert_kc_prices(conn)
    result = compute_regime_labels(conn)
    assert result.name == "regime"
    assert result.index.name == "date"


def test_compute_regime_labels_values_are_valid():
    conn = _make_conn()
    _insert_kc_prices(conn)
    result = compute_regime_labels(conn)
    assert set(result.unique()).issubset({"Low", "Medium", "High"})


def test_compute_regime_labels_non_empty():
    conn = _make_conn()
    _insert_kc_prices(conn, n_months=48)
    result = compute_regime_labels(conn)
    # 12-month rolling window → first 12 rows dropped; expect >0 labels
    assert len(result) > 0


def test_compute_regime_labels_index_is_datetime():
    conn = _make_conn()
    _insert_kc_prices(conn)
    result = compute_regime_labels(conn)
    assert pd.api.types.is_datetime64_any_dtype(result.index)


# ---------------------------------------------------------------------------
# load_residuals
# ---------------------------------------------------------------------------


def test_load_residuals_returns_correct_run():
    conn = _make_conn()
    run1 = _insert_vecm_run(conn)
    run2 = _insert_vecm_run(conn)
    _insert_residuals(conn, run1, n=5)
    _insert_residuals(conn, run2, n=3)
    result = load_residuals(conn, run_id=run1)
    assert len(result) == 5
    assert set(result.columns) >= {"date", "symbol", "residual"}


def test_load_residuals_empty_for_unknown_run():
    conn = _make_conn()
    result = load_residuals(conn, run_id=9999)
    assert result.empty


# ---------------------------------------------------------------------------
# build_gamlss_input
# ---------------------------------------------------------------------------


def _make_regime_series(date_regime_map: dict[str, str]) -> pd.Series:
    s = pd.Series(date_regime_map, name="regime")
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s


def test_build_gamlss_input_joins_on_date():
    residuals = pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "symbol": ["KC=F"] * 3,
        "residual": [0.01, -0.02, 0.03],
    })
    regime = _make_regime_series({"2020-01-01": "Low", "2020-02-01": "High"})
    result = build_gamlss_input(residuals, regime)
    assert len(result) == 2  # 2020-03-01 has no regime → dropped
    assert set(result["regime"]) == {"Low", "High"}


def test_build_gamlss_input_output_columns():
    residuals = pd.DataFrame({
        "date": ["2020-01-01"],
        "symbol": ["KC=F"],
        "residual": [0.0],
    })
    regime = _make_regime_series({"2020-01-01": "Medium"})
    result = build_gamlss_input(residuals, regime)
    assert list(result.columns) == ["date", "symbol", "residual", "regime"]


def test_build_gamlss_input_empty_when_no_overlap():
    residuals = pd.DataFrame({
        "date": ["2020-06-01"],
        "symbol": ["KC=F"],
        "residual": [0.0],
    })
    regime = _make_regime_series({"2020-01-01": "Low"})
    result = build_gamlss_input(residuals, regime)
    assert result.empty


# ---------------------------------------------------------------------------
# call_rscript
# ---------------------------------------------------------------------------


def test_call_rscript_raises_if_r_script_missing(tmp_path):
    missing = tmp_path / "nonexistent.R"
    with pytest.raises(FileNotFoundError, match="R script not found"):
        call_rscript("in.csv", "out.csv", r_script=missing)


def test_call_rscript_raises_on_nonzero_returncode(tmp_path):
    dummy_r = tmp_path / "fail.R"
    dummy_r.write_text("stop('deliberate failure')\n")
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    in_csv.write_text("date,symbol,residual,regime\n")

    try:
        from coffee_forecast.models.gamlss import _find_rscript
        _find_rscript()
    except FileNotFoundError:
        pytest.skip("Rscript not installed")

    with pytest.raises(RuntimeError, match="exited with code"):
        call_rscript(str(in_csv), str(out_csv), r_script=dummy_r)


def test_call_rscript_raises_filenotfounderror_when_rscript_missing(tmp_path):
    """When Rscript binary itself is absent, subprocess raises FileNotFoundError."""
    dummy_r = tmp_path / "ok.R"
    dummy_r.write_text("")
    with patch("subprocess.run", side_effect=FileNotFoundError("Rscript not found")):
        with pytest.raises(FileNotFoundError):
            call_rscript("in.csv", "out.csv", r_script=dummy_r)


# ---------------------------------------------------------------------------
# parse_gamlss_output
# ---------------------------------------------------------------------------


def test_parse_gamlss_output_valid(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text(_VALID_OUTPUT_CSV)
    df = parse_gamlss_output(str(p))
    assert len(df) == 3
    assert df.iloc[0]["symbol"] == "KC=F"
    assert df.iloc[0]["q50"] == pytest.approx(0.01)


def test_parse_gamlss_output_missing_column_raises(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text("symbol,regime,mu\nKC=F,Low,0.01\n")
    with pytest.raises(ValueError, match="missing columns"):
        parse_gamlss_output(str(p))


def test_parse_gamlss_output_na_rows_allowed(tmp_path):
    """Rows with NA params (convergence failures) must not raise."""
    p = tmp_path / "out.csv"
    p.write_text(_VALID_OUTPUT_CSV)
    df = parse_gamlss_output(str(p))
    # third row (High regime) has NA params
    assert pd.isna(df.iloc[2]["mu"])


# ---------------------------------------------------------------------------
# _float_or_none
# ---------------------------------------------------------------------------


def test_float_or_none_numeric():
    assert _float_or_none(1.5) == pytest.approx(1.5)


def test_float_or_none_nan_returns_none():
    assert _float_or_none(float("nan")) is None


def test_float_or_none_none_returns_none():
    assert _float_or_none(None) is None


def test_float_or_none_string_returns_none():
    assert _float_or_none("NA") is None


def test_float_or_none_zero():
    assert _float_or_none(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# write_gamlss_params
# ---------------------------------------------------------------------------


def _sample_params_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "symbol": "KC=F", "regime": "Low",
            "mu": 0.01, "sigma": 0.05, "nu": -0.10, "tau": 2.00,
            "q10": -0.07, "q25": -0.03, "q50": 0.01, "q75": 0.05, "q90": 0.09,
            "n_obs": 45,
        },
        {
            "symbol": "KC=F", "regime": "High",
            "mu": float("nan"), "sigma": float("nan"),
            "nu": float("nan"), "tau": float("nan"),
            "q10": float("nan"), "q25": float("nan"), "q50": float("nan"),
            "q75": float("nan"), "q90": float("nan"),
            "n_obs": 5,
        },
    ])


def test_write_gamlss_params_inserts_rows():
    conn = _make_conn()
    conn.execute(
        "INSERT INTO model_runs VALUES (1,'','gamlss','','','{}','{}','pending')"
    )
    conn.commit()
    write_gamlss_params(conn, run_id=1, params_df=_sample_params_df())
    rows = conn.execute("SELECT * FROM gamlss_params").fetchall()
    assert len(rows) == 2


def test_write_gamlss_params_nan_stored_as_null():
    conn = _make_conn()
    conn.execute(
        "INSERT INTO model_runs VALUES (1,'','gamlss','','','{}','{}','pending')"
    )
    conn.commit()
    write_gamlss_params(conn, run_id=1, params_df=_sample_params_df())
    row = conn.execute(
        "SELECT mu FROM gamlss_params WHERE regime = 'High'"
    ).fetchone()
    assert row[0] is None


def test_write_gamlss_params_upsert_is_idempotent():
    conn = _make_conn()
    conn.execute(
        "INSERT INTO model_runs VALUES (1,'','gamlss','','','{}','{}','pending')"
    )
    conn.commit()
    df = _sample_params_df().head(1)
    write_gamlss_params(conn, run_id=1, params_df=df)
    write_gamlss_params(conn, run_id=1, params_df=df)
    rows = conn.execute("SELECT * FROM gamlss_params").fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# run_gamlss_model — unit tests with mocked Rscript
# ---------------------------------------------------------------------------


def _make_full_conn(n_price_months: int = 60, n_resid: int = 36) -> tuple[sqlite3.Connection, int]:
    conn = _make_conn()
    _insert_kc_prices(conn, n_months=n_price_months)
    run_id = _insert_vecm_run(conn)
    _insert_residuals(conn, run_id, symbols=["KC=F", "RM=F"], n=n_resid)
    return conn, run_id


def _make_mock_rscript(tmp_path: Path, params_df: pd.DataFrame):
    """Return a callable that writes params_df as the R output CSV.

    The mock patches call_rscript at module level; run_gamlss_model calls it
    as call_rscript(input_csv, output_csv) without the r_script keyword arg.
    """
    def _side_effect(input_csv: str, output_csv: str) -> None:
        params_df.to_csv(output_csv, index=False)
    return _side_effect


def _minimal_params_df() -> pd.DataFrame:
    rows = []
    for sym in ("KC=F", "RM=F"):
        for reg in ("Low", "Medium", "High"):
            rows.append({
                "symbol": sym, "regime": reg,
                "mu": 0.00, "sigma": 0.05, "nu": 0.00, "tau": 2.00,
                "q10": -0.08, "q25": -0.04, "q50": 0.00,
                "q75": 0.04, "q90": 0.08,
                "n_obs": 12,
            })
    return pd.DataFrame(rows)


def test_run_gamlss_model_returns_minus_one_with_no_residuals():
    conn = _make_conn()
    _insert_kc_prices(conn)
    run_id = _insert_vecm_run(conn)
    # no residuals inserted
    result = run_gamlss_model(conn, run_id)
    assert result == -1


def test_run_gamlss_model_returns_minus_one_with_no_prices():
    conn = _make_conn()
    run_id = _insert_vecm_run(conn)
    _insert_residuals(conn, run_id)
    # no KC=F prices → no regime labels
    result = run_gamlss_model(conn, run_id)
    assert result == -1


def test_run_gamlss_model_writes_params_on_success(tmp_path):
    conn, run_id = _make_full_conn()
    fake_params = _minimal_params_df()

    with patch("coffee_forecast.models.gamlss.call_rscript", side_effect=_make_mock_rscript(tmp_path, fake_params)):
        gamlss_run_id = run_gamlss_model(conn, run_id)

    assert gamlss_run_id > 0
    rows = conn.execute(
        "SELECT * FROM gamlss_params WHERE run_id = ?", (gamlss_run_id,)
    ).fetchall()
    assert len(rows) == 6  # 2 symbols × 3 regimes


def test_run_gamlss_model_sets_status_success(tmp_path):
    conn, run_id = _make_full_conn()
    fake_params = _minimal_params_df()

    with patch("coffee_forecast.models.gamlss.call_rscript", side_effect=_make_mock_rscript(tmp_path, fake_params)):
        gamlss_run_id = run_gamlss_model(conn, run_id)

    status = conn.execute(
        "SELECT status FROM model_runs WHERE id = ?", (gamlss_run_id,)
    ).fetchone()[0]
    assert status == "success"


def test_run_gamlss_model_sets_status_failed_on_rscript_error(tmp_path):
    conn, run_id = _make_full_conn()

    with patch(
        "coffee_forecast.models.gamlss.call_rscript",
        side_effect=RuntimeError("Rscript died"),
    ):
        with pytest.raises(RuntimeError, match="Rscript died"):
            run_gamlss_model(conn, run_id)

    row = conn.execute(
        "SELECT status FROM model_runs WHERE model_type='gamlss' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "failed"


# ---------------------------------------------------------------------------
# Integration test — requires Rscript + gamlss package
# ---------------------------------------------------------------------------


@pytest.mark.r_required
def test_run_gamlss_model_integration(tmp_path):
    """Full end-to-end: Python → CSV → Rscript → CSV → DB.

    Skip with: pytest -m 'not r_required'
    Install R from https://cran.r-project.org/
    Then in R: install.packages('gamlss')
    """
    try:
        from coffee_forecast.models.gamlss import _find_rscript
        _find_rscript()
    except FileNotFoundError:
        pytest.skip("Rscript not installed")

    from coffee_forecast.db.migrations import ensure_schema

    conn = sqlite3.connect(str(tmp_path / "test.db"))
    ensure_schema(conn)

    _insert_kc_prices(conn, n_months=72)

    # Also insert RM=F prices (needed for regime join completeness)
    rng = np.random.default_rng(99)
    dates = pd.date_range("2016-01-01", periods=72, freq="MS")
    prices = 80.0 * np.exp(np.cumsum(rng.normal(0, 0.05, 72)))
    conn.executemany(
        "INSERT OR IGNORE INTO prices_monthly (date, symbol, adj_close) VALUES (?,?,?)",
        [(d.strftime("%Y-%m-%d"), "RM=F", float(p)) for d, p in zip(dates, prices)],
    )
    conn.commit()

    run_id = _insert_vecm_run(conn)
    _insert_residuals(conn, run_id, symbols=["KC=F", "RM=F"], n=48, seed=7)

    gamlss_run_id = run_gamlss_model(conn, run_id)
    assert gamlss_run_id > 0

    rows = conn.execute(
        "SELECT symbol, regime, mu, q50 FROM gamlss_params WHERE run_id = ?",
        (gamlss_run_id,),
    ).fetchall()
    # At minimum we expect rows for KC=F and RM=F across the three regimes
    assert len(rows) >= 2

    status = conn.execute(
        "SELECT status FROM model_runs WHERE id = ?", (gamlss_run_id,)
    ).fetchone()[0]
    assert status == "success"


# ---------------------------------------------------------------------------
# compute_regime_labels — max_date tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_conn() -> sqlite3.Connection:
    return _make_conn()


def test_compute_regime_labels_max_date_excludes_later_rows(mem_conn: sqlite3.Connection) -> None:
    """compute_regime_labels with max_date must not include prices after that date."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(7)
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.04, 60)))
    mem_conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    mem_conn.commit()

    series = compute_regime_labels(mem_conn, max_date="2020-12-01")

    assert not series.empty
    assert series.index.max() <= pd.Timestamp("2020-12-01")


def test_compute_regime_labels_max_date_none_uses_all_data(mem_conn: sqlite3.Connection) -> None:
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(8)
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.04, 60)))
    mem_conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    mem_conn.commit()

    series_all = compute_regime_labels(mem_conn, max_date=None)
    series_cut = compute_regime_labels(mem_conn, max_date="2020-12-01")

    assert len(series_all) > len(series_cut)
