import sqlite3

import numpy as np
import pandas as pd

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.spread import (
    build_spread_df,
    compute_spread,
    compute_zscore,
    fit_ar1,
    generate_signal,
    run_spread_model,
)


def _wide(kc: list[float], rm: list[float]) -> pd.DataFrame:
    """Helper: build a wide monthly price DataFrame from two lists."""
    dates = pd.date_range("2020-01-01", periods=len(kc), freq="MS")
    return pd.DataFrame({"KC=F": kc, "RM=F": rm}, index=dates)


def test_compute_spread_values() -> None:
    kc = [100.0, 200.0, 150.0]
    rm = [50.0, 100.0, 75.0]
    wide = _wide(kc, rm)
    result = compute_spread(wide)
    expected = np.log(np.array(kc)) - np.log(np.array(rm))
    np.testing.assert_allclose(np.asarray(result), expected)


def test_compute_spread_index_preserved() -> None:
    wide = _wide([100.0, 110.0], [50.0, 55.0])
    result = compute_spread(wide)
    assert list(result.index) == list(wide.index)


def test_fit_ar1_recovers_known_coefficient() -> None:
    rng = np.random.default_rng(42)
    n = 500
    s = np.zeros(n)
    rho_true = 0.7
    for t in range(1, n):
        s[t] = 0.05 + rho_true * s[t - 1] + rng.normal(0, 0.1)
    rho_est, _ = fit_ar1(pd.Series(s))
    assert abs(rho_est - rho_true) < 0.05


def test_fit_ar1_half_life_formula() -> None:
    # rho=0.5 → half-life = -ln(2)/ln(0.5) = 1.0 period exactly
    rng = np.random.default_rng(0)
    n = 2000
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.5 * s[t - 1] + rng.normal(0, 0.01)
    _, hl = fit_ar1(pd.Series(s))
    assert abs(hl - 1.0) < 0.15


def test_fit_ar1_non_stationary_returns_nan_halflife() -> None:
    # Explosive AR(1) with rho ≈ 1.05 → half-life is undefined
    s = pd.Series(np.array([1.05**i for i in range(50)]))
    _, hl = fit_ar1(s)
    assert np.isnan(hl)


def test_zscore_first_value_is_nan() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = compute_zscore(s)
    assert np.isnan(z.iloc[0])


def test_zscore_finite_from_index_1() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = compute_zscore(s)
    assert z.iloc[1:].notna().all()


def test_zscore_index_preserved() -> None:
    s = pd.Series([10.0, 20.0, 15.0], index=pd.date_range("2020-01", periods=3, freq="MS"))
    z = compute_zscore(s)
    assert list(z.index) == list(s.index)


def test_signal_entry_and_exit() -> None:
    # Entry long, hold, exit, entry short, hold, exit
    z = pd.Series([-3.0, -3.0, 0.3, 0.3, 3.0, 3.0, 0.3])
    sig = generate_signal(z)
    assert list(sig) == [1, 1, 0, 0, -1, -1, 0]


def test_signal_hold_in_dead_zone() -> None:
    # z in (0.5, 2.0) → hold previous signal
    z = pd.Series([3.0, 1.5, 1.5, 0.3])
    sig = generate_signal(z)
    assert sig.iloc[0] == -1  # entry short
    assert sig.iloc[1] == -1  # hold (1.5 is in (0.5, 2.0))
    assert sig.iloc[2] == -1  # hold
    assert sig.iloc[3] == 0  # exit


def test_signal_starts_flat() -> None:
    # No extreme z yet → stay flat
    z = pd.Series([1.0, 1.0, 1.0])
    sig = generate_signal(z)
    assert list(sig) == [0, 0, 0]


def test_signal_nan_preserves_state() -> None:
    # NaN at start doesn't trigger entry; position should stay 0
    z = pd.Series([float("nan"), float("nan"), 3.0, float("nan"), 0.3])
    sig = generate_signal(z)
    assert sig.iloc[0] == 0  # NaN → flat
    assert sig.iloc[1] == 0  # NaN → still flat
    assert sig.iloc[2] == -1  # entry short
    assert sig.iloc[3] == -1  # NaN → hold
    assert sig.iloc[4] == 0  # exit


def test_build_spread_df_columns() -> None:
    wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
    result = build_spread_df(wide)
    assert set(result.columns) >= {"date", "spread", "z_score", "signal", "half_life"}


def test_build_spread_df_row_count() -> None:
    wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
    result = build_spread_df(wide)
    assert len(result) == 20


def test_build_spread_df_signal_dtype() -> None:
    wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
    result = build_spread_df(wide)
    assert np.issubdtype(result["signal"].to_numpy().dtype, np.integer)


def test_build_spread_df_date_format() -> None:
    wide = _wide([100.0, 110.0], [50.0, 55.0])
    result = build_spread_df(wide)
    assert result["date"].iloc[0] == "2020-01-01"
    assert result["date"].iloc[1] == "2020-02-01"


def test_build_spread_df_early_halflife_nan() -> None:
    # First two rows don't have enough data for AR(1) (need >= 3 observations)
    wide = _wide([100.0 + i for i in range(10)], [50.0 + i * 0.5 for i in range(10)])
    result = build_spread_df(wide)
    assert np.isnan(result["half_life"].iloc[0])
    assert np.isnan(result["half_life"].iloc[1])
    assert not np.isnan(result["half_life"].iloc[-1])


# ---------------------------------------------------------------------------
# Integration tests — run_spread_model against an in-memory SQLite DB
# ---------------------------------------------------------------------------


def _make_db_with_monthly(n: int = 20) -> sqlite3.Connection:
    """Return an in-memory DB populated with n months of KC=F and RM=F data."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rows = []
    for i, d in enumerate(dates):
        rows.append((d.strftime("%Y-%m-%d"), "KC=F", 100.0 + i))
        rows.append((d.strftime("%Y-%m-%d"), "RM=F", 50.0 + i * 0.5))
    conn.executemany("INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)", rows)
    conn.commit()
    return conn


def test_run_spread_model_writes_correct_row_count() -> None:
    conn = _make_db_with_monthly(20)
    n = run_spread_model(conn)
    assert n == 20
    stored = conn.execute("SELECT COUNT(*) FROM spread_signals").fetchone()[0]
    assert stored == 20


def test_run_spread_model_is_idempotent() -> None:
    conn = _make_db_with_monthly(20)
    run_spread_model(conn)
    run_spread_model(conn)
    stored = conn.execute("SELECT COUNT(*) FROM spread_signals").fetchone()[0]
    assert stored == 20


def test_run_spread_model_values_match_build_spread_df() -> None:
    conn = _make_db_with_monthly(10)
    run_spread_model(conn)
    rows = conn.execute(
        "SELECT spread, z_score, signal FROM spread_signals ORDER BY date"
    ).fetchall()
    wide = _wide([100.0 + i for i in range(10)], [50.0 + i * 0.5 for i in range(10)])
    expected = build_spread_df(wide)
    for i, (db_spread, db_z, db_sig) in enumerate(rows):
        np.testing.assert_allclose(db_spread, expected["spread"].iloc[i], rtol=1e-9)
        assert db_sig == expected["signal"].iloc[i]
