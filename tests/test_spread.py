import numpy as np
import pandas as pd

from coffee_forecast.models.spread import compute_spread, compute_zscore, fit_ar1, generate_signal


def _wide(kc: list[float], rm: list[float]) -> pd.DataFrame:
    """Helper: build a wide monthly price DataFrame from two lists."""
    dates = pd.date_range("2020-01-01", periods=len(kc), freq="MS")
    return pd.DataFrame({"KC=F": kc, "RM=F": rm}, index=dates)


def test_compute_spread_values():
    kc = [100.0, 200.0, 150.0]
    rm = [50.0, 100.0, 75.0]
    wide = _wide(kc, rm)
    result = compute_spread(wide)
    expected = np.log(np.array(kc)) - np.log(np.array(rm))
    np.testing.assert_allclose(result.values, expected)


def test_compute_spread_index_preserved():
    wide = _wide([100.0, 110.0], [50.0, 55.0])
    result = compute_spread(wide)
    assert list(result.index) == list(wide.index)


def test_fit_ar1_recovers_known_coefficient():
    rng = np.random.default_rng(42)
    n = 500
    s = np.zeros(n)
    rho_true = 0.7
    for t in range(1, n):
        s[t] = 0.05 + rho_true * s[t - 1] + rng.normal(0, 0.1)
    rho_est, _ = fit_ar1(pd.Series(s))
    assert abs(rho_est - rho_true) < 0.05


def test_fit_ar1_half_life_formula():
    # rho=0.5 → half-life = -ln(2)/ln(0.5) = 1.0 period exactly
    rng = np.random.default_rng(0)
    n = 2000
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.5 * s[t - 1] + rng.normal(0, 0.01)
    _, hl = fit_ar1(pd.Series(s))
    assert abs(hl - 1.0) < 0.15


def test_fit_ar1_non_stationary_returns_nan_halflife():
    # Explosive AR(1) with rho ≈ 1.05 → half-life is undefined
    s = pd.Series(np.array([1.05**i for i in range(50)]))
    _, hl = fit_ar1(s)
    assert np.isnan(hl)


def test_zscore_first_value_is_nan():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = compute_zscore(s)
    assert np.isnan(z.iloc[0])


def test_zscore_finite_from_index_1():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = compute_zscore(s)
    assert z.iloc[1:].notna().all()


def test_zscore_index_preserved():
    s = pd.Series([10.0, 20.0, 15.0], index=pd.date_range("2020-01", periods=3, freq="MS"))
    z = compute_zscore(s)
    assert list(z.index) == list(s.index)


def test_signal_entry_and_exit():
    # Entry long, hold, exit, entry short, hold, exit
    z = pd.Series([-3.0, -3.0, 0.3, 0.3, 3.0, 3.0, 0.3])
    sig = generate_signal(z)
    assert list(sig) == [1, 1, 0, 0, -1, -1, 0]


def test_signal_hold_in_dead_zone():
    # z in (0.5, 2.0) → hold previous signal
    z = pd.Series([3.0, 1.5, 1.5, 0.3])
    sig = generate_signal(z)
    assert sig.iloc[0] == -1   # entry short
    assert sig.iloc[1] == -1   # hold (1.5 is in (0.5, 2.0))
    assert sig.iloc[2] == -1   # hold
    assert sig.iloc[3] == 0    # exit


def test_signal_starts_flat():
    # No extreme z yet → stay flat
    z = pd.Series([1.0, 1.0, 1.0])
    sig = generate_signal(z)
    assert list(sig) == [0, 0, 0]


def test_signal_nan_preserves_state():
    # NaN at start doesn't trigger entry; position should stay 0
    z = pd.Series([float("nan"), float("nan"), 3.0, float("nan"), 0.3])
    sig = generate_signal(z)
    assert sig.iloc[0] == 0    # NaN → flat
    assert sig.iloc[1] == 0    # NaN → still flat
    assert sig.iloc[2] == -1   # entry short
    assert sig.iloc[3] == -1   # NaN → hold
    assert sig.iloc[4] == 0    # exit
