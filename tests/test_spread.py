import numpy as np
import pandas as pd

from coffee_forecast.models.spread import compute_spread, fit_ar1


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
