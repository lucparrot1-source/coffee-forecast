import numpy as np
import pandas as pd
import pytest


from coffee_forecast.models.spread import compute_spread


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
