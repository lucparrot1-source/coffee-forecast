from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from coffee_forecast.data.providers import PriceProvider, YahooProvider, _COLUMNS


def _make_multi_index_df(ticker: str) -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    idx.name = "Date"
    cols = pd.MultiIndex.from_tuples([
        ("Open", ticker), ("High", ticker), ("Low", ticker),
        ("Close", ticker), ("Adj Close", ticker), ("Volume", ticker),
    ])
    return pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.5, 1000.0],
         [101.0, 102.0, 100.0, 101.5, 101.5, 1100.0]],
        index=idx, columns=cols,
    )


def test_price_provider_is_abstract():
    with pytest.raises(TypeError):
        PriceProvider()  # type: ignore[abstract]


def test_fetch_returns_expected_columns():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert list(df.columns) == _COLUMNS


def test_fetch_returns_correct_symbol():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert (df["symbol"] == "KC=F").all()


def test_fetch_correct_row_count():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert len(df) == 2


def test_fetch_dates_are_strings():
    mock_raw = _make_multi_index_df("KC=F")
    with patch("coffee_forecast.data.providers.yf.download", return_value=mock_raw):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert df["date"].iloc[0] == "2020-01-02"


def test_fetch_empty_returns_empty_dataframe():
    with patch("coffee_forecast.data.providers.yf.download", return_value=pd.DataFrame()):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 3))
    assert df.empty
    assert list(df.columns) == _COLUMNS


def test_fetch_single_ticker_flat_columns():
    """yfinance returns flat columns for a single ticker — YahooProvider must normalise."""
    idx = pd.to_datetime(["2020-01-02"])
    idx.name = "Date"
    flat_df = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.5, 1000.0]],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    )
    with patch("coffee_forecast.data.providers.yf.download", return_value=flat_df):
        df = YahooProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 2))
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "KC=F"
