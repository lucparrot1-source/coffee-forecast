"""Tests for FREDProvider, AlphaVantageProvider, CompositeProvider, and make_default_provider."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coffee_forecast.data.providers import (
    _COLUMNS,
    AlphaVantageProvider,
    CompositeProvider,
    FREDProvider,
    PriceProvider,
    make_default_provider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fred_raw(code: str, dates: list[str], values: list[float]) -> pd.DataFrame:
    """Minimal DataFrame matching what pandas-datareader returns from FRED."""
    idx = pd.to_datetime(dates)
    idx.name = "DATE"
    return pd.DataFrame({code: values}, index=idx)


def _av_response(dates: list[str], opens: list[float], closes: list[float]) -> dict[str, object]:
    """Minimal Alpha Vantage FX_MONTHLY JSON response."""
    ts: dict[str, object] = {}
    for d, o, c in zip(dates, opens, closes):
        ts[d] = {
            "1. open": str(o),
            "2. high": str(o + 0.01),
            "3. low": str(c - 0.01),
            "4. close": str(c),
        }
    return {"Time Series FX (Monthly)": ts}


# ---------------------------------------------------------------------------
# PriceProvider ABC
# ---------------------------------------------------------------------------


def test_price_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        PriceProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FREDProvider
# ---------------------------------------------------------------------------


class TestFREDProvider:
    def test_fetch_returns_expected_columns(self) -> None:
        raw = _fred_raw("PCOFFOTMUSDM", ["2020-01-01", "2020-02-01"], [150.0, 155.0])
        with patch("coffee_forecast.data.providers.web.DataReader", return_value=raw):
            df = FREDProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 2, 29))
        assert list(df.columns) == _COLUMNS

    def test_fetch_correct_symbol_stored(self) -> None:
        raw = _fred_raw("PCOFFOTMUSDM", ["2020-01-01"], [150.0])
        with patch("coffee_forecast.data.providers.web.DataReader", return_value=raw):
            df = FREDProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 31))
        assert (df["symbol"] == "KC=F").all()

    def test_fetch_adj_close_matches_raw(self) -> None:
        raw = _fred_raw("PCOFFROBUSDM", ["2020-01-01", "2020-02-01"], [80.0, 82.0])
        with patch("coffee_forecast.data.providers.web.DataReader", return_value=raw):
            df = FREDProvider().fetch(["RM=F"], date(2020, 1, 1), date(2020, 2, 29))
        assert list(df["adj_close"]) == [80.0, 82.0]

    def test_fetch_date_is_string(self) -> None:
        raw = _fred_raw("PCOFFOTMUSDM", ["2020-03-01"], [160.0])
        with patch("coffee_forecast.data.providers.web.DataReader", return_value=raw):
            df = FREDProvider().fetch(["KC=F"], date(2020, 3, 1), date(2020, 3, 31))
        assert df["date"].iloc[0] == "2020-03-01"

    def test_fetch_unknown_symbol_skipped(self) -> None:
        with patch("coffee_forecast.data.providers.web.DataReader") as mock_dr:
            df = FREDProvider().fetch(["UNKNOWN=F"], date(2020, 1, 1), date(2020, 1, 31))
        mock_dr.assert_not_called()
        assert df.empty
        assert list(df.columns) == _COLUMNS

    def test_fetch_empty_response_returns_empty(self) -> None:
        raw = _fred_raw("PCOFFOTMUSDM", [], [])
        with patch("coffee_forecast.data.providers.web.DataReader", return_value=raw):
            df = FREDProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 31))
        assert df.empty

    def test_fetch_fred_exception_skips_symbol(self) -> None:
        with patch(
            "coffee_forecast.data.providers.web.DataReader",
            side_effect=ConnectionError("FRED down"),
        ):
            df = FREDProvider().fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 31))
        assert df.empty

    def test_fetch_multiple_symbols(self) -> None:
        def fake_datareader(code: str, *args: object, **kwargs: object) -> pd.DataFrame:
            if code == "PCOFFOTMUSDM":
                return _fred_raw(code, ["2020-01-01"], [150.0])
            return _fred_raw(code, ["2020-01-01"], [80.0])

        with patch("coffee_forecast.data.providers.web.DataReader", side_effect=fake_datareader):
            df = FREDProvider().fetch(["KC=F", "RM=F"], date(2020, 1, 1), date(2020, 1, 31))
        assert set(df["symbol"]) == {"KC=F", "RM=F"}
        assert len(df) == 2


# ---------------------------------------------------------------------------
# AlphaVantageProvider
# ---------------------------------------------------------------------------


class TestAlphaVantageProvider:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY"):
            AlphaVantageProvider()

    def test_fetch_returns_expected_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        payload = _av_response(["2020-01-31", "2020-02-29"], [5.2, 5.3], [5.25, 5.35])
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        with patch("coffee_forecast.data.providers.requests.get", return_value=mock_resp):
            df = AlphaVantageProvider().fetch(["BRL=X"], date(2020, 1, 1), date(2020, 2, 29))
        assert list(df.columns) == _COLUMNS

    def test_fetch_correct_symbol_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        payload = _av_response(["2020-01-31"], [5.2], [5.25])
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        with patch("coffee_forecast.data.providers.requests.get", return_value=mock_resp):
            df = AlphaVantageProvider().fetch(["BRL=X"], date(2020, 1, 1), date(2020, 1, 31))
        assert (df["symbol"] == "BRL=X").all()

    def test_fetch_filters_by_date_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        # AV returns full history; provider should filter to [start, end]
        payload = _av_response(
            ["2019-12-31", "2020-01-31", "2020-02-29"],
            [5.0, 5.2, 5.3],
            [5.1, 5.25, 5.35],
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        with patch("coffee_forecast.data.providers.requests.get", return_value=mock_resp):
            df = AlphaVantageProvider().fetch(["BRL=X"], date(2020, 1, 1), date(2020, 2, 29))
        # 2019-12-31 is before start — should be excluded
        assert all(df["date"] >= "2020-01-01")

    def test_fetch_unknown_symbol_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        with patch("coffee_forecast.data.providers.requests.get") as mock_get:
            df = AlphaVantageProvider().fetch(["UNKNOWN=X"], date(2020, 1, 1), date(2020, 1, 31))
        mock_get.assert_not_called()
        assert df.empty

    def test_fetch_bad_response_skips_symbol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Note": "API rate limit reached"}
        mock_resp.raise_for_status.return_value = None
        with patch("coffee_forecast.data.providers.requests.get", return_value=mock_resp):
            df = AlphaVantageProvider().fetch(["BRL=X"], date(2020, 1, 1), date(2020, 1, 31))
        assert df.empty


# ---------------------------------------------------------------------------
# CompositeProvider
# ---------------------------------------------------------------------------


class TestCompositeProvider:
    def _mock_provider(self, symbol: str, value: float) -> MagicMock:
        """Return a mock PriceProvider that returns one row for the given symbol."""
        mock = MagicMock(spec=PriceProvider)
        mock.fetch.return_value = pd.DataFrame(
            [
                {
                    "date": "2020-01-01",
                    "symbol": symbol,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": value,
                    "volume": None,
                    "adj_close": value,
                }
            ]
        )
        return mock

    def test_routes_to_correct_provider(self) -> None:
        fred_mock = self._mock_provider("KC=F", 150.0)
        av_mock = self._mock_provider("BRL=X", 5.2)
        composite = CompositeProvider({"KC=F": fred_mock, "BRL=X": av_mock})
        df = composite.fetch(["KC=F", "BRL=X"], date(2020, 1, 1), date(2020, 1, 31))
        fred_mock.fetch.assert_called_once_with(["KC=F"], date(2020, 1, 1), date(2020, 1, 31))
        av_mock.fetch.assert_called_once_with(["BRL=X"], date(2020, 1, 1), date(2020, 1, 31))
        assert set(df["symbol"]) == {"KC=F", "BRL=X"}

    def test_unrouted_symbol_skipped(self) -> None:
        fred_mock = self._mock_provider("KC=F", 150.0)
        composite = CompositeProvider({"KC=F": fred_mock})
        df = composite.fetch(["KC=F", "UNKNOWN=X"], date(2020, 1, 1), date(2020, 1, 31))
        assert set(df["symbol"]) == {"KC=F"}

    def test_empty_routing_returns_empty(self) -> None:
        composite = CompositeProvider({})
        df = composite.fetch(["KC=F"], date(2020, 1, 1), date(2020, 1, 31))
        assert df.empty


# ---------------------------------------------------------------------------
# make_default_provider
# ---------------------------------------------------------------------------


class TestMakeDefaultProvider:
    def test_returns_fred_only_when_no_av_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        provider = make_default_provider()
        # Without AV key, falls back to a plain FREDProvider (not CompositeProvider)
        assert isinstance(provider, FREDProvider)

    def test_returns_composite_when_av_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
        provider = make_default_provider()
        assert isinstance(provider, CompositeProvider)
