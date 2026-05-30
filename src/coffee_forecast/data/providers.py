import logging
import os
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import pandas_datareader.data as web
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

# Internal symbol names used throughout the codebase.
# KC=F / RM=F  — coffee prices (FRED, physical ICO prices, US¢/lb)
# BRL=X        — Brazilian Real / USD (Alpha Vantage)
# VND=X        — Vietnamese Dong / USD (Alpha Vantage)
# IDR=X        — Indonesian Rupiah / USD (Alpha Vantage)
# DX-Y.NYB     — Broad USD index (FRED, DTWEXBGS)
TICKERS: list[str] = ["KC=F", "RM=F", "BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]

_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]

# ---------------------------------------------------------------------------
# FRED series codes (coffee prices + DXY)
# ---------------------------------------------------------------------------
_FRED_CODE: dict[str, str] = {
    "KC=F": "PCOFFOTMUSDM",  # Arabica (Other Mild Arabicas), US¢/lb, monthly
    "RM=F": "PCOFFROBUSDM",  # Robusta, US¢/lb, monthly
    "DX-Y.NYB": "DTWEXBGS",  # Broad Trade-Weighted USD Index, daily
}

# ---------------------------------------------------------------------------
# Alpha Vantage FX pairs (from_symbol, to_symbol) → rate = to per 1 from
# All quoted as foreign currency units per 1 USD
# ---------------------------------------------------------------------------
_AV_FX: dict[str, tuple[str, str]] = {
    "BRL=X": ("USD", "BRL"),  # Brazilian Reals per USD
    "VND=X": ("USD", "VND"),  # Vietnamese Dong per USD
    "IDR=X": ("USD", "IDR"),  # Indonesian Rupiah per USD
}


class PriceProvider(ABC):
    @abstractmethod
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return DataFrame with columns:
        date, symbol, open, high, low, close, volume, adj_close."""


# ---------------------------------------------------------------------------
# FRED provider — coffee prices and DXY
# ---------------------------------------------------------------------------


class FREDProvider(PriceProvider):
    """Fetches prices from the St. Louis Fed FRED API via pandas-datareader.

    No API key required. Coffee series are monthly physical ICO prices (US¢/lb).
    DXY uses the broad trade-weighted index (DTWEXBGS, starts 2006).
    """

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for sym in symbols:
            code = _FRED_CODE.get(sym)
            if code is None:
                log.warning("No FRED mapping for %s — skipping", sym)
                continue
            try:
                raw = self._get(code, start, end)
            except Exception:
                log.exception("Failed to fetch %s (%s) from FRED", sym, code)
                continue
            if raw.empty:
                log.warning("%s (%s): no data returned", sym, code)
                continue
            rows.append(self._normalise(raw, sym, code))
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))  # type: ignore[misc]
    def _get(self, code: str, start: date, end: date) -> pd.DataFrame:
        return web.DataReader(code, "fred", start=start.isoformat(), end=end.isoformat())  # type: ignore[no-any-return]

    def _normalise(self, raw: pd.DataFrame, sym: str, code: str) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        out["adj_close"] = raw[code]
        out["close"] = raw[code]
        out["open"] = None
        out["high"] = None
        out["low"] = None
        out["volume"] = None
        out["symbol"] = sym
        out = out.reset_index().rename(columns={"DATE": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        return out[_COLUMNS]


# ---------------------------------------------------------------------------
# Alpha Vantage provider — FX pairs (BRL, VND, IDR)
# ---------------------------------------------------------------------------


class AlphaVantageProvider(PriceProvider):
    """Fetches monthly FX rates from Alpha Vantage (FX_MONTHLY endpoint).

    Requires ALPHA_VANTAGE_API_KEY in the environment.
    Free tier: 25 API calls/day (1 call per FX pair).
    Dates are end-of-month; resampling in resample.py handles this correctly.
    """

    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self._api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set — cannot use AlphaVantageProvider")

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for sym in symbols:
            fx = _AV_FX.get(sym)
            if fx is None:
                log.warning("No Alpha Vantage mapping for %s — skipping", sym)
                continue
            try:
                raw = self._get_monthly(fx[0], fx[1])
            except Exception:
                log.exception("Failed to fetch %s (%s/%s) from Alpha Vantage", sym, *fx)
                continue
            if raw.empty:
                log.warning("%s: no data from Alpha Vantage", sym)
                continue
            rows.append(self._normalise(raw, sym, start, end))
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))  # type: ignore[misc]
    def _get_monthly(self, from_sym: str, to_sym: str) -> pd.DataFrame:
        resp = requests.get(
            self._BASE_URL,
            params={
                "function": "FX_MONTHLY",
                "from_symbol": from_sym,
                "to_symbol": to_sym,
                "outputsize": "full",
                "apikey": self._api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "Time Series FX (Monthly)" not in data:
            raise ValueError(f"Unexpected Alpha Vantage response keys: {list(data.keys())}")
        ts = data["Time Series FX (Monthly)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = pd.Index([c.split(". ")[1] for c in df.columns])  # "1. open" → "open"
        return df.astype(float).sort_index()

    def _normalise(self, raw: pd.DataFrame, sym: str, start: date, end: date) -> pd.DataFrame:
        mask = (raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))
        raw = raw[mask].copy()
        out = pd.DataFrame(index=raw.index)
        out["adj_close"] = raw["close"]
        out["close"] = raw["close"]
        out["open"] = raw.get("open")
        out["high"] = raw.get("high")
        out["low"] = raw.get("low")
        out["volume"] = None
        out["symbol"] = sym
        out = out.reset_index().rename(columns={"index": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        return out[_COLUMNS]


# ---------------------------------------------------------------------------
# Composite provider — routes each symbol to the right backend
# ---------------------------------------------------------------------------


class CompositeProvider(PriceProvider):
    """Routes each symbol to the appropriate underlying provider.

    routing: dict mapping symbol → PriceProvider instance.
    """

    def __init__(self, routing: dict[str, PriceProvider]) -> None:
        self._routing = routing

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        by_provider: dict[int, tuple[PriceProvider, list[str]]] = {}
        for sym in symbols:
            provider = self._routing.get(sym)
            if provider is None:
                log.warning("No provider routed for %s — skipping", sym)
                continue
            pid = id(provider)
            if pid not in by_provider:
                by_provider[pid] = (provider, [])
            by_provider[pid][1].append(sym)

        rows: list[pd.DataFrame] = []
        for provider, syms in by_provider.values():
            df = provider.fetch(syms, start, end)
            if not df.empty:
                rows.append(df)

        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(rows, ignore_index=True)


def make_default_provider() -> PriceProvider:
    """Build the default provider used by the ingest pipeline.

    Routes coffee prices and DXY to FRED (no key required).
    Routes BRL, VND, IDR to Alpha Vantage (requires ALPHA_VANTAGE_API_KEY).
    Falls back gracefully if the AV key is missing — logs a warning and
    skips the AV symbols rather than crashing the pipeline.
    """
    fred = FREDProvider()
    try:
        av = AlphaVantageProvider()
    except ValueError:
        log.warning("ALPHA_VANTAGE_API_KEY not set — BRL=X, VND=X, IDR=X will be skipped")
        return fred

    return CompositeProvider(
        {
            "KC=F": fred,
            "RM=F": fred,
            "DX-Y.NYB": fred,
            "BRL=X": av,
            "VND=X": av,
            "IDR=X": av,
        }
    )
