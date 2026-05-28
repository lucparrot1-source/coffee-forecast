import logging
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

TICKERS: list[str] = ["KC=F", "RM=F", "BRL=X", "VND=X", "DX-Y.NYB"]
_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


class PriceProvider(ABC):
    @abstractmethod
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return DataFrame with columns: date, symbol, open, high, low, close, volume, adj_close."""


class YahooProvider(PriceProvider):
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        raw = self._download(symbols, start, end)
        if raw.empty:
            return pd.DataFrame(columns=_COLUMNS)
        # yfinance returns flat columns for a single ticker; normalise to MultiIndex
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, symbols[0]) for col in raw.columns]
            )
        rows: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                sym_df = raw.xs(sym, level=1, axis=1).copy().reset_index()
            except KeyError:
                log.warning("No data returned for %s", sym)
                continue
            sym_df = sym_df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            sym_df["symbol"] = sym
            sym_df["date"] = pd.to_datetime(sym_df["date"]).dt.strftime("%Y-%m-%d")
            rows.append(sym_df[_COLUMNS])
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        return yf.download(  # type: ignore[no-any-return]
            symbols, start=start, end=end, auto_adjust=False, progress=False
        )
