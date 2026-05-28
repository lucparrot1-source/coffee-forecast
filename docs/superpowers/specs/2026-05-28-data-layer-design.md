# Data Layer Design — Step 2

**Date:** 2026-05-28
**Status:** Approved

---

## Summary

Build the plumbing between Yahoo Finance and SQLite. Defines a provider interface, a Yahoo implementation, an incremental daily ingestion CLI, and a monthly rollup CLI. The database schema already exists from Step 1.

---

## File Structure

```
src/coffee_forecast/
└── data/
    ├── __init__.py
    ├── providers.py   # PriceProvider ABC + YahooProvider
    ├── ingest.py      # CLI: fetch daily prices → prices table
    └── resample.py    # CLI: roll daily → prices_monthly table
```

---

## providers.py

### `PriceProvider` (ABC)

Single abstract method:

```python
def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame
```

Returns DataFrame with columns: `date, symbol, open, high, low, close, volume, adj_close`.

### `YahooProvider`

- Wraps `yfinance.download()` with a `tenacity` retry decorator (3 attempts, exponential backoff 2–10s).
- Handles multi-symbol download and reshapes the wide `yfinance` output into the standard flat format.
- Tickers: `KC=F` (Arabica), `RM=F` (Robusta), `BRL=X`, `VND=X`, `DX-Y.NYB` (DXY index).

---

## ingest.py

**Invocation:** `python -m coffee_forecast.data.ingest [--start YYYY-MM-DD] [--db PATH]`

**Behaviour:**
1. Opens SQLite DB and runs migrations if needed.
2. For each ticker, reads the latest stored date. Defaults to `2000-01-01` if no data exists.
3. `--start` flag overrides the auto-detected start date (for forced historical backfill).
4. Fetches `[latest_date + 1 day, today]` via `YahooProvider`.
5. Upserts rows using `INSERT OR IGNORE` (preserves existing data, no duplicates).
6. Logs rows inserted per ticker.
7. Wrapped in global Resend alert pattern on crash.

---

## resample.py

**Invocation:** `python -m coffee_forecast.data.resample [--db PATH]`

**Behaviour:**
1. Reads all rows from `prices` for months prior to the current month (no partial months).
2. Groups by `(year-month, symbol)`, computes mean of `adj_close` across all trading days.
3. Upserts into `prices_monthly` with date stored as `YYYY-MM-01` (INSERT OR REPLACE).
4. Logs months updated per ticker.
5. Wrapped in global Resend alert pattern on crash.

Safe to re-run: replaces with the same value unless new daily data arrived.

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Historical start date | 2000-01-01 | Maximum statistical power for cointegration tests |
| Monthly price metric | Mean of daily adj_close | Smooths noise; adjusted close corrects for futures rollovers |
| Re-run behaviour | Incremental (latest date → today) | Safe to schedule daily with no manual intervention |
| CLI structure | Two separate commands | Mirrors existing `db/` pattern; easier to debug/rerun independently |
| Provider structure | ABC + concrete class in `providers.py` | Swappable without touching callers |

---

## Testing

- Unit test for `YahooProvider.fetch()` using mocked `yfinance` response.
- Integration test for `ingest.py` against a temporary SQLite DB.
- Unit test for `resample.py` averaging logic with known fixture data.
