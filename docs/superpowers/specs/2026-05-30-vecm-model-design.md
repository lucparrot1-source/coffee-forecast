# Step 5 — VECM Model: Design Spec

**Date:** 2026-05-30  
**Status:** Approved for implementation

---

## What we're building

A Vector Error Correction Model (VECM) that forecasts Arabica (KC=F) and Robusta (RM=F) coffee prices at 1, 2, and 3 months ahead. The model captures the long-run tendency of the two prices to move together (cointegration), influenced by four currency drivers as exogenous inputs.

---

## Locked-in decisions carried forward

| Decision | Value |
|---|---|
| Endogenous variables | KC=F, RM=F (log prices) |
| Exogenous variables | BRL=X, VND=X, IDR=X, DX-Y.NYB (log prices) |
| Cointegration rank | r=1 (hardcoded — confirmed by Engle-Granger p=0.009 in EDA) |
| Lag selection | Automatic: test lags 1–12 on the VAR in levels, pick min AIC; cap at 12 |
| Forecast horizons | 1, 2, 3 months ahead |
| Training window | Common date range across all 6 symbols (inner join) |
| Price transform | Natural log before fitting; exp() to back-transform forecasts |
| Residual storage | Written to new `vecm_residuals` table (needed by GAMLSS in Step 6) |

---

## Architecture

Single file: `src/coffee_forecast/models/vecm.py`  
Mirrors the shape of `spread.py` — plain functions, a `run_vecm_model()` orchestrator, `main()` with argparse and Resend alert wrapper.

### Functions

| Function | Purpose |
|---|---|
| `load_aligned_data(conn)` | Query `prices_monthly`, pivot to wide, inner-join on common dates, log-transform. Returns `(endog_df, exog_df)`. |
| `select_lag_order(endog, exog, maxlags=12)` | Fit VAR in levels on endog+exog, return lag order minimising AIC. |
| `fit_vecm(endog, exog, lag_order)` | Fit `statsmodels.tsa.vector_ar.vecm.VECM` with `k_ar_diff=lag_order-1`, `coint_rank=1`, `exog=exog`. Returns fitted model. |
| `extract_residuals(result, endog)` | Pull in-sample residuals (one per endogenous variable per date). Returns long-format DataFrame with columns `(date, symbol, residual)`. |
| `generate_forecasts(result, steps=3)` | Call `result.predict(steps=steps, exog_fc=exog_fc)` where `exog_fc` is a zero matrix of shape `(steps, n_exog)` — naïve assumption: no change in exchange rates (Δexog = 0). Returns DataFrame with columns `(horizon, symbol, log_forecast, point_forecast)`. |
| `write_run(conn, params, metrics)` | Insert into `model_runs`; return `run_id`. |
| `write_forecasts(conn, run_id, forecasts_df, forecast_date)` | Insert into `forecasts` table. Only `point_forecast` is populated here; p10/p50/p90 are left NULL for GAMLSS to fill. |
| `write_residuals(conn, run_id, residuals_df)` | Insert into `vecm_residuals`. |
| `run_vecm_model(conn)` | Orchestrator: calls all of the above in order, logs key stats, returns `run_id`. |
| `main()` | `configure_logging()`, argparse `--db`, `get_connection()`, `ensure_schema()`, `run_vecm_model()`. |

---

## Database changes

### New table: `vecm_residuals`

```sql
CREATE TABLE IF NOT EXISTS vecm_residuals (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES model_runs(id),
    date     TEXT    NOT NULL,   -- YYYY-MM-01
    symbol   TEXT    NOT NULL,   -- KC=F or RM=F
    residual REAL    NOT NULL,
    UNIQUE (run_id, date, symbol)
);
```

Added to `schema.sql` alongside existing tables. `ensure_schema()` is idempotent so existing DBs upgrade automatically.

---

## Key modelling notes (plain English)

- **Cointegration rank hardcoded at r=1.** The EDA already proved this; re-testing every run would be wasteful and could produce instability if a short training window gives a noisy result.
- **Exog naïve assumption.** When forecasting 2–3 months ahead, we don't know future exchange rates. statsmodels VECM takes differenced exog at forecast time, so we pass zeros (Δexog = 0), meaning "assume exchange rates don't change". This is the standard naïve assumption; a more sophisticated exog forecast is out of scope for Step 5.
- **Log-scale residuals.** Residuals are on the log-price scale. GAMLSS (Step 6) will model them on that same scale.
- **`k_ar_diff` vs `k_ar`.** `statsmodels` VECM takes `k_ar_diff` = number of lagged difference terms = VAR lag order minus 1. So if AIC selects lag=2, we pass `k_ar_diff=1`.

---

## Params and metrics stored in `model_runs`

```json
{
  "params": {
    "lag_order": 2,
    "coint_rank": 1,
    "exog_symbols": ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"],
    "train_start": "2014-01-01",
    "train_end": "2026-04-01",
    "n_obs": 124
  },
  "metrics": {
    "aic": -1234.5,
    "log_likelihood": 678.9
  }
}
```

---

## Testing

| Test | Type |
|---|---|
| `test_select_lag_order_returns_int_in_range` | Unit — synthetic 2-col AR(1) data, assert result in 1–12 |
| `test_fit_vecm_returns_result_object` | Unit — synthetic cointegrated series, assert `.predict()` works |
| `test_extract_residuals_shape` | Unit — check output columns and row count |
| `test_generate_forecasts_horizons` | Unit — assert exactly 3 rows (h=1,2,3) per symbol |
| `test_generate_forecasts_back_transform` | Unit — verify exp(log_forecast) matches point_forecast |
| `test_write_run_inserts_model_run` | Integration — in-memory DB, assert row in model_runs |
| `test_write_forecasts_inserts_rows` | Integration — in-memory DB, assert 6 rows (2 symbols × 3 horizons) |
| `test_write_residuals_inserts_rows` | Integration — in-memory DB, assert rows in vecm_residuals |
| `test_run_vecm_model_smoke` | Integration — real DB (or in-memory with synthetic data), assert run_id returned and tables populated |

Total: ~9 tests. Synthetic cointegrated series generated with `numpy` (no need to load real DB data for unit tests).

---

## CLI entry point

```
python -m coffee_forecast.models.vecm [--db PATH]
```

Prints logged output (INFO level). On exception: sends Resend alert and re-raises.

---

## Out of scope for Step 5

- Probabilistic forecast intervals (p10/p50/p90) — that is Step 6 (GAMLSS)
- Exog forecasting beyond naïve forward-fill
- Model serialisation / pickle
- Rolling or expanding backtest — that is Step 8
