# Step 4 — Spread Model Design

**Date:** 2026-05-29  
**Status:** Approved  

---

## Goal

Build a mean-reversion signal on the Arabica–Robusta log price spread.  
Because the two coffees are cointegrated (Engle-Granger p = 0.009, confirmed in Step 3 EDA),  
the gap between their log prices tends to revert to a long-run equilibrium.  
This module quantifies how stretched that gap is and generates a +1/−1/0 trading signal.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Spread definition | `log(KC=F) − log(RM=F)` (1:1) | Matches locked-in spec; natural parity for two near-substitutes; no overfitting risk from OLS β |
| Z-score window | Expanding (all history to each point) | Stable over time; avoids look-ahead bias in backtesting |
| OU/AR(1) fitting | OLS regression of `s[t]` on `s[t-1]` | Simple, interpretable; half-life directly derivable |
| Half-life estimation | Expanding window, re-estimated each month | Consistent with expanding z-score |
| Signal exit | Flat at \|z\| < 0.5 | Captures full mean-reversion move; standard OU strategy rule |
| Signal entry | \|z\| > 2 | Locked-in spec threshold |
| Persistence | New `spread_signals` SQLite table | Fast dashboard queries; auditable history |
| Module location | `src/coffee_forecast/models/spread.py` | Starts the `models/` package that will also hold VECM (Step 5) and combiner (Step 7) |

---

## Architecture

### New package

```
src/coffee_forecast/models/
    __init__.py
    spread.py       ← core logic
```

The `models/` package will grow in Steps 5–7. `spread.py` is self-contained within it.

### Functions in `spread.py`

All pure functions — no classes, no global state.

| Function | Signature | Returns |
|---|---|---|
| `compute_spread` | `(wide: pd.DataFrame) → pd.Series` | Monthly `log(KC=F) − log(RM=F)`, index = date |
| `fit_ar1` | `(s: pd.Series) → tuple[float, float]` | `(ar1_coef, half_life_months)` |
| `compute_zscore` | `(s: pd.Series) → pd.Series` | Expanding z-score series |
| `generate_signal` | `(z: pd.Series, entry=2.0, exit_thresh=0.5) → pd.Series` | Integer series: +1, −1, 0 |
| `build_spread_df` | `(wide: pd.DataFrame) → pd.DataFrame` | Full table: date, spread, z_score, signal, half_life |
| `run_spread_model` | `(db_path: Path) → None` | Reads DB, computes, upserts to `spread_signals` |

### Signal logic (stateful, forward only)

```
for each month t:
    if z[t] > +entry:   signal[t] = -1   # spread too wide → short
    elif z[t] < -entry: signal[t] = +1   # spread too narrow → long
    elif |z[t]| < exit: signal[t] =  0   # near equilibrium → flat
    else:               signal[t] = signal[t-1]  # hold
```

Initial state before first signal: 0 (flat).

### AR(1) / OU relationship

Fit: `s[t] = α + ρ·s[t-1] + ε`  
Half-life: `h = −ln(2) / ln(|ρ|)` months  
Estimated at each date using all data up to and including that month (expanding).

---

## Schema change

New table added to `schema.sql` and `migrations.py`:

```sql
CREATE TABLE IF NOT EXISTS spread_signals (
    id         INTEGER PRIMARY KEY,
    date       TEXT    NOT NULL UNIQUE,  -- YYYY-MM-01
    spread     REAL,                     -- log(KC=F) - log(RM=F)
    z_score    REAL,                     -- expanding z-score
    signal     INTEGER,                  -- +1, -1, or 0
    half_life  REAL                      -- months, AR(1) expanding estimate
);
```

Upsert on `date` (INSERT OR REPLACE). Migration is additive — no existing tables affected.

---

## CLI

```
python -m coffee_forecast.models.spread [--db PATH]
```

- Reads `prices_monthly` from SQLite (both `KC=F` and `RM=F` must have data).
- Runs the full spread model.
- Upserts results to `spread_signals`.
- Logs half-life and current z-score on exit.
- Follows global alert pattern (Resend email on crash).

---

## Testing

`tests/test_spread.py` — no DB dependency, all synthetic data.

| Test | What it checks |
|---|---|
| `test_compute_spread` | Output equals log(KC=F) − log(RM=F) elementwise |
| `test_fit_ar1_known_coef` | Synthetic AR(1) series → recovered ρ within tolerance, half-life formula correct |
| `test_zscore_zero_mean` | Expanding z-score of a stationary series is zero-mean at the end |
| `test_signal_entry_exit` | z > 2 → −1; z < −2 → +1; \|z\| < 0.5 → 0; in-between → hold |
| `test_build_spread_df_columns` | Output DataFrame has all required columns |

---

## Out of scope for this step

- Portfolio P&L / backtest (Step 8)
- Dashboard tab (Step 9)
- GitHub Actions integration (Step 11)
- OLS β hedge ratio (deliberately excluded — 1:1 chosen)
