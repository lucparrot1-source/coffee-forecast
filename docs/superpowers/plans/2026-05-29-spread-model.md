# Step 4 — Spread Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mean-reversion signal on the Arabica–Robusta log price spread — fit an AR(1)/OU process, compute expanding z-scores, generate +1/−1/0 trading signals, and persist results to a new `spread_signals` SQLite table.

**Architecture:** Pure functions in `src/coffee_forecast/models/spread.py` compute the spread, z-score, AR(1) half-life, and signal from a wide DataFrame of monthly prices. A `run_spread_model` orchestrator reads `prices_monthly` from SQLite and upserts results to `spread_signals`. A `__main__` block exposes this as a CLI following the project's alerting and logging conventions.

**Tech Stack:** Python 3.11, NumPy (OLS via `np.linalg.lstsq`), pandas (expanding window), SQLite, pytest.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/coffee_forecast/db/schema.sql` | Add `spread_signals` table |
| Create | `src/coffee_forecast/models/__init__.py` | Package marker (empty) |
| Create | `src/coffee_forecast/models/spread.py` | All spread model logic + CLI |
| Create | `tests/test_spread.py` | All spread model tests |

---

## Task 1: Add `spread_signals` to the schema

**Files:**
- Modify: `src/coffee_forecast/db/schema.sql`

- [ ] **Step 1: Add the table definition**

  Append the following block to `src/coffee_forecast/db/schema.sql`, after the `accuracy_log` table and before the index definitions:

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

  Also append this index after the existing index lines:

  ```sql
  CREATE INDEX IF NOT EXISTS idx_spread_signals_date ON spread_signals (date);
  ```

  No change to `migrations.py` is needed — it already calls `executescript` on the full `schema.sql`, and `CREATE TABLE IF NOT EXISTS` is idempotent.

- [ ] **Step 2: Verify the schema applies cleanly**

  Run:
  ```
  python -c "
  import sqlite3, pathlib
  conn = sqlite3.connect(':memory:')
  conn.executescript(pathlib.Path('src/coffee_forecast/db/schema.sql').read_text())
  tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
  print([t[0] for t in tables])
  "
  ```

  Expected output includes `spread_signals` in the list.

- [ ] **Step 3: Commit**

  ```bash
  git add src/coffee_forecast/db/schema.sql
  git commit -m "feat: add spread_signals table to schema"
  ```

---

## Task 2: Create the `models` package and `spread.py` skeleton

**Files:**
- Create: `src/coffee_forecast/models/__init__.py`
- Create: `src/coffee_forecast/models/spread.py`
- Create: `tests/test_spread.py`

- [ ] **Step 1: Create the package marker**

  Create `src/coffee_forecast/models/__init__.py` with empty content (just a newline).

- [ ] **Step 2: Create the spread module skeleton**

  Create `src/coffee_forecast/models/spread.py` with this content (functions will be filled in across Tasks 3–7):

  ```python
  import argparse
  import logging
  import os
  import sqlite3
  import traceback
  from pathlib import Path

  import numpy as np
  import pandas as pd

  from coffee_forecast.alerts import send_pipeline_alert
  from coffee_forecast.db import get_connection
  from coffee_forecast.db.migrations import ensure_schema
  from coffee_forecast.logging_config import configure_logging

  log = logging.getLogger(__name__)
  ```

- [ ] **Step 3: Create the test file skeleton**

  Create `tests/test_spread.py` with this content:

  ```python
  import numpy as np
  import pandas as pd
  import pytest


  def _wide(kc: list[float], rm: list[float]) -> pd.DataFrame:
      """Helper: build a wide monthly price DataFrame from two lists."""
      dates = pd.date_range("2020-01-01", periods=len(kc), freq="MS")
      return pd.DataFrame({"KC=F": kc, "RM=F": rm}, index=dates)
  ```

- [ ] **Step 4: Verify the package imports cleanly**

  Run:
  ```
  python -c "import coffee_forecast.models.spread"
  ```

  Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

  ```bash
  git add src/coffee_forecast/models/__init__.py src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: add models package skeleton and spread.py stub"
  ```

---

## Task 3: `compute_spread`

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`
- Modify: `tests/test_spread.py`

- [ ] **Step 1: Write the failing test**

  Append to `tests/test_spread.py`:

  ```python
  from coffee_forecast.models.spread import compute_spread


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
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_compute_spread_values -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'compute_spread'`

- [ ] **Step 3: Implement `compute_spread`**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def compute_spread(wide: pd.DataFrame) -> pd.Series:
      """Return log(KC=F) - log(RM=F) as a monthly Series."""
      return np.log(wide["KC=F"]) - np.log(wide["RM=F"])
  ```

- [ ] **Step 4: Run to verify pass**

  ```
  pytest tests/test_spread.py::test_compute_spread_values tests/test_spread.py::test_compute_spread_index_preserved -v
  ```

  Expected: both `PASSED`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: implement compute_spread"
  ```

---

## Task 4: `fit_ar1`

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`
- Modify: `tests/test_spread.py`

`fit_ar1` fits `s[t] = α + ρ·s[t-1] + ε` via OLS and derives the half-life as `−ln(2) / ln(|ρ|)`. Returns `(rho, half_life)`. If `|ρ| == 0` or `|ρ| >= 1` (non-stationary or explosive), `half_life` is `nan`.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_spread.py`:

  ```python
  from coffee_forecast.models.spread import fit_ar1


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
      # rho >= 1.0 → half-life is undefined
      s = pd.Series(np.cumsum(np.ones(50)))  # rho ≈ 1
      _, hl = fit_ar1(s)
      assert np.isnan(hl)
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_fit_ar1_recovers_known_coefficient -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'fit_ar1'`

- [ ] **Step 3: Implement `fit_ar1`**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def fit_ar1(s: pd.Series) -> tuple[float, float]:
      """OLS AR(1) fit on spread series s. Returns (rho, half_life_months).

      half_life is nan when |rho| == 0 or |rho| >= 1 (non-stationary / explosive).
      """
      y = s.values[1:]
      x = s.values[:-1]
      X = np.column_stack([np.ones_like(x), x])
      coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
      rho = float(coefs[1])
      if 0.0 < abs(rho) < 1.0:
          half_life = -np.log(2) / np.log(abs(rho))
      else:
          half_life = float("nan")
      return rho, half_life
  ```

- [ ] **Step 4: Run to verify pass**

  ```
  pytest tests/test_spread.py -k "fit_ar1" -v
  ```

  Expected: all three `fit_ar1` tests `PASSED`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: implement fit_ar1 with OLS and half-life"
  ```

---

## Task 5: `compute_zscore`

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`
- Modify: `tests/test_spread.py`

Expanding z-score: `z[t] = (s[t] − mean(s[0..t])) / std(s[0..t])`. Uses pandas `expanding(min_periods=2)`, so index 0 is always `NaN` (need ≥ 2 points for a standard deviation).

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_spread.py`:

  ```python
  from coffee_forecast.models.spread import compute_zscore


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
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_zscore_first_value_is_nan -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'compute_zscore'`

- [ ] **Step 3: Implement `compute_zscore`**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def compute_zscore(s: pd.Series) -> pd.Series:
      """Expanding z-score: (s - expanding_mean) / expanding_std.

      Index 0 is NaN (need >= 2 points for std).
      """
      exp = s.expanding(min_periods=2)
      return (s - exp.mean()) / exp.std()
  ```

- [ ] **Step 4: Run to verify pass**

  ```
  pytest tests/test_spread.py -k "zscore" -v
  ```

  Expected: all three z-score tests `PASSED`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: implement compute_zscore with expanding window"
  ```

---

## Task 6: `generate_signal`

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`
- Modify: `tests/test_spread.py`

Stateful signal generator. Entry at `|z| > 2.0`, exit (go flat) at `|z| < 0.5`, hold otherwise. `NaN` z-scores do not change state.

Signal semantics:
- `+1`: long spread (buy Arabica, sell Robusta) — spread is below normal, expect it to rise
- `−1`: short spread (sell Arabica, buy Robusta) — spread is above normal, expect it to fall
- `0`: flat (no position)

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_spread.py`:

  ```python
  from coffee_forecast.models.spread import generate_signal


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
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_signal_entry_and_exit -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'generate_signal'`

- [ ] **Step 3: Implement `generate_signal`**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def generate_signal(
      z: pd.Series, entry: float = 2.0, exit_thresh: float = 0.5
  ) -> pd.Series:
      """Stateful trading signal from z-score series.

      +1 = long spread, -1 = short spread, 0 = flat.
      NaN z-scores leave the current position unchanged.
      """
      signals: list[int] = []
      current = 0
      for zi in z:
          if not np.isnan(zi):
              if zi > entry:
                  current = -1
              elif zi < -entry:
                  current = 1
              elif abs(zi) < exit_thresh:
                  current = 0
          signals.append(current)
      return pd.Series(signals, index=z.index, dtype=int)
  ```

- [ ] **Step 4: Run to verify pass**

  ```
  pytest tests/test_spread.py -k "signal" -v
  ```

  Expected: all four signal tests `PASSED`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: implement generate_signal with stateful entry/hold/exit logic"
  ```

---

## Task 7: `build_spread_df`

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`
- Modify: `tests/test_spread.py`

Orchestrates all four functions above into one DataFrame. Computes the expanding half-life by re-fitting AR(1) at each row using all data up to that point (O(n²) loop — fine for ~250 monthly rows).

Output columns: `date` (str, YYYY-MM-DD), `spread` (float), `z_score` (float), `signal` (int), `half_life` (float, NaN for first two rows).

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_spread.py`:

  ```python
  from coffee_forecast.models.spread import build_spread_df


  def test_build_spread_df_columns():
      wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
      result = build_spread_df(wide)
      assert set(result.columns) >= {"date", "spread", "z_score", "signal", "half_life"}


  def test_build_spread_df_row_count():
      wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
      result = build_spread_df(wide)
      assert len(result) == 20


  def test_build_spread_df_signal_dtype():
      wide = _wide([100.0 + i for i in range(20)], [50.0 + i * 0.5 for i in range(20)])
      result = build_spread_df(wide)
      assert np.issubdtype(result["signal"].dtype, np.integer)


  def test_build_spread_df_date_format():
      wide = _wide([100.0, 110.0], [50.0, 55.0])
      result = build_spread_df(wide)
      # Dates must be YYYY-MM-DD strings
      assert result["date"].iloc[0] == "2020-01-01"
      assert result["date"].iloc[1] == "2020-02-01"


  def test_build_spread_df_early_halflife_nan():
      # First two rows don't have enough data for AR(1) (need >= 3 observations)
      wide = _wide([100.0 + i for i in range(10)], [50.0 + i * 0.5 for i in range(10)])
      result = build_spread_df(wide)
      assert np.isnan(result["half_life"].iloc[0])
      assert np.isnan(result["half_life"].iloc[1])
      assert not np.isnan(result["half_life"].iloc[-1])
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_build_spread_df_columns -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'build_spread_df'`

- [ ] **Step 3: Implement `build_spread_df`**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def build_spread_df(wide: pd.DataFrame) -> pd.DataFrame:
      """Compute spread, z-score, signal, and expanding half-life for all months."""
      spread = compute_spread(wide)
      z = compute_zscore(spread)
      sig = generate_signal(z)

      half_lives: list[float] = []
      for i in range(len(spread)):
          s_slice = spread.iloc[: i + 1]
          if len(s_slice) < 3:
              half_lives.append(float("nan"))
          else:
              _, hl = fit_ar1(s_slice)
              half_lives.append(hl)

      return pd.DataFrame(
          {
              "date": spread.index.strftime("%Y-%m-%d"),
              "spread": spread.values,
              "z_score": z.values,
              "signal": sig.values.astype(int),
              "half_life": half_lives,
          }
      )
  ```

- [ ] **Step 4: Run to verify pass**

  ```
  pytest tests/test_spread.py -k "build_spread_df" -v
  ```

  Expected: all five `build_spread_df` tests `PASSED`.

- [ ] **Step 5: Run the full test suite**

  ```
  pytest tests/test_spread.py -v
  ```

  Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: implement build_spread_df orchestrator"
  ```

---

## Task 8: `run_spread_model` + CLI

**Files:**
- Modify: `src/coffee_forecast/models/spread.py`

Reads `prices_monthly` for `KC=F` and `RM=F`, builds the spread DataFrame, and upserts all rows to `spread_signals` using `INSERT OR REPLACE`. The CLI follows the same pattern as `data/ingest.py`: `main()` function, `argparse` for `--db`, Resend alert on crash.

- [ ] **Step 1: Write the failing integration test**

  Append to `tests/test_spread.py`:

  ```python
  import sqlite3

  from coffee_forecast.db import get_connection
  from coffee_forecast.db.migrations import ensure_schema
  from coffee_forecast.models.spread import run_spread_model


  def _insert_monthly(conn, rows):
      conn.executemany(
          "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)", rows
      )
      conn.commit()


  def test_run_spread_model_writes_rows(tmp_path, monkeypatch):
      monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
      conn = get_connection()
      ensure_schema(conn)
      rows = []
      for i in range(24):
          year = 2020 + i // 12
          month = (i % 12) + 1
          dt = f"{year}-{month:02d}-01"
          rows.append((dt, "KC=F", 100.0 + i))
          rows.append((dt, "RM=F", 50.0 + i * 0.5))
      _insert_monthly(conn, rows)
      conn.close()

      run_spread_model(tmp_path / "test.db")

      conn2 = sqlite3.connect(tmp_path / "test.db")
      count = conn2.execute("SELECT COUNT(*) FROM spread_signals").fetchone()[0]
      cols = [r[1] for r in conn2.execute("PRAGMA table_info(spread_signals)").fetchall()]
      conn2.close()

      assert count == 24
      assert set(cols) >= {"date", "spread", "z_score", "signal", "half_life"}


  def test_run_spread_model_idempotent(tmp_path, monkeypatch):
      monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
      conn = get_connection()
      ensure_schema(conn)
      rows = [(f"2020-{m:02d}-01", sym, 100.0 + m) for m in range(1, 13) for sym in ("KC=F", "RM=F")]
      _insert_monthly(conn, rows)
      conn.close()

      run_spread_model(tmp_path / "test.db")
      run_spread_model(tmp_path / "test.db")  # second run — should not duplicate

      conn2 = sqlite3.connect(tmp_path / "test.db")
      count = conn2.execute("SELECT COUNT(*) FROM spread_signals").fetchone()[0]
      conn2.close()
      assert count == 12


  def test_run_spread_model_no_data_no_error(tmp_path, monkeypatch):
      monkeypatch.setenv("COFFEE_DB_PATH", str(tmp_path / "test.db"))
      conn = get_connection()
      ensure_schema(conn)
      conn.close()
      run_spread_model(tmp_path / "test.db")  # empty DB — must not raise
  ```

- [ ] **Step 2: Run to verify failure**

  ```
  pytest tests/test_spread.py::test_run_spread_model_writes_rows -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'run_spread_model'`

- [ ] **Step 3: Implement `run_spread_model` and the CLI block**

  Append to `src/coffee_forecast/models/spread.py`:

  ```python
  def run_spread_model(db_path: Path) -> None:
      """Read prices_monthly, compute spread model, upsert to spread_signals."""
      conn = sqlite3.connect(db_path)
      try:
          df = pd.read_sql_query(
              "SELECT date, symbol, adj_close FROM prices_monthly"
              " WHERE symbol IN ('KC=F', 'RM=F') ORDER BY date",
              conn,
              parse_dates=["date"],
          )
      finally:
          conn.close()

      if df.empty:
          log.warning("No data in prices_monthly for KC=F / RM=F — nothing to do")
          return

      wide = df.pivot(index="date", columns="symbol", values="adj_close").dropna()
      if wide.empty or "KC=F" not in wide.columns or "RM=F" not in wide.columns:
          log.warning("Insufficient data to compute spread — need both KC=F and RM=F")
          return

      result = build_spread_df(wide)

      conn = sqlite3.connect(db_path)
      ensure_schema(conn)
      try:
          rows = result[["date", "spread", "z_score", "signal", "half_life"]].to_dict("records")
          conn.executemany(
              "INSERT OR REPLACE INTO spread_signals"
              " (date, spread, z_score, signal, half_life)"
              " VALUES (:date, :spread, :z_score, :signal, :half_life)",
              rows,
          )
          conn.commit()
      finally:
          conn.close()

      last = result.iloc[-1]
      log.info(
          "Spread model complete — %d rows written; latest z=%.2f signal=%+d half_life=%.1f months",
          len(result),
          last["z_score"] if pd.notna(last["z_score"]) else float("nan"),
          last["signal"],
          last["half_life"] if pd.notna(last["half_life"]) else float("nan"),
      )


  def main() -> None:
      configure_logging()
      parser = argparse.ArgumentParser(description="Run spread model and write to spread_signals")
      parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
      args = parser.parse_args()

      if args.db:
          os.environ["COFFEE_DB_PATH"] = args.db

      db_path = Path(os.getenv("COFFEE_DB_PATH", "data/coffee.db"))
      run_spread_model(db_path)


  if __name__ == "__main__":
      try:
          main()
      except Exception:
          send_pipeline_alert(__file__, traceback.format_exc())
          raise
  ```

- [ ] **Step 4: Run the integration tests**

  ```
  pytest tests/test_spread.py -k "run_spread_model" -v
  ```

  Expected: all three `run_spread_model` tests `PASSED`.

- [ ] **Step 5: Run the full test suite**

  ```
  pytest tests/ -v
  ```

  Expected: all tests `PASSED` (spread + existing ingest/resample/db/providers tests).

- [ ] **Step 6: Run the CLI against the real database**

  (Only if `data/coffee.db` exists and has been populated by Steps 2 ingest+resample.)

  ```
  python -m coffee_forecast.models.spread
  ```

  Expected log output similar to:
  ```
  2026-05-29 HH:MM:SS INFO — Spread model complete — 243 rows written; latest z=X.XX signal=±1 half_life=YY.Y months
  ```

  Then verify in SQLite:
  ```
  python -c "
  import sqlite3
  conn = sqlite3.connect('data/coffee.db')
  print(conn.execute('SELECT COUNT(*) FROM spread_signals').fetchone())
  print(conn.execute('SELECT * FROM spread_signals ORDER BY date DESC LIMIT 3').fetchall())
  "
  ```

- [ ] **Step 7: Run pre-commit hooks**

  ```
  pre-commit run --all-files
  ```

  Fix any ruff or mypy issues, then re-run until clean.

- [ ] **Step 8: Commit**

  ```bash
  git add src/coffee_forecast/models/spread.py tests/test_spread.py
  git commit -m "feat: Step 4 spread model — run_spread_model CLI and integration tests"
  ```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| `spread_signals` table in SQLite | Task 1 |
| `models/` package created | Task 2 |
| `compute_spread` — 1:1 log difference | Task 3 |
| `fit_ar1` — OLS AR(1), half-life formula | Task 4 |
| `compute_zscore` — expanding window | Task 5 |
| `generate_signal` — entry ±2, exit 0.5, hold | Task 6 |
| `build_spread_df` — orchestrator, expanding half-life per row | Task 7 |
| `run_spread_model` — reads DB, upserts | Task 8 |
| CLI `python -m coffee_forecast.models.spread` | Task 8 |
| Resend alert on crash | Task 8 |
| Logging (not print) | All tasks (imports + usage) |
| Tests for all functions | Tasks 3–8 |
