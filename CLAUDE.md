# CLAUDE.md — coffee-forecast

Project memory for the coffee price forecasting system. **Keep this file up to date at the end of every session.** Any new Claude session should be able to pick up work by reading this file alone.

---

## Project goal

Production-quality statistical coffee price forecasting system built as a portfolio piece. Hybrid **VECM + GAMLSS** model, **Arabica vs Robusta** spread component, rigorous backtesting, **live monthly ingestion** so forecasts are publicly falsifiable.

The user (Luc) is non-technical. Explain key concepts and decisions in plain English. Do not sacrifice engineering quality.

---

## Locked-in decisions (Phase 1)

| Area | Decision |
|---|---|
| Spread component | Arabica vs Robusta (inter-commodity) |
| Forecast horizon | 1, 2, 3 months ahead |
| Hosting | Local dev + GitHub Actions for monthly live runs |
| Output | Streamlit dashboard (Streamlit Cloud) + auto-generated monthly PDF (Quarto) |
| Language stack | Python primary; R subprocess for GAMLSS only |
| Data source | **FRED** (no key needed) for coffee prices: `PCOFFOTMUSDM` (Arabica), `PCOFFROBUSDM` (Robusta), `DTWEXBGS` (DXY broad index). **Alpha Vantage** (free key, `ALPHA_VANTAGE_API_KEY`) for FX monthly: BRL=X, VND=X, IDR=X via `FX_MONTHLY` endpoint. Provider routed via `CompositeProvider` / `make_default_provider()`. |
| FX in model | BRL, VND, IDR, DXY as **exogenous drivers** in VECM (not endogenous — Johansen found r=0 in 4-variable system) |
| TICKERS | `["KC=F", "RM=F", "BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]` — all stored in `prices` table |
| Combiner | Sequential — VECM produces point forecast, GAMLSS models distribution around it |
| GAMLSS family | BCT (Box-Cox-t) — fat tails + skew |
| Spread signal | OU / AR(1) on log(Arabica) − log(Robusta), entry threshold \|z\| > 2 |
| Storage | SQLite in repo |
| Repo name | `coffee-forecast` (initialised in this directory) |
| Ops alerting | Resend email to lucparrot1@gmail.com on pipeline failure |

---

## Architecture (one-paragraph summary)

A Python data layer ingests prices into SQLite behind a `PriceProvider` interface. `CompositeProvider` routes coffee + DXY symbols to `FREDProvider` and FX (BRL, VND, IDR) symbols to `AlphaVantageProvider`. A VECM (statsmodels) fits cointegration between Arabica and Robusta with BRL, VND, IDR, and DXY as exogenous drivers, producing point forecasts at 1/2/3-month horizons. A separate R subprocess fits a GAMLSS BCT distribution to the VECM forecast residuals conditional on regime indicators, giving full predictive distributions. A standalone spread model on log(Arabica/Robusta) provides a mean-reversion signal. A walk-forward backtest engine writes results to SQLite. Streamlit visualises everything; Quarto renders a monthly PDF. GitHub Actions runs the full pipeline on the 1st of each month and emails on failure.

See the full plan in conversation history of the initial planning session, or rebuild it from the locked-in decisions above.

---

## Build sequence (running TODO list)

**Update this list at the end of every session.** Tick off completed steps, add sub-tasks as they emerge, record any new [DECISION NEEDED] flags.

- [x] **Step 1 — Project skeleton**
  - [x] `git init` and first commit
  - [x] `pyproject.toml` (Python 3.11), pinned deps in `requirements.txt`
  - [x] `pre-commit` config (ruff, mypy)
  - [x] Logging boilerplate (see global standards)
  - [x] Resend alert wrapper for pipeline scripts
  - [x] SQLite schema + migrations (`prices`, `prices_monthly`, `model_runs`, `forecasts`, `backtest_results`, `accuracy_log`)
  - [x] Repo README placeholder
  - Note: `fx` merged into `prices` table — all symbols (BRL=X, VND=X, IDR=X, DX-Y.NYB) stored uniformly in `prices`
- [x] **Step 2 — Data layer**
  - [x] `PriceProvider` ABC (`src/coffee_forecast/data/providers.py`)
  - [x] `FREDProvider` (coffee prices + DXY), `AlphaVantageProvider` (BRL/VND/IDR FX), `CompositeProvider` routing, `make_default_provider()` factory
  - [x] Ingestion CLI (`python -m coffee_forecast.data.ingest [--start YYYY-MM-DD] [--db PATH]`) — single bulk fetch, `INSERT OR IGNORE`
  - [x] Month-end resampling job (`python -m coffee_forecast.data.resample [--db PATH]`) — mean `adj_close` per month, excludes current month
  - [x] `load_dotenv()` added to `configure_logging()` so `.env` is loaded by all pipeline scripts
  - Note: Monthly metric is **mean of daily adj_close** (chosen over last-close for noise smoothing). Historical start date: 2000-01-01.
  - Note: `db/__init__.py` patched to lazy-evaluate `COFFEE_DB_PATH` so `monkeypatch.setenv` works in tests.
  - Note: `data/` in `.gitignore` changed to `/data/` to avoid blocking the `src/coffee_forecast/data/` package.
  - Note: DB currently holds KC=F/RM=F ~315 months (FRED), DX-Y.NYB ~244 months (FRED), BRL=X ~316 months, VND=X/IDR=X ~137 months (Alpha Vantage)
- [x] **Step 3 — Exploratory analysis notebook** (`notebooks/01_eda.ipynb`)
  - [x] Price series plots (raw, log, log returns) — data from FRED, 243 monthly obs 2006–2026
  - [x] Stationarity: all 4 series confirmed I(1) ✅
  - [x] Johansen cointegration on 4 series: r=0 (no cointegrating vectors) ⚠️
  - [x] Engle-Granger Arabica–Robusta pairwise: p=0.009, cointegrated ✅
  - [x] ACF/PACF on differenced log prices for KC=F and RM=F
  - [x] Regime detection: rolling-vol gives balanced 76/79/76 Low/Med/High split ✅; HMM degenerated (240/2 split) — use rolling-vol for GAMLSS
  - **Key decision for Step 5:** model as 2-variable VECM [KC=F, RM=F] with BRL, VND, IDR, DXY as exogenous inputs — Johansen finds no cointegration in the 4-variable system, so FX stays exogenous
  - Note: new deps — `matplotlib==3.10.9`, `hmmlearn==0.3.3`, `jupyterlab==4.5.7`, `pandas-datareader==0.10.0` (yfinance and nasdaq-data-link removed)
  - Note: notebook TICKERS use `["KC=F", "RM=F", "BRL=X", "DX-Y.NYB"]`; VND=X and IDR=X data now in DB — can be added to notebook in a future pass
- [x] **Step 4 — Spread model** (Arabica–Robusta cointegration, OU fit, z-score, signal)
  - [x] `compute_spread`, `compute_zscore`, `fit_ar1`, `generate_signal`, `build_spread_df`
  - [x] `run_spread_model` — reads `prices_monthly`, upserts into `spread_signals` (INSERT OR REPLACE, idempotent)
  - [x] `main()` + `__main__` guard with Resend alert wrapper
  - [x] 20 unit + integration tests; smoke test confirmed 315 rows on live DB
  - Note: entry `|z| > 2.0`, exit `|z| < 0.5`, expanding-window z-score
- [x] **Step 5 — VECM model** (lag/rank selection, point forecasts, tests)
  - [x] 2-variable VECM [KC=F, RM=F] with 4 exogenous inputs (BRL=X, VND=X, IDR=X, DX-Y.NYB)
  - [x] Cointegration rank hardcoded to 1 (EDA confirmed), automatic AIC lag selection up to 12 lags
  - [x] Naïve exogenous forecast: Δexog = 0 (hold FX flat); refined forecast available via fallback to random walk
  - [x] `compute_vecm_forecast`, `run_vecm_model` — reads `prices_monthly`, upserts into `vecm_residuals` table (point forecast + residual)
  - [x] `main()` + `__main__` guard with Resend alert wrapper
  - [x] 47 unit + integration tests; full pipeline smoke test passed
  - Note: 2-variable VECM [KC=F, RM=F] with 4 exog (BRL=X, VND=X, IDR=X, DX-Y.NYB), coint_rank=1 hardcoded (EDA confirmed), AIC lag selection up to 12, naïve Δexog=0 forecast assumption, residuals stored in `vecm_residuals` table for GAMLSS (Step 6)
- [ ] **Step 6 — R/GAMLSS subprocess bridge** (BCT fit on VECM residuals)
- [ ] **Step 7 — Hybrid combiner** (sequential mean + distribution)
- [ ] **Step 8 — Backtest engine** (walk-forward expanding window, metrics, benchmarks)
- [ ] **Step 9 — Streamlit dashboard** (5 tabs: Current Forecast, Backtest, Live Accuracy, Spread Trade, Methodology)
- [ ] **Step 10 — Quarto PDF template** (monthly auto-rendered report)
- [ ] **Step 11 — GitHub Actions monthly pipeline** (ingest → refit → forecast → backtest → render PDF → commit → alert)
- [ ] **Step 12 — Methodology write-up + README polish**

### Open [DECISION NEEDED] flags
_(none currently open — all Phase 1 decisions resolved)_

---

## Working rules

- Before starting any new step, give a one-paragraph plain-English summary of what we are about to build and why.
- Explain significant technical decisions (model parameterisation, library choice) in non-technical terms.
- **Use `AskUserQuestion` liberally for every design decision.** Default to asking, not assuming. Anything that could reasonably be done two or more ways — library choice, model parameterisation, schema shape, UI layout, threshold values, naming, file structure — surface the options and let Luc choose. One good question beats building the wrong thing. Only skip the question for trivially reversible, obvious choices already implied by earlier decisions in this file.
- Production-quality code: modular, documented, tested, version-control ready.
- Flag every `[DECISION NEEDED]` in the open-decisions section above before continuing past it.
- Follow the global standards in `~/.claude/CLAUDE.md` (logging module not `print`, pinned deps, `tenacity` for HTTP, Resend alert wrapper on every pipeline script, `timeout-minutes` on GH Actions jobs).

---

## How to maintain this file

At the end of every session:
1. Tick off completed checkboxes in the build sequence.
2. Add any newly discovered sub-tasks.
3. Add new `[DECISION NEEDED]` items to the open-decisions section.
4. Update the locked-in decisions table if anything changed.
5. If a step's scope expanded meaningfully, add a short note under it.

Keep it concise. This file is the source of truth a new session reads first.
