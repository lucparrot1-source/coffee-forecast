# Coffee Price Forecast

A live statistical model for Arabica and Robusta coffee futures. Built to make real-world forecasts, be falsifiable, track performance, and improve over time.

**[→ Live Dashboard](https://coffee-forecast.streamlit.app)** &nbsp;·&nbsp; Reruns automatically on the 1st of each month

## About

I'm a commodity analyst and super avid coffee drinker. I built this project to pair those two together. What drives the price of coffee? Can a statistical model built over a weekend make a useful forecast for it?

This is a personal project, but the forecasts are very much real. This is ingested from live market data, refit monthly, and compared against actuals when they come in.

## How it works

The model is a hybrid in two-stages using different statistical models:

1. **VECM (Vector Error Correction Model)** — captures the long-run cointegration between Arabica and Robusta prices, with Brazilian Real, Vietnamese Dong, Indonesian Rupiah (major exporters), and the US Dollar index as exogenous drivers. Produces point forecasts at 1, 2, and 3 months ahead.

2. **GAMLSS (Generalized Additive Models for Location, Scale and Shape)** — fits a full probability distribution around the VECM point forecast, conditioned on the current volatility regime. Produces the uncertainty bands (p10–p90) you see on the dashboard.

A separate **spread signal** tracks the Arabica/Robusta price ratio and flags mean-reversion opportunities when the spread is statistically stretched. The literature identifies a degree of substitution effect when Arabica prices are significantly higher than Robusta, although I can personally vouch for and say the taste isn't exactly interchangeable.

## Stack

| Layer | Tools |
|---|---|
| Modelling | Python · statsmodels (VECM) · R / gamlss (GAMLSS) |
| Data | FRED · Alpha Vantage · SQLite |
| Dashboard | Streamlit · Plotly |
| Reports | Quarto · Claude API |
| Ops | GitHub Actions · Resend |

## Data sources

| Source | What it provides |
|---|---|
| [FRED](https://fred.stlouisfed.org) (Federal Reserve Economic Data) | Arabica and Robusta monthly price series, US Dollar index (DXY). Free, no API key required. |
| [Alpha Vantage](https://www.alphavantage.co) | Monthly FX rates for BRL, VND, IDR — the currencies of the three largest coffee-producing nations. |

## Pipeline

A GitHub Actions workflow runs on the 1st of each month: ingests new prices directly from the Federal Reserve and Alpha Vantage, refits the VECM and GAMLSS models, generates forecasts, runs a backtest, and renders a PDF summary. Forecast errors accumulate in SQLite and are visible on the Live Accuracy tab of the dashboard.
