import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import matplotlib

matplotlib.use("Agg")  # headless — no display needed

import matplotlib.pyplot as plt
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)


def fetch_latest_forecasts(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch forecasts from the latest successful hybrid model run.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with columns: horizon, symbol, point_forecast, p10, p25, p50, p75, p90
        Empty DataFrame if no successful hybrid run exists.
    """
    row = conn.execute(
        "SELECT id FROM model_runs "
        "WHERE model_type='hybrid' AND status='success' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return pd.DataFrame()
    return pd.read_sql(
        "SELECT horizon, symbol, point_forecast, p10, p25, p50, p75, p90 "
        "FROM forecasts WHERE run_id = ? ORDER BY symbol, horizon",
        conn,
        params=(row[0],),
    )


def fetch_last_prices(conn: sqlite3.Connection) -> dict[str, float]:
    """Fetch the latest prices for KC=F and RM=F.

    Args:
        conn: SQLite connection

    Returns:
        Dictionary mapping symbol to most recent adj_close price.
    """
    rows = conn.execute(
        "SELECT symbol, adj_close FROM prices_monthly "
        "WHERE symbol IN ('KC=F','RM=F') "
        "AND date = (SELECT MAX(date) FROM prices_monthly WHERE symbol = prices_monthly.symbol)"
    ).fetchall()
    return {symbol: price for symbol, price in rows}


def fetch_latest_backtest_metrics(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch the latest backtest accuracy metrics aggregated by symbol and horizon.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with columns: symbol, horizon, mae, mape, coverage_80
        Empty DataFrame if no accuracy_log data exists.
    """
    df = pd.read_sql(
        "SELECT horizon, symbol, mae, mape, coverage_80 "
        "FROM accuracy_log WHERE mae IS NOT NULL",
        conn,
    )
    if df.empty:
        return df
    return (
        df.groupby(["symbol", "horizon"])
        .agg(mae=("mae", "mean"), mape=("mape", "mean"), coverage_80=("coverage_80", "mean"))
        .reset_index()
    )


def fetch_spread_state(conn: sqlite3.Connection) -> dict[str, object]:
    """Fetch the latest spread signal state.

    Args:
        conn: SQLite connection

    Returns:
        Dictionary with keys: date, spread, z_score, signal, half_life
        Empty dictionary if no spread_signals data exists.
    """
    row = conn.execute(
        "SELECT date, spread, z_score, signal, half_life "
        "FROM spread_signals ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {}
    return {
        "date": row[0],
        "spread": row[1],
        "z_score": row[2],
        "signal": row[3],
        "half_life": row[4],
    }


def _build_analysis_prompt(
    forecasts_df: pd.DataFrame,
    last_prices: dict[str, float],
    backtest_df: pd.DataFrame,
    spread_state: dict,
) -> str:
    arabica_last = last_prices.get("KC=F", "N/A")
    robusta_last = last_prices.get("RM=F", "N/A")

    fc_rows: list[dict] = []
    if not forecasts_df.empty:
        for _, r in forecasts_df.iterrows():
            fc_rows.append({
                "symbol": r["symbol"],
                "horizon_months": int(r["horizon"]),
                "p50_forecast_cents_per_lb": round(float(r["p50"]), 1),
                "80pct_interval": [round(float(r["p10"]), 1), round(float(r["p90"]), 1)],
            })

    perf_rows: list[dict] = []
    if not backtest_df.empty:
        for _, r in backtest_df.iterrows():
            perf_rows.append({
                "symbol": r["symbol"],
                "horizon": int(r["horizon"]),
                "mean_mape_pct": round(float(r["mape"]), 1) if r["mape"] is not None else None,
                "coverage_80pct_actual": round(float(r["coverage_80"]) * 100, 1) if r["coverage_80"] is not None else None,
            })

    sig_map = {1: "buy_arabica", -1: "sell_arabica_buy_robusta", 0: "neutral"}
    spread_summary = {
        "z_score": round(spread_state.get("z_score", 0.0), 2),
        "signal": sig_map.get(spread_state.get("signal", 0), "neutral"),
        "half_life_months": round(spread_state.get("half_life", 0.0), 1),
    } if spread_state else {}

    data = {
        "last_observed_prices_cents_per_lb": {
            "arabica_KC=F": arabica_last,
            "robusta_RM=F": robusta_last,
        },
        "forecasts": fc_rows,
        "backtest_performance": perf_rows,
        "spread_signal": spread_summary,
    }

    return (
        "You are a commodity analyst writing for a non-technical audience. "
        "Based on the coffee price forecast data below, write a concise executive summary "
        "of 2-3 short paragraphs (around 120-150 words total). Cover: (1) what the model "
        "is currently forecasting for Arabica and Robusta prices over the next 3 months, "
        "(2) how the model has been performing historically (use MAPE and coverage_80 - "
        "explain coverage_80 as 'X% of our forecast ranges contained the actual price, "
        "versus an 80% target'), and (3) what the spread signal is saying in plain English. "
        "Do not use bullet points. Write in plain English. Do not mention the model internals "
        "(VECM, GAMLSS, SHASH). Use 'the model' or 'our forecast' instead. "
        "Do not use em dashes or en dashes. Do not use phrases typical of AI writing "
        "such as 'it is worth noting', 'importantly', 'in conclusion', 'delve', 'robust', "
        "'nuanced', or 'comprehensive'. Write like a straightforward analyst, not a language model.\n\n"
        f"Data (JSON):\n{json.dumps(data, indent=2)}"
    )


def generate_analysis(
    forecasts_df: pd.DataFrame,
    last_prices: dict[str, float],
    backtest_df: pd.DataFrame,
    spread_state: dict,
) -> str:
    """Generate a short plain-English executive summary using the Claude API.

    Args:
        forecasts_df: DataFrame with forecast quantiles.
        last_prices: Dict mapping symbol to latest observed price.
        backtest_df: DataFrame with aggregated backtest metrics.
        spread_state: Dict with spread signal state.

    Returns:
        A ~150-word analysis string, or a fallback message if the API is unavailable.
    """
    try:
        import anthropic
        from dotenv import dotenv_values
        from tenacity import retry, stop_after_attempt, wait_exponential

        # Read API key directly from .env so it isn't shadowed by the host environment
        api_key = dotenv_values().get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_analysis_prompt(forecasts_df, last_prices, backtest_df, spread_state)

        from tenacity import retry_if_exception_type

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(anthropic.RateLimitError),
            reraise=True,
        )
        def _call() -> str:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""

        return _call()
    except Exception as exc:
        log.warning("Claude API analysis unavailable: %s", exc)
        return "_AI-generated analysis unavailable for this report._"


_SPREAD_SIGNAL_LABELS = {1: "Buy Arabica", -1: "Sell Arabica / Buy Robusta", 0: "Neutral"}


def _build_success_email_body(
    month: str,
    forecasts_df: pd.DataFrame,
    last_prices: dict[str, float],
    spread_state: dict[str, object],
) -> str:
    """Build HTML body for the monthly success notification email."""
    rows = ""
    if not forecasts_df.empty:
        for symbol, label in [("KC=F", "Arabica"), ("RM=F", "Robusta")]:
            sub = forecasts_df[forecasts_df["symbol"] == symbol]
            last = last_prices.get(symbol, float("nan"))
            p50 = sub.set_index("horizon")["p50"]
            h1 = float(p50.get(1) or float("nan"))
            h2 = float(p50.get(2) or float("nan"))
            h3 = float(p50.get(3) or float("nan"))
            rows += (
                f"<tr><td>{label}</td>"
                f"<td>{last:.1f}</td>"
                f"<td>{h1:.1f}</td>"
                f"<td>{h2:.1f}</td>"
                f"<td>{h3:.1f}</td></tr>\n"
            )

    signal_text = _SPREAD_SIGNAL_LABELS.get(spread_state.get("signal", 0), "Neutral")
    zscore = spread_state.get("z_score", 0.0) or 0.0

    streamlit_url = os.getenv("STREAMLIT_URL", "")
    dashboard_link = (
        f'<p><a href="{streamlit_url}">View dashboard</a></p>' if streamlit_url else ""
    )

    return (
        f"<h2>Coffee Forecast &#8212; {month}</h2>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        "<tr><th>Symbol</th><th>Last (¢/lb)</th>"
        "<th>h=1 p50</th><th>h=2 p50</th><th>h=3 p50</th></tr>"
        f"{rows}"
        "</table>"
        f"<p>Spread signal: <strong>{signal_text}</strong> (z={zscore:.2f})</p>"
        f"{dashboard_link}"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _post_success_notification(api_key: str, alert_email: str, month: str, body: str) -> None:
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "onboarding@resend.dev",
            "to": [alert_email],
            "subject": f"[coffee-forecast] Monthly report ready: {month}",
            "html": body,
        },
        timeout=10,
    )
    resp.raise_for_status()


def send_success_email(month: str, db_path: "str | Path") -> None:
    """Send a success notification email via Resend after a monthly pipeline run.

    No-op if RESEND_API_KEY is not set.
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    alert_email = os.getenv("ALERT_EMAIL", "")
    if not api_key or not alert_email:
        log.warning("RESEND_API_KEY or ALERT_EMAIL not set — skipping success email")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        forecasts_df = fetch_latest_forecasts(conn)
        last_prices = fetch_last_prices(conn)
        spread_state = fetch_spread_state(conn)
    finally:
        conn.close()

    body = _build_success_email_body(month, forecasts_df, last_prices, spread_state)
    _post_success_notification(api_key, alert_email, month, body)
    log.info("Success email sent for %s", month)


_SYMBOL_COLORS = {"KC=F": "#B05C1A", "RM=F": "#4E7D3A"}
_SYMBOL_LABELS = {"KC=F": "Arabica (KC=F)", "RM=F": "Robusta (RM=F)"}


def build_forecast_chart(
    forecasts_df: pd.DataFrame,
    last_prices: dict[str, float],
) -> "matplotlib.figure.Figure":
    """Build a matplotlib figure showing forecast trajectories with confidence bands.

    Args:
        forecasts_df: DataFrame with columns: symbol, horizon, p10, p50, p90
        last_prices: Dictionary mapping symbol to most recent price

    Returns:
        A matplotlib Figure object. Empty forecasts render a placeholder message.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FBF7F2")
    ax.set_facecolor("#FBF7F2")

    if forecasts_df.empty:
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center", transform=ax.transAxes)
        return fig

    for symbol, color in _SYMBOL_COLORS.items():
        sub = forecasts_df[forecasts_df["symbol"] == symbol].sort_values("horizon")
        if sub.empty:
            continue
        horizons = [0] + sub["horizon"].tolist()
        last = last_prices.get(symbol, float("nan"))
        p50s = [last] + sub["p50"].tolist()
        p10s = [last] + sub["p10"].tolist()
        p90s = [last] + sub["p90"].tolist()
        ax.plot(horizons, p50s, marker="o", color=color, label=_SYMBOL_LABELS[symbol], linewidth=2)
        ax.fill_between(horizons, p10s, p90s, color=color, alpha=0.15)

    ax.set_xlabel("Months ahead", fontsize=10)
    ax.set_ylabel("Price (cents/lb)", fontsize=10)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Now", "h=1", "h=2", "h=3"])
    ax.legend(framealpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


_QMD_TEMPLATE = Path(__file__).parent.parent.parent / "reports" / "monthly_template.qmd"

# Common install locations for Quarto on Windows/macOS/Linux
_QUARTO_FALLBACK_DIRS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Quarto" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
]


def _find_quarto() -> str:
    """Return the quarto executable path, searching fallback dirs if not on PATH."""
    found = shutil.which("quarto")
    if found:
        return found
    for d in _QUARTO_FALLBACK_DIRS:
        candidate = d / ("quarto.exe" if sys.platform == "win32" else "quarto")
        if candidate.exists():
            return str(candidate)
    return "quarto"  # let subprocess raise FileNotFoundError with a clear message


def render_monthly_report(
    month: str,
    db_path: str | Path,
    output_dir: str | Path = Path("reports"),
) -> Path:
    """Invoke quarto render to produce a monthly PDF report.

    Args:
        month: Report month in YYYY-MM format.
        db_path: Path to the SQLite database.
        output_dir: Directory to write the PDF into. Created if absent.

    Returns:
        Resolved Path to the generated PDF file.

    Raises:
        RuntimeError: If quarto exits with a non-zero return code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / f"{month}.pdf").resolve()

    env = {
        **os.environ,
        "REPORT_DB_PATH": str(Path(db_path).resolve()),
        "REPORT_MONTH": month,
    }
    quarto_exe = _find_quarto()
    result = subprocess.run(
        [
            quarto_exe,
            "render",
            str(_QMD_TEMPLATE),
            "--to",
            "pdf",
            "--output",
            f"{month}.pdf",   # filename only — quarto rejects paths in --output
            "--output-dir",
            str(output_dir.resolve()),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        log.error("quarto render stderr:\n%s", result.stderr)
        raise RuntimeError(f"quarto render failed with code {result.returncode}")
    log.info("PDF written to %s", output_path)
    return output_path


def main() -> None:
    configure_logging()
    default_db = os.getenv("COFFEE_DB_PATH", "data/coffee.db")
    parser = argparse.ArgumentParser(description="Render monthly coffee forecast PDF or send success email")
    parser.add_argument(
        "--month",
        default=datetime.utcnow().strftime("%Y-%m"),
        help="Report month in YYYY-MM format (default: current month)",
    )
    parser.add_argument("--db", default=default_db, help="Path to SQLite database")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for the PDF",
    )
    parser.add_argument(
        "--success-email",
        action="store_true",
        help="Send monthly success notification email instead of rendering PDF",
    )
    args = parser.parse_args()

    if args.success_email:
        send_success_email(args.month, args.db)
    else:
        render_monthly_report(args.month, args.db, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
