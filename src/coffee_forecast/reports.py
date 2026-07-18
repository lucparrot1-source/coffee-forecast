import argparse
import base64
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


def fetch_spread_history(conn: sqlite3.Connection, months: int = 36) -> pd.DataFrame:
    """Fetch the last N months of spread z-score history.

    Args:
        conn: SQLite connection
        months: Number of months of history to return

    Returns:
        DataFrame with columns: date, z_score, signal. Sorted ascending by date.
    """
    df = pd.read_sql(
        "SELECT date, z_score, signal FROM spread_signals ORDER BY date DESC LIMIT ?",
        conn,
        params=(months,),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_spread_chart(spread_hist: pd.DataFrame) -> "matplotlib.figure.Figure":
    """Build a matplotlib z-score chart for the spread signal section.

    Args:
        spread_hist: DataFrame from fetch_spread_history

    Returns:
        A matplotlib Figure. Returns a placeholder figure if data is empty.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if spread_hist.empty:
        ax.text(0.5, 0.5, "No spread data", ha="center", va="center", transform=ax.transAxes)
        return fig

    dates = spread_hist["date"]
    z = spread_hist["z_score"]

    # Shaded threshold regions
    ax.axhspan(2, z.max() + 0.5, color="#C62828", alpha=0.06)
    ax.axhspan(z.min() - 0.5, -2, color="#2E7D32", alpha=0.06)

    # Reference lines
    for level, color, ls in [(2, "#C62828", "--"), (-2, "#2E7D32", "--"),
                              (0.5, "#888", ":"), (-0.5, "#888", ":")]:
        ax.axhline(level, color=color, linewidth=0.8, linestyle=ls)

    # Zero line
    ax.axhline(0, color="#CCCCCC", linewidth=0.6)

    # Z-score line
    ax.plot(dates, z, color="#B05C1A", linewidth=1.8)
    ax.fill_between(dates, z, 0, color="#B05C1A", alpha=0.08)

    ax.set_ylabel("Z-Score (standard deviations)", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("Entry zone (|z|>2)", xy=(dates.iloc[-1], 2.05),
                fontsize=7, color="#C62828", ha="right")
    ax.annotate("Entry zone (|z|>2)", xy=(dates.iloc[-1], -2.3),
                fontsize=7, color="#2E7D32", ha="right")
    fig.tight_layout()
    return fig


def fetch_recent_performance(conn: sqlite3.Connection, n_months: int = 3) -> pd.DataFrame:
    """Fetch the last N months of h=1 forecast vs actual for KC=F and RM=F.

    For each (target_date, symbol) pair picks the row with the most recent train_end
    (closest realistic one-month-ahead forecast).

    Returns:
        DataFrame with columns: target_date, symbol, point_forecast, actual, error_pct
        Empty DataFrame if no backtest data with actuals exists.
    """
    df = pd.read_sql(
        """
        SELECT b.target_date, b.symbol, b.point_forecast, b.actual
        FROM backtest_results b
        JOIN (
            SELECT target_date, symbol, MAX(train_end) AS latest_train
            FROM backtest_results
            WHERE horizon = 1 AND actual IS NOT NULL
              AND symbol IN ('KC=F', 'RM=F')
            GROUP BY target_date, symbol
        ) latest ON b.target_date = latest.target_date
               AND b.symbol = latest.symbol
               AND b.train_end = latest.latest_train
        WHERE b.horizon = 1 AND b.actual IS NOT NULL
        ORDER BY b.target_date DESC, b.symbol
        """,
        conn,
    )
    if df.empty:
        return df
    # Keep only the most recent n_months distinct target dates
    recent_dates = df["target_date"].unique()[:n_months]
    df = df[df["target_date"].isin(recent_dates)].copy()
    df["error_pct"] = (df["point_forecast"] - df["actual"]) / df["actual"] * 100
    return df


def generate_recent_commentary(recent_df: pd.DataFrame) -> str:
    """Generate a plain-English paragraph per month explaining recent h=1 forecast performance.

    Asks Claude to cover: what was forecast, what actually happened, size of miss,
    and plausible reasons a statistical model would not have anticipated the outcome.

    Returns:
        Multi-paragraph markdown string, or a fallback message if the API is unavailable.
    """
    if recent_df.empty:
        return "_No recent backtest data available._"

    rows = []
    for _, r in recent_df.iterrows():
        fc = float(r["point_forecast"]) if r["point_forecast"] is not None else None
        actual = float(r["actual"]) if r["actual"] is not None else None
        err = float(r["error_pct"]) if r["error_pct"] is not None else None
        rows.append({
            "month": str(r["target_date"])[:7],
            "symbol": r["symbol"],
            "forecast_cpl": round(fc, 1) if fc is not None else None,
            "actual_cpl": round(actual, 1) if actual is not None else None,
            "error_pct": round(err, 1) if err is not None else None,
            "direction": "overestimated" if err and err > 0 else "underestimated",
        })

    try:
        import anthropic
        from dotenv import dotenv_values
        from tenacity import retry_if_exception_type

        api_key = dotenv_values().get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            "You are a commodity analyst. The data below shows one-month-ahead coffee price forecasts "
            "versus the prices that were actually observed. "
            "Write one short paragraph per calendar month (group both symbols together per month). "
            "Each paragraph should cover: what the model forecast for Arabica and Robusta, what actually happened, "
            "the size of the miss in percentage terms, and one or two plausible reasons a purely statistical model "
            "working from historical price relationships might not have anticipated this outcome "
            "(e.g. unexpected supply shocks, FX moves, weather events, harvest disruptions, policy changes). "
            "Be specific but brief. Paragraphs should be 3-4 sentences. "
            "FORMATTING RULES: write the unit as *¢/lb* (asterisks for italics, no space). "
            "Do not use bullet points. Do not use em dashes or en dashes. "
            "Do not use AI clichés ('it is worth noting', 'importantly', 'robust', 'nuanced'). "
            "Write like a straightforward analyst.\n\n"
            f"Data (JSON):\n{json.dumps(rows, indent=2)}"
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(anthropic.RateLimitError),
            reraise=True,
        )
        def _call() -> str:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""

        return _call()
    except Exception as exc:
        log.warning("Recent commentary unavailable: %s", exc)
        return "_AI-generated commentary unavailable._"


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
            p10, p90 = round(float(r["p10"]), 1), round(float(r["p90"]), 1)
            fc_rows.append({
                "symbol": r["symbol"],
                "horizon_months": int(r["horizon"]),
                "p50_forecast": round(float(r["p50"]), 1),
                "80pct_band_width": round(p90 - p10, 1),
                "80pct_interval": [p10, p90],
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
        "FORMATTING RULES: "
        "Always write the unit as *¢/lb* (with asterisks so it renders in italics, no space between ¢ and /lb). "
        "When describing forecast uncertainty, express it as a band width, not raw bounds. "
        "For example: 'the 80% forecast range spans about 57 *¢/lb*' rather than 'between 283 and 340 cents'. "
        "Do not use bullet points. Do not mention the model internals "
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
def _post_success_notification(
    api_key: str,
    alert_email: str,
    month: str,
    body: str,
    pdf_bytes: bytes | None,
) -> None:
    payload: dict = {
        "from": "onboarding@resend.dev",
        "to": [alert_email],
        "subject": f"[coffee-forecast] Monthly report ready: {month}",
        "html": body,
    }
    if pdf_bytes is not None:
        payload["attachments"] = [
            {
                "filename": f"coffee-forecast-{month}.pdf",
                "content": base64.b64encode(pdf_bytes).decode(),
            }
        ]
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def send_success_email(month: str, db_path: "str | Path", output_dir: "str | Path" = "reports") -> None:
    """Send a success notification email via Resend after a monthly pipeline run.

    Attaches the rendered PDF if it exists at output_dir/{month}.pdf.
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

    pdf_path = Path(output_dir) / f"{month}.pdf"
    pdf_bytes: bytes | None = None
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
        log.info("Attaching PDF (%d KB): %s", len(pdf_bytes) // 1024, pdf_path)
    else:
        log.warning("PDF not found at %s — sending email without attachment", pdf_path)

    body = _build_success_email_body(month, forecasts_df, last_prices, spread_state)
    _post_success_notification(api_key, alert_email, month, body, pdf_bytes)
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
    ax.set_ylabel("Price (¢/lb)", fontsize=10)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Now", "h=1", "h=2", "h=3"])
    ax.legend(framealpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Broken-axis marks: two diagonal slashes at the base of the y-axis to
    # indicate the axis does not start at zero.
    d = 0.018
    for y0 in (0.0, 0.055):
        ax.plot(
            (-d, d),
            (y0 - d * 1.4, y0 + d * 1.4),
            transform=ax.transAxes,
            color="black",
            lw=1.1,
            clip_on=False,
        )

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
        # Pin Quarto to the same Python interpreter that installed our packages.
        # Without this, Quarto may resolve 'python3' to a different binary (e.g.
        # after r-lib/actions/setup-r modifies PATH) and fail to find pyyaml/ipykernel.
        "QUARTO_PYTHON": sys.executable,
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
            f"{month}.pdf",
        ],
        cwd=str(output_dir.resolve()),  # quarto --output-dir is unreliable; cwd is the output target
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
    matplotlib.use("Agg")  # headless — must be set before render_monthly_report creates figures
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
        send_success_email(args.month, args.db, args.output_dir)
    else:
        render_monthly_report(args.month, args.db, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
