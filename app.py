"""Coffee Price Forecast — Streamlit dashboard."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Coffee Price Forecast", layout="wide", page_icon="☕")

# ── CSS / design system ───────────────────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #FFFFFF !important; }
header[data-testid="stHeader"] { background: #FFFFFF !important; border-bottom: 1px solid #DDD0C0; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E0E0E0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #1A1A1A !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] strong { color: #B05C1A !important; }
[data-testid="stSidebarContent"] h3 {
    font-family: 'Lora', Georgia, serif !important;
    color: #1A1A1A !important; font-size: 1.0rem !important; font-weight: 700 !important;
    border-bottom: 2px solid #E0E0E0; padding-bottom: 8px; margin-bottom: 10px !important;
}

/* Tabs — Chrome-style */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #DDD0C0 !important;
    padding: 0 !important;
    gap: 0 !important;
    width: 100% !important;
    display: flex !important;
    overflow: visible !important;
}
/* target both div and button wrappers Streamlit may use */
div[data-baseweb="tab"],
button[data-baseweb="tab"] {
    flex: 1 1 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 14px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #8C6E52 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
    /* separator line between tabs */
    border-right: 1px solid #DDD0C0 !important;
}
div[data-baseweb="tab"]:last-child,
button[data-baseweb="tab"]:last-child {
    border-right: none !important;
}
div[data-baseweb="tab"]:hover,
button[data-baseweb="tab"]:hover {
    color: #2C1A0E !important;
    background: #F5EDE0 !important;
}
div[aria-selected="true"][data-baseweb="tab"],
button[aria-selected="true"][data-baseweb="tab"] {
    color: #B05C1A !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #B05C1A !important;
    background: transparent !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] { display: none !important; }
div[data-baseweb="tab-panel"] { padding-top: 28px !important; }

/* Headings */
h1, h2 { font-family: 'Lora', Georgia, serif !important; color: #1A1A1A !important; font-weight: 600 !important; }
h2 { font-size: 1.8rem !important; margin-bottom: 2px !important; }
h3 {
    font-family: 'Inter', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #1A1A1A !important; margin-bottom: 12px !important;
}
p { color: #1A1A1A; line-height: 1.65; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #DDD0C0; border-radius: 8px; padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Inter', sans-serif !important; font-size: 0.70rem !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    color: #1A1A1A !important; font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 1.75rem !important;
    font-weight: 600 !important; color: #2C1A0E !important; letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.76rem !important; }

/* Misc */
hr { border-color: #DDD0C0 !important; margin: 20px 0 !important; }
[data-testid="stCaptionContainer"] p { color: #3D3D3D !important; font-size: 0.77rem !important; }
[data-testid="stAlert"] {
    background: #F0E8DC !important; border: 1px solid #DDD0C0 !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] p { color: #4A2C1A !important; }
[data-testid="stExpander"] details {
    background: #F5EDE0 !important; border: 1px solid #DDD0C0 !important; border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #1A1A1A !important; font-size: 0.82rem !important; }

/* Badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #F0E8DC; border: 1px solid #DDD0C0; border-radius: 20px;
    padding: 4px 12px; font-size: 0.74rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase; color: #B05C1A;
    font-family: 'Inter', sans-serif;
}
.badge-neutral { color: #8C6E52; }
.badge-vol-low    { color: #2E7D32; background: #F1F8F1; border-color: #A5D6A7; }
.badge-vol-high   { color: #C62828; background: #FEF1F1; border-color: #FFCDD2; }

/* Signal card */
.signal-card {
    background: #FFFFFF; border: 1px solid #DDD0C0; border-radius: 12px;
    padding: 28px 32px; text-align: center;
}
.signal-card .sig-label {
    font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #8C6E52; margin-bottom: 8px;
}
.signal-card .sig-value {
    font-family: 'Lora', serif; font-size: 2.2rem; font-weight: 600;
    color: #B05C1A; margin-bottom: 4px;
}
.signal-card .sig-value.buy-arabica  { color: #2E7D32; }
.signal-card .sig-value.buy-robusta  { color: #C62828; }
.signal-card .sig-value.neutral      { color: #8C6E52; }
.signal-card .sig-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; color: #8C6E52; }
</style>
""")

# ── Plotly theme ──────────────────────────────────────────────────────────────
_BG     = "#FBF7F2"
_SURF   = "#FFFFFF"
_GRID   = "#EEE5D8"
_TXT    = "#3D2B1A"
_ACCENT = "#B05C1A"
_DARK   = "#2C1A0E"
_HIST   = "#C4A882"
_GREEN  = "#2E7D32"
_RED    = "#C62828"
_MONO   = "IBM Plex Mono, monospace"


def _base_layout(height: int = 360, title: str | None = None) -> dict:
    return dict(
        height=height,
        title=dict(text=title, font=dict(family="Lora, Georgia, serif", color=_DARK, size=14)) if title else None,
        paper_bgcolor=_SURF,
        plot_bgcolor=_SURF,
        font=dict(family="Inter, sans-serif", color=_TXT, size=11),
        margin=dict(l=0, r=0, t=36 if title else 8, b=0),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=_GRID,
                   tickfont=dict(family=_MONO, size=10, color=_TXT, weight=700)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False, linecolor=_GRID,
                   tickfont=dict(family=_MONO, size=10, color=_TXT, weight=700)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=_TXT),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_cached_conn(path: str) -> sqlite3.Connection:
    """Cache only when the DB file actually exists — keyed by path."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_conn() -> sqlite3.Connection | None:
    """Return a connection, or None if the DB doesn't exist yet.
    The None path is intentionally NOT cached so the app recovers
    automatically once the pipeline creates the DB."""
    db_path = Path(os.getenv("COFFEE_DB_PATH", "coffee.db"))
    if not db_path.exists():
        return None
    return _get_cached_conn(str(db_path))


def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a query; return empty DataFrame on any error or missing DB."""
    conn = _get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_price_history(symbol: str = "KC=F", months: int = 36) -> pd.DataFrame:
    df = _q(
        "SELECT date, adj_close FROM prices_monthly WHERE symbol = ?"
        " ORDER BY date DESC LIMIT ?",
        (symbol, months),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_latest_run() -> dict | None:
    """Return the latest successful hybrid run as a plain dict, or None."""
    df = _q(
        "SELECT * FROM model_runs WHERE model_type='hybrid' AND status='success'"
        " ORDER BY id DESC LIMIT 1"
    )
    return None if df.empty else df.iloc[0].to_dict()


@st.cache_data(ttl=300)
def load_latest_forecasts() -> pd.DataFrame:
    run = load_latest_run()
    if run is None:
        return pd.DataFrame()
    return _q(
        "SELECT horizon, symbol, forecast_date, target_date,"
        "       point_forecast, p10, p25, p50, p75, p90"
        " FROM forecasts WHERE run_id = ? ORDER BY horizon, symbol",
        (int(run["id"]),),
    )


@st.cache_data(ttl=300)
def load_backtest_results(symbol: str = "KC=F") -> pd.DataFrame:
    # Filter to the latest successful backtest run so re-running the backtest
    # replaces the display rather than stacking duplicate data points.
    df = _q(
        "SELECT train_end, target_date, horizon, actual,"
        "       point_forecast, p10, p25, p50, p75, p90"
        " FROM backtest_results"
        " WHERE symbol = ? AND actual IS NOT NULL"
        "   AND run_id = ("
        "       SELECT id FROM model_runs"
        "       WHERE model_type = 'backtest' AND status = 'success'"
        "       ORDER BY id DESC LIMIT 1"
        "   )"
        " ORDER BY target_date, horizon",
        (symbol,),
    )
    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data(ttl=300)
def load_accuracy_log(symbol: str = "KC=F") -> pd.DataFrame:
    # Join through backtest_results to anchor on the latest successful backtest run,
    # preventing duplicate metric rows from accumulating across multiple backtest runs.
    df = _q(
        "SELECT al.horizon, al.mae, al.mape, al.pinball_50, al.coverage_80,"
        "       f.forecast_date, f.target_date"
        " FROM accuracy_log al"
        " LEFT JOIN forecasts f ON al.forecast_id = f.id"
        " INNER JOIN backtest_results br ON al.forecast_id = br.forecast_id"
        "   AND br.run_id = ("
        "       SELECT id FROM model_runs"
        "       WHERE model_type = 'backtest' AND status = 'success'"
        "       ORDER BY id DESC LIMIT 1"
        "   )"
        " WHERE al.symbol = ? ORDER BY f.target_date",
        (symbol,),
    )
    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data(ttl=300)
def load_spread_signals(months: int = 60) -> pd.DataFrame:
    df = _q(
        "SELECT date, spread, z_score, signal, half_life"
        " FROM spread_signals ORDER BY date DESC LIMIT ?",
        (months,),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_latest_ingest_date() -> str:
    df = _q("SELECT MAX(date) AS d FROM prices_monthly")
    return df.iloc[0]["d"] if not df.empty and df.iloc[0]["d"] else "—"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _no_data(msg: str = "No model runs found yet. Run the pipeline first.") -> None:
    st.info(f"📭  {msg}")


def _fmt_price(v: float | None) -> str:
    return f"${v:.2f}" if v is not None and not np.isnan(v) else "—"


def _delta_str(forecast: float | None, last_actual: float | None) -> str:
    if forecast is None or last_actual is None:
        return ""
    d = forecast - last_actual
    sign = "+" if d >= 0 else "-"
    return f"{sign}${abs(d):.2f} vs last actual"


# ── Sidebar ───────────────────────────────────────────────────────────────────
run_info = load_latest_run()
data_through = load_latest_ingest_date()

with st.sidebar:
    st.markdown("### ☕ Coffee Price Forecast")
    st.markdown(
        "I drink a lot of coffee and got curious about what actually drives the price of it. "
        "This is a live statistical model — built to make real forecasts, be wrong in public, "
        "and improve over time."
    )
    st.html("<hr>")
    st.markdown("**Model status**")
    if run_info is not None:
        st.caption(f"Last run: **{str(run_info['run_at'])[:10]}**")
        st.caption(f"Status: ✅ **{run_info['status']}**")
    else:
        st.caption("Last run: **no runs yet**")
    st.caption(f"Data through: **{data_through}**")
    st.caption("Reruns on the **1st** of each month")
    st.html("<hr>")
    st.markdown("**Model**")
    st.caption("VECM + GAMLSS hybrid")
    st.caption("Arabica KC=F (primary)")
    st.caption("Robusta RM=F (secondary)")
    st.caption("FX drivers: BRL · VND · IDR · DXY")
    st.html("<hr>")
    st.markdown("**About**")
    st.caption("A personal project applying statistical methods to a real market I follow closely.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Current Forecast",
    "Track Record",
    "Live Accuracy",
    "Spread Signal",
    "Methodology",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · CURRENT FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Arabica Price Forecast")

    forecasts = load_latest_forecasts()
    price_hist = load_price_history("KC=F", months=36)

    if forecasts.empty:
        _no_data()
    else:
        kc = forecasts[forecasts["symbol"] == "KC=F"].sort_values("horizon")
        last_actual = float(price_hist["adj_close"].iloc[-1]) if not price_hist.empty else None

        # Volatility regime badge (derive from run params if present)
        regime_label = "—"
        if run_info is not None and run_info.get("params"):
            try:
                params = json.loads(run_info["params"])
                regime_label = params.get("regime", "—")
            except Exception:
                pass

        vol_class = {
            "Low": "badge-vol-low", "High": "badge-vol-high"
        }.get(regime_label, "badge-neutral")

        st.html(
            f'<span class="badge">KC=F · USD / lb</span>&nbsp;'
            f'<span class="badge {vol_class}">⚡ {regime_label} Volatility</span>'
        )
        st.html("<br>")

        # Hero metrics
        cols = st.columns(3)
        for col, (_, row) in zip(cols, kc.iterrows()):
            h = int(row["horizon"])
            p50 = row["p50"] if row["p50"] is not None else row["point_forecast"]
            col.metric(
                f"{h}-Month",
                _fmt_price(p50),
                _delta_str(p50, last_actual),
            )

        st.html("<br>")

        # Forecast chart
        fig = go.Figure()

        if not price_hist.empty:
            fig.add_trace(go.Scatter(
                x=price_hist["date"], y=price_hist["adj_close"],
                mode="lines", name="Historical (KC=F)",
                line=dict(color=_HIST, width=2),
            ))

        # Build forecast points
        fc_rows = kc.to_dict("records")
        fc_dates_list = [pd.to_datetime(r["target_date"]) for r in fc_rows]
        p50_vals = [r["p50"] if r["p50"] is not None else r["point_forecast"] for r in fc_rows]
        p10_vals = [r["p10"] for r in fc_rows]
        p90_vals = [r["p90"] for r in fc_rows]

        has_band = any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in p10_vals)
        null_horizons = [i + 1 for i, v in enumerate(p10_vals) if v is None or (isinstance(v, float) and np.isnan(v))]
        if has_band:
            p10_clean = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else p50_vals[i] for i, v in enumerate(p10_vals)]
            p90_clean = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else p50_vals[i] for i, v in enumerate(p90_vals)]
            # Anchor band to last actual so it connects with the history line
            anchor_date = price_hist["date"].iloc[-1] if not price_hist.empty else fc_dates_list[0]
            anchor_val = float(price_hist["adj_close"].iloc[-1]) if not price_hist.empty else p50_vals[0]
            band_x = [anchor_date] + fc_dates_list + list(reversed(fc_dates_list)) + [anchor_date]
            band_y = [anchor_val] + p90_clean + list(reversed(p10_clean)) + [anchor_val]
            fig.add_trace(go.Scatter(
                x=band_x, y=band_y,
                mode="lines",
                fill="toself", fillcolor="rgba(176,92,26,0.18)",
                line=dict(width=0), name="80% interval", hoverinfo="skip",
            ))
        if null_horizons:
            st.caption(f"⚠️ 80% interval unavailable for h={null_horizons} — GAMLSS did not converge for those horizons.")

        # Connect forecast line to last actual so there is no visual gap
        if not price_hist.empty:
            fc_x = [price_hist["date"].iloc[-1]] + fc_dates_list
            fc_y = [float(price_hist["adj_close"].iloc[-1])] + p50_vals
        else:
            fc_x, fc_y = fc_dates_list, p50_vals

        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines", name="Forecast (p50)",
            line=dict(color=_ACCENT, width=2.5, dash="solid"),
        ))

        if not price_hist.empty:
            last_data_str = price_hist["date"].iloc[-1].strftime("%Y-%m-%d")
            fig.add_vline(x=last_data_str, line_dash="dot", line_color=_HIST, line_width=1)

        # Mark today's date if it's in the forecast period
        today = pd.Timestamp.today()
        if fc_dates_list and today < fc_dates_list[-1]:
            fig.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
            fig.add_annotation(x=today, yref="paper", y=-0.08,
                              text="TODAY", showarrow=False,
                              font=dict(family=_MONO, size=8, color="#E74C3C"),
                              xanchor="center", yshift=-5)

        # "FORECAST PERIOD →" label centred in the forecast region
        if fc_dates_list and len(fc_dates_list) >= 2:
            mid_idx = len(fc_dates_list) // 2
            fig.add_annotation(
                x=fc_dates_list[mid_idx], y=1.0, yref="paper",
                text="FORECAST PERIOD →",
                showarrow=False,
                font=dict(family=_MONO, size=9, color=_TXT),
                xanchor="center", yshift=10,
                bgcolor="rgba(255,255,255,0.7)", borderpad=3,
            )

        layout1 = _base_layout(400, "Arabica Coffee Futures — 3-Month Price Forecast (KC=F)")
        layout1["yaxis"]["title"] = dict(text="USD / lb", font=dict(family=_MONO, size=10, color=_TXT))
        layout1["margin"]["b"] = 48
        layout1["legend"] = dict(
            bgcolor="rgba(255,255,255,0.88)", bordercolor=_GRID, borderwidth=1,
            font=dict(size=10, color=_TXT, family="Inter, sans-serif"),
            orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0,
        )
        fig.update_layout(**layout1)
        st.plotly_chart(fig, use_container_width=True)

        # Performance summary (from accuracy_log)
        acc = load_accuracy_log("KC=F")
        st.divider()
        st.markdown("### Model performance summary")
        if acc.empty:
            st.caption("No backtest accuracy data yet.")
        else:
            def _metric_for_horizon(h: int, col_name: str) -> str:
                sub = acc[acc["horizon"] == h][col_name].dropna()
                return f"{sub.mean():.3f}" if not sub.empty else "—"

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("MAE · h=1", f"${_metric_for_horizon(1, 'mae')}")
            p2.metric("MAE · h=2", f"${_metric_for_horizon(2, 'mae')}")
            p3.metric("MAE · h=3", f"${_metric_for_horizon(3, 'mae')}")
            cov_vals = acc["coverage_80"].dropna()
            cov_pct = f"{cov_vals.mean() * 100:.0f}%" if not cov_vals.empty else "—"
            p4.metric("80% Coverage", cov_pct,
                      help="% of actuals that fell within the p10–p90 forecast band")
            st.caption("Based on walk-forward backtest · Full detail in Track Record →")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · TRACK RECORD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Walk-Forward Track Record")
    st.caption("Each point is an out-of-sample forecast made at the time — no look-ahead.")
    st.html("<br>")

    bt = load_backtest_results("KC=F")

    if bt.empty:
        _no_data("No backtest results yet. Run the backtest pipeline first.")
    else:
        _mae_help = (
            "Mean Absolute Error — the average dollar distance between the forecast "
            "and the actual price across all backtest windows. Lower is better."
        )
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        for col, h, label in [(m1, 1, "MAE h=1"), (m2, 2, "MAE h=2"), (m3, 3, "MAE h=3")]:
            sub = bt[bt["horizon"] == h]
            forecast_col = sub["p50"] if "p50" in sub.columns and sub["p50"].notna().any() else sub["point_forecast"]
            err = (sub["actual"] - forecast_col).abs()
            col.metric(label, f"${err.mean():.3f}" if not err.empty else "—", help=_mae_help)
        acc_all = load_accuracy_log("KC=F")
        cov_vals = acc_all["coverage_80"].dropna() if not acc_all.empty else pd.Series([], dtype=float)
        cov_str = f"{cov_vals.mean() * 100:.0f}%" if not cov_vals.empty else "—"
        target_delta = f"{(cov_vals.mean() - 0.80) * 100:+.0f}pp vs 80% target" if not cov_vals.empty else ""
        m4.metric("80% Coverage", cov_str, target_delta,
                  help=(
                      "Of all the backtest forecasts, what share of actual prices landed inside "
                      "the shaded band (p10–p90)? A well-calibrated model hits 80%. "
                      "Below 80% means the bands are too narrow — the model is overconfident."
                  ))

        st.html("<br>")

        def _chart_layout(height: int, title: str, ytitle: str) -> dict:
            lay = _base_layout(height, title)
            lay["yaxis"]["title"] = dict(text=ytitle, font=dict(family=_MONO, size=10, color=_TXT))
            lay["margin"]["b"] = 48
            lay["legend"] = dict(
                bgcolor="rgba(255,255,255,0.88)", bordercolor=_GRID, borderwidth=1,
                font=dict(size=10, color=_TXT, family="Inter, sans-serif"),
                orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
            )
            return lay

        # Actuals vs h=1 forecasts
        st.html("""
        <p style="font-size:0.95rem; color:#1A1A1A; margin:0 0 16px 0; line-height:1.7;">
          Did the model track the actual price? The <strong style="color:#2563EB">blue line</strong> is what actually happened;
          the <strong style="color:#DC2626">red line</strong> is what the model predicted one month in advance.
          The shaded band is the <strong>80% confidence interval</strong> — the model expected
          the price to land inside it 80% of the time.
        </p>
        """)
        bt1 = bt[bt["horizon"] == 1].sort_values("target_date")
        fig2 = go.Figure()
        if bt1["p10"].notna().any():
            fig2.add_trace(go.Scatter(
                x=bt1["target_date"], y=bt1["p90"],
                mode="lines", fill=None, line=dict(width=0),
                name="80% interval", showlegend=True,
            ))
            fig2.add_trace(go.Scatter(
                x=bt1["target_date"], y=bt1["p10"],
                mode="lines", fill="tonexty", line=dict(width=0),
                fillcolor="rgba(176,92,26,0.18)", showlegend=False,
            ))
        fig2.add_trace(go.Scatter(
            x=bt1["target_date"], y=bt1["actual"],
            mode="lines", name="Actual price",
            line=dict(color="#2563EB", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=bt1["target_date"], y=bt1["point_forecast"],
            mode="lines", name="Forecast (1-month ahead)",
            line=dict(color="#DC2626", width=2),
        ))
        # Mark today
        today = pd.Timestamp.today()
        if not bt1.empty and today >= bt1["target_date"].min() and today <= bt1["target_date"].max():
            fig2.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
            fig2.add_annotation(x=today, yref="paper", y=-0.08,
                               text="TODAY", showarrow=False,
                               font=dict(family=_MONO, size=8, color="#E74C3C"),
                               xanchor="center", yshift=-5)
        fig2.update_layout(**_chart_layout(360, "Actual vs Forecast — 1-Month Ahead (KC=F)", "USD / lb"))
        st.plotly_chart(fig2, use_container_width=True)

        # Error over time by horizon
        st.html("""
        <p style="font-size:0.95rem; color:#1A1A1A; margin:32px 0 16px 0; line-height:1.7;">
          How wrong was the model, in dollar terms? The line shows the gap between the
          one-month-ahead forecast and the actual price each month.
          <strong>Spikes coincide with sudden market moves</strong> — droughts, crop disease,
          macro shocks — that a statistical model cannot anticipate in advance.
        </p>
        """)
        fig3 = go.Figure()
        cutoff = pd.Timestamp("2024-01-01")
        sub1 = bt[(bt["horizon"] == 1) & (bt["target_date"] >= cutoff)].sort_values("target_date")
        if not sub1.empty:
            err1 = (sub1["actual"] - sub1["point_forecast"]).abs()
            fig3.add_trace(go.Scatter(
                x=sub1["target_date"], y=err1,
                mode="lines", name="1-month ahead",
                line=dict(color=_ACCENT, width=2),
                showlegend=False,
            ))
        # Mark today
        today = pd.Timestamp.today()
        if not sub1.empty and today >= sub1["target_date"].min() and today <= sub1["target_date"].max():
            fig3.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
            fig3.add_annotation(x=today, yref="paper", y=-0.08,
                               text="TODAY", showarrow=False,
                               font=dict(family=_MONO, size=8, color="#E74C3C"),
                               xanchor="center", yshift=-5)
        lay3 = _chart_layout(260, "1-Month Forecast Error — 2024 to Present", "Error (USD / lb)")
        lay3["margin"]["b"] = 16  # no legend, so reclaim that space
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("View raw backtest results"):
            display_cols = ["target_date", "horizon", "actual", "point_forecast", "p10", "p50", "p90"]
            display_cols = [c for c in display_cols if c in bt.columns]
            st.dataframe(
                bt[display_cols].rename(columns={"target_date": "Date", "point_forecast": "Forecast"})
                  .round(3),
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · LIVE ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Live Accuracy Log")
    st.caption("Real forecasts verified against actual prices as each month closes.")
    st.html("<br>")

    acc = load_accuracy_log("KC=F")

    if acc.empty:
        _no_data("No accuracy data yet — runs automatically after each monthly close.")
    else:
        acc1 = acc[acc["horizon"] == 1].sort_values("target_date")
        last_12 = acc1.tail(12)

        a1, a2, a3 = st.columns(3)
        mae_12 = last_12["mae"].dropna()
        mape_12 = last_12["mape"].dropna()
        cov_12  = last_12["coverage_80"].dropna()
        a1.metric("Avg MAE (12m)", f"${mae_12.mean():.3f}" if not mae_12.empty else "—")
        a2.metric("Avg MAPE (12m)", f"{mape_12.mean():.1f}%" if not mape_12.empty else "—")
        a3.metric("80% Coverage (12m)",
                  f"{cov_12.mean() * 100:.0f}%" if not cov_12.empty else "—")

        st.html("<br>")

        # MAE over time
        fig4 = go.Figure()
        mae_all = acc1["mae"].dropna()
        if not mae_all.empty:
            fig4.add_trace(go.Scatter(
                x=acc1["target_date"], y=acc1["mae"],
                mode="lines+markers", name="MAE h=1",
                line=dict(color=_ACCENT, width=2),
                marker=dict(size=6, color=_ACCENT),
            ))
            fig4.add_hline(y=float(mae_all.mean()), line_dash="dash",
                           line_color=_HIST, line_width=1,
                           annotation_text="avg",
                           annotation_font=dict(family=_MONO, size=9))
            # Mark today
            today = pd.Timestamp.today()
            if today >= acc1["target_date"].min() and today <= acc1["target_date"].max():
                fig4.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
        fig4.update_layout(**_base_layout(220, "Monthly MAE — h=1 Forecasts (USD/lb)"))
        st.plotly_chart(fig4, use_container_width=True)

        # Coverage bar chart
        cov_data = acc1["coverage_80"].dropna()
        if not cov_data.empty:
            cov_dates = acc1.loc[cov_data.index, "target_date"]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=cov_dates, y=cov_data * 100,
                name="80% coverage",
                marker_color=[_GREEN if v >= 0.75 else _RED for v in cov_data],
                marker_line_width=0,
            ))
            fig5.add_hline(y=80, line_dash="dash", line_color=_HIST, line_width=1,
                           annotation_text="80% target",
                           annotation_font=dict(family=_MONO, size=9))
            # Mark today
            today = pd.Timestamp.today()
            if today >= cov_dates.min() and today <= cov_dates.max():
                fig5.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
            fig5.update_layout(**_base_layout(200, "Monthly 80% Interval Coverage (%)"))
            st.plotly_chart(fig5, use_container_width=True)

        with st.expander("View full accuracy log"):
            cols_present = [c for c in ["target_date", "horizon", "mae", "mape", "pinball_50", "coverage_80"] if c in acc.columns]
            disp = acc[cols_present].copy()
            disp["mape"] = disp["mape"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
            if "pinball_50" in disp.columns:
                disp["pinball_50"] = disp["pinball_50"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
            disp["coverage_80"] = disp["coverage_80"].map(
                lambda v: ("✅ Yes" if v == 1 else "❌ No") if pd.notna(v) else "—"
            )
            disp.rename(columns={
                "target_date": "Month", "horizon": "h",
                "mae": "MAE", "mape": "MAPE",
                "pinball_50": "Pinball (p50)", "coverage_80": "In 80% Band",
            }, inplace=True)
            st.dataframe(disp.sort_values("Month", ascending=False),
                         use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · SPREAD SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## Arabica / Robusta Spread Signal")
    st.caption("Does Arabica look cheap or expensive relative to Robusta right now?")
    st.html("<br>")

    spread = load_spread_signals(months=60)

    if spread.empty:
        _no_data("No spread signal data yet. Run the spread model first.")
    else:
        cur = spread.iloc[-1]
        cur_z   = float(cur["z_score"])
        cur_sig = int(cur["signal"]) if pd.notna(cur["signal"]) else 0
        half_life = cur["half_life"]

        if cur_sig == -1:
            sig_label = "Buy Arabica"
            sig_class = "buy-arabica"
            sig_desc = (
                "Arabica is historically <strong>cheap</strong> vs Robusta. "
                "The spread tends to mean-revert upward — a potential long Arabica edge."
            )
        elif cur_sig == 1:
            sig_label = "Buy Robusta"
            sig_class = "buy-robusta"
            sig_desc = (
                "Arabica is historically <strong>expensive</strong> vs Robusta. "
                "The spread tends to mean-revert downward — a potential long Robusta edge."
            )
        else:
            sig_label = "Neutral"
            sig_class = "neutral"
            sig_desc = (
                "No strong edge detected. The Arabica/Robusta price ratio is within its "
                "normal historical range — no mean-reversion trade indicated."
            )

        col_sig, col_detail = st.columns([1, 2])
        with col_sig:
            hl_str = f"{half_life:.1f} months" if pd.notna(half_life) else "—"
            st.html(f"""
            <div class="signal-card">
                <div class="sig-label">Current Signal</div>
                <div class="sig-value {sig_class}">{sig_label}</div>
                <div class="sig-sub">z = {cur_z:+.2f} · half-life ≈ {hl_str}</div>
            </div>
            """)

        with col_detail:
            st.markdown("#### What does this mean?")
            st.html(f"""
            <p>{sig_desc}</p>
            <p>The signal tracks the <strong>price ratio</strong> between Arabica and Robusta
            vs its long-run average. When the ratio drifts too far, it historically pulls back
            — a <em>mean-reversion</em> pattern confirmed by cointegration testing (p&nbsp;=&nbsp;0.009).</p>
            <p>The <strong>z-score</strong> measures how far the ratio is from its historical
            average, in units of standard deviations. A z-score of 0 means perfectly normal;
            +2 means Arabica is unusually expensive vs Robusta; −2 means unusually cheap.</p>
            <p><strong>Entry at |z| &gt; 2.0</strong> — a signal only fires when the ratio is more
            than 2 standard deviations from normal, which historically happens ~5% of the time
            and tends to revert. <strong>Exit at |z| &lt; 0.5</strong> — the trade closes once the
            ratio returns close to its average, rather than waiting for a full reversal which may
            never come. <strong>Half-life ({hl_str})</strong> is how long mean-reversion
            typically takes — a shorter half-life means a faster snap-back.</p>
            """)

        st.html("<br>")

        fig6 = go.Figure()
        fig6.add_hrect(y0=2, y1=spread["z_score"].max() + 0.5,
                       fillcolor="rgba(198,40,40,0.06)", line_width=0)
        fig6.add_hrect(y0=spread["z_score"].min() - 0.5, y1=-2,
                       fillcolor="rgba(46,125,50,0.06)", line_width=0)
        fig6.add_hline(y=2,    line_dash="dot", line_color=_RED,   line_width=1)
        fig6.add_hline(y=-2,   line_dash="dot", line_color=_GREEN, line_width=1)
        fig6.add_hline(y=0.5,  line_dash="dot", line_color=_HIST,  line_width=1)
        fig6.add_hline(y=-0.5, line_dash="dot", line_color=_HIST,  line_width=1)
        fig6.add_trace(go.Scatter(
            x=spread["date"], y=spread["z_score"],
            mode="lines", name="Spread z-score",
            line=dict(color=_ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(176,92,26,0.07)",
        ))
        fig6.add_annotation(
            x=spread["date"].iloc[-1], y=cur_z,
            text=f"  Now: {cur_z:+.2f}",
            showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=_ACCENT,
            ax=40, ay=-25,
            font=dict(family=_MONO, size=13, color=_ACCENT, weight=700),
        )
        today = pd.Timestamp.today()
        if today >= spread["date"].min() and today <= spread["date"].max():
            fig6.add_vline(x=today, line_dash="solid", line_color="#E74C3C", line_width=1.5)
        layout6 = _base_layout(340, "Arabica vs Robusta — Spread Z-Score")
        layout6["yaxis"]["title"] = dict(
            text="Z-Score",
            font=dict(family=_MONO, size=12, color="#000000", weight=700),
        )
        layout6["yaxis"]["dtick"] = 1
        layout6["yaxis"]["showgrid"] = True
        layout6["yaxis"]["gridcolor"] = "rgba(0,0,0,0.06)"
        fig6.update_layout(**layout6)
        st.plotly_chart(fig6, use_container_width=True)
        st.html(
            '<p style="font-size:13px; color:#000000; margin-top:-8px;">'
            "Red zone = Arabica expensive (z &gt; 2) · Green zone = Arabica cheap (z &lt; −2) "
            "· Dotted lines = exit bands (±0.5)</p>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## Methodology")
    st.caption("A plain-English guide to the data, the model, and what the numbers mean.")
    st.html("<br>")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Data sources")
        st.html("""
        <p>Prices are ingested monthly from two sources:</p>
        <ul>
          <li><strong>Arabica (KC=F)</strong> and <strong>Robusta (RM=F)</strong>
              coffee prices, plus the <strong>US Dollar index (DXY)</strong> —
              via <strong>FRED</strong> (Federal Reserve Economic Data), the St. Louis Fed's
              free public database of economic and financial time series. No API key required.</li>
          <li><strong>FX rates</strong> — Brazilian Real (BRL), Vietnamese Dong (VND),
              Indonesian Rupiah (IDR) — via <strong>Alpha Vantage</strong>, a financial data
              provider with a free tier. These currencies matter because Brazil, Vietnam, and
              Indonesia are the world's three largest coffee producers: when their currencies
              weaken against the dollar, local farmers receive less per bag sold, which affects
              supply and ultimately global prices.</li>
        </ul>
        <p>Training data runs from 2014 to present (~136 monthly observations).
        History before 2014 is excluded because VND and IDR series only start then.</p>
        """)

        st.markdown("#### The VECM model")
        st.html("""
        <p>A <strong>Vector Error Correction Model (VECM)</strong> captures the
        long-run cointegration between Arabica and Robusta — they tend to drift back
        together when their price ratio gets too extreme.</p>
        <p>Currency rates enter as <em>external drivers</em>: a weaker Brazilian Real
        makes Brazilian growers produce more Arabica for the same local income, which
        pushes global dollar prices down. The VECM produces a
        <strong>point forecast</strong> for 1, 2, and 3 months ahead.</p>
        """)

    with col_b:
        st.markdown("#### Uncertainty — the GAMLSS layer")
        st.html("""
        <p>A second model (GAMLSS with a SHASH distribution) is fitted to the
        VECM's historical forecast errors. This gives the <strong>probability
        band</strong> around each point forecast — the p10 to p90 range you see
        in the chart.</p>
        <p>The band width depends on the current <strong>volatility regime</strong>
        (Low / Medium / High), estimated from recent price swings. In high-volatility
        periods the band widens; in calm periods it narrows.</p>
        """)

        st.markdown("#### Known limitations")
        st.html("""
        <ul>
          <li><strong>Cointegration rank hardcoded at r=1</strong> — confirmed by testing
              (Engle-Granger p=0.009) but not re-tested on rolling windows. If the
              Arabica/Robusta relationship structurally breaks, the model won't detect it.</li>
          <li><strong>FX forecasts are naïve</strong> — currencies are held flat at
              their last observed value. A big BRL move would not be captured for h=2, h=3.</li>
          <li><strong>Lag order auto-selected but not bounded</strong> — AIC chose lag=1
              on the current dataset. Shocks that play out over 3–4 months (droughts,
              policy changes) may not be captured with a single lag.</li>
          <li><strong>Linear, time-invariant coefficients</strong> — the model doesn't
              auto-detect structural breaks like a major crop disease or policy shock.</li>
          <li><strong>GAMLSS assumes well-behaved residuals</strong> — if the VECM is
              misspecified, those errors carry through into the distribution model.
              Residual ACF and ARCH tests have not been run end-to-end.</li>
          <li><strong>Interval coverage ≈ 77% vs 80% target</strong> — bands are slightly
              too tight out-of-sample. Expected given GAMLSS is fit on in-sample residuals.</li>
          <li><strong>Training window limited to 2014+</strong> by VND/IDR availability.
              Dropping those FX drivers would unlock 15 more years of data.</li>
        </ul>
        """)

    st.divider()
    st.markdown("#### About this project")
    st.html("""
    <p>A personal project applying rigorous statistical methods to a real market.
    The full pipeline — ingestion, VECM, GAMLSS, backtest, and this dashboard — reruns
    automatically on the 1st of each month via GitHub Actions, making every forecast
    publicly falsifiable.</p>
    <p style="color:#8C6E52; font-size:0.8rem; margin-top:8px">
    Stack: Python · statsmodels VECM · R / GAMLSS · SQLite · Streamlit · Plotly ·
    GitHub Actions
    </p>
    """)
