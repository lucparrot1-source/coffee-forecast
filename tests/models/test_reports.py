import sqlite3

import pandas as pd
import pytest


@pytest.fixture
def conn_with_forecasts():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE model_runs (
            id INTEGER PRIMARY KEY, run_at TEXT NOT NULL DEFAULT '',
            model_type TEXT NOT NULL, train_start TEXT NOT NULL DEFAULT '',
            train_end TEXT NOT NULL DEFAULT '', params TEXT, metrics TEXT,
            status TEXT NOT NULL DEFAULT 'pending', notes TEXT
        );
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY, run_id INTEGER, forecast_date TEXT,
            target_date TEXT, horizon INTEGER, symbol TEXT,
            point_forecast REAL, p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL
        );
    """)
    conn.execute(
        "INSERT INTO model_runs (id, model_type, status) VALUES (1, 'hybrid', 'success')"
    )
    rows = [
        (1, 1, "2026-05-01", "2026-06-01", 1, "KC=F", 200.0, 180.0, 190.0, 200.0, 210.0, 220.0),
        (2, 1, "2026-05-01", "2026-07-01", 2, "KC=F", 202.0, 182.0, 192.0, 202.0, 212.0, 222.0),
        (3, 1, "2026-05-01", "2026-08-01", 3, "KC=F", 205.0, 185.0, 195.0, 205.0, 215.0, 225.0),
        (4, 1, "2026-05-01", "2026-06-01", 1, "RM=F", 100.0,  90.0,  95.0, 100.0, 105.0, 110.0),
        (5, 1, "2026-05-01", "2026-07-01", 2, "RM=F", 101.0,  91.0,  96.0, 101.0, 106.0, 111.0),
        (6, 1, "2026-05-01", "2026-08-01", 3, "RM=F", 103.0,  93.0,  98.0, 103.0, 108.0, 113.0),
    ]
    conn.executemany("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def test_fetch_latest_forecasts_returns_6_rows(conn_with_forecasts):
    from coffee_forecast.reports import fetch_latest_forecasts
    df = fetch_latest_forecasts(conn_with_forecasts)
    assert len(df) == 6


def test_fetch_latest_forecasts_has_required_columns(conn_with_forecasts):
    from coffee_forecast.reports import fetch_latest_forecasts
    df = fetch_latest_forecasts(conn_with_forecasts)
    for col in ["horizon", "symbol", "point_forecast", "p10", "p25", "p50", "p75", "p90"]:
        assert col in df.columns


def test_fetch_latest_forecasts_empty_when_no_run():
    from coffee_forecast.reports import fetch_latest_forecasts
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE model_runs (id INTEGER PRIMARY KEY, model_type TEXT, status TEXT);
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY, run_id INTEGER, horizon INTEGER, symbol TEXT,
            point_forecast REAL, p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL
        );
    """)
    df = fetch_latest_forecasts(conn)
    assert df.empty


@pytest.fixture
def conn_with_all(conn_with_forecasts):
    conn_with_forecasts.executescript("""
        CREATE TABLE prices_monthly (
            id INTEGER PRIMARY KEY, date TEXT, symbol TEXT, adj_close REAL,
            UNIQUE(date, symbol)
        );
        CREATE TABLE accuracy_log (
            id INTEGER PRIMARY KEY, logged_at TEXT NOT NULL DEFAULT '',
            forecast_id INTEGER NOT NULL DEFAULT 0, actual REAL NOT NULL DEFAULT 0.0,
            horizon INTEGER, symbol TEXT, mae REAL, mape REAL,
            pinball_50 REAL, coverage_80 INTEGER
        );
        CREATE TABLE spread_signals (
            id INTEGER PRIMARY KEY, date TEXT UNIQUE,
            spread REAL, z_score REAL, signal INTEGER, half_life REAL
        );
    """)
    conn_with_forecasts.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?,?,?)",
        [
            ("2026-05-01", "KC=F", 195.0),
            ("2026-05-01", "RM=F",  98.0),
        ],
    )
    conn_with_forecasts.executemany(
        "INSERT INTO accuracy_log (id, forecast_id, actual, horizon, symbol, mae, mape, "
        "coverage_80) VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 1, 200.0, 1, "KC=F", 5.0, 0.025, 1),
            (2, 2, 202.0, 2, "KC=F", 6.0, 0.030, 1),
            (3, 3, 100.0, 1, "RM=F", 3.0, 0.030, 0),
        ],
    )
    conn_with_forecasts.execute(
        "INSERT INTO spread_signals (date, spread, z_score, signal, half_life) "
        "VALUES ('2026-05-01', 0.7, 1.8, 0, 6.2)"
    )
    conn_with_forecasts.commit()
    return conn_with_forecasts


def test_fetch_last_prices_returns_both_symbols(conn_with_all):
    from coffee_forecast.reports import fetch_last_prices
    result = fetch_last_prices(conn_with_all)
    assert set(result.keys()) >= {"KC=F", "RM=F"}


def test_fetch_last_prices_values(conn_with_all):
    from coffee_forecast.reports import fetch_last_prices
    result = fetch_last_prices(conn_with_all)
    assert result["KC=F"] == pytest.approx(195.0)
    assert result["RM=F"] == pytest.approx(98.0)


def test_fetch_latest_backtest_metrics_columns(conn_with_all):
    from coffee_forecast.reports import fetch_latest_backtest_metrics
    df = fetch_latest_backtest_metrics(conn_with_all)
    for col in ["symbol", "horizon", "mae", "mape", "coverage_80"]:
        assert col in df.columns


def test_fetch_latest_backtest_metrics_empty_when_no_data():
    from coffee_forecast.reports import fetch_latest_backtest_metrics
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE accuracy_log ("
        "id INTEGER PRIMARY KEY, mae REAL, mape REAL, "
        "coverage_80 INTEGER, horizon INTEGER, symbol TEXT)"
    )
    assert fetch_latest_backtest_metrics(conn).empty


def test_fetch_spread_state_keys(conn_with_all):
    from coffee_forecast.reports import fetch_spread_state
    state = fetch_spread_state(conn_with_all)
    for key in ["date", "spread", "z_score", "signal", "half_life"]:
        assert key in state


def test_fetch_spread_state_empty_when_no_data():
    from coffee_forecast.reports import fetch_spread_state
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE spread_signals ("
        "id INTEGER PRIMARY KEY, date TEXT, spread REAL, "
        "z_score REAL, signal INTEGER, half_life REAL)"
    )
    assert fetch_spread_state(conn) == {}


def test_build_forecast_chart_returns_figure(conn_with_all):
    from coffee_forecast.reports import (
        build_forecast_chart,
        fetch_last_prices,
        fetch_latest_forecasts,
    )
    forecasts_df = fetch_latest_forecasts(conn_with_all)
    last_prices = fetch_last_prices(conn_with_all)
    fig = build_forecast_chart(forecasts_df, last_prices)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_build_forecast_chart_empty_forecasts_returns_figure():
    import matplotlib.figure

    from coffee_forecast.reports import build_forecast_chart

    fig = build_forecast_chart(pd.DataFrame(), {})
    assert isinstance(fig, matplotlib.figure.Figure)


def test_render_monthly_report_raises_on_bad_quarto(tmp_path, monkeypatch):
    """Quarto subprocess failure should raise RuntimeError, not swallow the error."""
    import subprocess

    from coffee_forecast.reports import render_monthly_report

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="fake quarto error"
        )

    monkeypatch.setattr("coffee_forecast.reports.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="quarto render failed"):
        render_monthly_report("2026-05", "data/coffee.db", tmp_path)


def test_render_monthly_report_returns_path_on_success(tmp_path, monkeypatch):
    import subprocess

    from coffee_forecast.reports import render_monthly_report

    def fake_run(*args, **kwargs):
        # Simulate quarto creating the output file
        out = tmp_path / "2026-05.pdf"
        out.write_bytes(b"%PDF-fake")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("coffee_forecast.reports.subprocess.run", fake_run)
    result = render_monthly_report("2026-05", "data/coffee.db", tmp_path)
    assert result == (tmp_path / "2026-05.pdf").resolve()


def test_generate_analysis_returns_non_empty_string(conn_with_all, monkeypatch):
    """generate_analysis returns a non-empty string (mocks the API call)."""
    from coffee_forecast.reports import (
        fetch_latest_forecasts, fetch_last_prices,
        fetch_latest_backtest_metrics, fetch_spread_state,
        generate_analysis,
    )
    import anthropic

    class FakeMessage:
        content = [type("Block", (), {"text": "Arabica prices rose this month."})()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeMessage()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeClient())

    forecasts_df  = fetch_latest_forecasts(conn_with_all)
    last_prices   = fetch_last_prices(conn_with_all)
    backtest_df   = fetch_latest_backtest_metrics(conn_with_all)
    spread_state  = fetch_spread_state(conn_with_all)

    result = generate_analysis(forecasts_df, last_prices, backtest_df, spread_state)
    assert isinstance(result, str)
    assert len(result) > 10


def test_generate_analysis_returns_fallback_on_missing_key(conn_with_all, monkeypatch):
    """If ANTHROPIC_API_KEY is absent, function returns a fallback string without raising."""
    from coffee_forecast.reports import (
        fetch_latest_forecasts, fetch_last_prices,
        fetch_latest_backtest_metrics, fetch_spread_state,
        generate_analysis,
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import anthropic
    import httpx

    def raise_auth(*args, **kwargs):
        raise anthropic.AuthenticationError(
            message="No API key",
            response=httpx.Response(401, text="Unauthorized"),
            body=None,
        )

    monkeypatch.setattr(anthropic, "Anthropic", raise_auth)

    forecasts_df  = fetch_latest_forecasts(conn_with_all)
    last_prices   = fetch_last_prices(conn_with_all)
    backtest_df   = fetch_latest_backtest_metrics(conn_with_all)
    spread_state  = fetch_spread_state(conn_with_all)

    result = generate_analysis(forecasts_df, last_prices, backtest_df, spread_state)
    assert isinstance(result, str)
    assert "unavailable" in result.lower()


# ── _build_success_email_body ──────────────────────────────────────────────

_SAMPLE_FORECASTS = pd.DataFrame([
    {"symbol": "KC=F", "horizon": 1, "p50": 200.0},
    {"symbol": "KC=F", "horizon": 2, "p50": 202.0},
    {"symbol": "KC=F", "horizon": 3, "p50": 205.0},
    {"symbol": "RM=F", "horizon": 1, "p50": 100.0},
    {"symbol": "RM=F", "horizon": 2, "p50": 101.0},
    {"symbol": "RM=F", "horizon": 3, "p50": 103.0},
])

_SAMPLE_PRICES = {"KC=F": 198.0, "RM=F": 99.0}
_SAMPLE_SPREAD = {"signal": 0, "z_score": 0.5, "half_life": 3.2}


def test_build_email_body_contains_month_and_symbols():
    from coffee_forecast.reports import _build_success_email_body
    body = _build_success_email_body("2026-05", _SAMPLE_FORECASTS, _SAMPLE_PRICES, _SAMPLE_SPREAD)
    assert "2026-05" in body
    assert "Arabica" in body
    assert "Robusta" in body


def test_build_email_body_contains_forecast_values():
    from coffee_forecast.reports import _build_success_email_body
    body = _build_success_email_body("2026-05", _SAMPLE_FORECASTS, _SAMPLE_PRICES, _SAMPLE_SPREAD)
    assert "200.0" in body   # KC=F h=1 p50
    assert "100.0" in body   # RM=F h=1 p50


def test_build_email_body_shows_spread_signal():
    from coffee_forecast.reports import _build_success_email_body
    body = _build_success_email_body("2026-05", _SAMPLE_FORECASTS, _SAMPLE_PRICES, {"signal": 1, "z_score": 2.3})
    assert "Buy Arabica" in body


def test_build_email_body_handles_empty_data():
    from coffee_forecast.reports import _build_success_email_body
    # Must not crash with empty inputs
    body = _build_success_email_body("2026-05", pd.DataFrame(), {}, {})
    assert "2026-05" in body


def test_build_email_body_handles_null_p50():
    from coffee_forecast.reports import _build_success_email_body
    df = pd.DataFrame([{"symbol": "KC=F", "horizon": 1, "p50": None}])
    body = _build_success_email_body("2026-05", df, {"KC=F": 200.0, "RM=F": 99.0}, {})
    assert "2026-05" in body  # must not crash on NULL p50


# ── send_success_email ────────────────────────────────────────────────────

@pytest.fixture
def db_with_forecast_data(tmp_path):
    """Write a minimal SQLite DB file with one hybrid run and forecast rows."""
    db_path = tmp_path / "coffee.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE model_runs (
            id INTEGER PRIMARY KEY, run_at TEXT NOT NULL DEFAULT '',
            model_type TEXT NOT NULL, train_start TEXT NOT NULL DEFAULT '',
            train_end TEXT NOT NULL DEFAULT '', params TEXT, metrics TEXT,
            status TEXT NOT NULL DEFAULT 'pending', notes TEXT
        );
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY, run_id INTEGER, forecast_date TEXT,
            target_date TEXT, horizon INTEGER, symbol TEXT,
            point_forecast REAL, p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL
        );
        CREATE TABLE prices_monthly (
            id INTEGER PRIMARY KEY, symbol TEXT, date TEXT, adj_close REAL
        );
        CREATE TABLE spread_signals (
            id INTEGER PRIMARY KEY, date TEXT, spread REAL, z_score REAL,
            signal INTEGER, half_life REAL
        );
    """)
    conn.execute(
        "INSERT INTO model_runs (id, model_type, status) VALUES (1, 'hybrid', 'success')"
    )
    conn.executemany(
        "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, 1, "2026-05-01", "2026-06-01", 1, "KC=F", 200.0, 180.0, 190.0, 200.0, 210.0, 220.0),
            (2, 1, "2026-05-01", "2026-07-01", 2, "KC=F", 202.0, 182.0, 192.0, 202.0, 212.0, 222.0),
            (3, 1, "2026-05-01", "2026-08-01", 3, "KC=F", 205.0, 185.0, 195.0, 205.0, 215.0, 225.0),
            (4, 1, "2026-05-01", "2026-06-01", 1, "RM=F", 100.0,  90.0,  95.0, 100.0, 105.0, 110.0),
            (5, 1, "2026-05-01", "2026-07-01", 2, "RM=F", 101.0,  91.0,  96.0, 101.0, 106.0, 111.0),
            (6, 1, "2026-05-01", "2026-08-01", 3, "RM=F", 103.0,  93.0,  98.0, 103.0, 108.0, 113.0),
        ],
    )
    conn.execute(
        "INSERT INTO prices_monthly (symbol, date, adj_close) VALUES ('KC=F', '2026-05-01', 198.0)"
    )
    conn.execute(
        "INSERT INTO prices_monthly (symbol, date, adj_close) VALUES ('RM=F', '2026-05-01', 99.0)"
    )
    conn.execute(
        "INSERT INTO spread_signals (date, spread, z_score, signal, half_life)"
        " VALUES ('2026-05-01', 0.7, 0.5, 0, 3.2)"
    )
    conn.commit()
    conn.close()
    return db_path


def test_send_success_email_posts_to_resend(db_with_forecast_data, monkeypatch):
    import requests as req_mod
    from coffee_forecast.reports import send_success_email

    monkeypatch.setenv("RESEND_API_KEY", "test-key")

    posted = []
    def _fake_post(url, **kwargs):
        posted.append(kwargs)
        mock_resp = type("MockResp", (), {"raise_for_status": lambda self: None})()
        return mock_resp
    monkeypatch.setattr(req_mod, "post", _fake_post)

    send_success_email("2026-05", db_with_forecast_data)

    assert len(posted) == 1
    payload = posted[0]["json"]
    assert "[coffee-forecast]" in payload["subject"]
    assert "2026-05" in payload["subject"]
    assert payload["to"] == ["lucparrot1@gmail.com"]
    assert "Bearer test-key" in posted[0]["headers"]["Authorization"]


def test_send_success_email_skips_without_api_key(db_with_forecast_data, monkeypatch):
    import requests as req_mod
    from coffee_forecast.reports import send_success_email

    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    posted = []
    monkeypatch.setattr(req_mod, "post", lambda *a, **k: posted.append(k))

    send_success_email("2026-05", db_with_forecast_data)

    assert posted == []


# ── CLI --success-email flag ───────────────────────────────────────────────

def test_cli_success_email_flag_calls_send_success_email(
    db_with_forecast_data, monkeypatch
):
    """--success-email routes to send_success_email, not render_monthly_report."""
    import requests as req_mod
    from coffee_forecast import reports as reports_mod

    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setenv("COFFEE_DB_PATH", str(db_with_forecast_data))

    posted = []
    def _fake_post(url, **kw):
        posted.append(kw)
        return type("MockResp", (), {"raise_for_status": lambda self: None})()
    monkeypatch.setattr(req_mod, "post", _fake_post)

    render_called = []
    monkeypatch.setattr(
        reports_mod, "render_monthly_report",
        lambda *a, **kw: render_called.append(True),
    )

    import sys
    monkeypatch.setattr(
        sys, "argv",
        ["reports", "--success-email", "--month", "2026-05", "--db", str(db_with_forecast_data)],
    )
    reports_mod.main()

    assert len(posted) == 1, "Resend was not called"
    assert render_called == [], "render_monthly_report should not be called"
