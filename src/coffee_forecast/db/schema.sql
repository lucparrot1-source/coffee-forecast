-- coffee_forecast database schema

CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY,
    date        TEXT    NOT NULL,  -- YYYY-MM-DD
    symbol      TEXT    NOT NULL,  -- e.g. KC=F, RM=F, BRL=X, VND=X, DX-Y.NYB
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    adj_close   REAL,
    UNIQUE (date, symbol)
);

CREATE TABLE IF NOT EXISTS prices_monthly (
    id          INTEGER PRIMARY KEY,
    date        TEXT    NOT NULL,  -- YYYY-MM-01 (month-start convention)
    symbol      TEXT    NOT NULL,
    close       REAL,
    adj_close   REAL,
    UNIQUE (date, symbol)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY,
    run_at      TEXT    NOT NULL,  -- ISO timestamp
    model_type  TEXT    NOT NULL,  -- vecm | gamlss | spread | hybrid
    train_start TEXT    NOT NULL,
    train_end   TEXT    NOT NULL,
    params      TEXT,              -- JSON blob
    metrics     TEXT,              -- JSON blob
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | success | failed
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES model_runs(id),
    forecast_date   TEXT    NOT NULL,  -- date forecast was generated
    target_date     TEXT    NOT NULL,  -- date being forecast
    horizon         INTEGER NOT NULL,  -- 1, 2, or 3 months ahead
    symbol          TEXT    NOT NULL,
    point_forecast  REAL,
    p10             REAL,
    p25             REAL,
    p50             REAL,
    p75             REAL,
    p90             REAL,
    UNIQUE (run_id, target_date, symbol)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES model_runs(id),
    backtest_date   TEXT    NOT NULL,
    train_end       TEXT    NOT NULL,
    target_date     TEXT    NOT NULL,
    horizon         INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    actual          REAL,
    point_forecast  REAL,
    p10             REAL,
    p50             REAL,
    p90             REAL
);

CREATE TABLE IF NOT EXISTS accuracy_log (
    id          INTEGER PRIMARY KEY,
    logged_at   TEXT    NOT NULL,
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    actual      REAL    NOT NULL,
    horizon     INTEGER NOT NULL,
    symbol      TEXT    NOT NULL,
    mae         REAL,
    mape        REAL,
    pinball_50  REAL,
    coverage_80 INTEGER  -- 1 if actual falls within [p10, p90], else 0
);

CREATE INDEX IF NOT EXISTS idx_prices_date_symbol       ON prices (date, symbol);
CREATE INDEX IF NOT EXISTS idx_prices_monthly_date      ON prices_monthly (date, symbol);
CREATE INDEX IF NOT EXISTS idx_forecasts_target         ON forecasts (target_date, symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_target          ON backtest_results (target_date, symbol);
