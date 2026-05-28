import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't already exist."""
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    log.info("Database schema applied")
