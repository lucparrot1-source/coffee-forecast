import os
import sqlite3
from pathlib import Path

_DB_PATH = Path(os.getenv("COFFEE_DB_PATH", "data/coffee.db"))


def get_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn
