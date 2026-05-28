import os
import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = "data/coffee.db"


def get_connection() -> sqlite3.Connection:
    db_path = Path(os.getenv("COFFEE_DB_PATH", _DEFAULT_DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn
