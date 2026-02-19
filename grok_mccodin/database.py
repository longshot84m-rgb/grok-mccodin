"""Database queries — SQLite built-in, optional PostgreSQL/MySQL support."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


# ---------------------------------------------------------------------------
# SQLite (built-in, no extra dependencies)
# ---------------------------------------------------------------------------


class SQLiteDB:
    """Simple SQLite wrapper for ad-hoc queries from the chat loop.

    Usage::

        db = SQLiteDB("mydata.db")
        rows = db.query("SELECT * FROM users WHERE age > ?", (18,))
        db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        schema = db.schema()
        db.close()
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a SELECT query and return rows as dicts."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """Execute a write statement (INSERT/UPDATE/DELETE). Returns rowcount."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"Execute failed: {exc}") from exc

    def execute_script(self, sql: str) -> None:
        """Execute a multi-statement SQL script."""
        conn = self._get_conn()
        try:
            conn.executescript(sql)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Script failed: {exc}") from exc

    def schema(self) -> str:
        """Return the database schema (all CREATE statements)."""
        rows = self.query(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view', 'index') "
            "AND sql IS NOT NULL ORDER BY type, name"
        )
        return "\n\n".join(row["sql"] for row in rows)

    def tables(self) -> list[str]:
        """Return a list of table names."""
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row["name"] for row in rows]

    def table_info(self, table: str) -> list[dict[str, Any]]:
        """Return column info for a table."""
        # Validate table exists via parameterized query to avoid injection
        if table not in self.tables():
            raise DatabaseError(f"Table not found: {table}")
        # PRAGMA doesn't support parameterized queries, so we must sanitize.
        # Since we validated against the actual table list above, we also
        # enforce identifier-safe characters as defense in depth.
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
            raise DatabaseError(f"Invalid table name: {table}")
        return self.query(f"PRAGMA table_info({table})")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Generic query runner (supports sqlite: and postgres: URIs)
# ---------------------------------------------------------------------------


def run_query(
    connection_string: str,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Run a query against a database identified by a connection string.

    Supported prefixes:
        - ``sqlite:path/to/db.sqlite`` — uses built-in sqlite3
        - ``postgres://...`` — requires psycopg2 (optional)
        - ``mysql://...`` — requires mysql-connector-python (optional)

    Returns rows as list of dicts.
    """
    if connection_string.startswith("sqlite:"):
        db_path = connection_string[len("sqlite:") :]
        with SQLiteDB(db_path) as db:
            return db.query(sql, params)

    if connection_string.startswith(("postgres://", "postgresql://")):
        return _query_postgres(connection_string, sql, params)

    if connection_string.startswith("mysql://"):
        return _query_mysql(connection_string, sql, params)

    raise DatabaseError(f"Unsupported connection string: {connection_string[:30]}...")


def _query_postgres(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Query PostgreSQL. Requires psycopg2."""
    try:
        import psycopg2  # type: ignore[import-untyped]
        import psycopg2.extras  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DatabaseError("psycopg2 not installed — run: pip install psycopg2-binary") from exc

    try:
        conn = psycopg2.connect(dsn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            conn.commit()
            return []
    except psycopg2.Error as exc:
        raise DatabaseError(f"PostgreSQL error: {exc}") from exc
    finally:
        conn.close()


def _query_mysql(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Query MySQL. Requires mysql-connector-python."""
    try:
        import mysql.connector  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DatabaseError(
            "mysql-connector-python not installed — run: pip install mysql-connector-python"
        ) from exc

    # Parse mysql://user:pass@host:port/db
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    config = {
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "database": parsed.path.lstrip("/"),
    }

    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params)
        if cursor.description:
            rows: list[dict[str, Any]] = cursor.fetchall()
            return rows
        conn.commit()
        return []
    except mysql.connector.Error as exc:
        raise DatabaseError(f"MySQL error: {exc}") from exc
    finally:
        conn.close()
