"""Tests for grok_mccodin.database."""

from __future__ import annotations

import pytest

from grok_mccodin.database import DatabaseError, SQLiteDB, run_query


class TestSQLiteDB:
    def test_create_and_query(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
            db.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Bob", 25))

            rows = db.query("SELECT * FROM users ORDER BY name")
            assert len(rows) == 2
            assert rows[0]["name"] == "Alice"
            assert rows[1]["age"] == 25

    def test_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE foo (id INTEGER)")
            db.execute("CREATE TABLE bar (id INTEGER)")
            tables = db.tables()
            assert "foo" in tables
            assert "bar" in tables

    def test_schema(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            schema = db.schema()
            assert "CREATE TABLE" in schema
            assert "users" in schema

    def test_table_info(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT NOT NULL)")
            info = db.table_info("items")
            col_names = [col["name"] for col in info]
            assert "id" in col_names
            assert "label" in col_names

    def test_table_info_missing(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE real_table (id INTEGER)")
            with pytest.raises(DatabaseError, match="not found"):
                db.table_info("nonexistent")

    def test_execute_script(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute_script(
                "CREATE TABLE a (id INTEGER);\n"
                "CREATE TABLE b (id INTEGER);\n"
                "INSERT INTO a VALUES (1);\n"
            )
            assert "a" in db.tables()
            assert "b" in db.tables()

    def test_bad_query(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            with pytest.raises(DatabaseError, match="Query failed"):
                db.query("SELECT * FROM nonexistent_table")

    def test_bad_execute(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            with pytest.raises(DatabaseError):
                db.execute("INSERT INTO nonexistent (x) VALUES (1)")

    def test_rowcount(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE nums (n INTEGER)")
            db.execute("INSERT INTO nums VALUES (1)")
            db.execute("INSERT INTO nums VALUES (2)")
            affected = db.execute("DELETE FROM nums WHERE n = 1")
            assert affected == 1


class TestRunQuery:
    def test_sqlite_prefix(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE t (id INTEGER)")
            db.execute("INSERT INTO t VALUES (42)")

        rows = run_query(f"sqlite:{db_path}", "SELECT * FROM t")
        assert len(rows) == 1
        assert rows[0]["id"] == 42

    def test_unsupported_prefix(self):
        with pytest.raises(DatabaseError, match="Unsupported"):
            run_query("redis://localhost", "PING")
