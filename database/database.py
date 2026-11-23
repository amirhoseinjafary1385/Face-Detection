"""
SQLite database layer for storing results and users.
"""

import os
import sqlite3
import hashlib
from datetime import datetime
from contextlib import contextmanager


DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def init_db(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    result_path TEXT,
                    face_count INTEGER DEFAULT 0,
                    kind TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

        # Seed default admin user if none
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
            if row["c"] == 0:
                self._create_user_raw("admin", "admin123")

    # ------------------------------------------------------------------
    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _create_user_raw(self, username: str, password: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                (username, self._hash(password), datetime.utcnow().isoformat()),
            )

    def create_user(self, username: str, password: str) -> bool:
        try:
            self._create_user_raw(username, password)
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username: str, password: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return False
        return row["password_hash"] == self._hash(password)

    # ------------------------------------------------------------------
    def add_result(self, image_path: str, result_path: str,
                   face_count: int, kind: str, detail: str = "") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO results
                   (image_path, result_path, face_count, kind, detail, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (image_path, result_path, face_count, kind, detail,
                 datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def get_all_results(self, limit: int = 100):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM results ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM results").fetchone()["c"]
            detects = conn.execute(
                "SELECT COUNT(*) AS c FROM results WHERE kind='detect'"
            ).fetchone()["c"]
            compares = conn.execute(
                "SELECT COUNT(*) AS c FROM results WHERE kind='compare'"
            ).fetchone()["c"]
            recognizes = conn.execute(
                "SELECT COUNT(*) AS c FROM results WHERE kind='recognize'"
            ).fetchone()["c"]
            faces = conn.execute(
                "SELECT COALESCE(SUM(face_count),0) AS s FROM results"
            ).fetchone()["s"]

        return {
            "total_operations": total,
            "detections": detects,
            "comparisons": compares,
            "recognitions": recognizes,
            "total_faces": faces,
        }