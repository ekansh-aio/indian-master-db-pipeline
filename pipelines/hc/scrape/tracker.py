"""
SQLite tracker for the HC scraper.

Two tables:
  files(path, dataset, size_bytes, uploaded_at)  — uploaded PDF + JSON records
  scrape_progress(court, bench, scraped_through)  — resume cursor per court

Thread safety: all public functions acquire _lock before touching the
connection. SQLite connection is opened with timeout=15 so workers queue
instead of immediately raising OperationalError under write contention.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

_DDL = """
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS files_dataset ON files (dataset);

CREATE TABLE IF NOT EXISTS scrape_progress (
    court           TEXT NOT NULL,
    bench           TEXT NOT NULL,
    scraped_through TEXT NOT NULL,
    PRIMARY KEY (court, bench)
);
"""

# Single lock for all DB access — WAL allows concurrent reads but we use
# a single connection shared across threads, so all calls must serialise.
_lock = threading.Lock()


def get_conn(db_path: Path) -> sqlite3.Connection:
    """
    Open (or create) the SQLite DB.

    timeout=15: worker threads wait up to 15 s on lock contention before
    raising OperationalError, rather than failing immediately.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    with _lock:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(_DDL)
        conn.commit()
    return conn


# ── file tracking ──────────────────────────────────────────────────────────────

def file_exists(conn: sqlite3.Connection, path: str) -> bool:
    with _lock:
        row = conn.execute("SELECT 1 FROM files WHERE path=?", (path,)).fetchone()
    return row is not None


def file_insert(conn: sqlite3.Connection, path: str, dataset: str,
                size_bytes: int, uploaded_at: str) -> None:
    """Record a successfully uploaded file. INSERT OR IGNORE — safe to call twice."""
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO files(path, dataset, size_bytes, uploaded_at)"
            " VALUES (?,?,?,?)",
            (path, dataset, size_bytes, uploaded_at),
        )
        conn.commit()


def file_count(conn: sqlite3.Connection, dataset: str) -> int:
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) FROM files WHERE dataset=?", (dataset,)
        ).fetchone()
    return row[0] if row else 0


def all_datasets(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Return [(dataset, count), ...] for all datasets in the tracker."""
    with _lock:
        rows = conn.execute(
            "SELECT dataset, COUNT(*) FROM files GROUP BY dataset ORDER BY dataset"
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


# ── scrape progress ────────────────────────────────────────────────────────────

def get_scraped_through(conn: sqlite3.Connection, court: str,
                        bench: str = "__all__") -> Optional[str]:
    """Return the last scraped_through date for this court/bench, or None."""
    with _lock:
        row = conn.execute(
            "SELECT scraped_through FROM scrape_progress WHERE court=? AND bench=?",
            (court, bench),
        ).fetchone()
    return row[0] if row else None


def set_scraped_through(conn: sqlite3.Connection, court: str,
                        bench: str, date_str: str) -> None:
    with _lock:
        conn.execute(
            "INSERT INTO scrape_progress(court, bench, scraped_through)"
            " VALUES (?,?,?)"
            " ON CONFLICT(court, bench)"
            " DO UPDATE SET scraped_through=excluded.scraped_through",
            (court, bench, date_str),
        )
        conn.commit()


def distinct_years(conn: sqlite3.Connection, dataset: str) -> list[int]:
    """Return sorted list of years that have tracked files (from year=YYYY/ prefix)."""
    with _lock:
        rows = conn.execute(
            "SELECT DISTINCT substr(path, 6, 4) FROM files WHERE dataset=? AND path LIKE 'year=%'",
            (dataset,),
        ).fetchall()
    return sorted(int(r[0]) for r in rows if r[0].isdigit())


def paths_for_year(conn: sqlite3.Connection, dataset: str, year: int) -> list[str]:
    """Return all tracked paths for a given year= partition."""
    with _lock:
        rows = conn.execute(
            "SELECT path FROM files WHERE dataset=? AND path LIKE ?",
            (dataset, f"year={year}/%"),
        ).fetchall()
    return sorted(r[0] for r in rows)


def all_progress(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    with _lock:
        rows = conn.execute(
            "SELECT court, bench, scraped_through"
            " FROM scrape_progress ORDER BY court, bench"
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]
