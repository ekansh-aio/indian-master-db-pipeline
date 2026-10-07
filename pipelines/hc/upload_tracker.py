"""
SQLite-backed upload tracker for the HC ingest pipeline.

Tracks which files have been successfully uploaded to ADLS so that
re-runs skip already-uploaded files.

Schema:
  files(path TEXT PRIMARY KEY, dataset TEXT, size_bytes INTEGER, uploaded_at TEXT)

  path        = relative path within the ADLS prefix, e.g.
                'year=2024/court=10_8/bench=xyz/BRHC01234.json'
  dataset     = dataset label, e.g. 'High_Court_Judgements'
  size_bytes  = file size in bytes (0 if populated from ADLS scan)
  uploaded_at = ISO-8601 UTC timestamp of upload (empty string if from ADLS scan)
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_dataset ON files(dataset);
"""


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
    conn.executescript(DDL)
    return conn


@contextmanager
def batch_inserter(conn: sqlite3.Connection, batch_size: int = 5000):
    """Buffer inserts and flush in batches for speed."""
    buf = []

    def insert(path: str, dataset: str, size_bytes: int = 0, uploaded_at: str = ""):
        buf.append((path, dataset, size_bytes, uploaded_at))
        if len(buf) >= batch_size:
            _flush(buf, conn)
            buf.clear()

    yield insert
    if buf:
        _flush(buf, conn)


def _flush(rows: list, conn: sqlite3.Connection):
    conn.executemany(
        "INSERT OR IGNORE INTO files (path, dataset, size_bytes, uploaded_at) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()


def exists(conn: sqlite3.Connection, path: str) -> bool:
    return conn.execute("SELECT 1 FROM files WHERE path=?", (path,)).fetchone() is not None


def insert_one(conn: sqlite3.Connection, path: str, dataset: str,
               size_bytes: int = 0, uploaded_at: str = ""):
    conn.execute(
        "INSERT OR IGNORE INTO files (path, dataset, size_bytes, uploaded_at) VALUES (?,?,?,?)",
        (path, dataset, size_bytes, uploaded_at),
    )
    conn.commit()


def count(conn: sqlite3.Connection, dataset: str | None = None) -> int:
    if dataset:
        return conn.execute(
            "SELECT COUNT(*) FROM files WHERE dataset=?", (dataset,)
        ).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]


def datasets(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT dataset, COUNT(*) FROM files GROUP BY dataset ORDER BY dataset"
    ).fetchall()
