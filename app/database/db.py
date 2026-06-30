# app/database/db.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from app.config.settings import settings

DB_PATH = Path(settings.db_path) if hasattr(settings, "db_path") else Path("data/anomaly_detection.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id  TEXT NOT NULL,
                ticker      TEXT,
                fiscal_year INTEGER,
                fiscal_quarter TEXT,
                anomaly_score   REAL NOT NULL,
                is_anomaly      INTEGER NOT NULL,
                risk_level      TEXT NOT NULL,
                explanation     TEXT,
                raw_metrics     TEXT,          -- JSON blob
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_company
                ON predictions(company_id);

            CREATE INDEX IF NOT EXISTS idx_predictions_created
                ON predictions(created_at);
        """)

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def insert_prediction(record: dict) -> int:
    sql = """
        INSERT INTO predictions
            (company_id, ticker, fiscal_year, fiscal_quarter,
             anomaly_score, is_anomaly, risk_level, explanation,
             raw_metrics, created_at)
        VALUES
            (:company_id, :ticker, :fiscal_year, :fiscal_quarter,
             :anomaly_score, :is_anomaly, :risk_level, :explanation,
             :raw_metrics, :created_at)
    """
    record.setdefault("created_at", datetime.utcnow().isoformat())
    if isinstance(record.get("raw_metrics"), dict):
        record["raw_metrics"] = json.dumps(record["raw_metrics"])
    with get_conn() as conn:
        cur = conn.execute(sql, record)
        return cur.lastrowid

def get_predictions(company_id: str | None = None, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        if company_id:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE company_id = ? ORDER BY created_at DESC LIMIT ?",
                (company_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]