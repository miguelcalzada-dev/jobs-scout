from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

from src.config import DB_PATH, settings
from src.scraper import JobOffer, JobSource

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT,
    location TEXT,
    url TEXT,
    description TEXT,
    description_html TEXT,
    salary TEXT,
    salary_min INTEGER DEFAULT 0,
    salary_max INTEGER DEFAULT 0,
    contract_type TEXT,
    remote INTEGER DEFAULT 0,
    hybrid INTEGER DEFAULT 0,
    onsite INTEGER DEFAULT 0,
    published_date TEXT,
    scrape_date TEXT,
    technologies_detected TEXT,
    seniority_level TEXT,
    company_size TEXT,
    sector TEXT,
    match_score REAL DEFAULT 0.0,
    notified INTEGER DEFAULT 0,
    seen INTEGER DEFAULT 0,
    applied INTEGER DEFAULT 0,
    discarded INTEGER DEFAULT 0,
    raw_data TEXT,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    offers_found INTEGER DEFAULT 0,
    duration_seconds REAL DEFAULT 0.0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON job_offers(source);
CREATE INDEX IF NOT EXISTS idx_jobs_scrape_date ON job_offers(scrape_date);
CREATE INDEX IF NOT EXISTS idx_jobs_score ON job_offers(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_notified ON job_offers(notified);
CREATE INDEX IF NOT EXISTS idx_jobs_unique ON job_offers(source, external_id);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), autocommit=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()
    logger.info("Database initialized")


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = _get_conn()
    try:
        yield conn
    finally:
        conn.close()


def insert_job(job: JobOffer) -> bool:
    with get_db() as conn:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO job_offers
                (source, external_id, title, company, location, url, description,
                 description_html, salary, salary_min, salary_max, contract_type,
                 remote, hybrid, onsite, published_date, scrape_date,
                 technologies_detected, seniority_level, company_size, sector, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.source.value,
                job.external_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.description[:10000],
                job.description_html,
                job.salary,
                job.salary_min,
                job.salary_max,
                job.contract_type,
                1 if job.remote else 0,
                1 if job.hybrid else 0,
                1 if job.onsite else 0,
                job.published_date,
                job.scrape_date,
                ",".join(job.technologies_detected),
                job.seniority_level,
                job.company_size,
                job.sector,
                str(job.raw_data) if job.raw_data else "{}",
            ))
            return True
        except Exception as e:
            logger.debug(f"Failed to insert job {job.external_id}: {e}")
            return False


def bulk_insert_jobs(jobs: list[JobOffer]) -> tuple[int, int]:
    if not jobs:
        return 0, 0
    with get_db() as conn:
        before = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
        data = [
            (
                job.source.value,
                job.external_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.description[:10000],
                job.description_html,
                job.salary,
                job.salary_min,
                job.salary_max,
                job.contract_type,
                1 if job.remote else 0,
                1 if job.hybrid else 0,
                1 if job.onsite else 0,
                job.published_date,
                job.scrape_date,
                ",".join(job.technologies_detected),
                job.seniority_level,
                job.company_size,
                job.sector,
                str(job.raw_data) if job.raw_data else "{}",
            )
            for job in jobs
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO job_offers "
            "(source, external_id, title, company, location, url, description, "
            "description_html, salary, salary_min, salary_max, contract_type, "
            "remote, hybrid, onsite, published_date, scrape_date, "
            "technologies_detected, seniority_level, company_size, sector, raw_data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )
        after = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
    inserted = after - before
    skipped = len(jobs) - inserted
    return inserted, skipped


def get_unscored_jobs(limit: int = 200, rescore_all: bool = False) -> list[dict]:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
        if rescore_all:
            rows = conn.execute("""
                SELECT * FROM job_offers
                WHERE discarded = 0 AND scrape_date >= ?
                ORDER BY scrape_date DESC LIMIT ?
            """, (cutoff, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM job_offers
                WHERE match_score = 0 AND discarded = 0 AND scrape_date >= ?
                ORDER BY scrape_date DESC LIMIT ?
            """, (cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def update_job_score(external_id: str, source: str, score: float, match_reason: str = "") -> None:
    with get_db() as conn:
        conn.execute("""
            UPDATE job_offers SET match_score = ?, raw_data = json_set(COALESCE(raw_data, '{}'), '$.match_reason', ?)
            WHERE external_id = ? AND source = ?
        """, (round(score, 4), match_reason, external_id, source))


def get_top_jobs(limit: int = 10, days: int = 3) -> list[dict]:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=max(days, 1))).isoformat()
        rows = conn.execute("""
            SELECT * FROM job_offers
            WHERE match_score > 0
              AND discarded = 0
              AND scrape_date >= ?
            ORDER BY match_score DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def get_pending_notification_jobs(limit: int = 10) -> list[dict]:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
        rows = conn.execute("""
            SELECT * FROM job_offers
            WHERE notified = 0
              AND match_score > 0
              AND discarded = 0
              AND scrape_date >= ?
            ORDER BY match_score DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def mark_as_notified(job_ids: list[int]) -> None:
    with get_db() as conn:
        conn.executemany(
            "UPDATE job_offers SET notified = 1 WHERE id = ?",
            [(job_id,) for job_id in job_ids],
        )


def mark_as_applied(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE job_offers SET applied = 1, seen = 1 WHERE id = ?",
            (job_id,),
        )


def mark_as_discarded(job_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE job_offers SET discarded = 1, seen = 1 WHERE id = ?",
            (job_id,),
        )


def log_scrape_run(source: str, status: str, offers_found: int, duration: float, error: str = "") -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO scrape_runs (run_date, source, status, offers_found, duration_seconds, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), source, status, offers_found, duration, error))


def cleanup_old_jobs(retention_days: int = 30) -> int:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = conn.execute(
            "DELETE FROM job_offers WHERE scrape_date < ?",
            (cutoff,),
        )
        return cursor.rowcount


def get_stats() -> dict:
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
        scored = conn.execute("SELECT COUNT(*) FROM job_offers WHERE match_score > 0").fetchone()[0]
        notified = conn.execute("SELECT COUNT(*) FROM job_offers WHERE notified = 1").fetchone()[0]
        applied = conn.execute("SELECT COUNT(*) FROM job_offers WHERE applied = 1").fetchone()[0]
        last_run = conn.execute(
            "SELECT run_date, COUNT(*) as cnt FROM scrape_runs GROUP BY run_date ORDER BY run_date DESC LIMIT 1"
        ).fetchone()
        return {
            "total_jobs": total,
            "scored_jobs": scored,
            "notified_jobs": notified,
            "applied_jobs": applied,
            "last_run": last_run["run_date"] if last_run else None,
            "last_run_sources": last_run["cnt"] if last_run else 0,
        }
