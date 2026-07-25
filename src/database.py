from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator

from src.config import DB_PATH, settings
from src.scraper import JobOffer

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
    is_external_redirect INTEGER DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS idx_runs_date ON scrape_runs(run_date);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), autocommit=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info(f"Added column {column} to {table}")


def init_db() -> None:
    conn = _get_conn()
    try:
        conn.executescript(SCHEMA)
        _ensure_column(conn, "job_offers", "is_external_redirect", "INTEGER DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_external ON job_offers(is_external_redirect)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_discarded ON job_offers(discarded)")
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


def clear_all_jobs() -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM job_offers")
        return cur.rowcount


def bulk_insert_jobs(jobs: list[JobOffer]) -> tuple[int, int]:
    if not jobs:
        return 0, 0
    with get_db() as conn:
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
                1 if job.is_external_redirect else 0,
                str(job.raw_data) if job.raw_data else "{}",
            )
            for job in jobs
        ]
        before = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO job_offers "
            "(source, external_id, title, company, location, url, description, "
            "description_html, salary, salary_min, salary_max, contract_type, "
            "remote, hybrid, onsite, published_date, scrape_date, "
            "technologies_detected, seniority_level, company_size, sector, "
            "is_external_redirect, raw_data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )
        after = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
    inserted = after - before
    skipped = len(jobs) - inserted
    return inserted, skipped


def get_unscored_jobs(limit: int = 300) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, source, external_id, title, company, location, url,
                   description, salary_min, technologies_detected, seniority_level,
                   remote, hybrid, onsite, scrape_date
            FROM job_offers
            WHERE match_score = 0
              AND discarded = 0
              AND is_external_redirect = 0
            ORDER BY scrape_date DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_all_active_job_ids() -> list[tuple[str, str]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT external_id, source FROM job_offers WHERE discarded = 0 AND is_external_redirect = 0"
        ).fetchall()
    return [(r["external_id"], r["source"]) for r in rows]


def update_job_score(external_id: str, source: str, score: float, match_reason: str = "") -> None:
    with get_db() as conn:
        conn.execute("""
            UPDATE job_offers SET match_score = ?, raw_data = json_set(COALESCE(raw_data, '{}'), '$.match_reason', ?)
            WHERE external_id = ? AND source = ?
        """, (round(score, 4), match_reason, external_id, source))


def reset_all_scores() -> int:
    with get_db() as conn:
        cur = conn.execute("UPDATE job_offers SET match_score = 0")
        return cur.rowcount


def get_top_jobs(limit: int = 50, days: int = 3) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=max(days, 1))).isoformat()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, source, external_id, title, company, location, url,
                   salary, match_score, remote, hybrid, onsite,
                   technologies_detected, seniority_level, scrape_date
            FROM job_offers
            WHERE match_score > 0
              AND discarded = 0
              AND is_external_redirect = 0
              AND scrape_date >= ?
            ORDER BY match_score DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def get_pending_notification_jobs(limit: int = 10) -> list[dict]:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
        rows = conn.execute("""
            SELECT id, source, external_id, title, company, location, url, salary,
                   match_score, remote, hybrid, technologies_detected, scrape_date
            FROM job_offers
            WHERE notified = 0
              AND match_score > 0
              AND discarded = 0
              AND is_external_redirect = 0
              AND scrape_date >= ?
            ORDER BY match_score DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
    return [dict(row) for row in rows]


def mark_as_notified(job_ids: list[int]) -> None:
    if not job_ids:
        return
    with get_db() as conn:
        conn.executemany(
            "UPDATE job_offers SET notified = 1 WHERE id = ?",
            [(job_id,) for job_id in job_ids],
        )


def mark_as_applied(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("UPDATE job_offers SET applied = 1, seen = 1 WHERE id = ?", (job_id,))


def mark_as_discarded(job_id: int) -> None:
    with get_db() as conn:
        conn.execute("UPDATE job_offers SET discarded = 1, seen = 1 WHERE id = ?", (job_id,))


def log_scrape_run(source: str, status: str, offers_found: int, duration: float, error: str = "") -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO scrape_runs (run_date, source, status, offers_found, duration_seconds, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), source, status, offers_found, duration, error))


def get_scrape_runs_paged(page: int = 1, per_page: int = 25, max_runs: int = 50) -> dict:
    """Return one page of the most recent scrape runs (server-side pagination)."""
    with get_db() as conn:
        total_count = conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0]
        capped_rows = conn.execute("""
            SELECT run_date, source, status, offers_found, duration_seconds
            FROM scrape_runs
            ORDER BY run_date DESC
            LIMIT ?
        """, (max_runs,)).fetchall()
    capped = [dict(r) for r in capped_rows]
    total_pages = max(1, (len(capped) + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    page_rows = capped[start:end]
    return {
        "runs": page_rows,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_shown": len(capped),
        "total_in_db": total_count,
    }


def cleanup_old_jobs(retention_days: int = 30) -> int:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cur = conn.execute("DELETE FROM job_offers WHERE scrape_date < ?", (cutoff,))
        return cur.rowcount


def cleanup_old_scrape_runs(retention_days: int = 30) -> int:
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cur = conn.execute("DELETE FROM scrape_runs WHERE run_date < ?", (cutoff,))
        return cur.rowcount


def get_stats() -> dict:
    cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_jobs,
                SUM(CASE WHEN match_score > 0 THEN 1 ELSE 0 END) AS scored_jobs,
                SUM(CASE WHEN notified = 1 THEN 1 ELSE 0 END) AS notified_jobs,
                SUM(CASE WHEN applied = 1 THEN 1 ELSE 0 END) AS applied_jobs,
                SUM(CASE WHEN discarded = 1 THEN 1 ELSE 0 END) AS discarded_jobs,
                SUM(CASE WHEN is_external_redirect = 1 THEN 1 ELSE 0 END) AS external_jobs
            FROM job_offers
            WHERE scrape_date >= ?
        """, (cutoff,)).fetchone()
        last_run = conn.execute(
            "SELECT run_date, COUNT(*) as cnt FROM scrape_runs GROUP BY run_date ORDER BY run_date DESC LIMIT 1"
        ).fetchone()
    return {
        "total_jobs": row["total_jobs"] or 0,
        "scored_jobs": row["scored_jobs"] or 0,
        "notified_jobs": row["notified_jobs"] or 0,
        "applied_jobs": row["applied_jobs"] or 0,
        "discarded_jobs": row["discarded_jobs"] or 0,
        "external_jobs": row["external_jobs"] or 0,
        "last_run": last_run["run_date"] if last_run else None,
        "last_run_sources": last_run["cnt"] if last_run else 0,
    }