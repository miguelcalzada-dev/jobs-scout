from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from src.config import (
    load_preferences,
    load_cv_profile,
    settings,
)
from src.database import (
    init_db,
    bulk_insert_jobs,
    get_top_jobs,
    get_pending_notification_jobs,
    mark_as_notified,
    cleanup_old_jobs,
    get_stats,
    log_scrape_run,
)
from src.delivery import send_daily_email
from src.matcher import score_all_unscored_jobs
from src.scrapers import TecnoempleoScraper, InfojobsScraper, LinkedInScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("jobs-scout")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

scheduler = AsyncIOScheduler(timezone=settings.tz)
_is_running = False
_last_run_status: dict = {"status": "idle", "last_run": None, "errors": []}


async def run_daily_job() -> dict:
    global _is_running, _last_run_status
    if _is_running:
        logger.warning("A daily run is already in progress, skipping")
        return {"status": "skipped", "reason": "already_running"}

    _is_running = True
    start_time = time.time()
    results = {"status": "running", "started_at": datetime.now().isoformat(), "sources": {}, "errors": []}

    try:
        prefs = load_preferences()
        cv = load_cv_profile()

        if not prefs.desired_titles:
            prefs.desired_titles = ["python", "desarrollador", "backend"]
            logger.info("Sin preferencias configuradas, usando búsqueda por defecto")
        if not prefs.tech_stack:
            prefs.tech_stack = ["python", "javascript", "docker", "sql"]
            logger.info("Sin stack configurado, usando stack por defecto")

        logger.info("=" * 60)
        logger.info("DAILY JOB SEARCH STARTING")
        logger.info(f"  Remoto: {prefs.remote_only}, Seniority: {prefs.seniority or 'indiferente'}")
        logger.info(f"  Cargos: {prefs.desired_titles}")
        logger.info("=" * 60)

        scrapers = [
            TecnoempleoScraper(max_offers=40, max_job_age_days=settings.max_job_age_days),
            InfojobsScraper(max_offers=40, max_job_age_days=settings.max_job_age_days),
            LinkedInScraper(max_offers=30, max_job_age_days=settings.max_job_age_days),
        ]

        all_offers = []
        for scraper in scrapers:
            logger.info(f"Scraping {scraper.source.value}...")
            try:
                result = await scraper.scrape()
                results["sources"][scraper.source.value] = {
                    "status": result.status.value,
                    "offers": result.offers_total,
                    "duration": round(result.duration_seconds, 1),
                }
                log_scrape_run(
                    source=scraper.source.value,
                    status=result.status.value,
                    offers_found=result.offers_total,
                    duration=result.duration_seconds,
                    error=result.error or "",
                )
                if result.offers:
                    inserted, skipped = bulk_insert_jobs(result.offers)
                    logger.info(
                        f"  {scraper.source.value}: {result.offers_total} found, "
                        f"{inserted} new, {skipped} duplicates"
                    )
                    all_offers.extend(result.offers)
                if result.error:
                    results["errors"].append(f"{scraper.source.value}: {result.error}")
            except Exception as e:
                logger.error(f"Scraper {scraper.source.value} crashed: {e}", exc_info=True)
                results["sources"][scraper.source.value] = {"status": "error", "error": str(e)}
                results["errors"].append(f"{scraper.source.value}: {str(e)}")
                log_scrape_run(
                    source=scraper.source.value,
                    status="error",
                    offers_found=0,
                    duration=0,
                    error=str(e)[:500],
                )

        total_found = len(all_offers)
        logger.info(f"Total offers scraped: {total_found} across all platforms")

        if all_offers:
            logger.info("Scoring offers...")
            scored = await score_all_unscored_jobs()
            logger.info(f"Scored {scored} offers")
            results["scored"] = scored

        top_jobs = get_pending_notification_jobs(limit=settings.max_jobs_per_day)

        if top_jobs:
            logger.info(f"Top {len(top_jobs)} jobs ready for delivery")
            for i, job in enumerate(top_jobs, 1):
                score = job.get("match_score", 0)
                logger.info(
                    f"  {i:2d}. [{job.get('source', ''):>12s}] {score:5.1f}% | "
                    f"{job.get('title', 'N/A')[:60]} | {job.get('company', 'N/A')[:30]}"
                )

            email_ok = await send_daily_email(top_jobs)
            if email_ok:
                mark_as_notified([j["id"] for j in top_jobs])
                results["email"] = "sent"
                logger.info(f"Email sent with {len(top_jobs)} jobs")
            else:
                results["email"] = "failed"
                logger.warning("Email delivery failed - jobs saved in DB for later")
        else:
            logger.info("No jobs matched the criteria today")
            results["email"] = "no_jobs"

        duration = time.time() - start_time
        results["status"] = "completed"
        results["duration_seconds"] = round(duration, 1)
        results["total_jobs_found"] = total_found
        results["total_jobs_delivered"] = len(top_jobs)

        _last_run_status = {
            "status": "ok",
            "last_run": datetime.now().isoformat(),
            "jobs_found": total_found,
            "jobs_delivered": len(top_jobs),
            "duration": round(duration, 1),
            "errors": results["errors"],
        }

        logger.info(f"DAILY RUN COMPLETED in {duration:.1f}s - {len(top_jobs)} jobs delivered")
        return results

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"DAILY RUN FAILED: {e}", exc_info=True)
        _last_run_status = {
            "status": "error",
            "last_run": datetime.now().isoformat(),
            "error": str(e),
            "duration": round(duration, 1),
        }
        results["status"] = "error"
        results["error"] = str(e)
        results["duration_seconds"] = round(duration, 1)
        return results

    finally:
        _is_running = False

        try:
            removed = cleanup_old_jobs(retention_days=30)
            if removed:
                logger.info(f"Cleaned up {removed} old jobs")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Jobs Scout starting up...")
    init_db()

    prefs = load_preferences()
    cv = load_cv_profile()

    if not cv and not prefs.tech_stack:
        logger.warning("No CV or preferences configured. Run: python src/setup.py")
    else:
        logger.info(f"CV loaded: {cv.full_name if cv else 'N/A'}")
        logger.info(f"Preferences: {prefs.desired_titles} | {prefs.location}")

    send_hour = settings.daily_send_hour
    send_minute = settings.daily_send_minute
    logger.info(f"Scheduler: daily at {send_hour:02d}:{send_minute:02d} ({settings.tz})")

    scheduler.add_job(
        run_daily_job,
        trigger=CronTrigger(hour=send_hour, minute=send_minute),
        id="daily_job_search",
        name="Daily job search and delivery",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    logger.info("Scheduler started")

    yield

    scheduler.shutdown(wait=False)
    logger.info("Jobs Scout shut down")


app = FastAPI(
    title="Jobs Scout",
    description="AI-powered daily job matching from LinkedIn, InfoJobs, and Tecnoempleo",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return JSONResponse({
        "service": "Jobs Scout",
        "version": "1.0.0",
        "status": "running",
        "last_run": _last_run_status,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return JSONResponse({
        "service_stats": get_stats(),
        "last_run": _last_run_status,
        "config": {
            "daily_send_hour": settings.daily_send_hour,
            "max_jobs_per_day": settings.max_jobs_per_day,
            "max_job_age_days": settings.max_job_age_days,
            "tz": settings.tz,
        },
    })


@app.get("/jobs/today")
async def jobs_today():
    jobs = get_top_jobs(limit=settings.max_jobs_per_day, days=settings.max_job_age_days)
    return JSONResponse({
        "date": datetime.now().isoformat(),
        "count": len(jobs),
        "jobs": [
            {
                "id": j["id"],
                "source": j["source"],
                "title": j["title"],
                "company": j["company"],
                "location": j["location"],
                "url": j["url"],
                "salary": j["salary"],
                "match_score": round(j["match_score"], 1),
                "remote": bool(j["remote"]),
                "hybrid": bool(j["hybrid"]),
                "seniority": j["seniority_level"],
                "technologies": (j["technologies_detected"] or "").split(",") if j["technologies_detected"] else [],
                "scrape_date": j["scrape_date"],
            }
            for j in jobs
        ],
    })


@app.post("/run")
async def trigger_run():
    if _is_running:
        raise HTTPException(status_code=429, detail="A job run is already in progress")

    task = asyncio.create_task(run_daily_job())
    return {"status": "triggered", "message": "Daily job search started"}


@app.post("/jobs/{job_id}/apply")
async def mark_applied(job_id: int):
    from src.database import mark_as_applied
    mark_as_applied(job_id)
    return {"status": "ok", "job_id": job_id, "action": "applied"}


@app.post("/jobs/{job_id}/discard")
async def mark_discarded(job_id: int):
    from src.database import mark_as_discarded
    mark_as_discarded(job_id)
    return {"status": "ok", "job_id": job_id, "action": "discarded"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    stats_data = get_stats()
    jobs = get_top_jobs(limit=50, days=settings.max_job_age_days)

    jobs_html = ""
    for j in jobs:
        score = j.get("match_score", 0)
        score_color = "#0d8a3a" if score >= 70 else "#b8760b" if score >= 40 else "#64748b"
        source_badge = {
            "linkedin": ("LinkedIn", "#0a66c2"),
            "infojobs": ("InfoJobs", "#e11d48"),
            "tecnoempleo": ("Tecnoempleo", "#065f46"),
        }.get(j.get("source", ""), (j.get("source", "?"), "#666"))

        badges = ""
        if j.get("remote"):
            badges += '<span style="background:#e0f2fe;color:#0369a1;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px;">Remoto</span>'
        if j.get("hybrid"):
            badges += '<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px;">Híbrido</span>'
        if j.get("sector"):
            badges += f'<span style="background:#f1f5f9;color:#475569;padding:2px 6px;border-radius:3px;font-size:11px;">{j["sector"]}</span>'

        techs = (j.get("technologies_detected") or "").split(",")
        techs_html = " ".join(
            f'<span style="background:#f1f5f9;color:#475569;padding:2px 6px;border-radius:3px;font-size:11px;">{t}</span>'
            for t in techs[:6] if t
        )

        jobs_html += f"""
        <div style="border:1px solid #e8ecf1;border-radius:10px;padding:16px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;">
                <h3 style="margin:0;font-size:16px;">{j.get('title', 'N/A')}</h3>
                <span style="color:{score_color};font-weight:700;font-size:14px;">{score:.0f}%</span>
            </div>
            <div style="margin:8px 0;color:#64748b;font-size:13px;">
                {j.get('company', '')} &bull; &#128205; {j.get('location', '')}
                <span style="background:{source_badge[1]};color:white;padding:2px 8px;border-radius:3px;font-size:11px;margin-left:8px;">{source_badge[0]}</span>
            </div>
            <div style="margin-bottom:8px;">{badges}</div>
            <div style="margin-bottom:8px;">{techs_html}</div>
            <div style="color:#64748b;font-size:13px;margin-bottom:8px;">&#128176; {j.get('salary') or 'No especificado'}</div>
            <a href="{j.get('url', '#')}" target="_blank" style="display:inline-block;padding:6px 16px;background:#667eea;color:white;text-decoration:none;border-radius:5px;font-size:13px;font-weight:600;">Ver oferta &rarr;</a>
            <a href="/jobs/{j.get('id')}/apply" style="display:inline-block;padding:6px 16px;background:#0d8a3a;color:white;text-decoration:none;border-radius:5px;font-size:13px;font-weight:600;margin-left:8px;">&#10003; Aplicada</a>
            <a href="/jobs/{j.get('id')}/discard" style="display:inline-block;padding:6px 16px;background:#e11d48;color:white;text-decoration:none;border-radius:5px;font-size:13px;font-weight:600;margin-left:8px;">&#10007; Descartar</a>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jobs Scout - Dashboard</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#f5f7fa; color:#1a1a2e; padding:20px; }}
        .container {{ max-width:800px; margin:0 auto; }}
        .header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:24px; border-radius:12px; margin-bottom:24px; }}
        .header h1 {{ font-size:24px; }}
        .stats {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
        .stat-card {{ flex:1; min-width:120px; background:white; padding:16px; border-radius:10px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
        .stat-card .num {{ font-size:28px; font-weight:700; color:#667eea; }}
        .stat-card .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; margin-top:4px; }}
        .refresh-btn {{ display:inline-block;padding:8px 20px;background:#667eea;color:white;text-decoration:none;border-radius:6px;font-weight:600;margin-bottom:16px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>&#128269; Jobs Scout Dashboard</h1>
            <p style="opacity:.8;margin-top:4px;">&Uacute;ltima ejecuci&oacute;n: {_last_run_status.get('last_run', 'Nunca')}</p>
        </div>
        <div class="stats">
            <div class="stat-card">
                <div class="num">{stats_data['total_jobs']}</div>
                <div class="label">Ofertas totales</div>
            </div>
            <div class="stat-card">
                <div class="num">{stats_data['scored_jobs']}</div>
                <div class="label">Evaluadas</div>
            </div>
            <div class="stat-card">
                <div class="num">{stats_data['notified_jobs']}</div>
                <div class="label">Notificadas</div>
            </div>
            <div class="stat-card">
                <div class="num">{stats_data['applied_jobs']}</div>
                <div class="label">Aplicadas</div>
            </div>
        </div>
        <a href="/run" class="refresh-btn">&#128260; Forzar b&uacute;squeda ahora</a>
        {jobs_html}
    </div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Jobs Scout")
    parser.add_argument("--run-now", action="store_true", help="Run a job search immediately and exit")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Port (default: {settings.port})")
    args = parser.parse_args()

    if args.run_now:
        logger.info("Running manual job search...")
        init_db()
        asyncio.run(run_daily_job())
        logger.info("Manual run complete. Exiting.")
        sys.exit(0)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
