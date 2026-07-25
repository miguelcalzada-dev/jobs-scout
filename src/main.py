from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import (
    TEMPLATES_DIR,
    load_preferences,
    save_preferences,
    load_cv_profile,
    save_cv_profile,
    settings,
    CVProfile,
)
from src.cv_parser import parse_cv_pdf
from src.database import (
    init_db,
    bulk_insert_jobs,
    get_top_jobs,
    get_pending_notification_jobs,
    mark_as_notified,
    cleanup_old_jobs,
    cleanup_old_scrape_runs,
    clear_all_jobs,
    get_stats,
    log_scrape_run,
    get_scrape_runs_paged,
    get_db,
)
from src.delivery import send_daily_email
from src.matcher import score_all_unscored_jobs
from src.scraper import ScrapeResult, ScrapeStatus
from src.scrapers import TecnoempleoScraper, InfojobsScraper
from src.autoapply import run_autoapply, get_autoapply_results

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

_RATE_LIMIT_SECONDS = 60
_last_trigger_time: datetime | None = None

# Per-source target so the combined batch is ~50 fresh offers.
_TARGET_PER_SOURCE = 25

_SCRAPER_MAP = {
    "tecnoempleo": TecnoempleoScraper,
    "infojobs": InfojobsScraper,
}

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def get_scrape_runs_page(page: int = 1, per_page: int = 25) -> dict:
    return get_scrape_runs_paged(page=page, per_page=per_page, max_runs=50)


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

        # Ensure non-fixed preferences always have sensible defaults (these do
        # NOT override values already saved by the user — only fill empties).
        if not prefs.desired_titles:
            prefs.desired_titles = list(_TITULOS_POR_DEFECTO)
        if not prefs.tech_stack:
            prefs.tech_stack = list(_TECH_STACK_POR_DEFECTO)

        logger.info("=" * 60)
        logger.info("JOB SEARCH STARTING")
        logger.info(f"  Remoto: {prefs.remote_only}, Seniority: {prefs.seniority or 'indiferente'}")
        logger.info(f"  Cargos: {prefs.desired_titles}")
        logger.info("=" * 60)

        # Fresh batch: wipe prior job_offers so each search is a clean 50-offer lot.
        removed = clear_all_jobs()
        if removed:
            logger.info(f"Cleared {removed} prior job rows for a fresh batch")

        scraper_classes = [
            cls for name, cls in _SCRAPER_MAP.items()
            if name in prefs.enabled_scrapers
        ]
        if not scraper_classes:
            logger.warning("No scrapers enabled in preferences")
            results["status"] = "completed"
            results["error"] = "no_scrapers_enabled"
            _is_running = False
            return results

        scrapers = [
            cls(max_offers=_TARGET_PER_SOURCE, max_job_age_days=settings.max_job_age_days)
            for cls in scraper_classes
        ]

        async def _scrape_with_log(scraper):
            logger.info(f"Scraping {scraper.source.value}...")
            try:
                result = await scraper.scrape()
                return scraper, result
            except Exception as e:
                logger.error(f"Scraper {scraper.source.value} crashed: {e}", exc_info=True)
                fake = ScrapeResult(
                    source=scraper.source,
                    status=ScrapeStatus.FAILED,
                    error=str(e),
                )
                return scraper, fake

        # Parallel scraping (asyncio.gather) — httpx-based scrapers don't block.
        scrape_tasks = [_scrape_with_log(s) for s in scrapers]
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        all_offers = []
        external_count = 0
        for item in scrape_results:
            if isinstance(item, BaseException):
                logger.error(f"Scraper task crashed: {item}")
                continue
            scraper, result = item
            source_name = scraper.source.value
            inserted_offers = [o for o in (result.offers or []) if not o.is_external_redirect]
            external_in_batch = sum(1 for o in (result.offers or []) if o.is_external_redirect)
            results["sources"][source_name] = {
                "status": result.status.value,
                "offers": len(inserted_offers),
                "external_skipped": external_in_batch,
                "duration": round(result.duration_seconds, 1),
            }
            log_scrape_run(
                source=source_name,
                status=result.status.value,
                offers_found=len(inserted_offers),
                duration=result.duration_seconds,
                error=result.error or "",
            )
            if result.offers:
                inserted, skipped = bulk_insert_jobs(result.offers)
                logger.info(
                    f"  {source_name}: {len(result.offers)} found "
                    f"({external_in_batch} external skipped), "
                    f"{inserted} new, {skipped} duplicates"
                )
                all_offers.extend(inserted_offers)
                external_count += external_in_batch
            if result.error:
                results["errors"].append(f"{source_name}: {result.error}")

        total_found = len(all_offers)
        logger.info(f"Total applicable offers scraped: {total_found} ({external_count} externas descartadas)")

        if all_offers:
            logger.info("Puntuando ofertas...")
            scored = await score_all_unscored_jobs()
            logger.info(f"Puntuadas {scored} ofertas")
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

        logger.info(f"RUN COMPLETED in {duration:.1f}s - {len(top_jobs)} jobs delivered")
        return results

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"RUN FAILED: {e}", exc_info=True)
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

        # Defensive: prune any stale rows older than 30 days (each run already
        # clears the table via clear_all_jobs, this just protects the scrape_runs
        # and any rows inserted by a manual DB poke).
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

    # On server restart we do NOT overwrite user preferences: only fill the
    # non-fixed list fields when they are empty (first run). Fixed fields
    # (seniority, location, etc.) are honored exactly as saved by the user.
    needs_save = False
    if not prefs.desired_titles:
        prefs.desired_titles = list(_TITULOS_POR_DEFECTO)
        needs_save = True
    if not prefs.tech_stack:
        prefs.tech_stack = list(_TECH_STACK_POR_DEFECTO)
        needs_save = True
    if not prefs.enabled_scrapers:
        prefs.enabled_scrapers = ["tecnoempleo", "infojobs"]
        needs_save = True
    if needs_save:
        save_preferences(prefs)

    cv = load_cv_profile()

    if not cv and not prefs.tech_stack:
        logger.warning("No CV or preferences configured via web dashboard")
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
    description="Buscador diario de ofertas con IA - LinkedIn, InfoJobs, Tecnoempleo",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")


# Inline SVG favicon (jobs-scout magnifying glass). Served as data so no extra
# file needs to ship — keeps Railway compatibility and avoids static mounts.
_FAVICON_SVG = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0a84ff"/>
      <stop offset="1" stop-color="#1d1d1f"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#g)"/>
  <circle cx="27" cy="27" r="13" fill="none" stroke="#fff" stroke-width="4"/>
  <line x1="37" y1="37" x2="50" y2="50" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
</svg>"""


@app.get("/favicon.ico")
async def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/health")
async def health():
    checks = {}

    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    try:
        prefs = load_preferences()
        enabled = [s for s in prefs.enabled_scrapers if s in _SCRAPER_MAP]
        checks["scrapers"] = {
            "enabled": enabled,
            "total": len(_SCRAPER_MAP),
        }
        checks["scrapers_enabled"] = len(enabled) > 0
    except Exception as e:
        checks["scrapers"] = f"error: {e}"
        checks["scrapers_enabled"] = False

    try:
        resend_key = os.getenv("RESEND_API_KEY", "")
        checks["email_configured"] = bool(resend_key)
    except Exception:
        checks["email_configured"] = False

    try:
        cv = load_cv_profile()
        checks["cv_loaded"] = cv is not None
    except Exception:
        checks["cv_loaded"] = False

    stats_data = get_stats()

    overall = all([
        checks.get("database") == "ok",
        checks.get("scrapers_enabled", False),
        checks.get("email_configured", False),
    ])

    return {
        "status": "healthy" if overall else "degraded",
        "checks": checks,
        "scheduler_running": scheduler.running,
        "last_run": _last_run_status,
        "stats": stats_data,
    }


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
    global _last_trigger_time
    if _is_running:
        raise HTTPException(status_code=429, detail="Una búsqueda ya está en progreso")
    if _last_trigger_time and (datetime.now() - _last_trigger_time).total_seconds() < _RATE_LIMIT_SECONDS:
        remaining = int(_RATE_LIMIT_SECONDS - (datetime.now() - _last_trigger_time).total_seconds())
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: espera {remaining}s antes de otra ejecución manual",
        )
    _last_trigger_time = datetime.now()
    task = asyncio.create_task(run_daily_job())
    return {"status": "triggered", "message": "Búsqueda iniciada — lote fresco de ~50 ofertas"}


@app.post("/rescore")
async def trigger_rescore():
    from src.matcher import score_all_unscored_jobs
    if _is_running:
        raise HTTPException(status_code=429, detail="Espera a que termine la búsqueda actual")
    scored = await score_all_unscored_jobs(rescore_all=True)
    return {"status": "ok", "scored": scored, "message": f"{scored} ofertas re-evaluadas con las reglas actuales"}


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


@app.get("/jobs/search")
async def search_jobs(q: str = "", source: str = "", min_score: float = 0, limit: int = 100):
    with get_db() as conn:
        conditions = ["match_score > 0", "discarded = 0", "is_external_redirect = 0"]
        params = []

        if q:
            conditions.append("(title LIKE ? OR company LIKE ? OR description LIKE ? OR technologies_detected LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like, like])
        if source and source in _SCRAPER_MAP:
            conditions.append("source = ?")
            params.append(source)
        if min_score > 0:
            conditions.append("match_score >= ?")
            params.append(min_score)

        cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
        conditions.append("scrape_date >= ?")
        params.append(cutoff)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"""SELECT id, source, external_id, title, company, location, url,
                      salary, match_score, remote, hybrid, technologies_detected,
                      seniority_level, scrape_date
                FROM job_offers WHERE {where}
                ORDER BY match_score DESC LIMIT ?""",
            params + [limit],
        ).fetchall()

    return {
        "count": len(rows),
        "jobs": [
            {
                "id": r["id"],
                "source": r["source"],
                "title": r["title"],
                "company": r["company"],
                "location": r["location"],
                "url": r["url"],
                "salary": r["salary"],
                "match_score": round(r["match_score"], 1),
                "remote": bool(r["remote"]),
                "hybrid": bool(r["hybrid"]),
                "technologies": (r["technologies_detected"] or "").split(",") if r["technologies_detected"] else [],
                "scrape_date": r["scrape_date"],
            }
            for r in rows
        ],
    }


@app.get("/stats/history")
async def stats_history():
    """Kept for backwards compatibility — returns a small summary, no charts."""
    return get_scrape_runs_paged(page=1, per_page=25, max_runs=50)


@app.get("/history")
async def history(page: int = 1, per_page: int = 25):
    page = max(1, min(page, 200))
    per_page = max(5, min(per_page, 100))
    return get_scrape_runs_paged(page=page, per_page=per_page, max_runs=50)


@app.post("/history/clear")
async def history_clear(days: int = 30):
    days = max(1, min(days, 365))
    removed = cleanup_old_scrape_runs(retention_days=days)
    logger.info(f"Cleared {removed} scrape_runs older than {days} days")
    return {"status": "ok", "removed": removed}


@app.post("/upload-cv")
async def upload_cv(cv_file: UploadFile = File(...)):
    if not cv_file.filename or not cv_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    cv_dir = Path(__file__).resolve().parent.parent / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    dest = cv_dir / cv_file.filename
    content = await cv_file.read()
    dest.write_bytes(content)

    try:
        profile = await parse_cv_pdf(dest)
        save_cv_profile(profile)

        _auto_update_preferences_from_cv(profile)

        return {
            "status": "ok",
            "filename": cv_file.filename,
            "full_name": profile.full_name,
            "technologies": profile.technologies,
            "experience_years": profile.experience_years,
        }
    except Exception as e:
        logger.error(f"Failed to parse CV: {e}")
        raise HTTPException(status_code=400, detail=f"Error al analizar el CV: {str(e)}")


_TITULOS_POR_DEFECTO = [
    "Desarrollador Backend", "Desarrollador Full-Stack", "Desarrollador IA",
    "Backend Developer", "Full-Stack Developer", "AI Developer",
    "Python Developer", "Software Engineer", "Backend Python Developer", "Flask Developer",
]

_TECH_STACK_POR_DEFECTO = [
    "python", "typescript", "java", "react", "angular", "next.js", "node.js",
    "flask", "spring boot", "spring", "docker", "kubernetes", "postgresql",
    "mysql", "mongodb", "rabbitmq", "sql", "nosql", "git", "agile", "scrum",
    "ci/cd", "microservicios", "machine learning", "deep learning", "nlp",
]

_PREFS_FIJOS = {
    "seniority": "junior",
    "location": "Madrid",
    "remote_only": False,
    "hybrid_allowed": True,
    "onsite_allowed": False,
    "min_salary": 0,
    "exclude_keywords": [],
    "exclude_sectors": [],
}

_TITULOS_DERIVADOS = {
    "python": ["Python Developer", "Backend Python Developer"],
    "django": ["Django Developer", "Python Backend Developer"],
    "fastapi": ["FastAPI Developer", "Backend API Developer"],
    "flask": ["Flask Developer", "Python Web Developer"],
    "javascript": ["Full-Stack Developer", "Frontend Developer"],
    "typescript": ["TypeScript Developer", "Full-Stack Developer"],
    "react": ["React Developer", "Full-Stack Developer"],
    "angular": ["Angular Developer", "Frontend Developer"],
    "vue": ["Vue Developer", "Frontend Developer"],
    "node.js": ["Node.js Developer", "Backend Developer"],
    "nodejs": ["Node.js Developer", "Backend Developer"],
    "java": ["Java Developer", "Backend Java Developer"],
    "c#": [".NET Developer", "C# Developer"],
    ".net": [".NET Developer", "Backend .NET Developer"],
    "go": ["Go Developer", "Backend Go Developer"],
    "golang": ["Go Developer", "Backend Go Developer"],
    "rust": ["Rust Developer", "Systems Developer"],
    "swift": ["iOS Developer", "Swift Developer"],
    "kotlin": ["Android Developer", "Kotlin Developer"],
    "aws": ["Cloud Engineer", "AWS Developer"],
    "azure": ["Cloud Engineer", "Azure Developer"],
    "gcp": ["Cloud Engineer", "GCP Developer"],
    "docker": ["DevOps Engineer", "Cloud Developer"],
    "kubernetes": ["DevOps Engineer", "Platform Engineer"],
    "terraform": ["DevOps Engineer", "Infrastructure Engineer"],
    "pytorch": ["Machine Learning Engineer", "AI Developer"],
    "tensorflow": ["Machine Learning Engineer", "AI Developer"],
    "machine learning": ["Machine Learning Engineer", "AI Engineer"],
    "deep learning": ["Deep Learning Engineer", "AI Engineer"],
    "nlp": ["NLP Engineer", "AI Developer"],
    "spark": ["Data Engineer", "Big Data Developer"],
    "hadoop": ["Data Engineer", "Big Data Developer"],
    "sql": ["Database Developer", "Backend Developer"],
    "postgresql": ["Backend Developer", "Database Developer"],
    "mongodb": ["Backend Developer", "Database Developer"],
    "graphql": ["Backend Developer", "API Developer"],
    "kafka": ["Data Engineer", "Backend Developer"],
}


def _auto_update_preferences_from_cv(profile):
    from src.config import JobPreferences, save_preferences

    prefs = load_preferences()

    if profile.technologies:
        existing_techs = set(t.lower().strip() for t in prefs.tech_stack if t)
        for tech in profile.technologies:
            t_lower = tech.lower().strip()
            if t_lower and t_lower not in existing_techs:
                prefs.tech_stack.append(tech)
                existing_techs.add(t_lower)

    prefs.desired_titles = list(dict.fromkeys(_TITULOS_POR_DEFECTO + prefs.desired_titles))[:15]

    for key, value in _PREFS_FIJOS.items():
        setattr(prefs, key, value)

    if profile.technologies:
        techs_lower = [t.lower() for t in profile.technologies]
        for tech_key, derived in _TITULOS_DERIVADOS.items():
            if tech_key in techs_lower:
                for d in derived:
                    if d not in _TITULOS_POR_DEFECTO and d not in prefs.desired_titles:
                        prefs.desired_titles.append(d)

    save_preferences(prefs)
    logger.info(
        f"Preferencias actualizadas desde CV: {len(prefs.tech_stack)} techs, "
        f"seniority={prefs.seniority}, {len(prefs.desired_titles)} títulos"
    )


@app.post("/preferences")
async def update_preferences(data: dict):
    try:
        from src.config import JobPreferences

        prefs = load_preferences()

        if "desired_titles" in data:
            prefs.desired_titles = data["desired_titles"]
        if "tech_stack" in data:
            prefs.tech_stack = data["tech_stack"]
        if "location" in data:
            prefs.location = data["location"]
        if "min_salary" in data:
            prefs.min_salary = int(data["min_salary"])
        if "seniority" in data:
            prefs.seniority = data["seniority"]
        if "remote_only" in data:
            prefs.remote_only = bool(data["remote_only"])
        if "hybrid_allowed" in data:
            prefs.hybrid_allowed = bool(data["hybrid_allowed"])
        if "onsite_allowed" in data:
            prefs.onsite_allowed = bool(data["onsite_allowed"])
        if "exclude_keywords" in data:
            prefs.exclude_keywords = data["exclude_keywords"]
        if "exclude_sectors" in data:
            prefs.exclude_sectors = data["exclude_sectors"]
        if "enabled_scrapers" in data:
            prefs.enabled_scrapers = data["enabled_scrapers"]

        save_preferences(prefs)
        return {"status": "ok", "message": "Preferencias guardadas correctamente"}

    except Exception as e:
        logger.error(f"Failed to save preferences: {e}")
        raise HTTPException(status_code=400, detail=f"Error al guardar preferencias: {str(e)}")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    stats_data = get_stats()
    jobs = get_top_jobs(limit=50, days=settings.max_job_age_days)
    cv = load_cv_profile()
    prefs = load_preferences()
    history_page = get_scrape_runs_page(page=1, per_page=25)

    last_run = _last_run_status.get("last_run")
    if last_run:
        last_run_text = f"Última búsqueda: {last_run[:16].replace('T', ' ')}"
    else:
        last_run_text = "Sin búsquedas aún"

    profile_feed_html = _build_profile_feed_html(cv) if cv else ""
    autoapply_data = get_autoapply_results()

    template = _jinja_env.get_template("dashboard.html")
    return template.render(
        stats=stats_data,
        jobs=jobs,
        cv=cv,
        prefs=prefs,
        history=history_page,
        last_run_text=last_run_text,
        settings=settings,
        profile_feed_html=profile_feed_html,
        autoapply=autoapply_data,
    )


def _build_profile_feed_html(cv) -> str:
    if not cv:
        return ""
    initials = "".join([w[0] for w in (cv.full_name or "CV").split()[:2]]).upper() or "?"

    html = '<div class="profile-header">'
    html += f'<div class="profile-avatar">{initials}</div>'
    html += '<div class="profile-name-box">'
    html += f'<h3>{cv.full_name or "Perfil CV"}</h3>'
    if cv.email:
        html += f'<div class="email">{cv.email}</div>'
    if cv.linkedin:
        html += f'<div class="linkedin-link">{cv.linkedin}</div>'
    html += '</div></div>'

    html += '<div class="profile-stats-row">'
    html += f'<div class="profile-stat"><span class="ps-val">{cv.experience_years}</span><span class="ps-lbl">Años exp.</span></div>'
    html += f'<div class="profile-stat"><span class="ps-val">{len(cv.technologies)}</span><span class="ps-lbl">Tecnologías</span></div>'
    html += f'<div class="profile-stat"><span class="ps-val">{len(cv.work_history)}</span><span class="ps-lbl">Empresas</span></div>'
    html += f'<div class="profile-stat"><span class="ps-val">{len(cv.languages)}</span><span class="ps-lbl">Idiomas</span></div>'
    html += '</div>'

    if cv.summary:
        html += f'<div class="profile-section"><h4>&#128172; Resumen</h4><div class="profile-summary">{cv.summary[:500]}</div></div>'

    if cv.technologies:
        html += '<div class="profile-section"><h4>&#128187; Tecnologías</h4><div class="tech-grid">'
        for tech in cv.technologies[:20]:
            html += f'<span class="tech-badge">{tech}</span>'
        html += '</div></div>'

    if cv.work_history:
        html += '<div class="profile-section"><h4>&#128188; Experiencia</h4>'
        for w in cv.work_history[:6]:
            html += '<div class="timeline-item">'
            html += f'<div class="ti-title">{w.get("title", "")}</div>'
            html += f'<div class="ti-sub">{w.get("company", "")}'
            if w.get("period"):
                html += f' &middot; {w["period"]}'
            html += '</div>'
            if w.get("description"):
                html += f'<div class="ti-desc">{str(w["description"])[:200]}</div>'
            html += '</div>'
        html += '</div>'

    if cv.education:
        html += '<div class="profile-section"><h4>&#127891; Formación</h4>'
        for e in cv.education[:4]:
            html += '<div class="timeline-item">'
            html += f'<div class="ti-title">{e.get("degree", "")}</div>'
            html += f'<div class="ti-sub">{e.get("institution", "")}'
            if e.get("year"):
                html += f' &middot; {e["year"]}'
            html += '</div></div>'
        html += '</div>'

    if cv.languages:
        html += '<div class="profile-section"><h4>&#127760; Idiomas</h4><div class="tech-grid">'
        for lang in cv.languages:
            html += f'<span class="tech-badge">{lang}</span>'
        html += '</div></div>'

    if cv.certs:
        html += '<div class="profile-section"><h4>&#127942; Certificaciones</h4><div class="tech-grid">'
        for cert in cv.certs:
            html += f'<span class="tech-badge">{cert}</span>'
        html += '</div></div>'

    return html


@app.get("/profile")
async def get_profile():
    cv = load_cv_profile()
    if not cv:
        raise HTTPException(status_code=404, detail="No hay CV cargado")
    return {
        "full_name": cv.full_name,
        "email": cv.email,
        "linkedin": cv.linkedin,
        "summary": cv.summary,
        "experience_years": cv.experience_years,
        "technologies": cv.technologies,
        "skills": cv.skills,
        "work_history": cv.work_history,
        "education": cv.education,
        "languages": cv.languages,
        "certifications": cv.certs,
    }


@app.post("/auto-apply")
async def trigger_autoapply(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    min_score = int(body.get("min_score", 75))
    email = body.get("email", "")
    password = body.get("password", "")
    if email:
        os.environ["TECNOEMPLEO_EMAIL"] = email
    if password:
        os.environ["TECNOEMPLEO_PASSWORD"] = password
    task = asyncio.create_task(run_autoapply(min_score=min_score))
    return {"status": "triggered", "min_score": min_score, "message": "Auto-apply iniciado en segundo plano"}


@app.get("/auto-apply/status")
async def autoapply_status():
    return get_autoapply_results()


@app.post("/auto-apply/cancel")
async def autoapply_cancel():
    from src.autoapply import cancel_autoapply
    cancel_autoapply()
    return {"status": "ok", "message": "Cancelación solicitada (se detendrá tras la oferta actual)"}


@app.get("/auto-apply/eligible-count")
async def eligible_count(min_score: int = 50):
    with get_db() as conn:
        cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
        count = conn.execute(
            "SELECT COUNT(*) FROM job_offers WHERE source = 'tecnoempleo' "
            "AND match_score >= ? AND applied = 0 AND discarded = 0 "
            "AND is_external_redirect = 0 AND scrape_date >= ?",
            (min_score, cutoff),
        ).fetchone()[0]
    return {"count": count, "min_score": min_score}


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Jobs Scout")
    parser.add_argument("--run-now", action="store_true", help="Ejecutar una búsqueda inmediata y salir")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Puerto (por defecto: {settings.port})")
    args = parser.parse_args()

    if args.run_now:
        logger.info("Ejecutando búsqueda manual...")
        init_db()
        asyncio.run(run_daily_job())
        logger.info("Búsqueda manual completada. Saliendo.")
        sys.exit(0)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
