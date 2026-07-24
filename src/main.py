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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import (
    TEMPLATES_DIR,
    load_preferences,
    save_preferences,
    load_cv_profile,
    save_cv_profile,
    load_config,
    save_config,
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
    get_stats,
    log_scrape_run,
    get_db,
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
_rescore_next = False
_last_run_status: dict = {"status": "idle", "last_run": None, "errors": []}

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def get_scrape_runs(limit: int = 30) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scrape_runs ORDER BY run_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


async def run_daily_job() -> dict:
    global _is_running, _last_run_status
    if _is_running:
        logger.warning("A daily run is already in progress, skipping")
        return {"status": "skipped", "reason": "already_running"}

    _is_running = True
    start_time = time.time()
    results = {"status": "running", "started_at": datetime.now().isoformat(), "sources": {}, "errors": []}

    try:
        global _rescore_next
        prefs = load_preferences()
        cv = load_cv_profile()

        if not prefs.desired_titles:
            prefs.desired_titles = list(_TITULOS_POR_DEFECTO)
        if not prefs.tech_stack:
            prefs.tech_stack = list(_TECH_STACK_POR_DEFECTO)

        for key, value in _PREFS_FIJOS.items():
            setattr(prefs, key, value)

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

        if all_offers or _rescore_next:
            logger.info("Puntuando ofertas...")
            scored = await score_all_unscored_jobs(rescore_all=_rescore_next)
            logger.info(f"Puntuadas {scored} ofertas")
            results["scored"] = scored
            _rescore_next = False

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
    global _rescore_next
    if _is_running:
        raise HTTPException(status_code=429, detail="Una búsqueda ya está en progreso")
    _rescore_next = True
    task = asyncio.create_task(run_daily_job())
    return {"status": "triggered", "message": "Búsqueda diaria iniciada"}


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
    scrape_runs = get_scrape_runs(limit=30)

    last_run = _last_run_status.get("last_run")
    if last_run:
        last_run_text = f"Última búsqueda: {last_run[:16].replace('T', ' ')}"
    else:
        last_run_text = "Sin búsquedas aún"

    profile_feed_html = _build_profile_feed_html(cv) if cv else ""

    template = _jinja_env.get_template("dashboard.html")
    return template.render(
        stats=stats_data,
        jobs=jobs,
        cv=cv,
        prefs=prefs,
        scrape_runs=scrape_runs,
        last_run_text=last_run_text,
        settings=settings,
        profile_feed_html=profile_feed_html,
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
