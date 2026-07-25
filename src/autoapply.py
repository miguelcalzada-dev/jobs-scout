from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import load_cv_profile, CV_DIR, settings

logger = logging.getLogger(__name__)

# Bounded result buffer (was unbounded — memory leak on long-running processes).
MAX_KEPT_RESULTS = 100
AUTOAPPLY_RESULTS: list[dict] = []
_autoapply_running = False
_min_auto_score = 75
_last_autoapply_error = ""

TECNOEMPLEO_LOGIN_URL = "https://www.tecnoempleo.com/login.php"
TECNOEMPLEO_BASE = "https://www.tecnoempleo.com"


async def run_autoapply(min_score: int = 75) -> dict:
    global _autoapply_running, AUTOAPPLY_RESULTS, _min_auto_score
    if _autoapply_running:
        return {"status": "skipped", "reason": "already_running"}

    email = os.getenv("TECNOEMPLEO_EMAIL", "") or settings.tecnoempleo_email
    password = os.getenv("TECNOEMPLEO_PASSWORD", "") or settings.tecnoempleo_password

    if not email or not password:
        _last_autoapply_error = "missing_credentials"
        return {"status": "error", "reason": "missing_credentials", "hint": "Configura TECNOEMPLEO_EMAIL y TECNOEMPLEO_PASSWORD en Railway o en el formulario"}

    _autoapply_running = True
    _min_auto_score = min_score
    _last_autoapply_error = ""
    AUTOAPPLY_RESULTS = []

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        _autoapply_running = False
        return {"status": "error", "reason": "playwright_not_installed"}

    from src.database import get_db, mark_as_applied

    cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, source, external_id, title, company, location, url,
                   description, match_score, scrape_date
            FROM job_offers
            WHERE source = 'tecnoempleo'
              AND match_score >= ?
              AND applied = 0
              AND discarded = 0
              AND is_external_redirect = 0
              AND scrape_date >= ?
            ORDER BY match_score DESC
        """, (min_score, cutoff)).fetchall()

    jobs = [dict(r) for r in rows]
    if not jobs:
        _autoapply_running = False
        logger.info("AutoApply: no hay ofertas que cumplan el criterio")
        return {"status": "completed", "applied": 0, "total": 0, "results": []}

    logger.info(f"AutoApply: {len(jobs)} ofertas candidatas (score >= {min_score})")

    cv = load_cv_profile()
    cv_pdf_path = _find_cv_pdf()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )

        try:
            context = await browser.new_context(
                viewport={"width": 1366, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                locale="es-ES",
            )
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)

            page = await context.new_page()

            logged_in = await _login_tecnoempleo(page, email, password)
            if not logged_in:
                _last_autoapply_error = "login_failed"
                _autoapply_running = False
                return {"status": "error", "reason": "login_failed", "hint": "Verifica email/contraseña de Tecnoempleo"}

            for job in jobs:
                if not _autoapply_running:
                    break
                result = await _apply_to_job(page, job, cv, cv_pdf_path)
                AUTOAPPLY_RESULTS.append(result)
                # Keep buffer bounded to avoid unbounded growth.
                if len(AUTOAPPLY_RESULTS) > MAX_KEPT_RESULTS:
                    del AUTOAPPLY_RESULTS[: len(AUTOAPPLY_RESULTS) - MAX_KEPT_RESULTS]
                logger.info(
                    f"AutoApply: [{result['status']}] {job['title'][:60]} "
                    f"({job['company']}) — {result.get('detail', '')}"
                )
                await asyncio.sleep(3.0)

        finally:
            await browser.close()

    applied_count = sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "applied")
    _autoapply_running = False
    logger.info(f"AutoApply completado: {applied_count}/{len(jobs)} aplicadas")
    return {
        "status": "completed",
        "applied": applied_count,
        "total": len(jobs),
        "results": AUTOAPPLY_RESULTS,
    }


async def _login_tecnoempleo(page, email: str, password: str) -> bool:
    try:
        logger.info("AutoApply: iniciando sesión en Tecnoempleo...")
        await page.goto(TECNOEMPLEO_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        email_input = await page.query_selector("input[type=email], input[name=email], input[id*=email], input[name*=email]")
        if not email_input:
            email_input = await page.query_selector("input[type=text]:not([name*=search])")

        pass_input = await page.query_selector("input[type=password]")

        if not email_input or not pass_input:
            logger.warning("AutoApply: no se encontraron campos de login")
            return False

        await email_input.fill(email)
        await pass_input.fill(password)

        submit_btn = await page.query_selector("button[type=submit], input[type=submit], button.btn-primary, button:has-text('Entrar'), button:has-text('Iniciar'), button:has-text('Acceder')")
        if not submit_btn:
            submit_btn = await page.query_selector("form button, form input[type=submit]")

        if submit_btn:
            await submit_btn.click()
        else:
            await pass_input.press("Enter")

        await page.wait_for_timeout(5000)

        current_url = page.url.lower()
        page_title = (await page.title()).lower()

        logged_in = (
            "login" not in current_url
            or "candidato" in current_url
            or "dashboard" in current_url
            or "mi-cuenta" in current_url
            or "micuenta" in current_url
            or "panel" in current_url
            or "cuenta" in page_title
            or "candidato" in page_title
        )

        if not logged_in:
            ctx = page.context
            cookies = await ctx.cookies()
            auth_cookies = [c for c in cookies if c.get("name", "").lower() in ("session", "tecnouser", "user", "login", "token", "phpsessid", "tecnosession")]
            if auth_cookies:
                logged_in = True

        if not logged_in:
            body_text = (await page.content()).lower()
            if any(kw in body_text for kw in ["mi cuenta", "micuenta", "cerrar sesión", "cerrar sesion", "panel de control", "mis ofertas", "mis candidaturas"]):
                logged_in = True

        if not logged_in:
            error_el = await page.query_selector("[class*=error], [class*=alert], .alert-danger")
            if error_el:
                error_text = await error_el.inner_text()
                logger.warning(f"AutoApply: login fallido — {error_text[:200]}")
            else:
                logger.warning(f"AutoApply: login fallido — URL actual: {current_url}")
            return False

        logger.info("AutoApply: sesión iniciada correctamente")
        return True

    except Exception as e:
        logger.error(f"AutoApply: error en login: {e}")
        return False


async def _apply_to_job(page, job: dict, cv, cv_pdf_path: Optional[Path]) -> dict:
    job_id = job["id"]
    job_url = job.get("url", "")
    job_title = job.get("title", "")
    job_company = job.get("company", "")
    job_desc = job.get("description", "")

    result = {
        "job_id": job_id,
        "title": job_title,
        "company": job_company,
        "url": job_url,
        "status": "skipped",
        "detail": "",
        "timestamp": datetime.now().isoformat(),
    }

    if not job_url:
        result["detail"] = "sin URL"
        return result

    try:
        await page.goto(job_url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(3000)

        current_url = page.url
        if TECNOEMPLEO_BASE not in current_url:
            result["status"] = "skipped"
            result["detail"] = "redirige a web externa"
            return result

        apply_btn = await page.query_selector(
            "a:has-text('Inscribirme'), a:has-text('Solicitar'), button:has-text('Inscribirme'), "
            "button:has-text('Solicitar'), a:has-text('inscribirme'), button:has-text('inscribirme'), "
            "a.btn-success, a.btn-primary, .btn-inscribirme, [class*=inscribir]"
        )

        if not apply_btn:
            apply_btn = await page.query_selector("a[href*=inscribir], a[href*=candidato], a[href*=aplicar]")

        if not apply_btn:
            result["status"] = "skipped"
            result["detail"] = "botón inscribirme no encontrado"
            return result

        href = await apply_btn.get_attribute("href")
        if href and TECNOEMPLEO_BASE not in href and href.startswith("http"):
            result["status"] = "skipped"
            result["detail"] = "inscripción en web externa"
            return result

        await apply_btn.click()
        await page.wait_for_timeout(4000)

        post_click_url = page.url
        if TECNOEMPLEO_BASE not in post_click_url:
            result["status"] = "skipped"
            result["detail"] = "tras clic redirige a web externa"
            return result

        cover_letter = await _generate_cover_letter(cv, job)

        letter_textarea = await page.query_selector(
            "textarea, textarea[name*=carta], textarea[name*=mensaje], textarea[name*=presentacion], "
            "textarea[id*=carta], textarea[id*=mensaje]"
        )

        if letter_textarea:
            await letter_textarea.fill(cover_letter)
            await page.wait_for_timeout(500)

        if cv_pdf_path and cv_pdf_path.exists():
            file_input = await page.query_selector("input[type=file]")
            if file_input:
                await file_input.set_input_files(str(cv_pdf_path))
                await page.wait_for_timeout(1000)

        questions_answered = await _answer_screening_questions(page, cv, job)

        submit_btn = await page.query_selector(
            "button[type=submit], input[type=submit], button:has-text('Enviar'), "
            "button:has-text('Solicitar'), button:has-text('Finalizar'), "
            "button:has-text('Confirmar'), button:has-text('Aplicar'), button.btn-success"
        )
        if submit_btn:
            await submit_btn.click()
            await page.wait_for_timeout(4000)

        post_click_url = page.url.lower()
        page_text = (await page.content()).lower()
        success_keywords = [
            "gracias", "éxito", "exito", "correctamente", "enviada",
            "inscrito", "candidatura enviada", "solicitud enviada",
            "te has inscrito", "inscripción correcta", "felicidades",
        ]
        url_success = any(kw in post_click_url for kw in ["mis-candidaturas", "mis-candidatures", "candidatura", "micuenta", "confirmacion"])
        if (any(kw in page_text for kw in success_keywords) or url_success):
            from src.database import mark_as_applied
            mark_as_applied(job_id)
            result["status"] = "applied"
            result["detail"] = "candidatura enviada" + (f", {questions_answered}" if questions_answered else "")
            return result

        # No confirma explícita -> conservador: marcamos como completado pero
        # NO como applied en BD para que el usuario pueda revisar.
        result["status"] = "skipped"
        result["detail"] = "sin confirmación de éxito tras enviar — revisa manualmente"
        return result

    except Exception as e:
        result["status"] = "error"
        result["detail"] = str(e)[:200]
        return result


async def _generate_cover_letter(cv, job: dict) -> str:
    if not cv:
        return "Estimado/a responsable de selección,\n\nAdjunto mi candidatura para esta posición.\n\nUn cordial saludo."

    job_title = job.get("title", "esta posición")
    job_company = job.get("company", "su empresa")
    job_desc = job.get("description", "")[:2000]

    try:
        import litellm

        cv_text = cv.to_text_block()[:3000]
        prompt = f"""Eres un asistente que redacta cartas de presentación profesionales en español para ofertas de tecnología.

CV DEL CANDIDATO:
{cv_text}

OFERTA DE TRABAJO:
Título: {job_title}
Empresa: {job_company}
Descripción: {job_desc}

Redacta una carta de presentación profesional en español, de 2-3 párrafos, que:
1. Mencione el interés por el puesto en {job_company}
2. Destaque 2-3 habilidades/experiencias del CV que encajen con la oferta
3. Sea natural y personal, NO genérica
4. Termine con un cierre cordial

NO uses placeholders como [Nombre] o [Empresa]. Solo el texto de la carta."""

        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un redactor profesional de cartas de presentación. Responde solo con la carta, sin introducciones."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
            api_key=settings.openai_api_key,
        )
        letter = response.choices[0].message.content.strip()
        return letter

    except Exception as e:
        logger.warning(f"AutoApply: error generando cover letter: {e}")

    techs = ", ".join(cv.technologies[:6]) if cv and cv.technologies else ""
    exp = cv.experience_years if cv else 0
    return (
        f"Estimado/a responsable de selección,\n\n"
        f"Me dirijo a ustedes para expresar mi interés en la posición de {job_title} en {job_company}.\n\n"
        f"Cuento con {exp} años de experiencia trabajando con tecnologías como {techs}, "
        f"y creo que mi perfil encaja bien con lo que buscan.\n\n"
        f"Quedo a su disposición para ampliar cualquier información en una entrevista.\n\n"
        f"Un cordial saludo."
    )


async def _answer_screening_questions(page, cv, job: dict) -> str:
    answered = 0
    try:
        questions = await page.query_selector_all("[class*=question], [class*=pregunta], .form-group label, form label")
        for q in questions[:8]:
            try:
                label_text = (await q.inner_text()).lower().strip()
                if not label_text or len(label_text) < 3:
                    continue

                parent = await q.evaluate("el => el.closest('.form-group, .row, .mb-3, .field, form > div') || el.parentElement")
                if not parent:
                    continue

                if any(kw in label_text for kw in ["experiencia", "años", "experience", "año"]):
                    exp = cv.experience_years if cv else 1
                    await _fill_input(parent, str(int(exp)))

                elif any(kw in label_text for kw in ["salario", "salary", "retribución", "pretensión", "aspiracion"]):
                    await _fill_input(parent, "Según convenio")

                elif any(kw in label_text for kw in ["disponibilidad", "incorporación", "incorporacion", "disponible"]):
                    await _fill_input(parent, "Inmediata")

                elif any(kw in label_text for kw in ["movilidad", "viajar", "desplazamiento"]):
                    await _fill_input(parent, "Sí")

                elif any(kw in label_text for kw in ["carnet", "conducir", "vehículo", "coche"]):
                    await _fill_input(parent, "Sí")

                elif any(kw in label_text for kw in ["ciudad", "residencia", "vivo", "resido", "ubicación"]):
                    await _fill_input(parent, "Madrid")

                elif any(kw in label_text for kw in ["estudios", "titulación", "formación", "nivel"]):
                    await _fill_input(parent, "Grado Universitario")

                elif any(kw in label_text for kw in ["inglés", "idioma"]):
                    await _fill_input(parent, "B2")

                else:
                    continue

                answered += 1
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"AutoApply: error answering questions: {e}")

    return f"{answered} preguntas respondidas" if answered else ""


async def _fill_input(parent_handle, value: str):
    try:
        inp = await parent_handle.query_selector("input[type=text], input[type=number], input:not([type]), textarea")
        if inp:
            await inp.fill(value)
            return

        select = await parent_handle.query_selector("select")
        if select:
            options = await select.query_selector_all("option")
            for opt in options:
                opt_text = (await opt.inner_text()).lower()
                opt_val = await opt.get_attribute("value")
                if value.lower() in opt_text or (opt_val and value.lower() in opt_val.lower()):
                    await select.select_option(value=opt_val or opt_text)
                    return
            if options:
                await select.select_option(index=1)
    except Exception:
        pass


def _find_cv_pdf() -> Optional[Path]:
    if CV_DIR.exists():
        pdfs = sorted(CV_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdfs:
            return pdfs[0]
    return None


def get_autoapply_results() -> dict:
    from src.database import get_db
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
    with get_db() as conn:
        eligible = conn.execute(
            "SELECT COUNT(*) FROM job_offers WHERE source = 'tecnoempleo' AND match_score > 0 AND applied = 0 AND discarded = 0 AND scrape_date >= ?",
            (cutoff,),
        ).fetchone()[0]

    return {
        "running": _autoapply_running,
        "min_score": _min_auto_score,
        "total": len(AUTOAPPLY_RESULTS),
        "applied": sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "applied"),
        "eligible_jobs": eligible,
        "last_error": _last_autoapply_error,
        "results": AUTOAPPLY_RESULTS[-50:],
    }
