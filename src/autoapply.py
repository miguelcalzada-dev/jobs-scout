from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import load_cv_profile, CV_DIR, settings

logger = logging.getLogger(__name__)

MAX_KEPT_RESULTS = 100
AUTOAPPLY_RESULTS: list[dict] = []
_autoapply_running = False
_cancel_requested = False
_min_auto_score = 75
_last_autoapply_error = ""

TECNOEMPLEO_LOGIN_URL = "https://www.tecnoempleo.com/login.php"
TECNOEMPLEO_BASE = "https://www.tecnoempleo.com"

SUCCESS_KEYWORDS = [
    "gracias por inscribirte", "candidatura enviada", "solicitud enviada",
    "te has inscrito", "inscripción correcta", "inscripcion correcta",
    "correctamente", "felicidades", "has sido registrado", "recibido tu cv",
    "guardamos tu candidatura", "inscripción completada", "inscripcion completada",
]
SUCCESS_URL_KEYWORDS = [
    "mis-candidaturas", "mis-candidatures", "candidatura", "micuenta",
    "confirmacion", "confirmacion-candidatura", "postulacion",
]
ALREADY_APPLIED_KEYWORDS = [
    "ya inscrito", "ya te has inscrito", "ya has solicitado",
    "inscrito en esta oferta", "candidatura ya existe",
]


async def run_autoapply(min_score: int = 75) -> dict:
    global _autoapply_running, AUTOAPPLY_RESULTS, _cancel_requested, _min_auto_score
    if _autoapply_running:
        return {"status": "skipped", "reason": "already_running"}

    email = os.getenv("TECNOEMPLEO_EMAIL", "") or settings.tecnoempleo_email
    password = os.getenv("TECNOEMPLEO_PASSWORD", "") or settings.tecnoempleo_password

    if not email or not password:
        _last_autoapply_error = "missing_credentials"
        return {
            "status": "error",
            "reason": "missing_credentials",
            "hint": "Configura TECNNOEMPLEO_EMAIL y TECNOEMPLEO_PASSWORD en Railway o en el formulario",
        }

    _autoapply_running = True
    _cancel_requested = False
    _min_auto_score = min_score
    _last_autoapply_error = ""
    AUTOAPPLY_RESULTS = []

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        _autoapply_running = False
        return {"status": "error", "reason": "playwright_not_installed"}

    from src.database import get_db

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

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-gpu",
                ],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": 1366, "height": 900},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    locale="es-ES",
                    accept_downloads=True,
                )
                await context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es', 'en']});
                """)

                page = await context.new_page()

                logged_in = await _login_tecnoempleo(page, email, password)
                if not logged_in:
                    _last_autoapply_error = "login_failed"
                    _autoapply_running = False
                    return {
                        "status": "error",
                        "reason": "login_failed",
                        "hint": "Verifica email/contraseña de Tecnoempleo",
                    }

                for job in jobs:
                    if _cancel_requested:
                        logger.info("AutoApply: cancelado por el usuario")
                        break
                    result = await _apply_to_job(page, job, cv, cv_pdf_path)
                    AUTOAPPLY_RESULTS.append(result)
                    if len(AUTOAPPLY_RESULTS) > MAX_KEPT_RESULTS:
                        del AUTOAPPLY_RESULTS[: len(AUTOAPPLY_RESULTS) - MAX_KEPT_RESULTS]
                    logger.info(
                        f"AutoApply: [{result['status']}] {job['title'][:60]} "
                        f"({job['company']}) — {result.get('detail', '')}"
                    )
                    await asyncio.sleep(4.0)

            finally:
                await browser.close()

        applied_count = sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "applied")
        already_count = sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "already_applied")
        skipped_count = sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "skipped")
        error_count = sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "error")
        logger.info(
            f"AutoApply completado: {applied_count} aplicadas, {already_count} ya inscritas, "
            f"{skipped_count} saltadas, {error_count} errores / {len(jobs)} totales"
        )
        return {
            "status": "completed",
            "applied": applied_count,
            "already_applied": already_count,
            "skipped": skipped_count,
            "errors": error_count,
            "total": len(jobs),
            "results": AUTOAPPLY_RESULTS,
        }

    except Exception as e:
        logger.error(f"AutoApply fatal: {e}", exc_info=True)
        _last_autoapply_error = str(e)[:150]
        return {
            "status": "error",
            "reason": "runtime_error",
            "error": str(e)[:200],
            "results": AUTOAPPLY_RESULTS,
        }
    finally:
        _autoapply_running = False


async def _login_tecnoempleo(page, email: str, password: str) -> bool:
    try:
        logger.info("AutoApply: iniciando sesión en Tecnoempleo...")
        await page.goto(TECNOEMPLEO_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1500)

        email_input = await page.query_selector(
            "input[type=email], input[name=email], input[id*=email], input[name*=email], "
            "input[name=correo], input[id*=correo], input[name=usuario]"
        )
        if not email_input:
            email_input = await page.query_selector(
                "input[type=text]:not([name*=search]):not([name*=keyword])"
            )

        pass_input = await page.query_selector("input[type=password]")

        if not email_input or not pass_input:
            logger.warning("AutoApply: no se encontraron campos de login")
            return False

        await email_input.fill(email)
        await pass_input.fill(password)

        submit_btn = await page.query_selector(
            "button[type=submit], input[type=submit], button.btn-primary, "
            "button:has-text('Entrar'), button:has-text('Iniciar'), button:has-text('Acceder'), "
            "button:has-text(' Entrar')"
        )
        if not submit_btn:
            submit_btn = await page.query_selector("form button, form input[type=submit]")

        if submit_btn:
            await submit_btn.click()
        else:
            await pass_input.press("Enter")

        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            await page.wait_for_timeout(5000)

        return await _check_logged_in(page)

    except Exception as e:
        logger.error(f"AutoApply: error en login: {e}")
        return False


async def _check_logged_in(page) -> bool:
    current_url = page.url.lower()
    page_title = (await page.title()).lower()
    page_html = ""
    try:
        page_html = (await page.content()).lower()
    except Exception:
        pass

    if "login" in current_url and not any(
        kw in current_url for kw in ["dashboard", "candidato", "mi-cuenta", "micuenta", "panel"]
    ):
        # Still on login page -> check for error message
        error_el = await page.query_selector("[class*=alert-danger], [class*=error-msg], .alert-warning")
        if error_el:
            error_text = await error_el.inner_text()
            logger.warning(f"AutoApply: login fallido — {error_text.strip()[:200]}")
        return False

    logged_in = any(kw in current_url for kw in ["candidato", "dashboard", "mi-cuenta", "micuenta", "panel", "cuenta"])
    if logged_in:
        logger.info("AutoApply: sesión iniciada (URL)")
        return True

    if any(kw in page_title for kw in ["cuenta", "candidato", "panel", "mi perfil"]):
        logger.info("AutoApply: sesión iniciada (title)")
        return True

    if any(kw in page_html for kw in [
        "cerrar sesión", "cerrar sesion", "mi cuenta", "micuenta",
        "panel de control", "mis candidaturas", "mi perfil", "logout",
        "subir cv", "mi cv", "modificar perfil",
    ]):
        logger.info("AutoApply: sesión iniciada (page content)")
        return True

    ctx = page.context
    cookies = await ctx.cookies()
    auth_cookies = [
        c for c in cookies
        if c.get("name", "").lower() in ("tecnouser", "tecnologin", "user", "login", "token", "phpsessid", "tecnosession")
    ]
    if auth_cookies:
        logger.info("AutoApply: sesión iniciada (cookies)")
        return True

    logger.warning(f"AutoApply: login fallido — URL: {current_url}")
    return False


async def _apply_to_job(page, job: dict, cv, cv_pdf_path: Optional[Path]) -> dict:
    job_id = job["id"]
    job_url = job.get("url", "")
    job_title = job.get("title", "")
    job_company = job.get("company", "")

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
        await page.goto(job_url, wait_until="domcontentloaded", timeout=25000)
        await page.wait_for_timeout(2500)

        current_url = page.url
        if TECNOEMPLEO_BASE not in current_url:
            result["status"] = "skipped"
            result["detail"] = "redirige a web externa"
            return result

        page_text = (await page.content()).lower()

        # Already applied? Tecnoempleo shows "Ya inscrito" instead of the button.
        if any(kw in page_text for kw in ALREADY_APPLIED_KEYWORDS):
            from src.database import mark_as_applied
            mark_as_applied(job_id)
            result["status"] = "already_applied"
            result["detail"] = "ya estabas inscrito en esta oferta"
            return result

        # Find the apply button. Tecnoempleo uses "Inscribirme" prominently.
        apply_btn = await page.query_selector(
            "a:has-text('Inscribirme'), button:has-text('Inscribirme'), "
            "a:has-text('inscribirme'), button:has-text('inscribirme'), "
            "a:has-text('Inscríbete'), button:has-text('Inscríbete'), "
            "a.btn-inscribirme, .btn-inscribirme, [class*=btn-inscribir], "
            "a:has-text('Postular'), a:has-text('Postularme'), "
            "a[href*=inscribir], a[href*=candidato/aplicar], a[href*=postular]"
        )

        if not apply_btn:
            result["status"] = "skipped"
            result["detail"] = "no hay botón de inscripción (¿ya inscrita o sin permiso?)"
            return result

        href = await apply_btn.get_attribute("href")
        if href and href.startswith("http") and TECNOEMPLEO_BASE not in href:
            result["status"] = "skipped"
            result["detail"] = "inscripción en web externa"
            return result

        await apply_btn.click()

        # Wait for the application form to load (navigation or modal).
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        await page.wait_for_timeout(2500)

        post_click_url = page.url
        if TECNOEMPLEO_BASE not in post_click_url:
            result["status"] = "skipped"
            result["detail"] = "tras clic redirige a web externa"
            return result

        # If a login wall appeared, the session may have expired.
        if "login" in post_click_url.lower():
            result["status"] = "error"
            result["detail"] = "sesión expirada durante el apply"
            return result

        # Fill cover letter.
        cover_letter = await _generate_cover_letter(cv, job)
        await _fill_cover_letter(page, cover_letter)

        # Upload CV if a file input is present.
        if cv_pdf_path and cv_pdf_path.exists():
            await _upload_cv(page, cv_pdf_path)

        # Answer screening questions.
        questions_answered = await _answer_screening_questions(page, cv)

        # Submit the application form.
        submitted = await _submit_application(page)
        if not submitted:
            result["status"] = "skipped"
            result["detail"] = "no se encontró botón de envío final"
            return result

        # Detect success / already-applied / failure.
        await page.wait_for_timeout(3500)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

        outcome = await _detect_outcome(page)

        if outcome == "applied":
            from src.database import mark_as_applied
            mark_as_applied(job_id)
            result["status"] = "applied"
            result["detail"] = "candidatura enviada" + (f", {questions_answered}" if questions_answered else "")
        elif outcome == "already_applied":
            from src.database import mark_as_applied
            mark_as_applied(job_id)
            result["status"] = "already_applied"
            result["detail"] = "ya estaba inscrito"
        else:
            result["status"] = "skipped"
            result["detail"] = "sin confirmación de éxito — revisa manualmente"
        return result

    except Exception as e:
        result["status"] = "error"
        result["detail"] = str(e)[:200]
        return result


async def _fill_cover_letter(page, letter: str) -> None:
    try:
        textarea = await page.query_selector(
            "textarea, textarea[name*=carta], textarea[name*=mensaje], "
            "textarea[name*=presentacion], textarea[id*=carta], textarea[id*=mensaje], "
            "#carta, #mensaje"
        )
        if textarea:
            await textarea.fill("")
            await textarea.type(letter, delay=5)
            await page.wait_for_timeout(400)
    except Exception as e:
        logger.debug(f"AutoApply: error rellenando carta: {e}")


async def _upload_cv(page, cv_path: Path) -> None:
    try:
        file_input = await page.query_selector(
            "input[type=file], input[name*=cv], input[name*=file], input[name*=pdf], input[accept*=pdf]"
        )
        if file_input:
            await file_input.set_input_files(str(cv_path))
            await page.wait_for_timeout(1200)
    except Exception as e:
        logger.debug(f"AutoApply: error subiendo CV: {e}")


async def _submit_application(page) -> bool:
    submit_btn = await page.query_selector(
        "button[type=submit], input[type=submit], button:has-text('Enviar'), "
        "button:has-text('Enviar candidatura'), button:has-text('Finalizar'), "
        "button:has-text('Confirmar'), button:has-text('Aplicar'), "
        "button:has-text('Inscribirme'), button:has-text('Enviar solicitud'), "
        "button.btn-success, button.btn-primary"
    )
    if not submit_btn:
        return False
    try:
        await submit_btn.click()
    except Exception:
        return False
    return True


async def _detect_outcome(page) -> str:
    try:
        current_url = page.url.lower()
        page_text = (await page.content()).lower()
    except Exception:
        return "unknown"

    url_ok = any(kw in current_url for kw in SUCCESS_URL_KEYWORDS)
    text_ok = any(kw in page_text for kw in SUCCESS_KEYWORDS)
    if url_ok or text_ok:
        return "applied"

    if any(kw in page_text for kw in ALREADY_APPLIED_KEYWORDS):
        return "already_applied"

    return "unknown"


async def _answer_screening_questions(page, cv) -> str:
    answered = 0
    try:
        # Tecnoempleo wraps each question in .form-group; we inspect labels and
        # find the associated input/select in the same group via DOM walking.
        groups = await page.query_selector_all(".form-group, .mb-3, .row.form-group, .campo, .field")
        for group in groups[:10]:
            try:
                label_el = await group.query_selector("label, .label, .control-label")
                if not label_el:
                    continue
                label_text = (await label_el.inner_text()).lower().strip()
                if not label_text or len(label_text) < 3:
                    continue

                value = _value_for_question(label_text, cv)
                if value is None:
                    continue

                filled = await _fill_field_in_group(group, value)
                if filled:
                    answered += 1
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"AutoApply: error en preguntas: {e}")

    return f"{answered} preguntas respondidas" if answered else ""


def _value_for_question(label_text: str, cv) -> Optional[str]:
    if any(kw in label_text for kw in ["experiencia", "años", "experience", "año"]):
        exp = int(cv.experience_years) if cv and cv.experience_years else 1
        return str(max(exp, 1))
    if any(kw in label_text for kw in ["salario", "salary", "retribución", "retencion", "pretensión", "aspiracion"]):
        return "Según convenio"
    if any(kw in label_text for kw in ["disponibilidad", "incorporación", "incorporacion", "disponible", "cuando"]):
        return "Inmediata"
    if any(kw in label_text for kw in ["movilidad", "viajar", "desplazamiento"]):
        return "Sí"
    if any(kw in label_text for kw in ["carnet", "conducir", "vehículo", "coche"]):
        return "Sí"
    if any(kw in label_text for kw in ["ciudad", "residencia", "vivo", "resido", "ubicación", "donde"]):
        return "Madrid"
    if any(kw in label_text for kw in ["estudios", "titulación", "titulacion", "formación", "formacion", "nivel"]):
        return "Grado Universitario"
    if any(kw in label_text for kw in ["inglés", "ingles", "idioma"]):
        return "B2"
    return None


async def _fill_field_in_group(group, value: str) -> bool:
    try:
        inp = await group.query_selector(
            "input[type=text], input[type=number], input:not([type]), textarea, input[type=email]"
        )
        if inp:
            await inp.fill(value)
            return True

        select = await group.query_selector("select")
        if select:
            options = await select.query_selector_all("option")
            for opt in options:
                opt_text = (await opt.inner_text()).lower()
                opt_val = await opt.get_attribute("value")
                if value.lower() in opt_text or (opt_val and value.lower() in opt_val.lower()):
                    await select.select_option(value=opt_val or opt_text)
                    return True
            if options:
                await select.select_option(index=1)
                return True

        # Radio / checkbox: try to find one matching the value.
        radios = await group.query_selector_all("input[type=radio], input[type=checkbox]")
        for r in radios:
            r_val = (await r.get_attribute("value")) or ""
            r_lbl = ""
            rid = await r.get_attribute("id")
            if rid:
                sib = await group.query_selector(f"label[for='{rid}']")
                if sib:
                    r_lbl = (await sib.inner_text()).lower()
            if value.lower() in r_val.lower() or value.lower() in r_lbl:
                await r.check()
                return True
    except Exception:
        pass
    return False


async def _generate_cover_letter(cv, job: dict) -> str:
    if not cv:
        return "Estimado/a responsable de selección,\n\nAdjunto mi candidatura para esta posición.\n\nUn cordial saludo."

    job_title = job.get("title", "esta posición")
    job_company = job.get("company", "su empresa")
    job_desc = job.get("description", "")[:2000]

    if settings.openai_api_key:
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
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"AutoApply: error generando cover letter con IA: {e}")

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


def _find_cv_pdf() -> Optional[Path]:
    if CV_DIR.exists():
        pdfs = sorted(CV_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdfs:
            return pdfs[0]
    return None


def cancel_autoapply() -> None:
    global _cancel_requested
    _cancel_requested = True


def get_autoapply_results() -> dict:
    from src.database import get_db

    cutoff = (datetime.now() - timedelta(days=max(settings.max_job_age_days, 1))).isoformat()
    with get_db() as conn:
        eligible = conn.execute(
            "SELECT COUNT(*) FROM job_offers WHERE source = 'tecnoempleo' "
            "AND match_score >= ? AND applied = 0 AND discarded = 0 "
            "AND is_external_redirect = 0 AND scrape_date >= ?",
            (_min_auto_score, cutoff),
        ).fetchone()[0]

    return {
        "running": _autoapply_running,
        "min_score": _min_auto_score,
        "total": len(AUTOAPPLY_RESULTS),
        "applied": sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "applied"),
        "already_applied": sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "already_applied"),
        "skipped": sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "skipped"),
        "errors": sum(1 for r in AUTOAPPLY_RESULTS if r["status"] == "error"),
        "eligible_jobs": eligible,
        "last_error": _last_autoapply_error,
        "results": AUTOAPPLY_RESULTS[-50:],
    }