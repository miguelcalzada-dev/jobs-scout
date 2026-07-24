from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import (
    TEMPLATES_DIR,
    load_cv_profile,
    load_preferences,
    settings,
)

logger = logging.getLogger(__name__)

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


async def send_daily_email(jobs: list[dict]) -> bool:
    if not settings.email_user or not settings.email_password:
        logger.warning("Email credentials not configured, cannot send email")
        return False

    if not jobs:
        logger.info("No jobs to send")
        return False

    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        prefs = load_preferences()
        cv = load_cv_profile()

        today = datetime.now().strftime("%d/%m/%Y")
        subject = f"🔎 Jobs Scout - {len(jobs)} ofertas para ti ({today})"

        template = _jinja_env.get_template("daily_email.html")
        html_content = template.render(
            jobs=jobs,
            date=today,
            prefs=prefs,
            cv=cv,
        )

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = settings.email_from or settings.email_user
        message["To"] = settings.email_to or settings.email_user

        text_content = _build_plain_text(jobs)
        message.attach(MIMEText(text_content, "plain", "utf-8"))
        message.attach(MIMEText(html_content, "html", "utf-8"))

        logger.info(f"Enviando email a {settings.email_to}...")

        await asyncio.wait_for(
            aiosmtplib.send(
                message,
                hostname=settings.email_host,
                port=settings.email_port,
                username=settings.email_user,
                password=settings.email_password,
                use_tls=False,
                start_tls=True,
            ),
            timeout=20.0,
        )

        logger.info(f"Email enviado a {settings.email_to} con {len(jobs)} ofertas")
        return True

    except ImportError:
        logger.warning("aiosmtplib no disponible, usando smtplib")
        return await _send_email_sync(jobs)
    except asyncio.TimeoutError:
        logger.error("Timeout enviando email (>20s)")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        return False


async def _send_email_sync(jobs: list[dict]) -> bool:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    today = datetime.now().strftime("%d/%m/%Y")
    subject = f"🔎 Jobs Scout - {len(jobs)} ofertas para ti ({today})"

    template = _jinja_env.get_template("daily_email.html")
    html_content = template.render(jobs=jobs, date=today)

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.email_from or settings.email_user
    message["To"] = settings.email_to or settings.email_user

    message.attach(MIMEText(_build_plain_text(jobs), "plain", "utf-8"))
    message.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        server = smtplib.SMTP(settings.email_host, settings.email_port, timeout=30)
        server.starttls()
        server.login(settings.email_user, settings.email_password)
        server.send_message(message)
        server.quit()
        logger.info(f"Email sent (sync) to {settings.email_to} with {len(jobs)} jobs")
        return True
    except Exception as e:
        logger.error(f"Failed to send email (sync): {e}")
        return False


def _build_plain_text(jobs: list[dict]) -> str:
    lines = [f"JOBS SCOUT - {len(jobs)} ofertas del día\n", "=" * 50, ""]
    for i, job in enumerate(jobs, 1):
        match_pct = int(job.get("match_score", 0))
        stars = "⭐" * min(5, match_pct // 20)
        lines.append(f"{i}. [{job.get('source', '').upper()}] {job.get('title', 'Sin título')}")
        lines.append(f"   {job.get('company', '')} | {job.get('location', '')}")
        lines.append(f"   Match: {match_pct}% {stars}")
        if job.get("salary"):
            lines.append(f"   Salario: {job.get('salary', '')}")
        lines.append(f"   {job.get('url', '')}")
        lines.append("")
    return "\n".join(lines)


async def test_email_connection() -> bool:
    try:
        import aiosmtplib

        await aiosmtplib.send(
            await _build_test_message(),
            hostname=settings.email_host,
            port=settings.email_port,
            username=settings.email_user,
            password=settings.email_password,
            use_tls=False,
            start_tls=True,
        )
        logger.info("Email connection test: OK")
        return True
    except Exception as e:
        logger.error(f"Email connection test failed: {e}")
        return False


async def _build_test_message():
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart()
    msg["Subject"] = "Jobs Scout - Test de conexión"
    msg["From"] = settings.email_from or settings.email_user
    msg["To"] = settings.email_to or settings.email_user
    msg.attach(MIMEText("Test de conexión exitoso. Jobs Scout está configurado correctamente.", "plain", "utf-8"))
    return msg
