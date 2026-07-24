from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import httpx
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

RESEND_API_URL = "https://api.resend.com/emails"


async def send_daily_email(jobs: list[dict]) -> bool:
    if not os.getenv("RESEND_API_KEY"):
        logger.warning("Resend API Key no configurada")
        return False

    if not jobs:
        logger.info("No hay ofertas que enviar")
        return False

    resend_key = os.getenv("RESEND_API_KEY", "")
    from_email = settings.email_from or settings.email_user or "Jobs Scout <onboarding@resend.dev>"
    to_email = settings.email_to or settings.email_user

    try:
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

        text_content = _build_plain_text(jobs)

        payload = {
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html_content,
            "text": text_content,
        }

        headers = {
            "Authorization": f"Bearer {resend_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Enviando email a {to_email} via Resend API...")

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(RESEND_API_URL, json=payload, headers=headers)

        if resp.status_code in (200, 201, 202):
            logger.info(f"Email enviado a {to_email} con {len(jobs)} ofertas")
            return True
        else:
            logger.error(f"Resend API error {resp.status_code}: {resp.text[:300]}")
            return False

    except Exception as e:
        logger.error(f"Error enviando email: {e}")
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
    resend_key = os.getenv("RESEND_API_KEY", "")
    if not resend_key:
        logger.warning("No RESEND_API_KEY configured")
        return False

    try:
        payload = {
            "from": "Jobs Scout <onboarding@resend.dev>",
            "to": [settings.email_to or settings.email_user],
            "subject": "Jobs Scout - Test de conexión",
            "html": "<p>Test de conexión exitoso. <strong>Jobs Scout</strong> está configurado correctamente.</p>",
        }
        headers = {"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(RESEND_API_URL, json=payload, headers=headers)

        if resp.status_code in (200, 201, 202):
            logger.info("Conexión Resend: OK")
            return True
        else:
            logger.error(f"Resend test failed: {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Resend test error: {e}")
        return False
