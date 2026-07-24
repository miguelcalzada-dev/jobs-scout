from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus

logger = logging.getLogger(__name__)

BASE_URL = "https://www.tecnoempleo.com"
SEARCH_URL = f"{BASE_URL}/ofertas-trabajo/"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
]


class TecnoempleoScraper(BaseScraper):
    source = JobSource.TECNOEMPLEO

    async def scrape(self) -> ScrapeResult:
        start = time.time()
        try:
            prefs = load_preferences()
            offers = await self._search(prefs)
            duration = time.time() - start
            logger.info(f"Tecnoempleo: scraped {len(offers)} offers in {duration:.1f}s")
            return self._build_result(ScrapeStatus.SUCCESS, offers, duration=duration)
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Tecnoempleo scraper failed: {e}", exc_info=True)
            return self._build_result(ScrapeStatus.FAILED, error=str(e), duration=duration)

    async def _search(self, prefs) -> list[JobOffer]:
        all_offers: list[JobOffer] = []
        titles = prefs.desired_titles or ["python", "desarrollador"]

        for title in titles[:3]:
            query = title.lower().strip().replace(" ", "-")
            offers = await self._scrape_page(query, page=1)
            all_offers.extend(offers)
            if len(all_offers) >= self.max_offers:
                break
            if len(titles) > 1:
                await asyncio.sleep(random.uniform(1.0, 2.5))

            if len(all_offers) < self.max_offers:
                extra = await self._scrape_page(query, page=2)
                all_offers.extend(extra)
                await asyncio.sleep(random.uniform(1.0, 2.0))

        seen = set()
        unique = []
        for o in all_offers:
            if o.external_id not in seen:
                seen.add(o.external_id)
                unique.append(o)
        return unique[: self.max_offers]

    async def _scrape_page(self, query: str, page: int = 1) -> list[JobOffer]:
        offers: list[JobOffer] = []
        url = f"{SEARCH_URL}{query}"
        params = {"pagina": str(page)} if page > 1 else {}

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "DNT": "1",
            "Connection": "keep-alive",
        }

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                html = resp.text
                break
            except requests.RequestException as e:
                logger.warning(f"Tecnoempleo attempt {attempt+1}/3 failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep((attempt + 1) * 2)

        soup = BeautifulSoup(html, "lxml")

        job_rows = soup.select("div.row.fs--15")

        for row in job_rows[: self.max_offers]:
            try:
                title_el = row.select_one("h3 a.text-cyan-700, h3 a.font-weight-bold")
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                url = title_el.get("href", "")
                if url and url.startswith("/"):
                    url = BASE_URL + url

                external_id = hashlib.md5((title + url).encode()).hexdigest()[:16]

                logo = row.select_one("img[alt]")
                company = ""
                if logo:
                    alt = logo.get("alt", "")
                    company = alt.replace("Logo ", "").strip()

                if not company:
                    company_link = row.select_one("a.text-primary.link-muted")
                    if company_link:
                        company = company_link.get_text(strip=True)

                location = ""
                location_b = row.select_one("b")
                if location_b:
                    location = location_b.get_text(strip=True)

                mobile_span = row.select_one("span.d-block.d-lg-none.text-gray-800")
                if mobile_span and not location:
                    b_tag = mobile_span.select_one("b")
                    if b_tag:
                        location = b_tag.get_text(strip=True)

                remote = any(kw in (title + " " + location).lower() for kw in ["remoto", "remote", "teletrabajo", "100% remoto"])
                hybrid = any(kw in (title + " " + location).lower() for kw in ["híbrido", "hibrido", "hybrid", "semanal"])

                description = ""
                desc_span = row.select_one("span.hidden-md-down.text-gray-800")
                if desc_span:
                    description = desc_span.get_text(" ", strip=True)[:3000]

                badge_texts = [b.get_text(strip=True).lower() for b in row.select("span.badge")]
                techs_from_badges = [b for b in badge_texts if b not in ["nueva", "nuevo", "actualizada", "urgente"]]

                all_techs = self._detect_techs(f"{title} {description}")
                all_techs = list(dict.fromkeys(techs_from_badges + all_techs))

                seniority = self._detect_seniority(f"{title} {description}")

                offers.append(JobOffer(
                    source=JobSource.TECNOEMPLEO,
                    external_id=external_id,
                    title=title,
                    company=company or "No especificada",
                    location=location or "No especificada",
                    url=url,
                    description=description,
                    remote=remote,
                    hybrid=hybrid,
                    onsite=not remote and not hybrid,
                    technologies_detected=all_techs,
                    seniority_level=seniority,
                ))
            except Exception as e:
                logger.debug(f"Failed to parse Tecnoempleo card: {e}")

        return offers

    def _detect_techs(self, text: str) -> list[str]:
        text_lower = text.lower()
        common_techs = [
            "python", "javascript", "typescript", "java", "c#", ".net", "php",
            "go", "golang", "rust", "ruby", "swift", "kotlin", "scala", "c++",
            "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
            "node.js", "nodejs", "django", "flask", "fastapi", "spring",
            "laravel", "rails", "express", "nest.js", "nestjs",
            "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
            "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
            "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
            "graphql", "rest", "soap", "grpc", "kafka", "rabbitmq",
            "sql", "nosql", "linux", "git", "agile", "scrum",
            "pandas", "numpy", "pytorch", "tensorflow", "spark", "hadoop",
            "power bi", "tableau", "excel", "jira",
        ]
        found = []
        for tech in common_techs:
            if re.search(rf'\b{re.escape(tech)}\b', text_lower):
                found.append(tech)
        return found

    def _detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["lead", "principal", "arquitecto", "architect", "director", "manager", "jefe"]):
            return "lead"
        if any(kw in text_lower for kw in ["senior", "sr.", "experto", "más de 5 años", "mas de 5 años", "+5 años"]):
            return "senior"
        if any(kw in text_lower for kw in ["mid", "semi-senior", "semi senior", "2-4 años", "2-5 años", "3-5 años"]):
            return "mid"
        if any(kw in text_lower for kw in ["junior", "jr.", "entry", "becario", "prácticas", "sin experiencia", "trainee"]):
            return "junior"
        return ""
