from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus
from src.scrapers._common import detect_techs, detect_seniority, is_remote_text, is_hybrid_text

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
        # Build queries from desired_titles so parameters are actually used.
        titles = [t for t in (prefs.desired_titles or []) if t]
        if not titles:
            titles = ["python", "desarrollador"]

        for title in titles[:3]:
            # Tecnnoempleo search uses technology/keyword as path segment.
            query = title.lower().strip().replace(" ", "-")
            offers = await self._scrape_page(query, page=1, location=prefs.location)
            all_offers.extend(offers)
            if len(all_offers) >= self.max_offers:
                break
            await asyncio.sleep(random.uniform(1.0, 2.5))

            if len(all_offers) < self.max_offers:
                extra = await self._scrape_page(query, page=2, location=prefs.location)
                all_offers.extend(extra)
                await asyncio.sleep(random.uniform(1.0, 2.0))

        seen = set()
        unique = []
        for o in all_offers:
            if o.external_id not in seen:
                seen.add(o.external_id)
                unique.append(o)
        return unique[: self.max_offers]

    async def _scrape_page(self, query: str, page: int = 1, location: str = "") -> list[JobOffer]:
        offers: list[JobOffer] = []
        url = f"{SEARCH_URL}{query}"
        params: dict[str, str] = {}
        if page > 1:
            params["pagina"] = str(page)
        if location:
            params["ubicacion"] = location

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "DNT": "1",
            "Connection": "keep-alive",
        }

        html = ""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    resp = await client.get(url, params=params, headers=headers)
                    resp.raise_for_status()
                    html = resp.text
                    break
                except httpx.HTTPError as e:
                    logger.warning(f"Tecnoempleo attempt {attempt+1}/3 failed: {e}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep((attempt + 1) * 2)

        if not html:
            return offers

        soup = BeautifulSoup(html, "lxml")
        job_rows = soup.select("div.row.fs--15")

        for row in job_rows[: self.max_offers]:
            try:
                offer = self._parse_row(row)
                if offer:
                    offers.append(offer)
            except Exception as e:
                logger.debug(f"Failed to parse Tecnoempleo card: {e}")

        return offers

    def _parse_row(self, row) -> Optional[JobOffer]:
        title_el = row.select_one("h3 a.text-cyan-700, h3 a.font-weight-bold")
        if not title_el:
            return None

        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        is_external = False
        if url:
            if url.startswith("/"):
                url = BASE_URL + url
            elif BASE_URL not in url:
                # Off-platform application link -> mark as not applicable.
                if not url.startswith("http"):
                    url = BASE_URL + "/" + url.lstrip("/")
                else:
                    is_external = True

        external_id = hashlib.md5((title + url).encode()).hexdigest()[:16]

        company = ""
        logo = row.select_one("img[alt]")
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

        combined_text = f"{title} {location}"
        remote = is_remote_text(combined_text)
        hybrid = is_hybrid_text(combined_text) and not remote

        description = ""
        desc_span = row.select_one("span.hidden-md-down.text-gray-800")
        if desc_span:
            description = desc_span.get_text(" ", strip=True)[:3000]

        badge_texts = [b.get_text(strip=True).lower() for b in row.select("span.badge")]
        techs_from_badges = [b for b in badge_texts if b not in ["nueva", "nuevo", "actualizada", "urgente"]]
        all_techs = list(dict.fromkeys(techs_from_badges + detect_techs(f"{title} {description}")))
        seniority = detect_seniority(f"{title} {description}")

        return JobOffer(
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
            is_external_redirect=is_external,
            technologies_detected=all_techs,
            seniority_level=seniority,
        )