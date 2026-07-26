from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus
from src.scrapers._common import detect_techs, detect_seniority, is_remote_text, is_hybrid_text

logger = logging.getLogger(__name__)

BASE_URL = "https://www.infojobs.net"
SEARCH_URL = f"{BASE_URL}/jobsearch/search-results/list.xhtml"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]


class InfojobsScraper(BaseScraper):
    """InfoJobs scraper (limited: site is a React SPA).

    The HTML search results page returns partial card data; we extract what's
    available and skip offers whose apply link redirects off-platform.
    """

    source = JobSource.INFOJOBS

    async def scrape(self) -> ScrapeResult:
        start = time.time()
        try:
            prefs = load_preferences()
            offers = await self._search(prefs)
            duration = time.time() - start
            logger.info(f"InfoJobs: scraped {len(offers)} offers in {duration:.1f}s")
            return self._build_result(ScrapeStatus.SUCCESS, offers, duration=duration)
        except Exception as e:
            duration = time.time() - start
            logger.error(f"InfoJobs scraper failed: {e}", exc_info=True)
            return self._build_result(ScrapeStatus.FAILED, error=str(e), duration=duration)

    async def _search(self, prefs) -> list[JobOffer]:
        all_offers: list[JobOffer] = []
        titles = [t for t in (prefs.desired_titles or []) if t]
        if not titles:
            titles = ["python developer", "backend developer"]

        shuffled_titles = list(titles)
        random.shuffle(shuffled_titles)

        tech_keywords = [t for t in (prefs.tech_stack or []) if t and len(t) > 2]
        random.shuffle(tech_keywords)

        search_terms = []
        for t in shuffled_titles[:6]:
            search_terms.append(t)
        for t in tech_keywords[:4]:
            search_terms.append(t)

        random.shuffle(search_terms)

        for query in search_terms:
            if len(all_offers) >= self.max_offers * 3:
                break
            offers = await self._scrape_page(query, location=prefs.location)
            all_offers.extend(offers)
            if len(search_terms) > 1:
                await asyncio.sleep(random.uniform(0.8, 1.6))

        seen = set()
        unique = []
        for o in all_offers:
            if o.external_id not in seen:
                seen.add(o.external_id)
                unique.append(o)
        return unique[: self.max_offers]

    async def _scrape_page(self, query: str, location: str = "") -> list[JobOffer]:
        offers: list[JobOffer] = []
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "DNT": "1",
        }
        params = {
            "keyword": query,
            "sortBy": "RELEVANCE",
            "sinceDate": "ANY",
        }
        if location:
            params["city"] = location

        html = ""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    resp = await client.get(SEARCH_URL, params=params, headers=headers)
                    resp.raise_for_status()
                    html = resp.text
                    break
                except httpx.HTTPError as e:
                    logger.warning(f"InfoJobs attempt {attempt+1}/3 failed: {e}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep((attempt + 1) * 2)

        if not html:
            return offers

        soup = BeautifulSoup(html, "lxml")
        cards = soup.select(".ij-OfferList-offerCardItem, [class*=offerCard]")
        if not cards:
            cards = soup.select("[class*=offer], [class*=result], [class*=vacancy]")

        for card in cards[: self.max_offers]:
            try:
                offer = self._parse_card(card)
                if offer:
                    offers.append(offer)
            except Exception as e:
                logger.debug(f"Failed to parse InfoJobs card: {e}")

        return offers

    def _parse_card(self, card) -> Optional[JobOffer]:
        title_el = card.select_one("h2, h3, a[class*=title], [class*=title]")
        company_el = card.select_one("[class*=company], [class*=subtitle], [class*=employer]")
        location_el = card.select_one("[class*=location], [class*=city], [class*=place]")

        if not title_el:
            title_el = card.select_one("a")
        if not title_el:
            return None

        title = title_el.get_text(strip=True)
        if len(title) < 3:
            return None

        is_external = False
        url = ""
        if title_el.name == "a":
            href = title_el.get("href", "")
            url = href if href.startswith("http") else urljoin(BASE_URL, href)
        else:
            title_link = title_el.select_one("a[href]")
            if title_link:
                href = title_link.get("href", "")
                url = href if href.startswith("http") else urljoin(BASE_URL, href)

        if not url:
            offer_link = card.select_one("a[href*='oferta'], a[href*='offer'], a[href*='job']")
            if not offer_link:
                offer_link = card.select_one("a[href*='infojobs.net']")
            if offer_link:
                href = offer_link.get("href", "")
                url = href if href.startswith("http") else urljoin(BASE_URL, href)

        if url and BASE_URL not in url and "infojobs" not in url:
            is_external = True

        external_id = hashlib.md5((title + url).encode()).hexdigest()[:16]
        company = company_el.get_text(strip=True) if company_el else ""

        location = location_el.get_text(strip=True) if location_el else ""

        card_text = card.get_text(" ", strip=True)
        combined = f"{title} {card_text}"
        remote = is_remote_text(combined)
        hybrid = is_hybrid_text(combined) and not remote

        description = card_text[:2000]
        techs = detect_techs(f"{title} {description}")
        seniority = detect_seniority(f"{title} {description}")

        contract_type = ""
        card_lower = card_text.lower()
        if "indefinido" in card_lower:
            contract_type = "indefinido"
        elif "temporal" in card_lower:
            contract_type = "temporal"

        return JobOffer(
            source=JobSource.INFOJOBS,
            external_id=external_id,
            title=title,
            company=company or "No especificada",
            location=location or "No especificada",
            url=url,
            description=description,
            contract_type=contract_type,
            remote=remote,
            hybrid=hybrid,
            onsite=not remote and not hybrid,
            is_external_redirect=is_external,
            technologies_detected=techs,
            seniority_level=seniority,
        )