from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
import requests
from bs4 import BeautifulSoup

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus

logger = logging.getLogger(__name__)

BASE_URL = "https://www.infojobs.net"
SEARCH_URL = f"{BASE_URL}/jobsearch/search-results/list.xhtml"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]

TECH_KEYWORDS = [
    "python", "javascript", "typescript", "java", "c#", ".net", "php", "go", "golang",
    "rust", "ruby", "swift", "kotlin", "scala", "c++",
    "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
    "node.js", "nodejs", "django", "flask", "fastapi", "spring", "spring boot",
    "laravel", "rails", "express", "nest.js", "nestjs",
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "graphql", "rest", "grpc", "kafka", "rabbitmq",
    "sql", "nosql", "linux", "git", "agile", "scrum",
    "pandas", "numpy", "pytorch", "tensorflow", "spark",
    "power bi", "tableau", "excel", "jira",
    "ci/cd", "microservicios", "microservices", "tdd",
    "machine learning", "deep learning", "nlp",
]


class InfojobsScraper(BaseScraper):
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
        titles = prefs.desired_titles or ["python developer", "backend developer"]

        for title in titles[:5]:
            if len(all_offers) >= self.max_offers:
                break
            offers = await self._scrape_page(title)
            all_offers.extend(offers)
            if len(titles) > 1:
                await asyncio.sleep(random.uniform(1.0, 2.0))

        seen = set()
        unique = []
        for o in all_offers:
            key = o.external_id
            if key not in seen:
                seen.add(key)
                unique.append(o)
        return unique[: self.max_offers]

    async def _scrape_page(self, query: str) -> list[JobOffer]:
        offers: list[JobOffer] = []
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "DNT": "1",
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(
                        SEARCH_URL,
                        params={"keyword": query, "sortBy": "RELEVANCE", "sinceDate": "ANY"},
                        headers=headers,
                    )
                    resp.raise_for_status()
                    html = resp.text
                    break
            except httpx.HTTPError as e:
                logger.warning(f"InfoJobs attempt {attempt+1}/3 failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep((attempt + 1) * 2)

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
        link_el = card.select_one("a[href*='infojobs.net']")

        if not title_el:
            title_el = card.select_one("a")
        if not title_el:
            return None

        title = title_el.get_text(strip=True)
        if len(title) < 3:
            return None

        url = ""
        if link_el:
            href = link_el.get("href", "")
            url = href if href.startswith("http") else urljoin(BASE_URL, href)

        external_id = hashlib.md5((title + url).encode()).hexdigest()[:16]
        company = company_el.get_text(strip=True) if company_el else ""

        location = ""
        if location_el:
            location = location_el.get_text(strip=True)

        card_text = card.get_text(" ", strip=True).lower()

        if not location:
            loc_match = re.search(
                r'(?:en\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)',
                card_text
            )
            if loc_match:
                location = loc_match.group(1)

        remote = any(kw in card_text for kw in [
            "remoto", "remote", "teletrabajo", "100% remoto",
            "trabajo desde casa", "work from home",
        ])
        hybrid = any(kw in card_text for kw in [
            "híbrido", "hibrido", "hybrid", "semanal",
        ]) and not remote

        salary = ""
        sal_match = re.search(r'(?:salario|salary|retribución)[:\s]*([\d.,]+\s*[€kK]*)', card_text, re.IGNORECASE)
        if sal_match:
            salary = sal_match.group(1)

        description = card_text[:2000]
        if url and not description:
            description = self._fetch_detail_description(url)

        techs = self._detect_techs(title + " " + description)
        seniority = self._detect_seniority(title + " " + description)

        contract_type = ""
        if "indefinido" in card_text:
            contract_type = "indefinido"
        elif "temporal" in card_text:
            contract_type = "temporal"

        return JobOffer(
            source=JobSource.INFOJOBS,
            external_id=external_id,
            title=title,
            company=company or "No especificada",
            location=location or "No especificada",
            url=url,
            description=description,
            salary=salary,
            contract_type=contract_type,
            remote=remote,
            hybrid=hybrid,
            onsite=not remote and not hybrid,
            technologies_detected=techs,
            seniority_level=seniority,
        )

    def _detect_techs(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for tech in TECH_KEYWORDS:
            if re.search(rf'\b{re.escape(tech)}\b', text_lower):
                found.append(tech)
        return list(dict.fromkeys(found))

    def _detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["lead", "principal", "arquitecto", "architect", "director", "manager", "jefe"]):
            return "lead"
        if any(kw in text_lower for kw in ["senior", "sr.", "experto", "+5", "más de 5"]):
            return "senior"
        if any(kw in text_lower for kw in ["mid", "semi-senior", "2-4 años", "3-5 años"]):
            return "mid"
        if any(kw in text_lower for kw in ["junior", "jr.", "entry", "becario", "prácticas", "trainee", "sin experiencia"]):
            return "junior"
        return ""

    def _fetch_detail_description(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            desc_el = soup.select_one("[class*=description], [class*=detail], [class*=requisitos], #job-description, article")
            if desc_el:
                return desc_el.get_text(" ", strip=True)[:5000]
            body_text = soup.body.get_text(" ", strip=True) if soup.body else ""
            return body_text[:5000]
        except Exception:
            return ""
