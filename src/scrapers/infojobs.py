from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, quote_plus

import httpx
from bs4 import BeautifulSoup

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus

logger = logging.getLogger(__name__)

BASE_URL = "https://www.infojobs.net"
SEARCH_URL = f"{BASE_URL}/jobsearch/search-results/list.xhtml"

REQUESTS_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": "https://www.infojobs.net/",
        "X-Requested-With": "XMLHttpRequest",
        "DNT": "1",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Referer": "https://www.infojobs.net/",
        "X-Requested-With": "XMLHttpRequest",
        "DNT": "1",
    },
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
        titles = prefs.desired_titles or ["desarrollador", "ingeniero software"]

        for title in titles[:3]:
            offers = await self._scrape_page(title, page=1)
            all_offers.extend(offers)
            if len(all_offers) >= self.max_offers:
                break
            await asyncio.sleep(random.uniform(1.5, 3.0))

        return all_offers[: self.max_offers]

    async def _scrape_page(self, query: str, page: int = 1) -> list[JobOffer]:
        offers: list[JobOffer] = []
        headers = random.choice(REQUESTS_HEADERS).copy()

        params = {
            "keyword": query,
            "page": str(page),
            "sortBy": "RELEVANCE",
            "offerLocations": "",
            "sinceDate": "ANY",
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(SEARCH_URL, params=params, headers=headers)
                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    if "json" in content_type:
                        data = resp.json()
                        offers = self._parse_json_response(data)
                    else:
                        offers = self._parse_html_response(resp.text)
                    break
            except httpx.HTTPError as e:
                logger.warning(f"InfoJobs attempt {attempt+1}/3 failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep((attempt + 1) * 2.5)

        return offers

    def _parse_json_response(self, data: dict) -> list[JobOffer]:
        offers: list[JobOffer] = []
        items = []
        if "offers" in data:
            items = data["offers"]
        elif "items" in data:
            items = data["items"]
        elif "results" in data:
            items = data["results"]

        for item in items[: self.max_offers]:
            try:
                external_id = str(item.get("id", ""))
                title = item.get("title", "")
                company = item.get("profile", {}).get("name", "No especificada") if isinstance(item.get("profile"), dict) else "No especificada"
                location = ""
                if isinstance(item.get("city"), str):
                    location = item["city"]
                elif isinstance(item.get("city"), dict):
                    location = item["city"].get("name", "")
                elif isinstance(item.get("location"), str):
                    location = item["location"]

                url = f"https://www.infojobs.net/ofertas-trabajo/{item.get('slug', '')}" if item.get("slug") else ""
                description = item.get("description", "") or item.get("summary", "") or ""
                salary_min = 0
                salary_max = 0
                if isinstance(item.get("salaryDescription"), str):
                    salary_match = re.search(r'[\d.,]+', item["salaryDescription"])
                    if salary_match:
                        salary_min = self._parse_salary(salary_match.group())

                remote = any(kw in str(item).lower() for kw in ["teletrabajo", "remote", "remoto"])
                techs = self._detect_techs(f"{title} {description}")
                seniority = self._detect_seniority(f"{title} {description}")

                offers.append(JobOffer(
                    source=JobSource.INFOJOBS,
                    external_id=external_id,
                    title=title,
                    company=company,
                    location=location,
                    url=url,
                    description=description[:3000],
                    salary=str(item.get("salaryDescription", "")),
                    salary_min=salary_min,
                    salary_max=salary_max,
                    remote=remote,
                    technologies_detected=techs,
                    seniority_level=seniority,
                    raw_data=item,
                ))
            except Exception as e:
                logger.debug(f"Failed to parse InfoJobs offer: {e}")

        return offers

    def _parse_html_response(self, html: str) -> list[JobOffer]:
        offers: list[JobOffer] = []
        soup = BeautifulSoup(html, "lxml")

        cards = soup.select("[class*=card], [class*=result], [class*=offer], [class*=vacancy]")
        if not cards:
            cards = soup.select("li, .list-item, article")

        for card in cards[: self.max_offers]:
            try:
                title_el = card.select_one("h2, h3, [class*=title]")
                company_el = card.select_one("[class*=company], [class*=subtitle]")
                location_el = card.select_one("[class*=location], [class*=city]")
                link_el = card.select_one("a[href]")

                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                url = urljoin(BASE_URL, link_el["href"]) if link_el and link_el.get("href") else ""
                external_id = hashlib.md5((title + url).encode()).hexdigest()[:16]
                company = company_el.get_text(strip=True) if company_el else "No especificada"
                location = location_el.get_text(strip=True) if location_el else "No especificada"

                offers.append(JobOffer(
                    source=JobSource.INFOJOBS,
                    external_id=external_id,
                    title=title,
                    company=company,
                    location=location,
                    url=url,
                    description="",
                    technologies_detected=self._detect_techs(title),
                ))
            except Exception:
                continue

        return offers

    def _parse_salary(self, text: str) -> int:
        text = text.replace(".", "").replace(",", "")
        try:
            return int(text)
        except ValueError:
            return 0

    def _detect_techs(self, text: str) -> list[str]:
        text_lower = text.lower()
        common_techs = [
            "python", "javascript", "typescript", "java", "c#", ".net", "php",
            "go", "golang", "rust", "ruby", "swift", "kotlin",
            "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
            "node.js", "nodejs", "django", "flask", "fastapi", "spring",
            "laravel", "rails", "express", "nest.js", "nestjs",
            "aws", "azure", "gcp", "docker", "kubernetes", "k8s",
            "terraform", "ansible", "jenkins", "github actions",
            "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
            "graphql", "rest", "grpc", "kafka", "rabbitmq",
            "sql", "nosql", "linux", "git",
        ]
        return [tech for tech in common_techs if tech in text_lower]

    def _detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["lead", "principal", "arquitecto", "director", "manager", "jefe"]):
            return "lead"
        if any(kw in text_lower for kw in ["senior", "sr.", "experto", "+5 años", "más de 5"]):
            return "senior"
        if any(kw in text_lower for kw in ["mid", "semi-senior", "2-4 años", "3-5 años"]):
            return "mid"
        if any(kw in text_lower for kw in ["junior", "jr.", "entry", "becario", "prácticas", "trainee"]):
            return "junior"
        return ""
