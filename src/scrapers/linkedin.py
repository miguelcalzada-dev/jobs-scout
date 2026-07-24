from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from typing import Optional
from urllib.parse import quote_plus

from src.config import load_preferences
from src.scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus

logger = logging.getLogger(__name__)

LINKEDIN_SEARCH_URL = "https://www.linkedin.com/jobs/search/"


class LinkedInScraper(BaseScraper):
    source = JobSource.LINKEDIN

    async def scrape(self) -> ScrapeResult:
        start = time.time()
        try:
            prefs = load_preferences()
            offers = await self._search(prefs)
            duration = time.time() - start
            logger.info(f"LinkedIn: scraped {len(offers)} offers in {duration:.1f}s")
            return self._build_result(ScrapeStatus.SUCCESS, offers, duration=duration)
        except Exception as e:
            duration = time.time() - start
            logger.error(f"LinkedIn scraper failed: {e}", exc_info=True)
            return self._build_result(ScrapeStatus.FAILED, error=str(e), duration=duration)

    async def _search(self, prefs) -> list[JobOffer]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed. Install with: playwright install chromium")
            return self._build_result(ScrapeStatus.FAILED, error="Playwright not installed").offers or []

        all_offers: list[JobOffer] = []
        titles = prefs.desired_titles or ["software engineer"]
        location = prefs.location or "España"

        for title in titles[:3]:
            offers = await self._scrape_with_playwright(title, location)
            all_offers.extend(offers)
            if len(all_offers) >= self.max_offers:
                break
            await asyncio.sleep(random.uniform(2.0, 4.0))

        return all_offers[: self.max_offers]

    async def _scrape_with_playwright(self, query: str, location: str) -> list[JobOffer]:
        offers: list[JobOffer] = []

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        async with async_playwright() as p:
            browser = None
            try:
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

                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    locale="es-ES",
                )

                await context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es', 'en']});
                """)

                page = await context.new_page()

                params = {
                    "keywords": query,
                    "location": location,
                    "f_TPR": f"r{self.max_job_age_days * 86400}" if self.max_job_age_days > 0 else "",
                    "geoId": "105646813",
                    "f_WT": "",
                    "position": "1",
                    "pageNum": "0",
                }

                search_url = LINKEDIN_SEARCH_URL + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v)

                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(random.randint(3000, 5000))

                for scroll_attempt in range(5):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(random.randint(1500, 3000))

                job_cards = await page.query_selector_all("li[class*=job-result], div[class*=job-card], li[class*=result]")
                if not job_cards:
                    job_cards = await page.query_selector_all("ul.jobs-search__results-list > li")

                for card in job_cards[: self.max_offers]:
                    try:
                        offer = await self._parse_card(card, page)
                        if offer:
                            offers.append(offer)
                    except Exception as e:
                        logger.debug(f"Failed to parse LinkedIn card: {e}")

            except Exception as e:
                logger.error(f"LinkedIn Playwright error: {e}", exc_info=True)
            finally:
                if browser:
                    await browser.close()

        return offers

    async def _parse_card(self, card, page) -> Optional[JobOffer]:
        title_el = await card.query_selector("[class*=title], h3, a[class*=title]")
        company_el = await card.query_selector("[class*=company], [class*=subtitle]")
        location_el = await card.query_selector("[class*=location], [class*=city]")
        link_el = await card.query_selector("a[href*=jobs]")

        if not title_el:
            return None

        title = (await title_el.inner_text()).strip()
        company = (await company_el.inner_text()).strip() if company_el else "No especificada"
        location = (await location_el.inner_text()).strip() if location_el else "No especificada"

        url = ""
        if link_el:
            href = await link_el.get_attribute("href")
            if href:
                url = href.split("?")[0] if "?" in href else href

        external_id = hashlib.md5((title + company + url).encode()).hexdigest()[:16]

        techs = self._detect_techs(title)
        remote = "remote" in title.lower() or "remoto" in title.lower() or "teletrabajo" in title.lower()

        return JobOffer(
            source=JobSource.LINKEDIN,
            external_id=external_id,
            title=title,
            company=company,
            location=location,
            url=url,
            description="",
            remote=remote,
            technologies_detected=techs,
            seniority_level=self._detect_seniority(title),
        )

    def _detect_techs(self, text: str) -> list[str]:
        text_lower = text.lower()
        common_techs = [
            "python", "javascript", "typescript", "java", "c#", ".net", "php",
            "go", "golang", "rust", "ruby", "swift", "kotlin",
            "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
            "node.js", "nodejs", "django", "flask", "fastapi", "spring",
            "laravel", "rails", "express", "nest.js", "nestjs",
            "aws", "azure", "gcp", "docker", "kubernetes", "k8s",
            "terraform", "ansible", "postgresql", "postgres", "mysql", "mongodb",
            "redis", "elasticsearch", "graphql", "rest", "grpc", "kafka",
            "sql", "nosql", "linux", "git",
        ]
        return [tech for tech in common_techs if tech in text_lower]

    def _detect_seniority(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["lead", "principal", "arquitecto", "director", "manager"]):
            return "lead"
        if any(kw in text_lower for kw in ["senior", "sr.", "experto"]):
            return "senior"
        if any(kw in text_lower for kw in ["mid", "semi-senior"]):
            return "mid"
        if any(kw in text_lower for kw in ["junior", "jr.", "entry", "trainee"]):
            return "junior"
        return ""
