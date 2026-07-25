from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class JobSource(Enum):
    LINKEDIN = "linkedin"
    INFOJOBS = "infojobs"
    TECNOEMPLEO = "tecnoempleo"


class ScrapeStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class JobOffer:
    source: JobSource
    external_id: str
    title: str
    company: str
    location: str
    url: str
    description: str
    description_html: str = ""
    salary: str = ""
    salary_min: int = 0
    salary_max: int = 0
    contract_type: str = ""
    remote: bool = False
    hybrid: bool = False
    onsite: bool = False
    published_date: Optional[str] = None
    scrape_date: str = field(default_factory=lambda: datetime.now().isoformat())
    technologies_detected: list[str] = field(default_factory=list)
    seniority_level: str = ""
    company_size: str = ""
    sector: str = ""
    is_external_redirect: bool = False
    raw_data: dict = field(default_factory=dict)

    @property
    def unique_key(self) -> str:
        return f"{self.source.value}:{self.external_id}"

    def to_searchable_text(self) -> str:
        parts = [
            self.title or "",
            self.company or "",
            self.description or "",
            self.location or "",
            self.contract_type or "",
            self.seniority_level or "",
            self.salary or "",
            " ".join(self.technologies_detected),
        ]
        return "\n".join(p for p in parts if p)


@dataclass
class ScrapeResult:
    source: JobSource
    status: ScrapeStatus
    offers: list[JobOffer] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    pages_scraped: int = 0
    offers_total: int = 0


class BaseScraper(ABC):
    source: JobSource

    def __init__(self, max_offers: int = 50, max_job_age_days: int = 3):
        self.max_offers = max_offers
        self.max_job_age_days = max_job_age_days
        self.session_offers: list[JobOffer] = []

    @abstractmethod
    async def scrape(self) -> ScrapeResult:
        ...

    def _build_result(
        self,
        status: ScrapeStatus,
        offers: Optional[list[JobOffer]] = None,
        error: Optional[str] = None,
        duration: float = 0.0,
        pages: int = 0,
    ) -> ScrapeResult:
        return ScrapeResult(
            source=self.source,
            status=status,
            offers=offers or [],
            error=error,
            duration_seconds=duration,
            pages_scraped=pages,
            offers_total=len(offers) if offers else 0,
        )
