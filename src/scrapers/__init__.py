from ..scraper import BaseScraper, JobOffer, JobSource, ScrapeResult, ScrapeStatus
from .tecnoempleo import TecnoempleoScraper
from .infojobs import InfojobsScraper

__all__ = [
    "BaseScraper",
    "JobOffer",
    "JobSource",
    "ScrapeResult",
    "ScrapeStatus",
    "TecnoempleoScraper",
    "InfojobsScraper",
]