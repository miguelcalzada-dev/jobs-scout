from __future__ import annotations

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CV_DIR = BASE_DIR / "cv"
TEMPLATES_DIR = BASE_DIR / "templates"
CONFIG_FILE = BASE_DIR / "config.yaml"
CV_PROFILE_FILE = BASE_DIR / "cv_profile.json"
DB_PATH = DATA_DIR / "jobs.db"

for d in [DATA_DIR, CV_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    openai_api_key: str = ""
    email_host: str = "smtp.gmail.com"
    email_port: int = 587
    email_user: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: str = ""
    daily_send_hour: int = 9
    daily_send_minute: int = 0
    max_jobs_per_day: int = 10
    port: int = 8080
    environment: str = "development"
    max_job_age_days: int = 3
    headless: bool = True
    tz: str = "Europe/Madrid"
    tecnoempleo_email: str = ""
    tecnoempleo_password: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


@dataclass
class JobPreferences:
    desired_titles: list[str] = field(default_factory=list)
    tech_stack: list[str] = field(default_factory=list)
    location: str = ""
    remote_only: bool = False
    hybrid_allowed: bool = True
    onsite_allowed: bool = False
    min_salary: int = 0
    seniority: str = ""
    exclude_keywords: list[str] = field(default_factory=list)
    company_size: str = ""
    languages: list[str] = field(default_factory=list)
    exclude_sectors: list[str] = field(default_factory=list)
    enabled_scrapers: list[str] = field(default_factory=lambda: ["tecnoempleo", "infojobs", "linkedin"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JobPreferences":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CVProfile:
    full_name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    summary: str = ""
    experience_years: float = 0.0
    work_history: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    certs: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    raw_text: str = ""
    profile_embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.profile_embedding is not None:
            d["profile_embedding"] = self.profile_embedding
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CVProfile":
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean)

    def to_text_block(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Resumen: {self.summary}")
        if self.technologies:
            parts.append(f"Tecnologías: {', '.join(self.technologies)}")
        if self.skills:
            parts.append(f"Habilidades: {', '.join(self.skills)}")
        if self.work_history:
            history = []
            for w in self.work_history:
                history.append(
                    f"- {w.get('title', '')} en {w.get('company', '')} ({w.get('period', '')}): {w.get('description', '')}"
                )
            parts.append(f"Experiencia:\n" + "\n".join(history))
        if self.education:
            edu = []
            for e in self.education:
                edu.append(f"- {e.get('degree', '')} en {e.get('institution', '')} ({e.get('year', '')})")
            parts.append(f"Educación:\n" + "\n".join(edu))
        if self.certs:
            parts.append(f"Certificaciones: {', '.join(self.certs)}")
        if self.languages:
            parts.append(f"Idiomas: {', '.join(self.languages)}")
        return "\n\n".join(parts)


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def load_preferences() -> JobPreferences:
    cfg = load_config()
    prefs = cfg.get("preferences", {})
    return JobPreferences.from_dict(prefs)


def save_preferences(prefs: JobPreferences) -> None:
    cfg = load_config()
    cfg["preferences"] = prefs.to_dict()
    save_config(cfg)


def load_cv_profile() -> Optional[CVProfile]:
    if CV_PROFILE_FILE.exists():
        with open(CV_PROFILE_FILE, "r") as f:
            data = json.load(f)
            return CVProfile.from_dict(data)
    return None


def save_cv_profile(profile: CVProfile) -> None:
    with open(CV_PROFILE_FILE, "w") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)


def copy_cv_to_dir(source: Path) -> Path:
    dest = CV_DIR / source.name
    shutil.copy2(source, dest)
    return dest
