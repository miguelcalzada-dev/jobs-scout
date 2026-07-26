"""Shared helpers for job scrapers: tech & seniority detection."""
from __future__ import annotations

import re

COMMON_TECHS = [
    "python", "javascript", "typescript", "java", "c#", ".net", "php",
    "go", "golang", "rust", "ruby", "swift", "kotlin", "scala", "c++",
    "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
    "node.js", "nodejs", "django", "flask", "fastapi", "spring boot",
    "spring", "laravel", "rails", "express", "nest.js", "nestjs",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "graphql", "rest", "soap", "grpc", "kafka", "rabbitmq",
    "sql", "nosql", "linux", "git", "agile", "scrum", "kanban",
    "pandas", "numpy", "pytorch", "tensorflow", "keras", "scikit-learn",
    "spark", "hadoop",
    "ci/cd", "microservicios", "microservices", "tdd",
    "machine learning", "deep learning", "nlp", "computer vision",
    "power bi", "tableau", "excel", "jira",
]

_SENIORITY_PATTERNS = [
    (["lead", "principal", "arquitecto", "architect", "director", "manager", "jefe"], "lead"),
    (["senior", "sr.", "experto", "más de 5 años", "mas de 5 años", "+5 años", "+5"], "senior"),
    (["mid", "semi-senior", "semi senior", "2-4 años", "2-5 años", "3-5 años"], "mid"),
    (["junior", "jr.", "entry", "becario", "prácticas", "practicas", "sin experiencia", "trainee"], "junior"),
]


def detect_techs(text: str) -> list[str]:
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for tech in COMMON_TECHS:
        if re.search(rf"\b{re.escape(tech)}\b", text_lower):
            found.append(tech)
    return list(dict.fromkeys(found))


def detect_seniority(text: str) -> str:
    if not text:
        return ""
    text_lower = text.lower()
    for keywords, label in _SENIORITY_PATTERNS:
        if any(kw in text_lower for kw in keywords):
            return label
    return ""


def is_remote_text(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in [
        "remoto", "remote", "teletrabajo", "100% remoto", "en remoto",
        "trabajo desde casa", "work from home", "full remote",
    ])


def is_hybrid_text(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in [
        "híbrido", "hibrido", "hybrid", "semanal", "2-3 días", "2 días",
        "modelo híbrido", "trabajo híbrido",
    ])


def detect_min_experience_years(text: str) -> int:
    if not text:
        return 0
    text_lower = text.lower()

    more_than = re.findall(
        r'(?:m[aá]s\s+de|superior\s+a|mayor\s+de)\s+(\d+)\s*(?:a[ñn]os?|years?)',
        text_lower,
    )
    if more_than:
        return int(more_than[0]) + 1

    plus = re.findall(r'(\d+)\+\s*(?:a[ñn]os?|years?)', text_lower)
    if plus:
        return int(plus[0])

    min_pat = re.findall(
        r'(?:m[ií]nimo|al\s+menos)\s+(\d+)\s*(?:a[ñn]os?|years?)',
        text_lower,
    )
    if min_pat:
        return int(min_pat[0])

    min_exp_pat = re.findall(
        r'(?:experiencia\s+)?m[ií]nima?\s+(?:de\s+)?(\d+)\s*(?:a[ñn]os?|years?)',
        text_lower,
    )
    if min_exp_pat:
        return int(min_exp_pat[0])

    between = re.findall(
        r'entre\s+(\d+)\s*(?:y|a|e|-)\s*\d+\s*(?:a[ñn]os?|years?)',
        text_lower,
    )
    if between:
        return int(between[0])

    exp_years = re.findall(
        r'(?:^|\s)(\d+)\s*(?:a[ñn]os?|years?)\s*(?:de\s+)?experiencia',
        text_lower,
    )
    if exp_years:
        return int(exp_years[0])

    return 0