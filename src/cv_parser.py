from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber

from src.config import CVProfile, save_cv_profile, settings

logger = logging.getLogger(__name__)

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:años|years|año|year)\s+(?:de\s+)?experiencia",
    r"experiencia\s+(?:de\s+)?(\d+)\+?\s*(?:años|years|año|year)",
    r"(\d+)\+?\s*(?:años|years|año|year)\s+(?:of\s+)?experience",
]

EDUCATION_PATTERNS = [
    r"(?:grado|ingenier[ií]a|licenciatura|m[aá]ster|master|phd|doctorado|bootcamp|fp|formaci[oó]n profesional|bachillerato)",
]

TECH_PATTERNS = [
    "python", "javascript", "typescript", "java", "c#", ".net", "php", "go",
    "golang", "rust", "ruby", "swift", "kotlin", "scala", "c++", "c",
    "react", "angular", "vue", "next.js", "nextjs", "nuxt", "svelte",
    "node.js", "nodejs", "django", "flask", "fastapi", "spring boot",
    "spring", "laravel", "rails", "express", "nestjs", "nest.js",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "graphql", "rest api", "rest", "soap", "grpc", "kafka", "rabbitmq",
    "sql", "nosql", "linux", "git", "agile", "scrum", "kanban",
    "html", "css", "sass", "less", "tailwind", "bootstrap",
    "figma", "jira", "confluence", "slack",
    "ci/cd", "microservicios", "microservices", "tdd",
    "power bi", "tableau", "excel", "spark", "hadoop",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
]


async def parse_cv_pdf(pdf_path: Path) -> CVProfile:
    try:
        full_text = _extract_text(pdf_path)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

    profile = CVProfile(raw_text=full_text)

    if settings.openai_api_key:
        try:
            profile = await _parse_with_llm(full_text)
        except Exception as e:
            logger.warning(f"LLM parsing failed, falling back to regex: {e}")
            profile = _parse_with_regex(full_text)
    else:
        logger.info("No OPENAI_API_KEY set, using regex-based CV parsing")
        profile = _parse_with_regex(full_text)

    save_cv_profile(profile)
    return profile


def _extract_text(pdf_path: Path) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    all_text.append(row_text)

    full_text = "\n".join(all_text)
    if not full_text.strip():
        raise ValueError("No text could be extracted from the PDF. It may be a scanned document.")

    return full_text


async def _parse_with_llm(full_text: str) -> CVProfile:
    import litellm

    prompt = _build_llm_prompt(full_text)
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un parser de CV. Responde SIEMPRE con JSON válido y nada más."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2000,
        api_key=settings.openai_api_key,
    )

    content = response.choices[0].message.content
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()

    data = json.loads(content)
    return CVProfile(
        full_name=data.get("full_name", ""),
        email=data.get("email", ""),
        phone=data.get("phone", ""),
        linkedin=data.get("linkedin", ""),
        summary=data.get("summary", ""),
        experience_years=float(data.get("experience_years", 0)),
        work_history=data.get("work_history", []),
        education=data.get("education", []),
        skills=data.get("skills", []),
        technologies=data.get("technologies", []),
        certs=data.get("certifications", []),
        languages=data.get("languages", []),
        raw_text=full_text,
    )


def _build_llm_prompt(full_text: str) -> str:
    return f"""Extrae la siguiente información del CV en formato JSON estricto. Si un campo no se encuentra, usa array vacío o string vacío.

CV:
{full_text[:8000]}

Formato requerido:
{{
    "full_name": "Nombre completo",
    "email": "email@ejemplo.com",
    "phone": "+34...",
    "linkedin": "url de linkedin",
    "summary": "resumen profesional en 1-2 frases",
    "experience_years": 5.5,
    "work_history": [{{"title": "puesto", "company": "empresa", "period": "2020-2023", "description": "logros y responsabilidades"}}],
    "education": [{{"degree": "título", "institution": "universidad", "year": "2020"}}],
    "skills": ["habilidad1", "habilidad2"],
    "technologies": ["python", "docker"],
    "certifications": ["cert1", "cert2"],
    "languages": ["español nativo", "inglés B2"]
}}"""


def _parse_with_regex(full_text: str) -> CVProfile:
    text_lower = full_text.lower()

    experience_years = _extract_experience_years(full_text, text_lower)

    technologies = []
    for tech in TECH_PATTERNS:
        if re.search(rf'\b{re.escape(tech)}\b', text_lower, re.IGNORECASE):
            technologies.append(tech)

    skills = _extract_section(full_text, ["habilidades", "skills", "competencias"], 500)
    if not skills:
        skills_lines = [line.strip().rstrip(",;") for line in full_text.split("\n")
                       if len(line.strip()) < 80 and line.strip() and not line.startswith("#")]
        skills = [s for s in skills_lines if s][:20]

    education = _extract_education(full_text, text_lower)

    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text)
    phone_match = re.search(r'(\+?\d{1,3}[-. ]?)?\d{9,}', full_text)

    languages = []
    if re.search(r'ingl[ée]s.*(?:b|B)(\d)', full_text):
        m = re.search(r'ingl[ée]s.*(?:b|B)(\d)', full_text)
        languages.append(f"inglés B{m.group(1)}")
    lang_keywords = ["español", "catalán", "francés", "alemán", "italiano", "portugués", "valenciano"]
    for lang in lang_keywords:
        if lang in text_lower:
            languages.append(lang)

    names = _extract_potential_name(full_text)

    return CVProfile(
        full_name=names,
        email=email_match.group() if email_match else "",
        phone=phone_match.group() if phone_match else "",
        summary=full_text[:300].strip(),
        experience_years=experience_years,
        work_history=[],
        education=education,
        skills=skills[:30],
        technologies=technologies,
        languages=languages,
        raw_text=full_text,
    )


def _extract_experience_years(full_text: str, text_lower: str) -> float:
    for pattern in EXPERIENCE_PATTERNS:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            return float(m.group(1))

    years_matches = re.findall(r'(?:20\d\d|19\d\d)\s*[-–/]\s*(?:20\d\d|presente|actualidad|present|actual)', text_lower)
    if years_matches:
        return float(len(years_matches) * 1.5)
    return 1.0


def _extract_section(text: str, keywords: list[str], max_chars: int = 500) -> list[str]:
    for kw in keywords:
        pattern = rf'{re.escape(kw)}[:\s]*(.*?)(?:\n\n|\n[A-Z]|$)'
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            content = m.group(1).strip()[:max_chars]
            return [s.strip().lstrip("-•·* ") for s in re.split(r'[,;\n•·*\-]', content) if s.strip()]
    return []


def _extract_education(full_text: str, text_lower: str) -> list[dict]:
    education = []
    for pattern in EDUCATION_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            start = max(0, m.start() - 20)
            end = min(len(full_text), m.end() + 100)
            context = full_text[start:end].strip()
            line = context.split("\n")[0] if "\n" in context else context
            education.append({"degree": line[:200], "institution": "", "year": ""})
    return education[:5]


def _extract_potential_name(full_text: str) -> str:
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    for line in lines[:5]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() and len(w) > 1 and not w.isupper() for w in words if w.isalpha()):
            if not any(kw in line.lower() for kw in ["curriculum", "cv", "experiencia", "teléfono", "email", "profile"]):
                return line
    return ""
