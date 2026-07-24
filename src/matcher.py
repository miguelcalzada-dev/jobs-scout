from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_preferences, load_cv_profile, settings
from src.database import get_unscored_jobs, update_job_score

logger = logging.getLogger(__name__)

_embedding_model = None

TECH_KEYWORDS_EXPANDED = {
    "python": ["python", "django", "flask", "fastapi", "pytest", "pandas", "numpy"],
    "javascript": ["javascript", "js", "node", "nodejs", "node.js", "express"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "spring", "springboot", "spring boot", "j2ee", "hibernate"],
    "c#": ["c#", ".net", "dotnet", "asp.net", "entity framework"],
    "react": ["react", "reactjs", "react.js", "nextjs", "next.js", "redux"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js", "nuxt"],
    "aws": ["aws", "amazon web services", "lambda", "s3", "ec2", "cloudformation"],
    "docker": ["docker", "docker-compose", "container"],
    "kubernetes": ["kubernetes", "k8s", "kubectl", "helm"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "graphql": ["graphql", "apollo"],
    "kafka": ["kafka", "event-driven"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
    "ci/cd": ["ci/cd", "jenkins", "github actions", "gitlab ci", "argocd"],
    "machine learning": ["machine learning", "ml", "deep learning", "ai", "nlp", "computer vision"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
}


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Cargando modelo de embeddings...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Modelo de embeddings cargado")
    return _embedding_model


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _compute_similarity(emb1, emb2) -> float:
    return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])


def _compute_tech_match_score(job_text: str, job_techs: str, user_techs: list[str]) -> float:
    job_lower = job_text.lower()
    job_techs_lower = job_techs.lower()

    matched = 0
    matched_categories = set()

    for user_tech in user_techs:
        user_lower = user_tech.lower().strip()

        if user_lower in job_techs_lower:
            matched += 1
            continue

        if user_lower in job_lower:
            matched += 1
            continue

        for category_key, aliases in TECH_KEYWORDS_EXPANDED.items():
            if user_lower == category_key or user_lower in aliases:
                for alias in aliases:
                    if alias in job_lower:
                        matched += 1
                        break
                break

    return matched


def _score_from_preferences(job: dict, prefs) -> float:
    score = 0.0

    job_title = (job.get("title") or "").lower()
    job_desc = (job.get("description") or "").lower()
    job_text = f"{job_title} {job_desc}"
    job_location = (job.get("location") or "").lower()
    job_techs = (job.get("technologies_detected") or "").lower()
    job_seniority = (job.get("seniority_level") or "").lower()
    job_salary = job.get("salary_min") or 0
    job_company = (job.get("company") or "").lower()

    user_techs = prefs.tech_stack if prefs.tech_stack else []
    tech_matches = _compute_tech_match_score(job_text, job_techs, user_techs)

    if user_techs:
        tech_ratio = tech_matches / max(len(user_techs), 1)
        score += tech_ratio * 35.0
    else:
        score += tech_matches * 3.0

    title_matches = 0
    for t in prefs.desired_titles:
        t_lower = t.lower().strip()
        if t_lower in job_title:
            title_matches += 1

    title_words_in_job = 0
    for t in prefs.desired_titles:
        for word in t.lower().split():
            if len(word) > 2 and word in job_title:
                title_words_in_job += 1
                break

    score += title_matches * 12.0
    score += title_words_in_job * 4.0

    if prefs.location:
        loc_lower = prefs.location.lower()
        if loc_lower in job_location:
            score += 10.0
        elif loc_lower in job_text:
            score += 5.0
        elif any(word in job_location for word in loc_lower.split() if len(word) > 2):
            score += 4.0

    if job.get("remote") and prefs.remote_only:
        score += 10.0
    elif job.get("remote") and (prefs.hybrid_allowed or not prefs.remote_only):
        score += 6.0
    elif job.get("hybrid") and prefs.hybrid_allowed:
        score += 5.0

    if prefs.min_salary > 0 and job_salary > 0:
        if job_salary >= prefs.min_salary:
            score += 8.0
        elif job_salary >= prefs.min_salary * 0.8:
            score += 4.0

    if prefs.seniority:
        sen_lower = prefs.seniority.lower()
        if sen_lower in job_seniority:
            score += 8.0
        elif sen_lower == "senior" and any(kw in job_text for kw in ["senior", "sr.", "experto"]):
            score += 5.0
        elif sen_lower == "mid" and any(kw in job_text for kw in ["mid", "semi"]):
            score += 5.0

    for kw in prefs.exclude_keywords:
        if kw.lower() in job_text:
            score -= 25.0

    for sector in prefs.exclude_sectors:
        if sector.lower() in job_text or sector.lower() in job_company:
            score -= 20.0

    salary_keywords = {
        "20k": 20000, "25k": 25000, "30k": 30000, "35k": 35000,
        "40k": 40000, "45k": 45000, "50k": 50000, "55k": 55000,
        "60k": 60000, "65k": 65000, "70k": 70000, "80k": 80000,
        "90k": 90000, "100k": 100000,
    }
    for kw, val in salary_keywords.items():
        if kw in job_text and val >= prefs.min_salary:
            score += 3.0
            break

    return score


async def score_all_unscored_jobs() -> int:
    prefs = load_preferences()
    cv_profile = load_cv_profile()

    jobs = get_unscored_jobs(limit=200)
    if not jobs:
        logger.info("No hay ofertas nuevas para puntuar")
        return 0

    logger.info(f"Puntuando {len(jobs)} ofertas nuevas...")

    if cv_profile:
        cv_text = cv_profile.to_text_block()
    else:
        cv_text = " ".join(prefs.tech_stack) if prefs.tech_stack else "tecnologías programación"

    job_texts = []
    for job in jobs:
        title = job.get("title", "")
        desc = (job.get("description") or "")[:1500]
        techs = (job.get("technologies_detected") or "")
        location = job.get("location") or ""
        jt = f"{title} | {techs} | {location} | {desc}"
        job_texts.append(jt)

    try:
        cv_emb = _encode([cv_text])[0]
        job_embs = _encode(job_texts)
    except Exception as e:
        logger.error(f"Error en embeddings: {e}, usando solo reglas")
        cv_emb = np.zeros(384)
        job_embs = np.zeros((len(jobs), 384))

    scored_count = 0
    for i, job in enumerate(jobs):
        try:
            similarity = _compute_similarity(cv_emb, job_embs[i]) if job_embs[i].any() else 0.0

            pref_score = _score_from_preferences(job, prefs)

            final_score = (similarity * 40.0) + pref_score
            final_score = max(0.0, min(100.0, final_score))

            update_job_score(
                external_id=job["external_id"],
                source=job["source"],
                score=final_score,
                match_reason=f"sim_semantica={similarity:.3f}, reglas={pref_score:.1f}",
            )
            scored_count += 1
        except Exception as e:
            logger.debug(f"Error puntuando oferta {job.get('external_id')}: {e}")

        if scored_count % 50 == 0:
            await asyncio.sleep(0.01)

    logger.info(f"Puntuadas {scored_count}/{len(jobs)} ofertas")
    return scored_count
