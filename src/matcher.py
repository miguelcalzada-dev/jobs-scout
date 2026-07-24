from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_preferences, load_cv_profile, settings
from src.database import get_unscored_jobs, update_job_score

logger = logging.getLogger(__name__)

_embedding_model = None


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

    user_techs = [t.lower().strip() for t in (prefs.tech_stack or [])]
    matched_techs = 0
    for tech in user_techs:
        if tech in job_techs:
            matched_techs += 1
        elif tech in job_text:
            matched_techs += 1
    score += matched_techs * 8.0

    for title_str in prefs.desired_titles:
        title_lower = title_str.lower().strip()
        if title_lower in job_title:
            score += 14.0
            break
    else:
        title_word_hits = 0
        for title_str in prefs.desired_titles:
            for word in title_str.lower().split():
                if len(word) > 3 and word in job_title:
                    title_word_hits += 1
        score += title_word_hits * 6.0

    if prefs.location:
        loc_lower = prefs.location.lower()
        if loc_lower in job_location:
            score += 12.0
        elif loc_lower in job_text:
            score += 6.0
        else:
            for word in loc_lower.split():
                if len(word) > 2 and word in job_location:
                    score += 3.0
                    break

    if job.get("remote") and prefs.remote_only:
        score += 12.0
    elif job.get("remote") and not prefs.remote_only:
        score += 8.0
    elif job.get("hybrid") and prefs.hybrid_allowed:
        score += 6.0

    if prefs.min_salary > 0 and job_salary > 0:
        if job_salary >= prefs.min_salary:
            score += 10.0
        elif job_salary >= prefs.min_salary * 0.8:
            score += 5.0

    if prefs.seniority:
        sen_lower = prefs.seniority.lower()
        if sen_lower in job_seniority:
            score += 10.0
        elif job_title and (sen_lower in job_title or "senior" in job_title or "lead" in job_title):
            score += 5.0

    for kw in prefs.exclude_keywords:
        if kw.lower() in job_text:
            score -= 30.0

    for sector in prefs.exclude_sectors:
        if sector.lower() in job_text or sector.lower() in job_company:
            score -= 25.0

    return score


async def score_all_unscored_jobs(rescore_all: bool = False) -> int:
    prefs = load_preferences()
    cv_profile = load_cv_profile()

    jobs = get_unscored_jobs(limit=300, rescore_all=rescore_all)
    if not jobs:
        logger.info("No hay ofertas para puntuar")
        return 0

    logger.info(f"Puntuando {len(jobs)} ofertas...")

    if cv_profile and cv_profile.technologies:
        cv_text = cv_profile.to_text_block()
    else:
        cv_text = " ".join(prefs.tech_stack) if prefs.tech_stack else "python backend developer"

    job_texts = []
    for job in jobs:
        title = job.get("title", "")
        techs = (job.get("technologies_detected") or "")
        location = job.get("location") or ""
        desc = (job.get("description") or "")[:1200]
        jt = f"{title}. Tecnologías: {techs}. Ubicación: {location}. {desc}"
        job_texts.append(jt)

    try:
        cv_emb = _encode([cv_text])[0]
        job_embs = _encode(job_texts)
    except Exception as e:
        logger.error(f"Error embeddings: {e}, usando solo reglas")
        cv_emb = np.zeros(384)
        job_embs = np.zeros((len(jobs), 384))

    scored_count = 0
    for i, job in enumerate(jobs):
        try:
            sim = _compute_similarity(cv_emb, job_embs[i]) if job_embs[i].any() else 0.0

            pref_score = _score_from_preferences(job, prefs)

            final_score = (sim * 50.0) + pref_score
            final_score = max(0.0, min(100.0, final_score))

            update_job_score(
                external_id=job["external_id"],
                source=job["source"],
                score=final_score,
                match_reason=f"s={sim:.3f}_r={pref_score:.1f}",
            )
            scored_count += 1
        except Exception as e:
            logger.debug(f"Error puntuando {job.get('external_id')}: {e}")

    logger.info(f"Puntuadas {scored_count}/{len(jobs)} ofertas")
    return scored_count
