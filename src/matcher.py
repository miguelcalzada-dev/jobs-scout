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


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (first run, this may take a moment)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Embedding model loaded")
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

    user_techs = [t.lower() for t in prefs.tech_stack]
    for tech in user_techs:
        if tech in job_text or tech in job_techs:
            score += 2

    title_matches = [t.lower() for t in prefs.desired_titles]
    for t in title_matches:
        if t in job_title:
            score += 3

    if prefs.location:
        loc_lower = prefs.location.lower()
        if loc_lower in job_location:
            score += 5
        elif any(word in job_location for word in loc_lower.split()):
            score += 2

    if prefs.remote_only and (job.get("remote") or "remoto" in job_text or "remote" in job_text):
        score += 5
    elif prefs.hybrid_allowed and (job.get("hybrid") or "híbrido" in job_text or "hibrido" in job_text):
        score += 3

    if prefs.min_salary > 0 and job_salary > 0:
        if job_salary >= prefs.min_salary:
            score += 5
        elif job_salary >= prefs.min_salary * 0.7:
            score += 2

    if prefs.seniority:
        sen_lower = prefs.seniority.lower()
        if sen_lower in job_seniority:
            score += 4

    if prefs.exclude_keywords:
        for kw in prefs.exclude_keywords:
            if kw.lower() in job_text:
                score -= 20

    if prefs.exclude_sectors:
        for sector in prefs.exclude_sectors:
            if sector.lower() in job_text:
                score -= 15

    salary_keywords = {
        "20k": 20000, "25k": 25000, "30k": 30000, "35k": 35000,
        "40k": 40000, "45k": 45000, "50k": 50000, "55k": 55000,
        "60k": 60000, "65k": 65000, "70k": 70000, "80k": 80000,
    }
    for kw, val in salary_keywords.items():
        if kw in job_text and val >= prefs.min_salary:
            score += 2
            break

    return score


async def score_all_unscored_jobs() -> int:
    prefs = load_preferences()
    cv_profile = load_cv_profile()

    jobs = get_unscored_jobs(limit=200)
    if not jobs:
        logger.info("No new jobs to score")
        return 0

    logger.info(f"Scoring {len(jobs)} new jobs...")

    if cv_profile:
        cv_text = cv_profile.to_text_block()
    else:
        cv_text = " ".join(prefs.tech_stack) if prefs.tech_stack else "technologies"

    job_texts = []
    for job in jobs:
        jt = job.get("title", "") + " | " + job.get("company", "") + " | "
        jt += job.get("description", "")[:1000] + " | " + (job.get("technologies_detected") or "")
        job_texts.append(jt)

    try:
        cv_emb = _encode([cv_text])[0]
        job_embs = _encode(job_texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}, using rule-based scoring only")
        cv_emb = np.zeros(384)
        job_embs = np.zeros((len(jobs), 384))

    scored_count = 0
    for i, job in enumerate(jobs):
        try:
            similarity = _compute_similarity(cv_emb, job_embs[i]) if job_embs[i].any() else 0.0

            pref_score = _score_from_preferences(job, prefs)

            final_score = (similarity * 60.0) + pref_score
            final_score = max(0.0, min(100.0, final_score))

            update_job_score(
                external_id=job["external_id"],
                source=job["source"],
                score=final_score,
                match_reason=f"embedding_similarity={similarity:.3f}, pref_score={pref_score:.1f}",
            )
            scored_count += 1
        except Exception as e:
            logger.debug(f"Failed to score job {job.get('external_id')}: {e}")

        if scored_count % 50 == 0:
            await asyncio.sleep(0.01)

    logger.info(f"Scored {scored_count}/{len(jobs)} jobs")
    return scored_count
