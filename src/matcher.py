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

CIUDADES_ESPANA = {
    "madrid", "barcelona", "valencia", "sevilla", "bilbao", "málaga", "malaga",
    "zaragoza", "murcia", "palma", "las palmas", "granada", "alicante",
    "valladolid", "coruña", "coruna", "vigo", "gijón", "gijon", "oviedo",
    "santander", "pamplona", "san sebastián", "san sebastian", "donostia",
    "cádiz", "cadiz", "tenerife", "salamanca", "toledo", "córdoba", "cordoba",
    "almería", "almeria", "tarragona", "lleida", "castellón", "castellon",
    "badajoz", "cáceres", "caceres", "logroño", "logrono", "guadalajara",
    "huelva", "jaén", "jaen", "lugo", "orense", "pontevedra", "segovia",
    "soria", "zamora", "ávila", "avila", "burgos", "león", "leon", "palencia",
    "cuenca", "ciudad real", "albacete", "teruel", "huesca",
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


def _detect_modality_from_text(text: str) -> dict:
    text_lower = text.lower()
    is_remote = any(kw in text_lower for kw in [
        "remoto", "remote", "teletrabajo", "100% remoto", "en remoto",
        "trabajo desde casa", "work from home", "full remote",
    ])
    is_hybrid = any(kw in text_lower for kw in [
        "híbrido", "hibrido", "hybrid", "semanal", "2-3 días", "2 días",
        "modelo híbrido", "trabajo híbrido",
    ])
    is_onsite = any(kw in text_lower for kw in [
        "presencial", "on-site", "onsite", "en oficina", "trabajo presencial",
        "jornada presencial",
    ]) and not is_remote and not is_hybrid

    return {"remote": is_remote, "hybrid": is_hybrid, "onsite": is_onsite}


def _detect_city_from_location(location: str, description: str) -> list[str]:
    combined = (location + " " + description[:500]).lower()
    found = set()
    for city in CIUDADES_ESPANA:
        if city in combined:
            found.add(city)
    return list(found)


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
    score += matched_techs * 9.0

    for title_str in prefs.desired_titles:
        title_lower = title_str.lower().strip()
        if title_lower in job_title:
            score += 15.0
            break
    else:
        title_word_hits = 0
        for title_str in prefs.desired_titles:
            for word in title_str.lower().split():
                if len(word) > 3 and word in job_title:
                    title_word_hits += 1
        score += title_word_hits * 7.0

    modality = _detect_modality_from_text(f"{job_title} {job_desc} {job_location}")

    is_remote = job.get("remote") or modality["remote"]
    is_hybrid = job.get("hybrid") or modality["hybrid"]
    is_onsite = (job.get("onsite") and not is_remote and not is_hybrid) or modality["onsite"]

    if is_remote:
        score += 15.0
    elif is_hybrid:
        if prefs.remote_only:
            score -= 35.0
        elif not prefs.hybrid_allowed:
            score -= 25.0
        else:
            score += 10.0
    elif is_onsite:
        if prefs.remote_only:
            score -= 50.0
        elif not prefs.onsite_allowed:
            score -= 40.0

    cities_found = _detect_city_from_location(job_location, job_desc)
    location_ok = prefs.location.lower() in job_location
    location_remoto = is_remote or "remoto" in job_location or "españa" in job_location or "espana" in job_location

    if prefs.location:
        if location_ok:
            score += 12.0
        elif location_remoto and is_remote:
            score += 8.0
        elif location_remoto:
            score += 5.0
        elif cities_found and "madrid" not in [c.lower() for c in cities_found]:
            score -= 30.0

    if prefs.min_salary > 0 and job_salary > 0:
        if job_salary >= prefs.min_salary:
            score += 10.0
        elif job_salary >= prefs.min_salary * 0.8:
            score += 5.0

    if prefs.seniority:
        sen_lower = prefs.seniority.lower()
        if sen_lower in job_seniority:
            score += 10.0

    for kw in prefs.exclude_keywords:
        if kw.lower() in job_text:
            score -= 35.0

    for sector in prefs.exclude_sectors:
        if sector.lower() in job_text or sector.lower() in job_company:
            score -= 30.0

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
            if final_score < 0:
                final_score = 0.0
            elif final_score > 100:
                final_score = 100.0

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
