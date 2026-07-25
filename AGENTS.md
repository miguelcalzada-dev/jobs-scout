# AGENTS.md

## Project overview

Jobs Scout — AI-powered daily job scraper and matcher. Scrapes **Tecnoempleo** and **InfoJobs** in parallel (httpx async, ~50 fresh offers per manual/daily run), filters out offers that redirect off-platform, matches offers against the user's CV using sentence-transformers embeddings + rule-based scoring, and emails the top N every evening via the Resend HTTP API. The dashboard is a single-page UI with light/dark mode (persisted), paginated history (no charts), and Auto Apply on Tecnoempleo via Playwright.

**Stack:** Python 3.12+, FastAPI, SQLite (stdlib `sqlite3`, WAL mode), APScheduler, Jinja2, sentence-transformers, BeautifulSoup4 + lxml, httpx. Deployed on Railway ($5 plan).

## Commands

```bash
# Install deps (playwright is optional, only for Auto Apply)
pip install -r requirements.txt

# Interactive first-time setup (upload CV, configure preferences)
python3 src/setup.py

# Start the server (daily scrape at 20:00 Europe/Madrid, configurable via DAILY_SEND_HOUR)
python3 src/main.py

# Manual scrape + score + email, then exit
python3 src/main.py --run-now

# Start on a different port
python3 src/main.py --port 8081
```

## Architecture

```
src/
  main.py         FastAPI + APScheduler (entry point, all routes, lifecycle)
  config.py       Settings (pydantic-settings), CVProfile/JobPreferences dataclasses, file I/O
  setup.py        Interactive CLI wizard (CV upload + preference questions)
  cv_parser.py    PDF -> CVProfile (LLM via litellm or regex fallback)
  database.py     SQLite via stdlib sqlite3 (autocommit=WAL), all CRUD + pagination
  scraper.py      BaseScraper, JobOffer dataclass (with is_external_redirect), ScrapeResult
  scrapers/
    _common.py       Shared tech & seniority detection (DRY)
    tecnoempleo.py   httpx async, 25 offers/scrape, full descriptions, external-link detection
    infojobs.py      httpx async, 25 offers/scrape (limited — site is a React SPA)
  matcher.py      Embedding model + rule-based scoring -> 0-100%
  autoapply.py    Playwright-driven Tecnoempleo Auto Apply (login, cover letter, screening Qs)
  delivery.py     Resend HTTP API for email (NOT SMTP — Railway blocks outbound SMTP)
templates/
  dashboard.html    Full web UI (5 tabs: Ofertas, Mi CV, Preferencias, Historial, Auto Apply)
  daily_email.html  Email template
data/jobs.db     SQLite database (gitignored)
```

## Critical gotchas

### Path handling
`main.py` and `setup.py` have `sys.path.insert(0, ...)` at the top. This is REQUIRED because the app runs as `python3 src/main.py`, not as a module.

### Railway deployment
- **Start command** in `railway.toml` MUST use `--port 8080` literally. Do NOT use `${PORT:-8080}` — Nixpacks does not expand shell variables.
- **Volume** at `/app/data` is needed for SQLite persistence across deploys.
- Playwright Chromium is intentionally NOT installed on Railway ($5 plan too small). Only the Auto Apply scraper needs it; the rest works without it.
- **Python version** pinned to 3.12 in the Dockerfile. Nixpacks also uses Python 3.12.

### Fresh-batch search model
- Each manual (`POST /run`) or scheduled run calls `database.clear_all_jobs()` BEFORE scraping, so every batch is a clean ~50-offer lot (25 per source). Prior jobs, scores, and applied/discarded flags are wiped on each run by design.
- `asyncio.gather` runs both scrapers in parallel — all I/O is non-blocking httpx, so the parallelism is real (the previous `requests`-based Tecnoempleo blocked the event loop and cancelled the benefit).
- Offers whose apply link redirects off-platform are marked `is_external_redirect = 1` and excluded from scoring, dashboard, notifications, and Auto Apply.

### Preferences (fixed fields)
Six preference fields are HARD-CODED as defaults in `main.py` (`_PREFS_FIJOS`):
`seniority="junior"`, `location="Madrid"`, `remote_only=False`, `hybrid_allowed=True`, `onsite_allowed=False`, `min_salary=0`.
- The web form lets the user edit them freely (saved verbatim by `POST /preferences`).
- On **CV upload** (`_auto_update_preferences_from_cv`) those six are FORCED back to the defaults (only techs + derived titles are *added*, never removed).
- On **server restart** (`lifespan`) the saved preferences are HONORED — only the list fields (desired_titles / tech_stack / enabled_scrapers) are filled with sensible defaults when empty. Nothing is overwritten.

### SQLite quirks
- `sqlite3.connect(..., autocommit=True)` is used because Python 3.13 changed the default autocommit behavior.
- WAL mode + `synchronous=NORMAL` for concurrent read performance.
- Schema migrations are done in `database.py` via `_ensure_column()` (`PRAGMA table_info` + `ALTER TABLE ADD COLUMN`) since SQLite lacks `ADD COLUMN IF NOT EXISTS`.

### Embedding model
- Uses `sentence-transformers/all-MiniLM-L6-v2` (~90MB, cached by HuggingFace).
- Loaded lazily on first scoring call.
- If download fails, scoring falls back to rule-based only (`np.zeros(384)`).

### Scoring formula (matcher.py)
- `final_score = (cosine_similarity * 50.0) + rule_score`, capped 0-100.
- Rule weights: tech match +9 each, title match +15, remote +15, hybrid +10 (or penalties based on prefs), onsite -40/-50, wrong Spanish city -30, location match +12, salary +10/+5, seniority match +10, excluded keyword -35, excluded sector -30.

### Auto Apply
- Only runs against `source='tecnoempleo'`, `is_external_redirect = 0`, `applied = 0`, `discarded = 0`, `match_score >= min_score`.
- Skips any offer whose apply button/href redirects off Tecnoempleo.
- Considers the application "applied" only when it detects success keywords in the resulting page or a success URL (`mis-candidaturas`, `micuenta`, `confirmacion`). Otherwise it marks "skipped — review manually" and does NOT flip `applied=1` in the DB.
- `AUTOAPPLY_RESULTS` is capped to the last 100 entries (was unbounded — memory leak on long-running processes).

### Environment variables
Set these in Railway (not in `.env`):
- `RESEND_API_KEY` — Resend API key for email
- `EMAIL_TO` — recipient email
- `EMAIL_USER` — sender email (used as Resend `from` address, must match verified domain or `onboarding@resend.dev` in test mode)
- `DAILY_SEND_HOUR`, `DAILY_SEND_MINUTE` — schedule (default 09:00; setup wizard sets 20)
- `OPENAI_API_KEY` — optional, enables LLM-based CV parsing (otherwise regex)
- `TECNOEMPLEO_EMAIL`, `TECNOEMPLEO_PASSWORD` — optional, for Auto Apply

### Legacy env vars
The Settings model uses `extra="ignore"` so old `.env` files that still contain `EMAIL_HOST`/`EMAIL_PORT`/`EMAIL_PASSWORD` from the deleted SMTP path will NOT crash startup (they are simply ignored).

## File conventions
- `.env` — gitignored (use `.env.example` as template)
- `config.yaml` — user preferences (gitignored, generated by setup)
- `cv_profile.json` — parsed CV data (gitignored, generated by setup)
- `data/jobs.db` — SQLite database (gitignored)
- `cv/*.pdf` — uploaded CVs (gitignored)
- Templates use standard Jinja2 syntax. The dashboard is a single-page app with inline JS, no build step.
- Logging format: `%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`
