# SOC Buddy

AI virtual assistant for the School of Computing, Federal University of Technology Akure (FUTA).
Answers questions from the department handbooks in `knowledge_base/`.

Runs at zero cost: Groq free tier (LLM), in-process BM25 search (no vector DB), SQLite chat history locally,
Render free tier + Neon free Postgres when hosted.

## Run locally

Requires Python 3.13 (Chainlit does not yet work on 3.14 — its `nest_asyncio` patch breaks event-loop detection).

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env          # fill in GROQ_API_KEY and the admin login at minimum
.venv/Scripts/chainlit run app.py -w
```

Open http://localhost:8000. Tests: `.venv/Scripts/python -m pytest`.

## Deploy free on Render

1. Create a free Postgres database at https://neon.com and copy its connection string.
2. On https://render.com: **New > Blueprint**, pick this repo — `render.yaml` defines the service.
3. Fill in the prompted values: `GROQ_API_KEY`, admin login, `DATABASE_URL` (Neon), `CHAINLIT_URL`
   (e.g. `https://soc-buddy.onrender.com`) and, for Google login, the OAuth client ID/secret.
4. In Google Cloud Console add `<CHAINLIT_URL>/auth/oauth/google/callback` as an authorised redirect URI.

The free instance sleeps after 15 idle minutes; the first request after that takes about a minute.

## Updating the knowledge

Drop a `.docx` handbook into `knowledge_base/` and restart. See `.claude/skills/knowledge-base/SKILL.md`.
