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
   (e.g. `https://soc-buddy.onrender.com`), the Brevo values below and, for Google login, the OAuth
   client ID/secret.
4. In Google Cloud Console add `<CHAINLIT_URL>/auth/oauth/google/callback` as an authorised redirect URI.

### Account emails (Brevo)

Sign-up confirmation and password reset are emailed through Brevo's free plan (300 emails a day, no
card). Render's free tier blocks outgoing SMTP, so the app uses Brevo's HTTPS API instead.

1. Create a free account at https://www.brevo.com (menu names below may differ slightly in its dashboard).
2. **Senders, domains & dedicated IPs > Senders**: add the address emails should come from and confirm
   the code Brevo emails to it.
3. **SMTP & API > API keys**: create a key.
4. In Render, set `BREVO_API_KEY` to the key and `EMAIL_FROM` to the sender address.

A free address (e.g. @gmail.com) cannot be authenticated, so Brevo rewrites the visible "From"
address to one of its own (the "SOC Buddy" name stays) and more emails may land in spam. Sending
from a domain you own and have authenticated in Brevo fixes both; only `EMAIL_FROM` changes.
Without `BREVO_API_KEY`, sign-up and password reset answer "not available yet"; existing accounts
still sign in. Locally, set `EMAIL_OUTBOX_DIR=data/outbox` to have emails written there as files.

The free instance sleeps after 15 idle minutes; the first request after that takes about a minute.

The app must be served at the domain root: the sign-up page and `public/custom.js` use root-relative
paths, so `CHAINLIT_ROOT_PATH` is not supported.

## Updating the knowledge

Drop a `.docx` handbook into `knowledge_base/` and restart. See `.claude/skills/knowledge-base/SKILL.md`.

## Changing the database schema

Add a numbered file to both `buddy/schema/sqlite/` and `buddy/schema/postgres/`, e.g. `002_add_x.sql`.
Pending migrations run once at startup and are recorded in the `schema_migrations` table. Never edit a
migration that has already run anywhere; write a new one. Write each so a rerun is harmless
(`IF NOT EXISTS`), and never put a `;` inside a string literal (files are split on `;`).
`tests/test_history.py` fails if Chainlit starts writing a column the schema lacks.

## Operations

**Health.** `/health` answers whenever the process is up; Render's health check uses it. `/ready` also
checks the database and that `GROQ_API_KEY` is set, answering 503 otherwise and logging which check
failed (`not_ready` event).

**Uptime monitoring (UptimeRobot, free).** Create two HTTP(s) monitors at https://uptimerobot.com
(check that its current free-plan terms still fit this non-commercial use):

| Monitor | URL | Interval | Why |
|---|---|---|---|
| SOC Buddy awake | `<CHAINLIT_URL>/health` | 10 minutes | Keeps the free Render instance from sleeping, so nobody waits ~1 minute for a cold start. Does not touch the database. |
| SOC Buddy ready | `<CHAINLIT_URL>/ready` | 30 minutes | Alerts you (email) when the database or Groq key is broken. |

Cost check: staying awake uses about 720–744 of Render's 750 free instance hours a month, which only
fits if SOC Buddy is your only free Render service. The 30-minute database check keeps Neon well
inside its free 100 compute-hours a month; checking every 5 minutes would keep Neon awake permanently
and could exhaust them.

**Logs.** One JSON object per line in the Render dashboard's Logs tab, with `level`, `event` and,
inside a chat, `session`. Message text and passwords are never logged. Events worth alerting on:

| `event` | Meaning | Response |
|---|---|---|
| `database_migration_failed` | Startup could not reach or migrate the database; history and sign-in are down | Check `DATABASE_URL` and the Neon console, then redeploy |
| `not_ready` / `database_unreachable` | A `/ready` check failed; the log names which | Same as above if it repeats |
| `email_failed` | An account email could not be handed to Brevo (bad key, quota, outage) | Check the Brevo dashboard; `email_unconfigured` at startup means no key is set |
| `email_throttled` | One address was sent too many account emails in an hour and further ones were skipped | Usually harmless; many of these suggest someone is abusing the forms |
| `hashing_busy` | Sign-ins/sign-ups refused because too many were queued (likely a flood) | Look for the source in `rate_limited` events |
| `groq_unconfigured` | No `GROQ_API_KEY`; every question gets a "not available" reply | Set the key in Render |
| `groq_error` (many) | Groq calls failing (rate limit, bad key, outage) | Check the Groq console for quota or key problems |
| `rate_limited` (many) | Sign-in, sign-up or chat limits being hit | Look for one address or account hammering the app |

**Rate limits** (in memory, reset on restart): sign-in 30 per minute per IP; failed sign-ins 10 per
15 minutes per account and address, and 50 per 15 minutes per account; sign-up 30 per hour per IP and
300 accepted per hour in total; chat 10 questions per minute and 150 per
day per user. Per-IP limits are loose because a campus lab may share one address. They live in `app.py`. The per-IP limits read the first `X-Forwarded-For` address set by Render's proxy. After
deploying, confirm it is the real client address, e.g. by triggering a `rate_limited` log from your own
connection.

**Backups.** Neon keeps a point-in-time restore window; check its length for your plan in the Neon
console. Practise a restore at least once: create a branch from a past point in time, point a local
`.env` `DATABASE_URL` at that branch, run the app and confirm old chats appear, then delete the branch.

**Staging.** For risky changes, deploy the branch as a second free Render service pointed at a Neon
branch of the database, never at the production database. Free Render instance hours are shared across
services, so delete the staging service afterwards.

**Secrets.** Use different `CHAINLIT_AUTH_SECRET` and database credentials for local, staging and
production. Rotating `CHAINLIT_AUTH_SECRET` signs everyone out.
