"""SOC Buddy — Chainlit entry point. Run with `chainlit run app.py`.

Presentation layer only: wires Chainlit events to the buddy/ package (CLAUDE.md architecture rule 1).
"""

import asyncio
import functools
import logging
import re
from collections.abc import Coroutine
from pathlib import Path
from typing import Optional

import chainlit as cl
import chainlit.server
import groq
from chainlit.context import ChainlitContextException, context as chainlit_context
from chainlit.server import app as server
from chainlit.types import ThreadDict
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from buddy.accounts import (
    RESET_REQUESTED_MESSAGE,
    SIGNUP_CHECK_EMAIL_MESSAGE,
    AccountStore,
    AccountsUnavailable,
    Outcome,
    authenticate,
    confirm_signup,
    reset_password,
    reset_request_problem,
    start_password_reset,
    start_signup,
)
from buddy.assistant import (
    FAILED_REPLY_FLAG,
    AnswerFailed,
    build_messages,
    history_from_steps,
    standalone_question,
    stream_answer,
)
from buddy.auth import (
    MAX_EMAIL_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PASSWORD_LENGTH,
    google_user_identifier,
    normalize_email,
    signup_problem,
)
from buddy.config import load_settings
from buddy.history import create_data_layer, database_reachable, migrate
from buddy.http_guard import HttpGuard, refuse_sign_in, sign_in_client_ip
from buddy.knowledge import KnowledgeBase
from buddy.logging_setup import configure_logging
from buddy.mailer import BrevoMailer, Mailer, OutboxMailer, ThrottledMailer
from buddy.rate_limit import RateLimiter, allow
from buddy.theme import early_theme_script, inject_into_head, load_theme_variables

# Questions longer than this are almost always pasted documents; they burn the free token quota.
MAX_QUESTION_CHARS = 2000
# Deliberately vague: which setting is missing is for the logs, not for users.
MISSING_KEY_REPLY = "I'm not available right now. Please tell the administrator."
QUESTIONS_PER_MINUTE_REPLY = "You're asking questions faster than I can keep up. Please wait a minute and try again."
QUESTIONS_PER_DAY_REPLY = "You've reached today's question limit. Please come back tomorrow."
TOO_MANY_SIGN_INS_REPLY = "Too many sign-in attempts for this account. Please wait 15 minutes and try again."
TOO_MANY_SIGNUPS_REPLY = "Too many sign-ups right now. Please try again later."
SIGN_IN_UNAVAILABLE_REPLY = "Sign-in is unavailable right now. Please try again later."
EMAIL_UNCONFIGURED_REPLY = "This is not available yet. Please ask the administrator."
PUBLIC_DIR = Path(__file__).parent / "public"
# Our own pages, served with the early theme snippet like Chainlit's.
PAGES = {
    "/signup": "signup.html",
    "/verify-email": "verify-email.html",
    "/forgot-password": "forgot-password.html",
    "/reset-password": "reset-password.html",
}
OUTCOME_STATUS = {Outcome.DONE: 200, Outcome.INVALID: 400, Outcome.LINK_EXPIRED: 410, Outcome.UNAVAILABLE: 503}
# token_urlsafe(32) is 43 characters; anything far longer is not a token we issued.
MAX_TOKEN_LENGTH = 100
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")

# Rate limits (chat limits agreed with the owner 2026-09-30). In memory: they reset on restart.
# Per-IP limits are loose because a whole campus lab can share one address (NAT); the
# per-account limit and the hashing cap in buddy/accounts.py do the precise protecting.
SIGN_INS_PER_IP = RateLimiter(limit=30, window_seconds=60)
# Only failed sign-ins count, so the real owner is never locked out by their own successes. The
# tight limit is per account *and* address, so a stranger cannot lock someone out cheaply; the
# loose per-account one still stops a guesser who spreads attempts over many (forged) addresses.
FAILED_SIGN_INS_PER_ACCOUNT_AND_IP = RateLimiter(limit=10, window_seconds=15 * 60)
FAILED_SIGN_INS_PER_ACCOUNT = RateLimiter(limit=50, window_seconds=15 * 60)
SIGNUPS_PER_IP = RateLimiter(limit=30, window_seconds=60 * 60)
# Backstop if X-Forwarded-For is forged: bounds how fast anyone can fill the free database.
# Counts accepted sign-ups only, so invalid submissions cannot use it up.
SIGNUPS_TOTAL = RateLimiter(limit=300, window_seconds=60 * 60)
SIGNUPS_TOTAL_KEY = "all"
RESET_REQUESTS_PER_IP = RateLimiter(limit=10, window_seconds=60 * 60)
LINK_USES_PER_IP = RateLimiter(limit=30, window_seconds=60 * 60)
# The sign-up and reset forms accept any address, so cap how often one inbox can be emailed, and
# keep the daily total under Brevo's free 300 so a flood cannot use up the day's real emails.
# In memory like the others: a restart (e.g. Render's free instance waking up) resets them.
EMAILS_PER_RECIPIENT = RateLimiter(limit=3, window_seconds=60 * 60)
EMAILS_PER_DAY = RateLimiter(limit=250, window_seconds=24 * 60 * 60)
QUESTIONS_PER_MINUTE = RateLimiter(limit=10, window_seconds=60)
QUESTIONS_PER_DAY = RateLimiter(limit=150, window_seconds=24 * 60 * 60)


def _chat_session_id() -> str | None:
    try:
        return chainlit_context.session.id
    except ChainlitContextException:
        # Startup and plain HTTP routes run outside any chat, so there is no session to name.
        return None


configure_logging(_chat_session_id)
logger = logging.getLogger("soc_buddy")

settings = load_settings()
knowledge_base = KnowledgeBase.from_directory(settings.knowledge_dir)
logger.info("Loaded %d handbook chunks", len(knowledge_base.chunks))
groq_client = groq.AsyncGroq(api_key=settings.groq_api_key) if settings.groq_api_key else None
if groq_client is None:
    logger.error("GROQ_API_KEY is not set; every question gets the not-configured reply", extra={"event": "groq_unconfigured"})
history_store = create_data_layer(settings.database_url)
account_store = AccountStore(history_store.engine)
EARLY_THEME_SCRIPT = early_theme_script(load_theme_variables(PUBLIC_DIR / "theme.json"))


def _build_mailer() -> Mailer | None:
    if settings.brevo_api_key:
        sender = BrevoMailer(settings.brevo_api_key, settings.email_from, settings.email_from_name)
    elif settings.email_outbox_dir is not None:
        sender = OutboxMailer(settings.email_outbox_dir)
    else:
        logger.error("BREVO_API_KEY is not set; sign-up and password reset are switched off", extra={"event": "email_unconfigured"})
        return None
    return ThrottledMailer(sender, EMAILS_PER_RECIPIENT, EMAILS_PER_DAY)


mailer = _build_mailer()
# Strong references: the event loop keeps only weak ones, and a collected task never finishes.
_background_tasks: set[asyncio.Task] = set()


def _run_in_background(work: Coroutine) -> None:
    # Sign-up and reset requests are answered before any lookup, hashing or email happens, so
    # neither the reply nor its timing can show whether an address has an account.
    task = asyncio.create_task(work)
    _background_tasks.add(task)
    task.add_done_callback(_finish_background_task)


def _finish_background_task(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    # The flows log the failures they expect; anything else would otherwise vanish with the task.
    if not task.cancelled() and task.exception() is not None:
        logger.error("Background task failed", exc_info=task.exception(), extra={"event": "background_task_failed"})


@cl.on_app_startup
async def prepare_database():
    try:
        applied = await migrate(history_store.engine)
    except Exception:
        # Chainlit only logs startup errors and keeps serving; without this the app would run
        # with every history write silently failing. No URL in the message — it holds the password.
        logger.exception("CHAT HISTORY DISABLED: could not reach or migrate the DATABASE_URL database",
                         extra={"event": "database_migration_failed"})
        return
    if applied:
        logger.info("Applied database migrations %s", applied)


@cl.on_app_shutdown
async def close_mailer():
    if mailer is not None:
        await mailer.close()


@cl.data_layer
def get_data_layer():
    return history_store


@cl.password_auth_callback
async def password_login(username: str, password: str) -> Optional[cl.User]:
    account = normalize_email(username)
    failure_limits = [
        (FAILED_SIGN_INS_PER_ACCOUNT_AND_IP, f"{account} {sign_in_client_ip()}"),
        (FAILED_SIGN_INS_PER_ACCOUNT, account),
    ]
    if any(limiter.is_limited(key) for limiter, key in failure_limits):
        logger.warning("Rate limit hit", extra={"event": "rate_limited", "path": "/login"})
        refuse_sign_in(429, TOO_MANY_SIGN_INS_REPLY)
        return None
    try:
        user = await authenticate(account_store, username, password, settings)
    except AccountsUnavailable:
        refuse_sign_in(503, SIGN_IN_UNAVAILABLE_REPLY)
        return None
    if user is None:
        for limiter, key in failure_limits:
            limiter.record(key)
        return None
    metadata = {"provider": "credentials", "name": user.name}
    if user.is_admin:
        metadata["role"] = "admin"
    return cl.User(identifier=user.identifier, display_name=user.name, metadata=metadata)


# Generous caps stop oversized bodies early; auth.py's checks give the friendly messages.
class SignupRequest(BaseModel):
    name: str = Field(max_length=MAX_NAME_LENGTH * 2)
    email: str = Field(max_length=MAX_EMAIL_LENGTH * 2)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH * 2)


class EmailLinkRequest(BaseModel):
    token: str = Field(max_length=MAX_TOKEN_LENGTH)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH * 2)


class ResetRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH * 2)


class NewPasswordRequest(BaseModel):
    token: str = Field(max_length=MAX_TOKEN_LENGTH)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH * 2)


@server.post("/auth/signup")
async def signup(details: SignupRequest) -> JSONResponse:
    if mailer is None:
        return JSONResponse({"detail": EMAIL_UNCONFIGURED_REPLY}, status_code=503)
    problem = signup_problem(details.name, details.email, details.password)
    if problem:
        return JSONResponse({"detail": problem}, status_code=400)
    if not allow(SIGNUPS_TOTAL_KEY, [SIGNUPS_TOTAL]):
        logger.warning("Rate limit hit", extra={"event": "rate_limited", "path": "/auth/signup"})
        return JSONResponse({"detail": TOO_MANY_SIGNUPS_REPLY}, status_code=429)
    _run_in_background(start_signup(account_store, mailer, settings.app_url, details.name, details.email, details.password))
    return JSONResponse({"detail": SIGNUP_CHECK_EMAIL_MESSAGE}, status_code=202)


@server.post("/auth/verify-email")
async def verify_email(link: EmailLinkRequest) -> JSONResponse:
    outcome, message = await confirm_signup(account_store, link.token, link.password)
    return JSONResponse({"detail": message}, status_code=OUTCOME_STATUS[outcome])


@server.post("/auth/password-reset")
async def request_password_reset(details: ResetRequest) -> JSONResponse:
    if mailer is None:
        return JSONResponse({"detail": EMAIL_UNCONFIGURED_REPLY}, status_code=503)
    problem = reset_request_problem(details.email)
    if problem:
        return JSONResponse({"detail": problem}, status_code=400)
    _run_in_background(start_password_reset(account_store, mailer, settings.app_url, details.email))
    return JSONResponse({"detail": RESET_REQUESTED_MESSAGE}, status_code=202)


@server.post("/auth/password-reset/confirm")
async def confirm_password_reset(details: NewPasswordRequest) -> JSONResponse:
    outcome, message = await reset_password(account_store, details.token, details.password)
    return JSONResponse({"detail": message}, status_code=OUTCOME_STATUS[outcome])


def _serve_page(file_name: str):
    async def page() -> HTMLResponse:
        return HTMLResponse(inject_into_head((PUBLIC_DIR / file_name).read_text(encoding="utf-8"), EARLY_THEME_SCRIPT))

    return page


for _path, _file_name in PAGES.items():
    server.add_api_route(_path, _serve_page(_file_name), methods=["GET"], include_in_schema=False)


@server.get("/ready", include_in_schema=False)
async def readiness() -> JSONResponse:
    """For an uptime monitor. Render's own health check uses /health.

    What is broken goes to the logs (event "not_ready"), not to whoever asks.
    """
    failing = [
        name
        for name, ok in (("database", await database_reachable(history_store.engine)), ("groq_key", groq_client is not None))
        if not ok
    ]
    if failing:
        logger.error("Not ready: %s", ", ".join(failing), extra={"event": "not_ready"})
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ready"})


def _move_ahead_of_chainlit(path: str) -> None:
    # Chainlit registered a catch-all GET route that serves its own app for every path, so ours
    # must come first. `chainlit run -w` re-imports this file on each save, leaving stale copies:
    # keep only the newest.
    routes = server.router.routes
    ours = [route for route in routes if getattr(route, "path", None) == path]
    for route in ours:
        routes.remove(route)
    routes.insert(0, ours[-1])


for _path in (*PAGES, "/auth/signup", "/auth/verify-email", "/auth/password-reset", "/auth/password-reset/confirm", "/ready"):
    _move_ahead_of_chainlit(_path)

# `chainlit run -w` re-imports this file into the running server, where adding middleware raises.
if not any(middleware.cls is HttpGuard for middleware in server.user_middleware):
    server.add_middleware(
        HttpGuard,
        trusted_origins=[settings.public_url],
        ip_limits={
            "/login": [SIGN_INS_PER_IP],
            "/auth/signup": [SIGNUPS_PER_IP],
            "/auth/password-reset": [RESET_REQUESTS_PER_IP],
            "/auth/verify-email": [LINK_USES_PER_IP],
            "/auth/password-reset/confirm": [LINK_USES_PER_IP],
        },
    )

# Chainlit has no hook for the top of <head>, so wrap the function its catch-all route calls to
# build every page. Unwrap first: `chainlit run -w` re-imports this file on each save.
_chainlit_page_html = getattr(chainlit.server.get_html_template, "__wrapped__", chainlit.server.get_html_template)


@functools.wraps(_chainlit_page_html)
def _page_html_with_early_theme(root_path: str) -> str:
    return inject_into_head(_chainlit_page_html(root_path), EARLY_THEME_SCRIPT)


chainlit.server.get_html_template = _page_html_with_early_theme


# Chainlit refuses to start if an oauth callback exists without a configured provider.
if settings.google_oauth_enabled:

    @cl.oauth_callback
    def google_login(
        provider_id: str, token: str, raw_user_data: dict, default_user: cl.User, id_token: Optional[str] = None
    ) -> Optional[cl.User]:
        identifier = google_user_identifier(raw_user_data)
        if identifier is None:
            return None
        default_user.identifier = identifier
        default_user.metadata = {**default_user.metadata, "name": raw_user_data.get("name") or identifier}
        return default_user


def greeting(name: str) -> str:
    # Names are typed by users at sign-up; escaped so one cannot smuggle links or formatting in.
    plain_name = MARKDOWN_SPECIAL.sub(r"\\\1", name)
    return (
        f"Hello {plain_name}.\n\n"
        "I am SOC Buddy, an AI Virtual Assistant designated to the School of Computing, FUTA.\n"
        "How may I be of service to you today?"
    )


@cl.on_chat_start
async def greet():
    user = cl.user_session.get("user")
    await cl.Message(greeting(user.metadata.get("name", user.identifier) if user else "there")).send()


@cl.on_chat_resume
async def resume_chat(thread: ThreadDict):
    """Defining this handler is what makes Chainlit offer resuming. History is rebuilt per message."""


@cl.on_message
async def answer(message: cl.Message):
    if groq_client is None:
        await cl.Message(MISSING_KEY_REPLY).send()
        return
    question = message.content.strip()
    if len(question) > MAX_QUESTION_CHARS:
        await cl.Message(f"That message is too long — please keep questions under {MAX_QUESTION_CHARS} characters.").send()
        return
    limit_reply = _question_limit_reply()
    if limit_reply:
        await cl.Message(limit_reply).send()
        return

    history = _history_before(message)
    search_query = await standalone_question(groq_client, settings.groq_model, question, history)
    reference = knowledge_base.retrieve(search_query)
    course_facts = knowledge_base.course_facts(search_query)
    messages = build_messages(question, reference, history, course_facts)

    reply = cl.Message(content="")
    try:
        async for token in stream_answer(groq_client, settings.groq_model, messages):
            await reply.stream_token(token)
    except AnswerFailed as failure:
        reply.metadata = {**(reply.metadata or {}), FAILED_REPLY_FLAG: True}
        await reply.stream_token(failure.reply)
    await reply.send()


def _question_limit_reply() -> str | None:
    user = cl.user_session.get("user")
    key = user.identifier if user else "anonymous"
    if QUESTIONS_PER_DAY.is_limited(key):
        reply = QUESTIONS_PER_DAY_REPLY
    elif not allow(key, [QUESTIONS_PER_MINUTE, QUESTIONS_PER_DAY]):
        reply = QUESTIONS_PER_MINUTE_REPLY
    else:
        return None
    logger.warning("Question limit hit", extra={"event": "rate_limited", "path": "chat"})
    return reply


def _history_before(message: cl.Message) -> list[dict]:
    # Read from Chainlit's chat context, not kept separately: it already drops the messages after
    # an edited one, holds a resumed thread's messages, and cannot drift from what the user sees.
    earlier = []
    for chat_message in cl.chat_context.get():
        if chat_message.id == message.id:
            break
        earlier.append(chat_message.to_dict())
    return history_from_steps(earlier)
