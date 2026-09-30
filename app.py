"""SOC Buddy — Chainlit entry point. Run with `chainlit run app.py`.

Presentation layer only: wires Chainlit events to the buddy/ package (CLAUDE.md architecture rule 1).
"""

import logging
from typing import Optional

import chainlit as cl
import groq
from chainlit.types import ThreadDict

from buddy.assistant import (
    HISTORY_MESSAGES,
    AnswerFailed,
    build_messages,
    history_from_steps,
    standalone_question,
    stream_answer,
)
from buddy.auth import google_user_identifier, is_valid_password_login
from buddy.config import load_settings
from buddy.history import create_data_layer, ensure_schema
from buddy.knowledge import KnowledgeBase

# Questions longer than this are almost always pasted documents; they burn the free token quota.
MAX_QUESTION_CHARS = 2000
MISSING_KEY_REPLY = "I'm not configured yet (GROQ_API_KEY is missing). Please tell the administrator."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soc_buddy")

settings = load_settings()
knowledge_base = KnowledgeBase.from_directory(settings.knowledge_dir)
logger.info("Loaded %d handbook chunks", len(knowledge_base.chunks))
groq_client = groq.AsyncGroq(api_key=settings.groq_api_key) if settings.groq_api_key else None


@cl.on_app_startup
async def prepare_history_tables():
    try:
        await ensure_schema(settings.database_url)
    except Exception:
        # Chainlit only logs startup errors and keeps serving; without this the app would run
        # with every history write silently failing. No URL in the message — it holds the password.
        logger.exception("CHAT HISTORY DISABLED: could not reach or prepare the DATABASE_URL database")


@cl.data_layer
def get_data_layer():
    return create_data_layer(settings.database_url)


@cl.password_auth_callback
def password_login(username: str, password: str) -> Optional[cl.User]:
    if not is_valid_password_login(username, password, settings):
        return None
    # The configured name, not what was typed: "Admin" and "admin" must share one account and history.
    admin = settings.admin_username
    return cl.User(identifier=admin, metadata={"role": "admin", "provider": "credentials", "name": admin})


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


@cl.on_chat_start
async def greet():
    user = cl.user_session.get("user")
    name = user.metadata.get("name", user.identifier) if user else "there"
    cl.user_session.set("history", [])
    await cl.Message(
        f"Hello {name}.\n\nI am SOC Buddy, an AI Virtual Assistant designated to the School of Computing, FUTA.\n"
        "How may I be of service to you today?"
    ).send()


@cl.on_chat_resume
async def restore_history(thread: ThreadDict):
    cl.user_session.set("history", history_from_steps(thread["steps"]))


@cl.on_message
async def answer(message: cl.Message):
    if groq_client is None:
        await cl.Message(MISSING_KEY_REPLY).send()
        return
    question = message.content.strip()
    if len(question) > MAX_QUESTION_CHARS:
        await cl.Message(f"That message is too long — please keep questions under {MAX_QUESTION_CHARS} characters.").send()
        return

    history = cl.user_session.get("history") or []
    search_query = await standalone_question(groq_client, settings.groq_model, question, history)
    reference = knowledge_base.retrieve(search_query)
    course_facts = knowledge_base.course_facts(search_query)
    messages = build_messages(question, reference, history, course_facts)

    reply = cl.Message(content="")
    try:
        async for token in stream_answer(groq_client, settings.groq_model, messages):
            await reply.stream_token(token)
    except AnswerFailed as failure:
        await reply.stream_token(failure.reply)
        await reply.send()
        return  # not added to history: the model must not see error text as its own answer
    await reply.send()

    history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply.content}]
    cl.user_session.set("history", history[-HISTORY_MESSAGES:])
