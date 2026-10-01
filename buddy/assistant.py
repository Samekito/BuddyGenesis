"""Builds SOC Buddy's prompt and streams answers from Groq.

Consumed by app.py. The only module that talks to the LLM (CLAUDE.md architecture rule 5).
"""

import json
import logging
from collections.abc import AsyncIterator

import groq

from buddy.knowledge import Chunk

logger = logging.getLogger(__name__)

# Two exchanges, as in the original app: enough for follow-up questions ("what about 200 level?")
# while keeping each request small for Groq's free-tier token-per-minute limit.
HISTORY_MESSAGES = 4

# Low but non-zero: answers must stay faithful to the handbook text, and the original used 0.
TEMPERATURE = 0.2

# Caps what one reply can spend of the shared free quota. gpt-oss counts its hidden reasoning
# against this too, so it is set well above the longest real answer (a full level's course table).
ANSWER_MAX_TOKENS = 4096
REWRITE_MAX_TOKENS = 512
# Only these models accept `reasoning_effort`; any other model rejects the whole request.
REASONING_EFFORT_MODEL_PREFIXES = ("openai/gpt-oss",)

# Set in a failed reply's metadata. The reply is still shown and saved with the thread, so this
# is what keeps its error text out of the history the model sees, now and after a resume.
FAILED_REPLY_FLAG = "failedReply"

SYSTEM_PROMPT = """You are SOC Buddy, a helpful virtual assistant for the School of Computing at the \
Federal University of Technology Akure (FUTA), Nigeria. The School has five departments: Computer \
Science, Cyber Security, Information Systems, Information Technology and Software Engineering.

Answer questions using only the information under "Reference" below. The user cannot see it, so \
never refer to it: do not use words like "reference", "excerpt", "handbook", "provided text" or \
"documents" — answer as if you simply know. If the information needed is not there, say plainly \
that you do not know and suggest the student confirm with their department. Greetings and small \
talk do not need the reference.

Each reference section starts with the department (or School) it comes from. Many courses \
(especially at 100 level) appear in several departments' programmes, so never say a course \
belongs to or is offered by only one department unless the text states that explicitly.

Lines under "Course index" are complete and exact: when asked which departments or programmes \
have a course, answer from the course index, never from the excerpts. For any other list, the \
reference holds only the most relevant parts and may be incomplete, so never present a list as \
complete: say "including" or "the ones I know of" unless the text itself says it is complete.

Be concise. Use short lists or tables when listing courses.

Course index:
{course_index}

Reference:
{reference}"""

FALLBACK_REPLIES = {
    groq.RateLimitError: "I'm getting a lot of questions right now. Please wait a minute and try again.",
    groq.AuthenticationError: "I'm not configured correctly (the AI service rejected my key). Please tell the administrator.",
    groq.APIConnectionError: "I couldn't reach my AI service. Please check back in a moment.",
}
GENERIC_FAILURE_REPLY = "Something went wrong while I was answering. Please try again."

CONDENSE_PROMPT = (
    "Rewrite the user's latest message as a standalone question that names its subject explicitly "
    "(course codes, department, level), using the conversation for context. If it is already "
    "standalone, return it unchanged. Reply with the question only."
)

# Earlier answers can be long tables; the rewrite only needs their gist.
CONDENSE_TURN_CHARS = 500


async def standalone_question(client: groq.AsyncGroq, model: str, question: str, history: list[dict]) -> str:
    """Rewrites a follow-up ("which semester is it taken?") into a self-contained search query.

    BM25 can only match words it is given, and follow-ups leave the subject ("CSC 101") in
    earlier turns. First messages skip the extra call. On any API error the raw question is used.
    """
    if not any(message["role"] == "user" for message in history):
        return question
    transcript = "\n".join(
        f"{message['role']}: {message['content'][:CONDENSE_TURN_CHARS]}" for message in history[-HISTORY_MESSAGES:]
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONDENSE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
            ],
            temperature=0,
            max_completion_tokens=REWRITE_MAX_TOKENS,
            **_low_reasoning_effort(model),
        )
    except groq.APIError as error:
        logger.warning("Follow-up rewrite failed, searching with the raw question: %s", type(error).__name__)
        return question
    return (response.choices[0].message.content or "").strip() or question


def build_messages(
    question: str, reference: list[Chunk], history: list[dict], course_facts: list[str] | None = None
) -> list[dict]:
    reference_text = "\n\n---\n\n".join(chunk.text for chunk in reference) or "(no matching excerpts)"
    course_index = "\n".join(course_facts or []) or "(no course codes in this question)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(course_index=course_index, reference=reference_text)},
        *history[-HISTORY_MESSAGES:],
        {"role": "user", "content": question},
    ]


class AnswerFailed(Exception):
    """Raised by stream_answer when Groq fails. `reply` is the friendly message to show instead.

    A distinct signal (not a yielded token) so callers can keep the error text out of the
    conversation history — otherwise the model sees "please try again" as its own past answer.
    """

    def __init__(self, reply: str):
        super().__init__(reply)
        self.reply = reply


async def stream_answer(client: groq.AsyncGroq, model: str, messages: list[dict]) -> AsyncIterator[str]:
    """Yields answer tokens. Raises AnswerFailed (with a user-friendly reply) on API failure."""
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, temperature=TEMPERATURE, max_completion_tokens=ANSWER_MAX_TOKENS, stream=True
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                yield token
    except groq.APIError as error:
        # Log the error type only; the message can echo request content.
        logger.error("Groq request failed: %s", type(error).__name__, extra={"event": "groq_error"})
        reply = next(
            (reply for error_type, reply in FALLBACK_REPLIES.items() if isinstance(error, error_type)),
            GENERIC_FAILURE_REPLY,
        )
        raise AnswerFailed(reply) from error


def history_from_steps(steps: list[dict]) -> list[dict]:
    """Rebuilds conversation history from Chainlit step dicts (oldest first).

    Assistant messages before the first user message are the greeting, which is not part of the
    conversation. Failed replies are dropped so the model never sees "please try again" as its own answer.
    """
    roles = {"user_message": "user", "assistant_message": "assistant"}
    history: list[dict] = []
    for step in steps:
        role = roles.get(step.get("type"))
        if role is None or not step.get("output") or _step_metadata(step).get(FAILED_REPLY_FLAG):
            continue
        if role == "assistant" and not history:
            continue
        history.append({"role": role, "content": step["output"]})
    return history[-HISTORY_MESSAGES:]


def _low_reasoning_effort(model: str) -> dict:
    # A rewrite needs no deliberation; low effort keeps the extra call to ~1-2 s.
    return {"reasoning_effort": "low"} if model.startswith(REASONING_EFFORT_MODEL_PREFIXES) else {}


def _step_metadata(step: dict) -> dict:
    # Live messages carry a dict; steps read back from the database carry the stored JSON text.
    metadata = step.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}
