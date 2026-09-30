from types import SimpleNamespace

import groq
import httpx

import pytest

from buddy.assistant import (
    AnswerFailed,
    FALLBACK_REPLIES,
    GENERIC_FAILURE_REPLY,
    HISTORY_MESSAGES,
    build_messages,
    history_from_steps,
    standalone_question,
    stream_answer,
)
from buddy.knowledge import Chunk


def _token_chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class FakeGroq:
    """Stands in for groq.AsyncGroq at the service boundary — tests never hit the network."""

    def __init__(self, tokens=None, reply=None, error=None):
        self.requests = []

        async def create(**kwargs):
            self.requests.append(kwargs)
            if error:
                raise error
            if not kwargs.get("stream"):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

            async def stream():
                for token in tokens:
                    yield _token_chunk(token)
            return stream()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _status_error(error_type, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    return error_type("failed", response=response, body=None)


async def _collect(client, messages=None):
    return "".join([t async for t in stream_answer(client, "test-model", messages or [])])


async def _failure_reply(client):
    with pytest.raises(AnswerFailed) as failure:
        await _collect(client)
    return failure.value.reply


def test_build_messages_puts_reference_in_system_prompt_and_question_last():
    messages = build_messages("How many units?", [Chunk("CS", "CSC 101 is 2 units")], [])

    assert messages[0]["role"] == "system"
    assert "CSC 101 is 2 units" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "How many units?"}


def test_build_messages_keeps_only_recent_history():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]

    messages = build_messages("latest", [], history)

    assert messages[1:-1] == history[-HISTORY_MESSAGES:]


def test_build_messages_includes_course_index_facts():
    facts = ["CSC 101 appears in these programmes (complete list): A; B."]

    system = build_messages("which departments offer CSC 101?", [], [], facts)[0]["content"]

    assert facts[0] in system


def test_build_messages_marks_missing_course_index():
    assert "(no course codes in this question)" in build_messages("hi", [], [])[0]["content"]


def test_build_messages_marks_empty_reference():
    assert "(no matching excerpts)" in build_messages("hi", [], [])[0]["content"]


async def test_stream_answer_yields_tokens_and_skips_empty_deltas():
    client = FakeGroq(tokens=["Hello", None, " there"])

    assert await _collect(client) == "Hello there"
    assert client.requests[0]["stream"] is True
    assert client.requests[0]["model"] == "test-model"


async def test_stream_answer_turns_rate_limit_into_friendly_reply():
    client = FakeGroq(error=_status_error(groq.RateLimitError, 429))

    assert await _failure_reply(client) == FALLBACK_REPLIES[groq.RateLimitError]


async def test_stream_answer_turns_bad_key_into_friendly_reply():
    client = FakeGroq(error=_status_error(groq.AuthenticationError, 401))

    assert await _failure_reply(client) == FALLBACK_REPLIES[groq.AuthenticationError]


async def test_stream_answer_turns_network_failure_into_friendly_reply():
    client = FakeGroq(error=groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")))

    assert await _failure_reply(client) == FALLBACK_REPLIES[groq.APIConnectionError]


async def test_stream_answer_has_generic_reply_for_other_api_errors():
    client = FakeGroq(error=_status_error(groq.InternalServerError, 500))

    assert await _failure_reply(client) == GENERIC_FAILURE_REPLY


FOLLOW_UP_HISTORY = [
    {"role": "user", "content": "How many units is CSC 101 and what is it about?"},
    {"role": "assistant", "content": "CSC 101 is 2 units."},
]
FOLLOW_UP = "Which department offers it, and in which semester is it taken?"


async def test_first_message_is_searched_as_is_without_an_api_call():
    client = FakeGroq(reply="should not be used")

    assert await standalone_question(client, "m", "How many units is CSC 101?", []) == "How many units is CSC 101?"
    assert client.requests == []


async def test_follow_up_is_rewritten_with_conversation_context():
    client = FakeGroq(reply="  In which semester is CSC 101 taken?  ")

    result = await standalone_question(client, "m", FOLLOW_UP, FOLLOW_UP_HISTORY)

    assert result == "In which semester is CSC 101 taken?"
    sent = client.requests[0]["messages"][-1]["content"]
    assert "CSC 101" in sent and FOLLOW_UP in sent


async def test_follow_up_falls_back_to_raw_question_on_api_error():
    client = FakeGroq(error=_status_error(groq.RateLimitError, 429))

    assert await standalone_question(client, "m", FOLLOW_UP, FOLLOW_UP_HISTORY) == FOLLOW_UP


async def test_follow_up_falls_back_to_raw_question_on_empty_rewrite():
    client = FakeGroq(reply="")

    assert await standalone_question(client, "m", FOLLOW_UP, FOLLOW_UP_HISTORY) == FOLLOW_UP



def _step(step_type, output):
    return {"type": step_type, "output": output}


def test_history_from_steps_drops_greeting_and_keeps_order():
    steps = [
        _step("assistant_message", "Hello admin."),
        _step("user_message", "How many units is CSC 101?"),
        _step("assistant_message", "2 units."),
    ]

    assert history_from_steps(steps) == [
        {"role": "user", "content": "How many units is CSC 101?"},
        {"role": "assistant", "content": "2 units."},
    ]


def test_history_from_steps_skips_empty_outputs_and_other_step_types():
    steps = [_step("user_message", "hi"), _step("run", "internal"), _step("assistant_message", ""), _step("assistant_message", "hello")]

    assert history_from_steps(steps) == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_history_from_steps_keeps_only_recent_messages():
    steps = [_step("user_message" if i % 2 == 0 else "assistant_message", f"m{i}") for i in range(10)]

    assert [m["content"] for m in history_from_steps(steps)] == ["m6", "m7", "m8", "m9"]


def test_history_from_steps_handles_thread_without_messages():
    assert history_from_steps([]) == []
