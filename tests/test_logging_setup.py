import json
import logging

from buddy.logging_setup import JsonFormatter


def _format(formatter, **extra):
    record = logging.LogRecord("soc_buddy", logging.INFO, __file__, 1, "Loaded %d chunks", (7,), None)
    record.__dict__.update(extra)
    return json.loads(formatter.format(record))


def test_log_line_is_json_with_level_logger_and_message():
    line = _format(JsonFormatter())

    assert (line["level"], line["logger"], line["message"]) == ("INFO", "soc_buddy", "Loaded 7 chunks")
    assert line["time"].endswith("+00:00")


def test_event_name_is_carried_for_searching_and_alerting():
    assert _format(JsonFormatter(), event="rate_limited")["event"] == "rate_limited"


def test_chat_session_is_attached_when_there_is_one():
    assert _format(JsonFormatter(lambda: "session-1"))["session"] == "session-1"


def test_no_session_field_outside_a_chat():
    assert "session" not in _format(JsonFormatter())
