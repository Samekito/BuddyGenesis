"""One-line JSON logs, so the host's log search can filter by level, event and chat session.

Consumed by app.py at import. Messages must never carry secrets or user text (CLAUDE.md code style 7);
the `event` extra is the stable name to search and alert on.
"""

import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone

# Fields passed via `extra=` that are copied into the JSON line.
EXTRA_FIELDS = ("event", "path")


class JsonFormatter(logging.Formatter):
    def __init__(self, correlation_id: Callable[[], str | None] = lambda: None):
        super().__init__()
        self._correlation_id = correlation_id

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        line.update({field: getattr(record, field) for field in EXTRA_FIELDS if hasattr(record, field)})
        session = self._correlation_id()
        if session:
            line["session"] = session
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure_logging(correlation_id: Callable[[], str | None] = lambda: None) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(correlation_id))
    root = logging.getLogger()
    # Replaced, not appended: `chainlit run -w` re-imports app.py on every save.
    root.handlers = [handler]
    root.setLevel(logging.INFO)
