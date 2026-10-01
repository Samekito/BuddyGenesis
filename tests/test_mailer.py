import json

import httpx
import pytest

from buddy import mailer as mailer_module
from buddy.mailer import BREVO_SEND_URL, BrevoMailer, Email, EmailFailed, OutboxMailer, ThrottledMailer
from buddy.rate_limit import RateLimiter

EMAIL = Email(to="ada@example.com", subject="Confirm", text="Hello Ada", html="<p>Hello Ada</p>")


def _brevo(statuses, requests):
    """A BrevoMailer whose HTTP client answers with `statuses` in turn (mocked at the boundary)."""
    answers = iter(statuses)

    def handle(request):
        requests.append(request)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(answer, json={"messageId": "<1@brevo>"} if answer == 201 else {"message": "no"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return BrevoMailer("key-123", "samuel@example.com", "SOC Buddy", client=client)


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(mailer_module, "RETRY_DELAY_SECONDS", 0)


async def test_brevo_request_has_the_documented_shape():
    requests = []

    await _brevo([201], requests).send(EMAIL)

    request = requests[0]
    assert (request.method, str(request.url)) == ("POST", BREVO_SEND_URL)
    assert request.headers["api-key"] == "key-123"
    assert json.loads(request.content) == {
        "sender": {"email": "samuel@example.com", "name": "SOC Buddy"},
        "to": [{"email": "ada@example.com"}],
        "subject": "Confirm",
        "textContent": "Hello Ada",
        "htmlContent": "<p>Hello Ada</p>",
    }


async def test_brevo_blip_is_retried_once():
    requests = []

    await _brevo([503, 201], requests).send(EMAIL)

    assert len(requests) == 2


async def test_brevo_network_error_is_retried_once():
    requests = []

    await _brevo([httpx.ConnectError("down"), 201], requests).send(EMAIL)

    assert len(requests) == 2


async def test_brevo_refusal_is_not_retried_and_raises():
    requests = []

    with pytest.raises(EmailFailed, match="HTTP 401"):
        await _brevo([401, 201], requests).send(EMAIL)
    assert len(requests) == 1


async def test_brevo_outage_raises_after_the_retry():
    with pytest.raises(EmailFailed, match="HTTP 503"):
        await _brevo([503, 503], []).send(EMAIL)


async def test_outbox_writes_the_email_to_a_file(tmp_path):
    await OutboxMailer(tmp_path / "outbox").send(EMAIL)

    [written] = (tmp_path / "outbox").iterdir()
    assert "To: ada@example.com" in written.read_text(encoding="utf-8") and "Hello Ada" in written.read_text(encoding="utf-8")


class Recorder:
    def __init__(self):
        self.sent = []

    async def send(self, email):
        self.sent.append(email)

    async def close(self):
        """Nothing to release."""


async def test_one_address_is_not_emailed_more_than_its_limit():
    inner = Recorder()
    throttled = ThrottledMailer(inner, RateLimiter(limit=2, window_seconds=3600), RateLimiter(limit=100, window_seconds=86400))

    for _ in range(4):
        await throttled.send(EMAIL)

    assert len(inner.sent) == 2


async def test_address_limit_ignores_letter_case():
    inner = Recorder()
    throttled = ThrottledMailer(inner, RateLimiter(limit=1, window_seconds=3600), RateLimiter(limit=100, window_seconds=86400))

    await throttled.send(EMAIL)
    await throttled.send(Email(to="ADA@EXAMPLE.COM", subject="s", text="t", html="h"))

    assert len(inner.sent) == 1


async def test_daily_total_raises_once_used_up():
    throttled = ThrottledMailer(Recorder(), RateLimiter(limit=10, window_seconds=3600), RateLimiter(limit=1, window_seconds=86400))
    await throttled.send(EMAIL)

    with pytest.raises(EmailFailed, match="daily"):
        await throttled.send(Email(to="grace@example.com", subject="s", text="t", html="h"))
