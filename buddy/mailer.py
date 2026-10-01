"""Sends SOC Buddy's account emails: sign-up confirmation, "already registered" and password reset.

Consumed by app.py, which builds one mailer at startup. Brevo's HTTP API, not SMTP: Render's free
tier blocks outbound SMTP ports. OutboxMailer writes files instead, for local development only
(buddy/config.py refuses it on Render). Recipient addresses and links are never logged.
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx

from buddy.rate_limit import RateLimiter, allow

BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"
BREVO_SENT_STATUS = 201
SEND_TIMEOUT_SECONDS = 10
# One retry, with jitter, for a blip; any other refusal is a configuration problem that a retry cannot fix.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_DELAY_SECONDS = 1.0
DAILY_TOTAL_KEY = "all"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    text: str
    html: str


class EmailFailed(Exception):
    """The email could not be handed to the provider. The message is safe to log (no address, no link)."""


class Mailer(Protocol):
    async def send(self, email: Email) -> None: ...

    async def close(self) -> None: ...


class BrevoMailer:
    def __init__(self, api_key: str, sender_email: str, sender_name: str, client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._sender = {"email": sender_email, "name": sender_name}
        self._client = client or httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS)

    async def send(self, email: Email) -> None:
        payload = {
            "sender": self._sender,
            "to": [{"email": email.to}],
            "subject": email.subject,
            "textContent": email.text,
            "htmlContent": email.html,
        }
        headers = {"api-key": self._api_key, "accept": "application/json"}
        failure = ""
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(RETRY_DELAY_SECONDS * (1 + random.random()))
            try:
                response = await self._client.post(BREVO_SEND_URL, json=payload, headers=headers)
            except httpx.TransportError as error:
                failure = type(error).__name__
                continue
            if response.status_code == BREVO_SENT_STATUS:
                return
            failure = f"Brevo answered HTTP {response.status_code}"
            if response.status_code not in RETRYABLE_STATUSES:
                break
        raise EmailFailed(failure)

    async def close(self) -> None:
        await self._client.aclose()


class OutboxMailer:
    """Writes each email to a text file, so sign-up and reset can be tried locally without a provider."""

    def __init__(self, directory: Path):
        self._directory = directory

    async def send(self, email: Email) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self._directory / f"{stamp}-{re.sub(r'[^a-z0-9]+', '_', email.to.lower())}.txt"
        path.write_text(f"To: {email.to}\nSubject: {email.subject}\n\n{email.text}", encoding="utf-8")

    async def close(self) -> None:
        """Nothing to release; present so every mailer can be closed the same way."""


class ThrottledMailer:
    """Limits how often one address is emailed and how many emails go out per day.

    The sign-up and reset forms accept anyone's address, so without a per-address limit they
    could be used to flood someone's inbox. Over that limit the email is skipped silently: the
    form's reply must not change, or it would reveal whether the address has an account.
    """

    def __init__(self, inner: Mailer, per_recipient: RateLimiter, per_day: RateLimiter):
        self._inner = inner
        self._per_recipient = per_recipient
        self._per_day = per_day

    async def send(self, email: Email) -> None:
        recipient = email.to.lower()
        if self._per_recipient.is_limited(recipient):
            logger.warning("Email skipped: address emailed too often", extra={"event": "email_throttled"})
            return
        if not allow(DAILY_TOTAL_KEY, [self._per_day]):
            raise EmailFailed("daily email limit reached")
        self._per_recipient.record(recipient)
        await self._inner.send(email)

    async def close(self) -> None:
        await self._inner.close()
