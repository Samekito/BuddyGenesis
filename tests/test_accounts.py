import asyncio
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from buddy import accounts
from buddy.accounts import (
    EMAIL_CONFIRMED_MESSAGE,
    WRONG_SIGNUP_PASSWORD_MESSAGE,
    INVALID_EMAIL_MESSAGE,
    PASSWORD_CHANGED_MESSAGE,
    AccountStore,
    AccountsUnavailable,
    Outcome,
    SignupOutcome,
    authenticate,
    confirm_signup,
    reset_password,
    reset_request_problem,
    start_password_reset,
    start_signup,
)
from buddy.config import Settings
from buddy.history import migrate
from buddy.mailer import EmailFailed

APP_URL = "https://soc-buddy.example"
SETTINGS = Settings(
    groq_api_key="", groq_model="m", database_url="sqlite+aiosqlite:///:memory:",
    admin_username="admin", admin_password="s3cret!", google_oauth_enabled=False,
    knowledge_dir=Path("."),
)
TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class RecordingMailer:
    """Stands in for Brevo at the service boundary: keeps every email instead of sending it."""

    def __init__(self):
        self.sent = []

    async def send(self, email):
        self.sent.append(email)

    async def close(self):
        """Nothing to release."""


class FailingMailer(RecordingMailer):
    async def send(self, email):
        raise EmailFailed("Brevo answered HTTP 401")


class UnreachableStore:
    """Stands in for a database that is asleep or missing its tables."""

    async def find(self, email):
        raise OperationalError("SELECT", {}, Exception("unreachable"))

    async def token_email(self, token_hash, purpose):
        raise OperationalError("SELECT", {}, Exception("unreachable"))

    async def confirm_signup(self, token_hash):
        raise OperationalError("UPDATE", {}, Exception("unreachable"))


class RefusedConnectionStore(UnreachableStore):
    """Stands in for asyncpg failing to connect: it raises plain OSErrors that SQLAlchemy does not wrap."""

    async def find(self, email):
        raise ConnectionRefusedError("connection refused")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
async def store(tmp_path, clock):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'accounts.db').as_posix()}")
    await migrate(engine)
    yield AccountStore(engine, clock=clock)
    await engine.dispose()


@pytest.fixture
def mailer():
    return RecordingMailer()


def _token(email):
    return TOKEN_IN_LINK.search(email.text).group(1)


async def _create_account(store, mailer, name="Ada Lovelace", email="ada@example.com", password="correct-horse"):
    await start_signup(store, mailer, APP_URL, name, email, password)
    await confirm_signup(store, _token(mailer.sent[-1]), password)


# Sign-up by emailed link

async def test_signup_emails_a_confirmation_link_and_creates_nothing_yet(store, mailer):
    outcome = await start_signup(store, mailer, APP_URL, "Ada Lovelace", " Ada@Example.com ", "correct-horse")

    assert outcome is SignupOutcome.PENDING
    assert [email.to for email in mailer.sent] == ["ada@example.com"]
    assert f"{APP_URL}/verify-email?token=" in mailer.sent[0].text
    assert await store.find("ada@example.com") is None


async def test_confirmation_link_creates_the_account(store, mailer):
    await start_signup(store, mailer, APP_URL, "Ada Lovelace", "ada@example.com", "correct-horse")

    outcome, message = await confirm_signup(store, _token(mailer.sent[0]), "correct-horse")

    assert (outcome, message) == (Outcome.DONE, EMAIL_CONFIRMED_MESSAGE)
    user = await authenticate(store, "ADA@example.com", "correct-horse", SETTINGS)
    assert (user.identifier, user.name, user.is_admin) == ("password:ada@example.com", "Ada Lovelace", False)


async def test_confirmation_link_works_only_once(store, mailer):
    await start_signup(store, mailer, APP_URL, "Ada", "ada@example.com", "correct-horse")
    token = _token(mailer.sent[0])
    await confirm_signup(store, token, "correct-horse")

    assert (await confirm_signup(store, token, "correct-horse"))[0] is Outcome.LINK_EXPIRED


async def test_confirmation_link_expires_after_a_day(store, mailer, clock):
    await start_signup(store, mailer, APP_URL, "Ada", "ada@example.com", "correct-horse")

    clock.now += timedelta(hours=accounts.SIGNUP_LINK_HOURS, minutes=1)

    assert (await confirm_signup(store, _token(mailer.sent[0]), "correct-horse"))[0] is Outcome.LINK_EXPIRED
    assert await store.find("ada@example.com") is None


async def test_made_up_token_is_refused(store):
    assert (await confirm_signup(store, "not-a-real-token", "correct-horse"))[0] is Outcome.LINK_EXPIRED


async def test_signup_for_a_registered_email_sends_a_notice_instead(store, mailer):
    await _create_account(store, mailer)

    outcome = await start_signup(store, mailer, APP_URL, "Impostor", "ada@example.com", "other-horse-99")

    assert outcome is SignupOutcome.DUPLICATE
    notice = mailer.sent[-1]
    assert "already has one" in notice.text and f"{APP_URL}/forgot-password" in notice.text
    assert "token=" not in notice.text


async def test_signup_for_a_registered_email_does_not_change_its_password(store, mailer):
    await _create_account(store, mailer)
    await start_signup(store, mailer, APP_URL, "Impostor", "ada@example.com", "other-horse-99")

    assert await authenticate(store, "ada@example.com", "other-horse-99", SETTINGS) is None
    assert await authenticate(store, "ada@example.com", "correct-horse", SETTINGS) is not None


async def test_owner_gets_their_own_password_even_via_a_strangers_link(store, mailer):
    # A stranger started a sign-up with Ada's address; both emails reach Ada, who clicks the stranger's.
    await start_signup(store, mailer, APP_URL, "Stranger", "ada@example.com", "stranger-pass-1")
    await start_signup(store, mailer, APP_URL, "Ada", "ada@example.com", "correct-horse")

    outcome, _ = await confirm_signup(store, _token(mailer.sent[0]), "correct-horse")

    assert outcome is Outcome.DONE
    assert (await store.find("ada@example.com")).name == "Ada"
    assert await authenticate(store, "ada@example.com", "stranger-pass-1", SETTINGS) is None
    assert await authenticate(store, "ada@example.com", "correct-horse", SETTINGS) is not None


async def test_confirming_with_the_wrong_password_is_refused_and_the_link_still_works(store, mailer):
    await start_signup(store, mailer, APP_URL, "Ada", "ada@example.com", "correct-horse")
    token = _token(mailer.sent[0])

    assert await confirm_signup(store, token, "a-guess-1234") == (Outcome.INVALID, WRONG_SIGNUP_PASSWORD_MESSAGE)
    assert (await confirm_signup(store, token, "correct-horse"))[0] is Outcome.DONE


async def test_confirmation_email_does_not_repeat_the_typed_name(store, mailer):
    await start_signup(store, mailer, APP_URL, "Visit evil.example now", "ada@example.com", "correct-horse")

    assert "evil.example" not in mailer.sent[0].text and "evil.example" not in mailer.sent[0].html


async def test_signup_reports_unavailable_when_the_database_fails(mailer):
    assert await start_signup(UnreachableStore(), mailer, APP_URL, "Ada", "ada@example.com", "correct-horse") is SignupOutcome.UNAVAILABLE
    assert mailer.sent == []


async def test_signup_reports_unavailable_when_the_connection_itself_fails(mailer):
    outcome = await start_signup(RefusedConnectionStore(), mailer, APP_URL, "Ada", "ada@example.com", "correct-horse")

    assert outcome is SignupOutcome.UNAVAILABLE


async def test_email_failure_is_logged_not_raised(store, caplog):
    outcome = await start_signup(store, FailingMailer(), APP_URL, "Ada", "ada@example.com", "correct-horse")

    assert outcome is SignupOutcome.PENDING
    assert "Email not sent: Brevo answered HTTP 401" in caplog.text
    assert "ada@example.com" not in caplog.text


async def test_confirmation_reports_unavailable_when_the_database_fails():
    assert (await confirm_signup(UnreachableStore(), "token", "correct-horse"))[0] is Outcome.UNAVAILABLE


# Password reset

async def test_reset_request_emails_a_link_for_a_registered_address(store, mailer):
    await _create_account(store, mailer)

    sent = await start_password_reset(store, mailer, APP_URL, "ADA@example.com")

    assert sent
    assert f"{APP_URL}/reset-password?token=" in mailer.sent[-1].text


async def test_reset_request_for_an_unknown_address_sends_nothing(store, mailer):
    assert not await start_password_reset(store, mailer, APP_URL, "nobody@example.com")
    assert mailer.sent == []


async def test_reset_link_sets_the_new_password(store, mailer):
    await _create_account(store, mailer)
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")

    outcome, message = await reset_password(store, _token(mailer.sent[-1]), "brand-new-secret")

    assert (outcome, message) == (Outcome.DONE, PASSWORD_CHANGED_MESSAGE)
    assert await authenticate(store, "ada@example.com", "brand-new-secret", SETTINGS) is not None
    assert await authenticate(store, "ada@example.com", "correct-horse", SETTINGS) is None


async def test_reset_link_works_only_once(store, mailer):
    await _create_account(store, mailer)
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")
    token = _token(mailer.sent[-1])
    await reset_password(store, token, "brand-new-secret")

    assert (await reset_password(store, token, "another-new-secret"))[0] is Outcome.LINK_EXPIRED


async def test_older_reset_links_stop_working_after_a_reset(store, mailer):
    await _create_account(store, mailer)
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")
    older = _token(mailer.sent[-1])
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")

    await reset_password(store, _token(mailer.sent[-1]), "brand-new-secret")

    assert (await reset_password(store, older, "another-new-secret"))[0] is Outcome.LINK_EXPIRED


async def test_reset_link_expires_after_an_hour(store, mailer, clock):
    await _create_account(store, mailer)
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")

    clock.now += timedelta(minutes=accounts.RESET_LINK_MINUTES + 1)

    assert (await reset_password(store, _token(mailer.sent[-1]), "brand-new-secret"))[0] is Outcome.LINK_EXPIRED


async def test_weak_new_password_is_refused_and_the_link_still_works(store, mailer):
    await _create_account(store, mailer)
    await start_password_reset(store, mailer, APP_URL, "ada@example.com")
    token = _token(mailer.sent[-1])

    outcome, message = await reset_password(store, token, "password123")

    assert outcome is Outcome.INVALID and "too easy" in message
    assert (await reset_password(store, token, "brand-new-secret"))[0] is Outcome.DONE


async def test_reset_reports_unavailable_when_the_database_fails():
    assert (await reset_password(UnreachableStore(), "token", "brand-new-secret"))[0] is Outcome.UNAVAILABLE


def test_reset_request_only_checks_the_address_shape():
    assert reset_request_problem("not-an-email") == INVALID_EMAIL_MESSAGE
    assert reset_request_problem("nobody@example.com") is None


# Sign-in

async def test_wrong_password_is_refused(store, mailer):
    await _create_account(store, mailer)

    assert await authenticate(store, "ada@example.com", "wrong-horse", SETTINGS) is None


async def test_unknown_email_is_refused(store):
    assert await authenticate(store, "nobody@example.com", "correct-horse", SETTINGS) is None


async def test_admin_is_checked_before_accounts(store):
    user = await authenticate(store, "Admin", "s3cret!", SETTINGS)

    assert user.identifier == "admin"
    assert user.is_admin


async def test_sign_in_reports_the_database_outage_instead_of_a_wrong_password():
    with pytest.raises(AccountsUnavailable):
        await authenticate(UnreachableStore(), "ada@example.com", "correct-horse", SETTINGS)


async def test_sign_in_reports_an_outage_when_the_connection_itself_fails():
    with pytest.raises(AccountsUnavailable):
        await authenticate(RefusedConnectionStore(), "ada@example.com", "correct-horse", SETTINGS)


async def test_admin_can_still_sign_in_when_database_fails():
    assert (await authenticate(UnreachableStore(), "admin", "s3cret!", SETTINGS)).is_admin


async def test_blank_login_is_refused_when_admin_is_unconfigured(store):
    unconfigured = replace(SETTINGS, admin_username="", admin_password="")

    assert await authenticate(store, "", "", unconfigured) is None


# Password hashing capacity

async def test_no_more_than_the_allowed_number_of_hashes_run_at_once(monkeypatch, store, mailer):
    lock, running, peak = threading.Lock(), 0, 0

    def slow_hash(password):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return "scrypt$1$1$1$00$00"

    monkeypatch.setattr(accounts, "hash_password", slow_hash)

    await asyncio.gather(*(start_signup(store, mailer, APP_URL, "Ada", f"ada{n}@example.com", "correct-horse") for n in range(8)))

    assert peak == accounts.MAX_CONCURRENT_HASHES


async def test_requests_beyond_the_hashing_queue_are_turned_away_at_once(monkeypatch, store, mailer):
    monkeypatch.setattr(accounts, "_hashing", accounts._HashingQueue(slots=1, max_waiting=1))
    monkeypatch.setattr(accounts, "hash_password", lambda password: time.sleep(0.2) or "scrypt$1$1$1$00$00")

    outcomes = await asyncio.gather(*(start_signup(store, mailer, APP_URL, "Ada", f"ada{n}@example.com", "correct-horse") for n in range(3)))

    assert sorted(outcome.value for outcome in outcomes) == ["pending", "pending", "unavailable"]
    assert accounts._hashing._in_flight == 0


async def test_sign_in_turned_away_by_a_full_hashing_queue_reports_an_outage(monkeypatch, store):
    busy = accounts._HashingQueue(slots=1, max_waiting=0)
    busy._in_flight = 1
    monkeypatch.setattr(accounts, "_hashing", busy)

    with pytest.raises(AccountsUnavailable):
        await authenticate(store, "ada@example.com", "correct-horse", SETTINGS)


async def _add_google_user(store, email):
    # How Chainlit records a Google sign-in: a users row whose identifier is the email (app.google_login).
    async with store._engine.begin() as connection:
        await connection.execute(
            text('INSERT INTO users ("id", "identifier", "metadata", "createdAt") VALUES (:id, :identifier, :metadata, :created_at)'),
            {"id": str(uuid.uuid4()), "identifier": email, "metadata": "{}", "created_at": "2026-10-01T00:00:00Z"},
        )


async def test_reset_request_for_a_google_account_explains_how_to_sign_in(store, mailer):
    await _add_google_user(store, "grace@gmail.com")

    sent = await start_password_reset(store, mailer, APP_URL, "Grace@Gmail.com")

    assert sent
    notice = mailer.sent[-1]
    assert notice.to == "grace@gmail.com"
    assert "Continue with Google" in notice.text and f"{APP_URL}/login" in notice.text
    assert "token=" not in notice.text


async def test_reset_request_for_an_account_with_a_password_still_sends_a_reset_link(store, mailer):
    await _create_account(store, mailer, email="ada@gmail.com")
    await _add_google_user(store, "ada@gmail.com")

    await start_password_reset(store, mailer, APP_URL, "ada@gmail.com")

    assert "reset-password?token=" in mailer.sent[-1].text
