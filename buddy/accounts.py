"""Self-service email + password accounts: storage, sign-up by emailed link, sign-in and password reset.

Consumed by app.py, which only maps these results onto Chainlit users and HTTP responses. Accounts
and email tokens live in the chat-history database (DATABASE_URL) on the data layer's engine; their
tables come from history.migrate. Hashing and validation rules live in buddy/auth.py.
An account is created only when the owner of the inbox clicks the confirmation link, and every
reply a stranger can see is the same whether or not an email is registered (OWASP Authentication
Cheat Sheet: no account enumeration).
"""

import asyncio
import hashlib
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from buddy import emails
from buddy.auth import (
    hash_password,
    is_plausible_email,
    is_valid_password_login,
    normalize_email,
    password_problem,
    password_user_identifier,
    verify_password,
)
from buddy.config import Settings
from buddy.history import DATABASE_ERRORS
from buddy.mailer import Email, EmailFailed, Mailer

SIGNUP_CHECK_EMAIL_MESSAGE = (
    "Check your email: we have sent you a link to finish creating your account. "
    "It can take a few minutes, so check your spam folder too."
)
RESET_REQUESTED_MESSAGE = (
    "If an account uses this email, we have sent it a link to choose a new password. "
    "It can take a few minutes, so check your spam folder too."
)
EMAIL_CONFIRMED_MESSAGE = "Your email is confirmed and your account is ready. Please sign in."
WRONG_SIGNUP_PASSWORD_MESSAGE = "That is not the password you chose when signing up. Please try again."
PASSWORD_CHANGED_MESSAGE = "Your password has been changed. Please sign in with the new one."
LINK_EXPIRED_MESSAGE = "This link has expired or has already been used. Please start again."
UNAVAILABLE_MESSAGE = "This is unavailable right now. Please try again later."
INVALID_EMAIL_MESSAGE = "Please enter a valid email address."
SIGNUP_PURPOSE = "signup"
RESET_PURPOSE = "reset"
SIGNUP_LINK_HOURS = 24
# Short: until it is used, a reset link is as good as the password.
RESET_LINK_MINUTES = 60
# 256 random bits: guessing a live token is not feasible, however many attempts are made.
TOKEN_BYTES = 32
# Several sign-ups for one address can be pending (retries, or a stranger typing someone else's
# address). Confirming checks the typed password against each; this bounds that scrypt work.
MAX_PENDING_SIGNUPS_CHECKED = 5
# Each scrypt hash holds ~16 MB for its duration. Unbounded, a burst of logins could exhaust
# the 512 MB host; two at a time keeps hashing near 32 MB and queues the rest.
MAX_CONCURRENT_HASHES = 2
# Past this many sign-ins/sign-ups already waiting for a slot, answer "try again later" at once
# instead of queueing: a flood with made-up emails would otherwise make real users wait minutes.
MAX_WAITING_HASHES = 20
# Checked when the email is unknown, so those logins take as long as real ones and response
# timing does not reveal which emails are registered.
TIMING_DECOY_HASH = hash_password("timing-decoy")
# ISO-8601 UTC strings of one format compare correctly as text, in SQLite and Postgres alike.
LIVE_TOKEN_CONDITION = '"tokenHash" = :token_hash AND "purpose" = :purpose AND "usedAt" IS NULL AND "expiresAt" > :now'

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Account:
    email: str
    name: str
    password_hash: str


@dataclass(frozen=True)
class SignedInUser:
    identifier: str
    name: str
    is_admin: bool


class SignupOutcome(Enum):
    """What happened behind the uniform sign-up reply: for logs and tests, never for the reply."""

    PENDING = "pending"
    DUPLICATE = "duplicate"
    UNAVAILABLE = "unavailable"


class Outcome(Enum):
    DONE = "done"
    INVALID = "invalid"
    LINK_EXPIRED = "link_expired"
    UNAVAILABLE = "unavailable"


class AccountsUnavailable(Exception):
    """Raised by authenticate when the accounts database cannot be reached or hashing is overloaded."""


class HashingBusy(Exception):
    """Raised when too many password hashes are already waiting to run."""


class _HashingQueue:
    def __init__(self, slots: int, max_waiting: int):
        self._slots = asyncio.Semaphore(slots)
        self._max_in_flight = slots + max_waiting
        self._in_flight = 0

    async def run[T](self, function: Callable[..., T], *args) -> T:
        if self._in_flight >= self._max_in_flight:
            raise HashingBusy()
        self._in_flight += 1
        try:
            async with self._slots:
                # scrypt is deliberately slow; in a thread so one sign-in does not stall every chat.
                return await asyncio.to_thread(function, *args)
        finally:
            self._in_flight -= 1


_hashing = _HashingQueue(MAX_CONCURRENT_HASHES, MAX_WAITING_HASHES)


class AccountStore:
    def __init__(self, engine: AsyncEngine, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._engine = engine
        self._clock = clock

    async def find(self, email: str) -> Account | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    text('SELECT "email", "name", "passwordHash" FROM accounts WHERE "email" = :email'),
                    {"email": email},
                )
            ).first()
        return Account(email=row[0], name=row[1], password_hash=row[2]) if row else None

    async def is_google_user(self, email: str) -> bool:
        """Whether this address has signed in with Google (Chainlit keys those users by their email)."""
        async with self._engine.connect() as connection:
            row = (await connection.execute(text('SELECT 1 FROM users WHERE "identifier" = :email'), {"email": email})).first()
        return row is not None

    async def save_token(
        self,
        token_hash: str,
        purpose: str,
        email: str,
        lifetime: timedelta,
        name: str | None = None,
        password_hash: str | None = None,
    ) -> None:
        now = self._clock()
        async with self._engine.begin() as connection:
            # Housekeeping: lapsed tokens can never be used, so drop them whenever a new one is made.
            await connection.execute(text('DELETE FROM email_tokens WHERE "expiresAt" < :now'), {"now": now.isoformat()})
            await connection.execute(
                text(
                    'INSERT INTO email_tokens ("tokenHash", "purpose", "email", "name", "passwordHash", "createdAt", "expiresAt") '
                    "VALUES (:token_hash, :purpose, :email, :name, :password_hash, :created_at, :expires_at)"
                ),
                {
                    "token_hash": token_hash,
                    "purpose": purpose,
                    "email": email,
                    "name": name,
                    "password_hash": password_hash,
                    "created_at": now.isoformat(),
                    "expires_at": (now + lifetime).isoformat(),
                },
            )

    async def token_email(self, token_hash: str, purpose: str) -> str | None:
        """The email a live (unused, unexpired) token belongs to, without using it up."""
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    text(f'SELECT "email" FROM email_tokens WHERE {LIVE_TOKEN_CONDITION}'),
                    self._token_params(token_hash, purpose),
                )
            ).first()
        return row[0] if row else None

    async def pending_signups(self, email: str, limit: int) -> list[tuple[str, str]]:
        """(token hash, password hash) of the newest live sign-up tokens for an email."""
        async with self._engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        'SELECT "tokenHash", "passwordHash" FROM email_tokens WHERE "email" = :email AND "purpose" = :purpose '
                        'AND "usedAt" IS NULL AND "expiresAt" > :now ORDER BY "createdAt" DESC LIMIT :limit'
                    ),
                    {"email": email, "purpose": SIGNUP_PURPOSE, "now": self._clock().isoformat(), "limit": limit},
                )
            ).all()
        return [(row[0], row[1]) for row in rows]

    async def confirm_signup(self, token_hash: str) -> bool:
        """Uses a sign-up token and creates its account, in one transaction. False if the token is not live."""
        async with self._engine.begin() as connection:
            pending = await self._use_token(connection, token_hash, SIGNUP_PURPOSE)
            if pending is None:
                return False
            email, name, password_hash = pending
            # An earlier link to the same inbox may already have created the account: same owner, nothing to do.
            await connection.execute(
                text(
                    'INSERT INTO accounts ("email", "name", "passwordHash", "createdAt") '
                    "VALUES (:email, :name, :password_hash, :created_at) ON CONFLICT DO NOTHING"
                ),
                {"email": email, "name": name, "password_hash": password_hash, "created_at": self._clock().isoformat()},
            )
            await self._retire_tokens(connection, email, SIGNUP_PURPOSE)
        return True

    async def reset_password(self, token_hash: str, password_hash: str) -> bool:
        """Uses a reset token and sets the new password, in one transaction. False if the token is not live."""
        async with self._engine.begin() as connection:
            used = await self._use_token(connection, token_hash, RESET_PURPOSE)
            if used is None:
                return False
            email = used[0]
            await connection.execute(
                text('UPDATE accounts SET "passwordHash" = :password_hash WHERE "email" = :email'),
                {"password_hash": password_hash, "email": email},
            )
            # Other reset links sent earlier must stop working once the password has changed.
            await self._retire_tokens(connection, email, RESET_PURPOSE)
        return True

    async def _use_token(self, connection: AsyncConnection, token_hash: str, purpose: str) -> tuple | None:
        # One UPDATE ... RETURNING, so two clicks racing on the same link cannot both succeed.
        row = (
            await connection.execute(
                text(f'UPDATE email_tokens SET "usedAt" = :now WHERE {LIVE_TOKEN_CONDITION} RETURNING "email", "name", "passwordHash"'),
                self._token_params(token_hash, purpose),
            )
        ).first()
        return tuple(row) if row else None

    async def _retire_tokens(self, connection: AsyncConnection, email: str, purpose: str) -> None:
        await connection.execute(
            text('UPDATE email_tokens SET "usedAt" = :now WHERE "email" = :email AND "purpose" = :purpose AND "usedAt" IS NULL'),
            {"now": self._clock().isoformat(), "email": email, "purpose": purpose},
        )

    def _token_params(self, token_hash: str, purpose: str) -> dict:
        return {"token_hash": token_hash, "purpose": purpose, "now": self._clock().isoformat()}


async def start_signup(store: AccountStore, mailer: Mailer, app_url: str, name: str, email: str, password: str) -> SignupOutcome:
    """Emails a confirmation link, or an "already registered" notice. Meant to run in the background.

    The caller has already checked the details with auth.signup_problem and given its uniform
    reply; nothing here may change that reply, so failures are only logged.
    """
    address = normalize_email(email)
    try:
        existing = await store.find(address)
        if existing is None:
            password_hash = await _hashing.run(hash_password, password)
            token = secrets.token_urlsafe(TOKEN_BYTES)
            await store.save_token(
                _digest(token), SIGNUP_PURPOSE, address, timedelta(hours=SIGNUP_LINK_HOURS),
                name=name.strip(), password_hash=password_hash,
            )
    except (*DATABASE_ERRORS, HashingBusy):
        logger.exception("Could not start a sign-up")
        return SignupOutcome.UNAVAILABLE
    if existing is not None:
        await _send(mailer, emails.already_registered(address, existing.name, f"{app_url}/login", f"{app_url}/forgot-password"))
        return SignupOutcome.DUPLICATE
    await _send(mailer, emails.signup_confirmation(address, f"{app_url}/verify-email?token={token}", SIGNUP_LINK_HOURS))
    return SignupOutcome.PENDING


async def confirm_signup(store: AccountStore, token: str, password: str) -> tuple[Outcome, str]:
    """Creates the account from whichever pending sign-up for this inbox was made with `password`.

    The link alone is not enough: a stranger can start a sign-up with someone else's address and
    their own password, and that email reaches the real owner. Asking for the password means the
    owner's click creates the account with the owner's password, whichever link they use.
    """
    try:
        email = await store.token_email(_digest(token), SIGNUP_PURPOSE)
        pending = await store.pending_signups(email, MAX_PENDING_SIGNUPS_CHECKED) if email else []
    except DATABASE_ERRORS:
        logger.exception("Could not look up a sign-up")
        return Outcome.UNAVAILABLE, UNAVAILABLE_MESSAGE
    if not pending:
        return Outcome.LINK_EXPIRED, LINK_EXPIRED_MESSAGE
    try:
        matching = [token_hash for token_hash, password_hash in pending if await _hashing.run(verify_password, password, password_hash)]
        confirmed = bool(matching) and await store.confirm_signup(matching[0])
    except (*DATABASE_ERRORS, HashingBusy):
        logger.exception("Could not confirm a sign-up")
        return Outcome.UNAVAILABLE, UNAVAILABLE_MESSAGE
    if not matching:
        return Outcome.INVALID, WRONG_SIGNUP_PASSWORD_MESSAGE
    if not confirmed:
        return Outcome.LINK_EXPIRED, LINK_EXPIRED_MESSAGE
    logger.info("Account created", extra={"event": "signup_confirmed"})
    return Outcome.DONE, EMAIL_CONFIRMED_MESSAGE


def reset_request_problem(email: str) -> str | None:
    """A shape check only: it says nothing about whether the address has an account."""
    return None if is_plausible_email(normalize_email(email)) else INVALID_EMAIL_MESSAGE


async def start_password_reset(store: AccountStore, mailer: Mailer, app_url: str, email: str) -> bool:
    """Emails a reset link if the address has a password account. Meant to run in the background.

    A Google-only address has no password to reset, so it gets a note saying to use Google instead,
    rather than nothing (which left people waiting for an email that never came). Returns whether
    an email was sent.
    """
    address = normalize_email(email)
    try:
        account = await store.find(address)
        if account is None:
            if not await store.is_google_user(address):
                return False
            return await _send(mailer, emails.google_account_notice(address, f"{app_url}/login"))
        token = secrets.token_urlsafe(TOKEN_BYTES)
        await store.save_token(_digest(token), RESET_PURPOSE, address, timedelta(minutes=RESET_LINK_MINUTES))
    except DATABASE_ERRORS:
        logger.exception("Could not start a password reset")
        return False
    return await _send(mailer, emails.password_reset(address, account.name, f"{app_url}/reset-password?token={token}", RESET_LINK_MINUTES))


async def reset_password(store: AccountStore, token: str, password: str) -> tuple[Outcome, str]:
    token_hash = _digest(token)
    try:
        email = await store.token_email(token_hash, RESET_PURPOSE)
        account = await store.find(email) if email else None
    except DATABASE_ERRORS:
        logger.exception("Could not look up a password reset")
        return Outcome.UNAVAILABLE, UNAVAILABLE_MESSAGE
    if account is None:
        return Outcome.LINK_EXPIRED, LINK_EXPIRED_MESSAGE
    problem = password_problem(password, account.name, account.email)
    if problem:
        return Outcome.INVALID, problem
    try:
        password_hash = await _hashing.run(hash_password, password)
        changed = await store.reset_password(token_hash, password_hash)
    except (*DATABASE_ERRORS, HashingBusy):
        logger.exception("Could not reset a password")
        return Outcome.UNAVAILABLE, UNAVAILABLE_MESSAGE
    if not changed:
        return Outcome.LINK_EXPIRED, LINK_EXPIRED_MESSAGE
    logger.info("Password reset", extra={"event": "password_reset"})
    return Outcome.DONE, PASSWORD_CHANGED_MESSAGE


async def authenticate(store: AccountStore, username: str, password: str, settings: Settings) -> SignedInUser | None:
    """Checks the configured admin first, then self-service accounts. None means the login is refused.

    Raises AccountsUnavailable when the database is down, so the user is not told their password is wrong.
    """
    if is_valid_password_login(username, password, settings):
        # The configured name, not what was typed: "Admin" and "admin" must share one account and history.
        return SignedInUser(identifier=settings.admin_username, name=settings.admin_username, is_admin=True)
    try:
        account = await store.find(normalize_email(username))
    except DATABASE_ERRORS as error:
        logger.exception("Could not look up an account during sign-in")
        raise AccountsUnavailable() from error
    stored_hash = account.password_hash if account else TIMING_DECOY_HASH
    try:
        password_ok = await _hashing.run(verify_password, password, stored_hash)
    except HashingBusy as error:
        logger.warning("Sign-in refused: hashing queue full", extra={"event": "hashing_busy"})
        raise AccountsUnavailable() from error
    if account is None or not password_ok:
        return None
    return SignedInUser(identifier=password_user_identifier(account.email), name=account.name, is_admin=False)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _send(mailer: Mailer, email: Email) -> bool:
    try:
        await mailer.send(email)
    except EmailFailed as failure:
        logger.error("Email not sent: %s", failure, extra={"event": "email_failed"})
        return False
    return True
