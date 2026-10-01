"""HTTP-level tests of app.py against the real Chainlit server, on a throwaway SQLite database.

These guard the Chainlit internals app.py relies on (route order, the page-template wrapper,
the password callback's status override), so a Chainlit upgrade that breaks them fails here.
"""

import asyncio
import importlib
import itertools
import json
import re
import time

import pytest
from starlette.testclient import TestClient

from buddy.accounts import (
    EMAIL_CONFIRMED_MESSAGE,
    LINK_EXPIRED_MESSAGE,
    PASSWORD_CHANGED_MESSAGE,
    RESET_REQUESTED_MESSAGE,
    SIGNUP_CHECK_EMAIL_MESSAGE,
)
from buddy.config import PROJECT_ROOT
from buddy.rate_limit import RateLimiter

# Explicit values win over .env (python-dotenv never overrides), so no real key or database is used.
TEST_ENV = {
    "DATABASE_URL": "",  # filled with a temp file per session
    "GROQ_API_KEY": "test-key-never-sent",
    "BUDDY_ADMIN_USERNAME": "admin",
    "BUDDY_ADMIN_PASSWORD": "admin-password-for-tests",
    "CHAINLIT_AUTH_SECRET": "test-secret-" + "x" * 40,
    "OAUTH_GOOGLE_CLIENT_ID": "",
    "OAUTH_GOOGLE_CLIENT_SECRET": "",
    "CHAINLIT_URL": "",
    "RENDER": "",
    "BREVO_API_KEY": "",
    "EMAIL_FROM": "",
    "EMAIL_OUTBOX_DIR": "",  # filled with a temp folder per session: emails land there as files
}
HOST = "testserver"
PASSWORD = "correct-horse-battery"
TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")
# Sign-up and reset emails are written by a background task; allow it this long to finish.
EMAIL_WAIT_SECONDS = 10
_addresses = (f"203.0.113.{n}" for n in itertools.count(1))


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    folder = tmp_path_factory.mktemp("app")
    overrides = {"DATABASE_URL": f"sqlite+aiosqlite:///{(folder / 'chainlit.db').as_posix()}", "EMAIL_OUTBOX_DIR": str(folder / "outbox")}
    with pytest.MonkeyPatch.context() as env:
        for name, value in {**TEST_ENV, **overrides}.items():
            env.setenv(name, value)
        yield importlib.import_module("app")


@pytest.fixture(scope="module")
def client(app_module):
    async def server_without_lifespan(scope, receive, send):
        if scope["type"] == "lifespan":
            # Chainlit's lifespan may open a browser and ends with os._exit(0), which would kill pytest.
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})
            return
        await app_module.server(scope, receive, send)

    # One `with` block = one event loop for every request, which the pooled SQLite connections need.
    with TestClient(server_without_lifespan, base_url=f"http://{HOST}") as test_client:
        test_client.portal.call(app_module.migrate, app_module.history_store.engine)
        yield test_client
        test_client.portal.call(app_module.history_store.engine.dispose)


def _fresh_ip():
    # The per-IP limits are shared by every test in this module; a new address per test keeps them apart.
    return {"x-forwarded-for": next(_addresses)}


def _signup(client, email, password=PASSWORD, name="Ada Lovelace", headers=None):
    return client.post("/auth/signup", json={"name": name, "email": email, "password": password}, headers=headers or _fresh_ip())


def _emails_to(app_module, address):
    outbox = app_module.settings.email_outbox_dir
    slug = re.sub(r"[^a-z0-9]+", "_", address.lower())
    return sorted(outbox.glob(f"*-{slug}.txt")) if outbox.exists() else []


def _next_email(app_module, address, already_seen):
    deadline = time.monotonic() + EMAIL_WAIT_SECONDS
    while time.monotonic() < deadline:
        files = _emails_to(app_module, address)
        if len(files) > already_seen:
            return files[-1].read_text(encoding="utf-8")
        time.sleep(0.05)
    raise AssertionError(f"no email to {address} within {EMAIL_WAIT_SECONDS}s")


def _token_from_next_email(client, app_module, address, send):
    seen = len(_emails_to(app_module, address))
    send()
    return TOKEN_IN_LINK.search(_next_email(app_module, address, seen)).group(1)


def _create_account(client, app_module, email, password=PASSWORD):
    token = _token_from_next_email(client, app_module, email, lambda: _signup(client, email, password))
    assert client.post("/auth/verify-email", json={"token": token, "password": password}, headers=_fresh_ip()).status_code == 200


def _login(client, username, password, headers=None):
    return client.post("/login", data={"username": username, "password": password}, headers=headers or _fresh_ip())


def test_signup_page_is_served_ahead_of_chainlits_catch_all(client):
    response = client.get("/signup")

    assert response.status_code == 200
    assert "Create your account" in response.text and "vite-ui-theme" in response.text


def test_chainlit_pages_get_the_early_theme_snippet(client):
    assert "vite-ui-theme" in client.get("/login").text


def test_pages_carry_the_security_headers(client):
    headers = client.get("/login").headers

    assert headers["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/verify-email", "/forgot-password", "/reset-password"])
def test_account_pages_are_served_ahead_of_chainlits_catch_all(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert "auth.js" in response.text and "vite-ui-theme" in response.text


def test_signup_for_new_and_registered_emails_gets_identical_responses(client, app_module):
    _create_account(client, app_module, "grace@example.com")

    new = _signup(client, "brand-new@example.com")
    registered = _signup(client, "grace@example.com", password="another-long-pass")

    assert (new.status_code, new.json()) == (registered.status_code, registered.json()) == (202, {"detail": SIGNUP_CHECK_EMAIL_MESSAGE})


def test_account_exists_only_after_the_emailed_link_is_confirmed(client, app_module):
    token = _token_from_next_email(client, app_module, "linda@example.com", lambda: _signup(client, "linda@example.com"))
    assert _login(client, "linda@example.com", PASSWORD).status_code == 401

    confirmed = client.post("/auth/verify-email", json={"token": token, "password": PASSWORD}, headers=_fresh_ip())

    assert (confirmed.status_code, confirmed.json()["detail"]) == (200, EMAIL_CONFIRMED_MESSAGE)
    assert _login(client, "linda@example.com", PASSWORD).status_code == 200


def test_used_or_made_up_links_get_410(client):
    response = client.post("/auth/verify-email", json={"token": "made-up", "password": PASSWORD}, headers=_fresh_ip())

    assert (response.status_code, response.json()["detail"]) == (410, LINK_EXPIRED_MESSAGE)


def test_password_reset_end_to_end(client, app_module):
    _create_account(client, app_module, "hedy@example.com")
    token = _token_from_next_email(
        client, app_module, "hedy@example.com",
        lambda: client.post("/auth/password-reset", json={"email": "hedy@example.com"}, headers=_fresh_ip()),
    )

    changed = client.post("/auth/password-reset/confirm", json={"token": token, "password": "a-new-secret-77"}, headers=_fresh_ip())

    assert (changed.status_code, changed.json()["detail"]) == (200, PASSWORD_CHANGED_MESSAGE)
    assert _login(client, "hedy@example.com", "a-new-secret-77").status_code == 200
    assert _login(client, "hedy@example.com", PASSWORD).status_code == 401


def test_reset_request_reply_is_the_same_for_unknown_addresses(client, app_module):
    _create_account(client, app_module, "known@example.com")

    known = client.post("/auth/password-reset", json={"email": "known@example.com"}, headers=_fresh_ip())
    unknown = client.post("/auth/password-reset", json={"email": "unknown@example.com"}, headers=_fresh_ip())

    assert (known.status_code, known.json()) == (unknown.status_code, unknown.json()) == (202, {"detail": RESET_REQUESTED_MESSAGE})


def test_weak_new_password_is_explained(client, app_module):
    _create_account(client, app_module, "barbara@example.com")
    token = _token_from_next_email(
        client, app_module, "barbara@example.com",
        lambda: client.post("/auth/password-reset", json={"email": "barbara@example.com"}, headers=_fresh_ip()),
    )

    response = client.post("/auth/password-reset/confirm", json={"token": token, "password": "password123"}, headers=_fresh_ip())

    assert response.status_code == 400 and "too easy" in response.json()["detail"]


def test_signup_and_reset_are_off_without_an_email_provider(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "mailer", None)

    signup = _signup(client, "someone@example.com")
    reset = client.post("/auth/password-reset", json={"email": "someone@example.com"}, headers=_fresh_ip())

    assert signup.status_code == reset.status_code == 503


def test_cross_site_reset_confirm_is_refused(client):
    headers = {**_fresh_ip(), "origin": "https://evil.example"}

    response = client.post("/auth/password-reset/confirm", json={"token": "t", "password": "a-new-secret-77"}, headers=headers)

    assert response.status_code == 403


def test_overlong_token_is_rejected_before_any_lookup(client):
    assert client.post("/auth/verify-email", json={"token": "x" * 500, "password": PASSWORD}, headers=_fresh_ip()).status_code == 422


def test_invalid_signup_is_explained(client):
    response = _signup(client, "not-an-email")

    assert response.status_code == 400 and "valid email" in response.json()["detail"]


def test_signed_up_user_can_log_in_and_gets_an_httponly_cookie(client, app_module):
    _create_account(client, app_module, "katherine@example.com")

    response = _login(client, "katherine@example.com", PASSWORD)

    assert response.status_code == 200
    assert "httponly" in response.headers["set-cookie"].lower()


def test_wrong_password_gets_chainlits_generic_401(client, app_module):
    _create_account(client, app_module, "dorothy@example.com")

    assert _login(client, "dorothy@example.com", "wrong-password").status_code == 401


def test_repeated_failed_logins_for_one_account_from_one_address_get_429(client, app_module):
    address = _fresh_ip()
    attempts = app_module.FAILED_SIGN_INS_PER_ACCOUNT_AND_IP.limit + 1

    statuses = [_login(client, "target@example.com", "guess", headers=address).status_code for _ in range(attempts)]

    assert statuses[-1] == 429 and set(statuses[:-1]) == {401}


def test_failures_from_another_address_do_not_lock_the_owner_out(client, app_module):
    _create_account(client, app_module, "owner@example.com")
    attacker = _fresh_ip()
    for _ in range(app_module.FAILED_SIGN_INS_PER_ACCOUNT_AND_IP.limit + 1):
        _login(client, "owner@example.com", "guess", headers=attacker)

    assert _login(client, "owner@example.com", PASSWORD).status_code == 200


def test_failures_spread_over_many_addresses_still_hit_the_account_limit(client, app_module):
    statuses = [_login(client, "spread@example.com", "guess").status_code for _ in range(app_module.FAILED_SIGN_INS_PER_ACCOUNT.limit + 1)]

    assert statuses[-1] == 429


def test_successful_logins_do_not_count_towards_the_limit(client, app_module):
    _create_account(client, app_module, "frequent@example.com")
    address = _fresh_ip()

    statuses = [
        _login(client, "frequent@example.com", PASSWORD, headers=address).status_code
        for _ in range(app_module.FAILED_SIGN_INS_PER_ACCOUNT_AND_IP.limit + 1)
    ]

    assert set(statuses) == {200}


def test_login_page_has_text_for_every_key_custom_js_sends():
    errors = json.loads((PROJECT_ROOT / ".chainlit" / "translations" / "en-US.json").read_text(encoding="utf-8"))["auth"]["login"]["errors"]
    custom_js = (PROJECT_ROOT / "public" / "custom.js").read_text(encoding="utf-8")

    for key in ("toomanysignins", "signinunavailable"):
        assert f'"{key}"' in custom_js and key in errors


def test_too_many_logins_from_one_address_get_429(client, app_module):
    address = _fresh_ip()

    statuses = [_login(client, f"user{n}@example.com", "guess", headers=address).status_code for n in range(app_module.SIGN_INS_PER_IP.limit + 1)]

    assert statuses[-1] == 429


def test_sign_in_reports_503_when_the_database_is_down(client, app_module, monkeypatch):
    async def unreachable(email):
        from sqlalchemy.exc import OperationalError
        raise OperationalError("SELECT", {}, Exception("down"))

    monkeypatch.setattr(app_module.account_store, "find", unreachable)

    response = _login(client, "someone@example.com", "whatever-password")

    assert response.status_code == 503 and response.json()["detail"] == app_module.SIGN_IN_UNAVAILABLE_REPLY


def test_cross_site_login_is_refused(client):
    headers = {**_fresh_ip(), "origin": "https://evil.example"}

    assert _login(client, "admin", "admin-password-for-tests", headers=headers).status_code == 403


def test_admin_can_log_in(client):
    assert _login(client, "Admin", "admin-password-for-tests").status_code == 200


def test_signup_total_cap_applies_across_addresses(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "SIGNUPS_TOTAL", RateLimiter(limit=1, window_seconds=3600))

    statuses = [_signup(client, f"cap{n}@example.com").status_code for n in range(2)]

    assert statuses == [202, 429]


def test_invalid_signups_do_not_use_up_the_total_cap(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "SIGNUPS_TOTAL", RateLimiter(limit=1, window_seconds=3600))
    for n in range(3):
        _signup(client, f"not-an-email-{n}")

    assert _signup(client, "valid-after-invalid@example.com").status_code == 202


def test_file_upload_route_is_blocked(client):
    assert client.post("/project/file?session_id=x", files={"file": ("a.txt", b"hello")}).status_code == 403


def test_ready_when_database_and_key_are_there(client):
    assert client.get("/ready").json() == {"status": "ready"}


def test_not_ready_names_the_failing_check(client, app_module, monkeypatch):
    async def unreachable(engine):
        return False

    monkeypatch.setattr(app_module, "database_reachable", unreachable)

    response = client.get("/ready")

    assert response.status_code == 503 and response.json() == {"status": "unavailable"}


def test_greeting_escapes_markdown_in_the_users_name(app_module):
    assert "\\[click\\]\\(https://evil\\.example\\)" in app_module.greeting("[click](https://evil.example)")


class _FakeUserSession:
    def __init__(self, identifier):
        self._user = type("User", (), {"identifier": identifier})()

    def get(self, key):
        return self._user if key == "user" else None


def test_eleventh_question_in_a_minute_gets_the_slow_down_reply(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cl, "user_session", _FakeUserSession("password:minute@example.com"))

    replies = [app_module._question_limit_reply() for _ in range(11)]

    assert replies[:10] == [None] * 10 and replies[10] == app_module.QUESTIONS_PER_MINUTE_REPLY


def test_question_past_the_daily_limit_gets_the_come_back_tomorrow_reply(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cl, "user_session", _FakeUserSession("password:daily@example.com"))
    monkeypatch.setattr(app_module, "QUESTIONS_PER_DAY", RateLimiter(limit=2, window_seconds=86400))

    replies = [app_module._question_limit_reply() for _ in range(3)]

    assert replies == [None, None, app_module.QUESTIONS_PER_DAY_REPLY]


def test_question_limits_are_per_user(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cl, "user_session", _FakeUserSession("password:busy@example.com"))
    for _ in range(10):
        app_module._question_limit_reply()

    monkeypatch.setattr(app_module.cl, "user_session", _FakeUserSession("password:other@example.com"))

    assert app_module._question_limit_reply() is None


def test_unexpected_background_failure_is_logged(client, app_module, caplog):
    async def broken():
        raise RuntimeError("bug in a background flow")

    client.portal.call(_run_and_wait, app_module, broken())

    assert "Background task failed" in caplog.text and "bug in a background flow" in caplog.text


async def _run_and_wait(app_module, work):
    app_module._run_in_background(work)
    while app_module._background_tasks:
        await asyncio.sleep(0.01)
