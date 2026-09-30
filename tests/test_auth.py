from dataclasses import replace
from pathlib import Path

from buddy.auth import google_user_identifier, is_valid_password_login
from buddy.config import Settings

SETTINGS = Settings(
    groq_api_key="", groq_model="m", database_url="sqlite+aiosqlite:///:memory:",
    admin_username="admin", admin_password="s3cret!", google_oauth_enabled=False,
    knowledge_dir=Path("."),
)


def test_correct_credentials_are_accepted():
    assert is_valid_password_login("admin", "s3cret!", SETTINGS)


def test_wrong_password_is_rejected():
    assert not is_valid_password_login("admin", "admin", SETTINGS)


def test_wrong_username_is_rejected():
    assert not is_valid_password_login("root", "s3cret!", SETTINGS)


def test_password_login_is_disabled_when_unconfigured():
    unconfigured = replace(SETTINGS, admin_username="", admin_password="")

    assert not is_valid_password_login("", "", unconfigured)


def test_non_ascii_input_does_not_crash():
    assert not is_valid_password_login("ádmin", "pässword", SETTINGS)


def test_google_user_is_keyed_by_lowercased_email():
    assert google_user_identifier({"email": "Ada@Gmail.com", "verified_email": True, "name": "Ada"}) == "ada@gmail.com"


def test_google_user_without_email_is_rejected():
    assert google_user_identifier({"name": "Ada"}) is None


def test_google_user_with_unverified_email_is_rejected():
    assert google_user_identifier({"email": "ada@gmail.com", "verified_email": False}) is None
