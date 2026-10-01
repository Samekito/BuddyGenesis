from dataclasses import replace
from pathlib import Path

from buddy.auth import (
    MAX_PASSWORD_LENGTH,
    google_user_identifier,
    hash_password,
    is_valid_password_login,
    password_user_identifier,
    signup_problem,
    verify_password,
)
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


def test_username_is_case_insensitive():
    assert is_valid_password_login("Admin", "s3cret!", SETTINGS)
    assert is_valid_password_login("ADMIN", "s3cret!", SETTINGS)


def test_password_stays_case_sensitive():
    assert not is_valid_password_login("admin", "S3CRET!", SETTINGS)


def test_hashed_password_verifies():
    assert verify_password("correct horse", hash_password("correct horse"))


def test_wrong_password_does_not_verify_against_hash():
    assert not verify_password("wrong horse", hash_password("correct horse"))


def test_same_password_hashes_differently_each_time():
    assert hash_password("correct horse") != hash_password("correct horse")


def test_hash_never_contains_the_password():
    assert "correct horse" not in hash_password("correct horse")


def test_malformed_stored_hash_fails_closed():
    assert not verify_password("anything", "not-a-hash")


def test_stored_hash_with_non_hex_key_fails_closed():
    assert not verify_password("anything", "scrypt$16384$8$1$00$zz")


def test_unknown_hash_scheme_is_rejected():
    assert not verify_password("anything", "bcrypt$1$1$1$00$00")


def test_valid_signup_has_no_problem():
    assert signup_problem("Ada", "ada@example.com", "longenough") is None


def test_signup_needs_a_name():
    assert "name" in signup_problem("   ", "ada@example.com", "longenough")


def test_signup_rejects_malformed_email():
    assert "email" in signup_problem("Ada", "ada@example", "longenough")


def test_signup_rejects_short_password():
    assert "at least" in signup_problem("Ada", "ada@example.com", "short")


def test_signup_rejects_oversized_password():
    assert "at most" in signup_problem("Ada", "ada@example.com", "x" * (MAX_PASSWORD_LENGTH + 1))


def test_password_account_never_shares_a_google_identifier():
    google = google_user_identifier({"email": "Ada@Gmail.com", "verified_email": True})

    assert password_user_identifier(" Ada@Gmail.com ") != google
    assert password_user_identifier(" Ada@Gmail.com ") == "password:ada@gmail.com"


def test_google_user_without_a_verified_flag_is_rejected():
    assert google_user_identifier({"email": "ada@gmail.com", "name": "Ada"}) is None


def test_common_password_is_refused_at_signup():
    assert "too easy" in signup_problem("Ada", "ada@example.com", "Password123")


def test_single_repeated_character_password_is_refused():
    assert "too easy" in signup_problem("Ada", "ada@example.com", "zzzzzzzzzz")


def test_password_containing_the_email_name_is_refused():
    assert "too easy" in signup_problem("Grace", "hopper1906@example.com", "hopper1906!")


def test_password_containing_the_persons_name_is_refused():
    assert "too easy" in signup_problem("Ada Lovelace", "ada@example.com", "lovelace-rocks")


def test_uncommon_password_is_accepted():
    assert signup_problem("Ada Lovelace", "ada@example.com", "correct-horse-battery") is None
