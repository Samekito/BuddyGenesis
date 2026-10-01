from pathlib import Path

import pytest

from buddy.config import DEFAULT_DATABASE_URL, Settings, load_settings

POSTGRES_URL = "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require"


def _settings(**overrides):
    values = dict(
        groq_api_key="", groq_model="m", database_url=DEFAULT_DATABASE_URL, admin_username="admin",
        admin_password="s3cret!", google_oauth_enabled=False, knowledge_dir=Path("."),
    )
    return Settings(**{**values, **overrides})


def test_sqlite_on_render_refuses_to_start():
    with pytest.raises(ValueError, match="Postgres on Render"):
        _settings(on_render=True)


def test_postgres_on_render_is_accepted():
    assert _settings(on_render=True, database_url=POSTGRES_URL).on_render


def test_sqlite_is_fine_locally():
    assert _settings().database_url == DEFAULT_DATABASE_URL


def test_email_shaped_admin_username_is_refused():
    with pytest.raises(ValueError, match="must not be an email"):
        _settings(admin_username="admin@example.com")


def test_render_is_detected_from_its_environment_variable(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)

    assert load_settings().on_render


def test_public_url_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv("CHAINLIT_URL", "https://soc-buddy.onrender.com/")

    assert load_settings().public_url == "https://soc-buddy.onrender.com"


def test_brevo_key_needs_a_sender_address():
    with pytest.raises(ValueError, match="EMAIL_FROM"):
        _settings(brevo_api_key="key", public_url="http://localhost:8000")


def test_outbox_is_refused_on_render():
    with pytest.raises(ValueError, match="local development only"):
        _settings(on_render=True, database_url=POSTGRES_URL, email_outbox_dir=Path("outbox"))


def test_emailed_links_need_the_public_url_whenever_real_emails_go_out():
    with pytest.raises(ValueError, match="CHAINLIT_URL"):
        _settings(brevo_api_key="key", email_from="me@example.com")


def test_links_point_at_the_public_url_when_set():
    assert _settings(public_url="https://soc-buddy.onrender.com").app_url == "https://soc-buddy.onrender.com"


def test_links_point_at_the_local_server_by_default():
    assert _settings().app_url == "http://localhost:8000"


def test_email_is_off_without_brevo_or_outbox():
    assert not _settings().email_enabled
    assert _settings(brevo_api_key="key", email_from="me@example.com", public_url="http://localhost:8000").email_enabled
