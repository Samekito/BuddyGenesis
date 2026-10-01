"""Single source of runtime settings for SOC Buddy.

Every environment variable the app reads is read here (CLAUDE.md architecture rule 2).
Chainlit itself also reads CHAINLIT_AUTH_SECRET, CHAINLIT_URL and OAUTH_GOOGLE_* directly.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Largest model open to Groq free-tier accounts (checked 2026-09-30 via the models API);
# llama-3.3-70b-versatile is listed in the docs but not served to free accounts.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'chainlit.db').as_posix()}"
LOCAL_URL = "http://localhost:8000"
DEFAULT_EMAIL_FROM_NAME = "SOC Buddy"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    database_url: str
    admin_username: str
    admin_password: str
    google_oauth_enabled: bool
    knowledge_dir: Path
    # The app's public address (CHAINLIT_URL), e.g. https://soc-buddy.onrender.com; "" locally.
    public_url: str = ""
    on_render: bool = False
    # Sign-up confirmation and password-reset emails go out through Brevo's HTTP API (Render's
    # free tier blocks SMTP). Without a key, sign-up and password reset are switched off.
    brevo_api_key: str = ""
    email_from: str = ""
    email_from_name: str = DEFAULT_EMAIL_FROM_NAME
    # Local development only: write emails as files here instead of sending them.
    email_outbox_dir: Path | None = None

    def __post_init__(self):
        # Render's disk is wiped on spin-down: a forgotten DATABASE_URL would silently lose every
        # account and chat (architecture rule 6), so refuse to start instead.
        if self.on_render and self.database_url.startswith("sqlite"):
            raise ValueError("DATABASE_URL must point to Postgres on Render; SQLite files are wiped on spin-down.")
        # Google users are identified by email, so an email-shaped admin name would share the
        # admin's identifier and chat history with that Google account.
        if "@" in self.admin_username:
            raise ValueError("BUDDY_ADMIN_USERNAME must not be an email address.")
        if self.brevo_api_key and not self.email_from:
            raise ValueError("EMAIL_FROM must be set to the Brevo sender address when BREVO_API_KEY is set.")
        # Outbox files hold live sign-in links; they must never exist on a server.
        if self.on_render and self.email_outbox_dir is not None:
            raise ValueError("EMAIL_OUTBOX_DIR is for local development only; use BREVO_API_KEY on Render.")
        # Links in emails must point at the real site, never at whatever Host a request claimed,
        # nor at the local fallback address when real emails go out.
        if self.brevo_api_key and not self.public_url:
            raise ValueError("CHAINLIT_URL must be set when BREVO_API_KEY is, so emailed links point at the app.")

    @property
    def email_enabled(self) -> bool:
        return bool(self.brevo_api_key) or self.email_outbox_dir is not None

    @property
    def app_url(self) -> str:
        """Base for links in emails: the configured public address, or the local dev server."""
        return self.public_url or LOCAL_URL


def load_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
        database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        admin_username=os.getenv("BUDDY_ADMIN_USERNAME", ""),
        admin_password=os.getenv("BUDDY_ADMIN_PASSWORD", ""),
        google_oauth_enabled=bool(
            os.getenv("OAUTH_GOOGLE_CLIENT_ID") and os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")
        ),
        knowledge_dir=PROJECT_ROOT / "knowledge_base",
        public_url=os.getenv("CHAINLIT_URL", "").rstrip("/"),
        # Render sets RENDER=true in every service's environment.
        on_render=bool(os.getenv("RENDER")),
        brevo_api_key=os.getenv("BREVO_API_KEY", ""),
        email_from=os.getenv("EMAIL_FROM", "").strip(),
        email_from_name=os.getenv("EMAIL_FROM_NAME") or DEFAULT_EMAIL_FROM_NAME,
        email_outbox_dir=Path(os.environ["EMAIL_OUTBOX_DIR"]) if os.getenv("EMAIL_OUTBOX_DIR") else None,
    )
