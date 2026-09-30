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


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    database_url: str
    admin_username: str
    admin_password: str
    google_oauth_enabled: bool
    knowledge_dir: Path


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
    )
