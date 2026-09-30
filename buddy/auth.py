"""Login checks for the Chainlit auth callbacks in app.py.

Kept free of Chainlit imports so the rules can be unit-tested directly.
"""

import hmac

from buddy.config import Settings


def is_valid_password_login(username: str, password: str, settings: Settings) -> bool:
    # No configured credentials means password login is switched off — never fall back to a default.
    if not settings.admin_username or not settings.admin_password:
        return False
    # compare_digest avoids leaking how many leading characters matched via response timing.
    # Usernames ignore case ("Admin" == "admin"), like most logins; passwords never do.
    username_ok = hmac.compare_digest(username.strip().casefold().encode(), settings.admin_username.casefold().encode())
    password_ok = hmac.compare_digest(password.encode(), settings.admin_password.encode())
    return username_ok and password_ok


def google_user_identifier(raw_user_data: dict) -> str | None:
    """Returns the verified email to key the user's chat history on, or None to reject the login."""
    email = raw_user_data.get("email")
    # Chainlit fetches Google's userinfo/v2/me, which names the flag `verified_email`.
    if not email or raw_user_data.get("verified_email") is False:
        return None
    return email.lower()
