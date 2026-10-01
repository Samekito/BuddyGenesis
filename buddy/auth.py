"""Login checks for the Chainlit auth callbacks in app.py.

Kept free of Chainlit imports so the rules can be unit-tested directly.
"""

import hashlib
import hmac
import re
import secrets

from buddy.config import Settings

MIN_PASSWORD_LENGTH = 8
# scrypt's cost does not grow with input length, but an unbounded body is still pointless to accept.
MAX_PASSWORD_LENGTH = 128
MAX_EMAIL_LENGTH = 254
MAX_NAME_LENGTH = 80
# Password accounts and Google logins prove an address in different ways (our emailed link vs
# Google), so they are kept apart: never sharing an identifier, and therefore never chat history.
PASSWORD_ACCOUNT_PREFIX = "password:"
# A deliberately loose shape check; only a sent email could prove the address is real.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# OWASP's scrypt baseline (N=2^17 needs 128 MB; 2^14 keeps each hash at 16 MB on the 512 MB host).
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
SCRYPT_SALT_BYTES = 16
SCRYPT_KEY_BYTES = 64
# The most-used passwords from public breach lists, plus this app's own names. A short local list,
# not an online breach lookup, so sign-up never sends anything about a password to a third party.
COMMON_PASSWORDS = frozenset(
    "password password1 password12 password123 passw0rd p@ssw0rd 12345678 123456789 1234567890 "
    "12341234 87654321 11111111 00000000 88888888 qwerty12 qwerty123 qwertyui qwertyuiop 1q2w3e4r "
    "1qaz2wsx asdfghjk asdf1234 zxcvbnm1 abc12345 abcd1234 iloveyou princess sunshine football "
    "baseball welcome1 welcome123 admin123 letmein1 trustno1 superman dragon123 monkey123 "
    "socbuddy socbuddy1 socbuddy123 futa1234 futaakure computing".split()
)


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
    # Only an explicit True counts: a missing flag must not be read as verified.
    if not email or raw_user_data.get("verified_email") is not True:
        return None
    return email.lower()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def signup_problem(name: str, email: str, password: str) -> str | None:
    """Returns a message to show the person signing up, or None when the details are acceptable."""
    if not name.strip():
        return "Please enter your name."
    if len(name.strip()) > MAX_NAME_LENGTH:
        return f"Your name must be at most {MAX_NAME_LENGTH} characters."
    email = normalize_email(email)
    if not is_plausible_email(email):
        return "Please enter a valid email address."
    return password_problem(password, name, email)


def password_problem(password: str, name: str, email: str) -> str | None:
    """The password rules alone, shared by sign-up and password reset. None means acceptable."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Your password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"Your password must be at most {MAX_PASSWORD_LENGTH} characters."
    if _is_guessable(password, name, normalize_email(email)):
        return "That password is too easy to guess. Please choose a less common one."
    return None


def is_plausible_email(email: str) -> bool:
    return len(email) <= MAX_EMAIL_LENGTH and bool(EMAIL_PATTERN.match(email))


def hash_password(password: str) -> str:
    """Returns a self-describing `scrypt$N$r$p$salt$key` string, so the cost can be raised later."""
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES)
    key = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, n, r, p, salt, key = stored_hash.split("$")
        if scheme != "scrypt":
            return False
        candidate = _scrypt(password, bytes.fromhex(salt), int(n), int(r), int(p))
        expected = bytes.fromhex(key)
    except ValueError:
        # A malformed stored hash must fail closed, never crash the login.
        return False
    return hmac.compare_digest(candidate, expected)


def password_user_identifier(email: str) -> str:
    return PASSWORD_ACCOUNT_PREFIX + normalize_email(email)


def _is_guessable(password: str, name: str, email: str) -> bool:
    lowered = password.casefold()
    personal = {email.split("@")[0], email, *name.casefold().split()}
    return (
        lowered in COMMON_PASSWORDS
        or len(set(lowered)) == 1
        or any(len(part) >= 3 and part in lowered for part in personal)
    )


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    # scrypt needs a little over 128 * N * r bytes; set maxmem so a raised N never hits OpenSSL's 32 MB default.
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=256 * n * r, dklen=SCRYPT_KEY_BYTES)
