"""HTTP hardening that Chainlit does not do itself, installed by app.py around Chainlit's app.

Adds security headers and Secure cookies, refuses cross-site posts to the sign-in/sign-up forms
(login CSRF), caps their body size and per-IP rate, and blocks the disabled file-upload route,
which Chainlit reads fully into memory before checking whether uploads are enabled. Also lets
the login callback answer with a real status (see refuse_sign_in).
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from buddy.rate_limit import RateLimiter, allow, client_ip

SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    # Only directives that cannot break Chainlit's bundle (which needs inline scripts and CDN fonts).
    (b"content-security-policy", b"frame-ancestors 'none'; base-uri 'self'; object-src 'none'; form-action 'self'"),
)
HSTS_HEADER = (b"strict-transport-security", b"max-age=31536000")
# Posts that change who is signed in or an account: a cross-site post to one of these is CSRF.
SAME_ORIGIN_POSTS = frozenset(
    {"/login", "/logout", "/auth/signup", "/auth/verify-email", "/auth/password-reset", "/auth/password-reset/confirm"}
)
BLOCKED_POSTS = frozenset({"/project/file"})
# A sign-in or sign-up form is well under 1 KB; anything bigger is not a real form.
MAX_FORM_BYTES = 16 * 1024
LOGIN_PATH = "/login"
RATE_LIMITED_REPLY = "Too many attempts. Please wait a few minutes and try again."

logger = logging.getLogger(__name__)


@dataclass
class _SignInRequest:
    client_ip: str
    status: int | None = None
    detail: str = ""


_sign_in_request: ContextVar[_SignInRequest | None] = ContextVar("sign_in_request", default=None)


def sign_in_client_ip() -> str | None:
    """The client address of the login being checked, for Chainlit's password callback (which gets no request)."""
    request = _sign_in_request.get()
    return request.client_ip if request else None


def refuse_sign_in(status: int, detail: str) -> None:
    """Called from Chainlit's password callback: answer this login with `status` and `detail`.

    Chainlit catches every exception the callback raises and always answers a refused login
    with a generic 401, so "too many attempts" or "database down" could not reach the user otherwise.
    Chainlit's login page shows only the status, which public/custom.js turns into a message.
    """
    request = _sign_in_request.get()
    if request is not None:
        request.status, request.detail = status, detail


class HttpGuard:
    def __init__(
        self,
        app: ASGIApp,
        trusted_origins: Iterable[str] = (),
        ip_limits: Mapping[str, Sequence[RateLimiter]] | None = None,
    ):
        self.app = app
        self.trusted_origins = frozenset(origin for origin in trusted_origins if origin)
        self.ip_limits = ip_limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]}
        is_https = scope.get("scheme") == "https" or headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"

        async def send_hardened(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": harden_headers(message.get("headers", []), is_https)}
            await send(message)

        refusal = self._refusal(scope, headers)
        if refusal is not None:
            await refusal(scope, receive, send_hardened)
            return
        if scope["method"] == "POST" and scope["path"] == LOGIN_PATH:
            await self._login(scope, receive, send_hardened, client_ip(headers.get("x-forwarded-for"), _peer(scope)))
            return
        await self.app(scope, receive, send_hardened)

    async def _login(self, scope: Scope, receive: Receive, send: Send, ip: str) -> None:
        sign_in = _SignInRequest(client_ip=ip)
        token = _sign_in_request.set(sign_in)
        replaced = False

        async def send_or_replace(message: Message) -> None:
            nonlocal replaced
            if message["type"] == "http.response.start" and sign_in.status is not None:
                replaced = True
                response = JSONResponse({"detail": sign_in.detail}, status_code=sign_in.status)
                await response(scope, receive, send)
            elif not replaced:
                await send(message)

        try:
            await self.app(scope, receive, send_or_replace)
        finally:
            _sign_in_request.reset(token)

    def _refusal(self, scope: Scope, headers: dict[str, str]) -> JSONResponse | None:
        if scope["method"] != "POST":
            return None
        path = scope["path"]
        if path in BLOCKED_POSTS:
            return JSONResponse({"detail": "File uploads are disabled."}, status_code=403)
        if path not in SAME_ORIGIN_POSTS:
            return None
        if not self._is_same_origin(headers):
            return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
        if not _is_small_form(headers):
            return JSONResponse({"detail": "Request too large."}, status_code=413)
        limiters = self.ip_limits.get(path, ())
        if limiters and not allow(client_ip(headers.get("x-forwarded-for"), _peer(scope)), limiters):
            logger.warning("Rate limit hit", extra={"event": "rate_limited", "path": path})
            retry_after = str(int(max(limiter.window_seconds for limiter in limiters)))
            return JSONResponse(
                {"detail": RATE_LIMITED_REPLY},
                status_code=429,
                headers={"retry-after": retry_after},
            )
        return None

    def _is_same_origin(self, headers: dict[str, str]) -> bool:
        origin = headers.get("origin")
        # Browsers send Origin on every POST; without one the caller is not a browser, so no CSRF.
        if origin is None:
            return True
        if origin in self.trusted_origins:
            return True
        return origin != "null" and urlsplit(origin).netloc == headers.get("host")


def harden_headers(raw_headers: Iterable[tuple[bytes, bytes]], is_https: bool) -> list[tuple[bytes, bytes]]:
    """Adds the security headers a response lacks and, over HTTPS, marks cookies Secure.

    Chainlit only sets Secure on its cookies when SameSite=None, so its login cookie would
    otherwise also be sent over plain HTTP.
    """
    headers = []
    for name, value in raw_headers:
        if is_https and name.lower() == b"set-cookie" and b"secure" not in value.lower():
            value += b"; Secure"
        headers.append((name, value))
    present = {name.lower() for name, _ in headers}
    extra = [*SECURITY_HEADERS, HSTS_HEADER] if is_https else list(SECURITY_HEADERS)
    return headers + [(name, value) for name, value in extra if name not in present]


def _is_small_form(headers: dict[str, str]) -> bool:
    # Only a declared length can be checked up front. A body without one (chunked, which a proxy
    # may produce) is let through: refusing it could lock everyone out behind such a proxy.
    length = headers.get("content-length")
    return length is None or (length.isdigit() and int(length) <= MAX_FORM_BYTES)


def _peer(scope: Scope) -> str | None:
    client = scope.get("client")
    return client[0] if client else None
