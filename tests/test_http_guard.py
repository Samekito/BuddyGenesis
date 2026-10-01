import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from buddy.http_guard import MAX_FORM_BYTES, HttpGuard, harden_headers, refuse_sign_in, sign_in_client_ip
from buddy.rate_limit import RateLimiter

HOST = "soc-buddy.example"


async def _ok(request):
    response = PlainTextResponse("ok")
    response.set_cookie("access_token", "jwt", httponly=True, samesite="lax")
    return response


async def _login(request):
    # Stands in for Chainlit's /login: its callback may call refuse_sign_in, then a generic 401 follows.
    form = await request.form()
    if form.get("username") == "who-am-i":
        return JSONResponse({"ip": sign_in_client_ip()})
    if form.get("username") == "limited":
        refuse_sign_in(429, "Too many sign-in attempts.")
    if form.get("password") == "right":
        return JSONResponse({"success": True})
    return JSONResponse({"detail": "credentialssignin"}, status_code=401)


def _client(**guard_options):
    app = Starlette(routes=[
        Route("/", _ok),
        Route("/login", _login, methods=["POST"]),
        Route("/auth/signup", _ok, methods=["POST"]),
        Route("/project/file", _ok, methods=["POST"]),
    ])
    return TestClient(HttpGuard(app, **guard_options), base_url=f"http://{HOST}")


def test_every_response_gets_the_security_headers():
    headers = _client().get("/").headers

    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["content-security-policy"]


def test_https_responses_get_hsts_and_secure_cookies():
    headers = _client().get("/", headers={"x-forwarded-proto": "https"}).headers

    assert headers["strict-transport-security"].startswith("max-age=")
    assert "Secure" in headers["set-cookie"]


def test_plain_http_responses_get_neither_hsts_nor_secure_cookies():
    headers = _client().get("/").headers

    assert "strict-transport-security" not in headers
    assert "Secure" not in headers["set-cookie"]


def test_existing_headers_are_not_duplicated():
    headers = harden_headers([(b"x-frame-options", b"SAMEORIGIN")], is_https=False)

    assert [value for name, value in headers if name == b"x-frame-options"] == [b"SAMEORIGIN"]


def test_file_upload_route_is_blocked():
    assert _client().post("/project/file", content=b"x" * 1024).status_code == 403


def test_cross_site_login_post_is_refused():
    response = _client().post("/login", data={"password": "right"}, headers={"origin": "https://evil.example"})

    assert response.status_code == 403


def test_same_origin_login_post_is_allowed():
    response = _client().post("/login", data={"password": "right"}, headers={"origin": f"http://{HOST}"})

    assert response.status_code == 200


def test_login_from_the_configured_public_url_is_allowed():
    client = _client(trusted_origins=["https://soc-buddy.onrender.com"])

    response = client.post("/login", data={"password": "right"}, headers={"origin": "https://soc-buddy.onrender.com"})

    assert response.status_code == 200


def test_opaque_null_origin_is_refused():
    assert _client().post("/login", data={"password": "right"}, headers={"origin": "null"}).status_code == 403


def test_post_without_origin_is_allowed_because_it_is_not_from_a_browser():
    assert _client().post("/login", data={"password": "right"}).status_code == 200


def test_oversized_auth_form_is_refused():
    assert _client().post("/auth/signup", content=b"x" * (MAX_FORM_BYTES + 1)).status_code == 413


@pytest.mark.parametrize("length", ["abc", "-1"])
def test_malformed_content_length_is_refused(length):
    response = _client().post("/auth/signup", content=b"{}", headers={"content-length": length})

    assert response.status_code == 413


def test_auth_posts_over_the_per_ip_limit_get_429_with_retry_after():
    client = _client(ip_limits={"/auth/signup": [RateLimiter(limit=2, window_seconds=3600)]})

    responses = [client.post("/auth/signup", headers={"x-forwarded-for": "203.0.113.9"}) for _ in range(3)]

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert responses[-1].headers["retry-after"] == "3600"


def test_per_ip_limit_does_not_affect_other_addresses():
    client = _client(ip_limits={"/auth/signup": [RateLimiter(limit=1, window_seconds=3600)]})
    client.post("/auth/signup", headers={"x-forwarded-for": "203.0.113.9"})

    assert client.post("/auth/signup", headers={"x-forwarded-for": "198.51.100.4"}).status_code == 200


def test_login_callback_can_replace_the_generic_401():
    response = _client().post("/login", data={"username": "limited", "password": "wrong"})

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many sign-in attempts."}
    assert response.headers["x-frame-options"] == "DENY"


def test_ordinary_failed_login_keeps_its_401():
    assert _client().post("/login", data={"username": "a", "password": "wrong"}).status_code == 401


def test_refuse_sign_in_outside_a_login_request_is_harmless():
    refuse_sign_in(429, "ignored")


def test_login_callback_can_see_the_client_address():
    response = _client().post("/login", data={"username": "who-am-i"}, headers={"x-forwarded-for": "203.0.113.9"})

    assert response.json() == {"ip": "203.0.113.9"}


def test_client_address_is_unknown_outside_a_login_request():
    assert sign_in_client_ip() is None


def test_auth_form_without_a_declared_length_is_let_through():
    def chunks():
        yield b"username=a&password=right"

    assert _client().post("/login", content=chunks(), headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 200
