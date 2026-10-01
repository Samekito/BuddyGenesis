from buddy import emails


def test_confirmation_email_carries_the_link_in_text_and_html():
    email = emails.signup_confirmation("ada@example.com", "https://x.example/verify-email?token=abc", 24)

    assert email.to == "ada@example.com"
    assert "https://x.example/verify-email?token=abc" in email.text
    assert 'href="https://x.example/verify-email?token=abc"' in email.html
    assert "24 hours" in email.text


def test_names_are_escaped_in_html():
    email = emails.password_reset("ada@example.com", "<script>alert(1)</script>", "https://x.example/r", 60)

    assert "<script>" not in email.html and "&lt;script&gt;" in email.html


def test_already_registered_email_offers_sign_in_and_reset_but_no_token():
    email = emails.already_registered("ada@example.com", "Ada", "https://x.example/login", "https://x.example/forgot-password")

    assert "https://x.example/login" in email.text and "https://x.example/forgot-password" in email.text
    assert "token" not in email.text


def test_confirmation_email_greets_without_a_name():
    email = emails.signup_confirmation("ada@example.com", "https://x.example/verify-email?token=abc", 24)

    assert email.text.startswith("Hello,")
