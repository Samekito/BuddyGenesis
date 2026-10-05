"""The text of SOC Buddy's account emails, as plain text and simple HTML.

Consumed by buddy/accounts.py. Names are typed by users at sign-up, so they are HTML-escaped. The
confirmation email greets nobody by name: it goes to an address nobody has proven yet, so a
typed name would let strangers put their own words into our email to any inbox.
"""

from html import escape

from buddy.mailer import Email

APP_NAME = "SOC Buddy"


def signup_confirmation(to: str, link: str, valid_hours: int) -> Email:
    return _email(
        to,
        subject=f"Confirm your {APP_NAME} account",
        greeting_name=None,
        paragraphs=[
            f"Confirm your email address to finish creating your {APP_NAME} account. You will be asked for the password you chose.",
            f"This link works once and expires in {valid_hours} hours.",
        ],
        link=link,
        link_label="Confirm my email",
        footer="If you did not sign up, ignore this email and no account will be created.",
    )


def already_registered(to: str, name: str, sign_in_link: str, reset_link: str) -> Email:
    return _email(
        to,
        subject=f"You already have a {APP_NAME} account",
        greeting_name=name,
        paragraphs=[
            f"Someone tried to create a {APP_NAME} account with this email address, but it already has one.",
            f"If that was you, sign in instead: {sign_in_link}",
            "If you have forgotten your password, you can reset it with the button below.",
        ],
        link=reset_link,
        link_label="Reset my password",
        footer="If this was not you, ignore this email. Your account has not been changed.",
    )


def password_reset(to: str, name: str, link: str, valid_minutes: int) -> Email:
    return _email(
        to,
        subject=f"Reset your {APP_NAME} password",
        greeting_name=name,
        paragraphs=[
            f"Use the button below to choose a new {APP_NAME} password.",
            f"This link works once and expires in {valid_minutes} minutes.",
        ],
        link=link,
        link_label="Choose a new password",
        footer=(
            "If you did not ask for this, ignore this email. Your password has not been changed. "
            "Changing it does not sign out devices already signed in; those sign-ins end within a day."
        ),
    )


def google_account_notice(to: str, sign_in_link: str) -> Email:
    return _email(
        to,
        subject=f"How to sign in to {APP_NAME}",
        greeting_name=None,
        paragraphs=[
            f"Someone asked to reset the {APP_NAME} password for this email address, but this address signs in with Google, so there is no {APP_NAME} password to reset.",
            "Use \"Continue with Google\" on the sign-in page instead. If you cannot get into your Google account, use Google's own account recovery.",
        ],
        link=sign_in_link,
        link_label="Go to sign in",
        footer="If you did not ask for this, ignore this email. Nothing has been changed.",
    )


def _email(
    to: str, subject: str, greeting_name: str | None, paragraphs: list[str], link: str, link_label: str, footer: str
) -> Email:
    greeting = f"Hello {greeting_name}," if greeting_name else "Hello,"
    text = "\n\n".join([greeting, *paragraphs, f"{link_label}: {link}", footer, f"— {APP_NAME}"])
    body = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
    html = (
        f'<div style="font-family:system-ui,sans-serif;font-size:15px;line-height:1.6;color:#2b1f08;max-width:520px">'
        f"<p>{escape(greeting)}</p>{body}"
        f'<p><a href="{escape(link)}" style="display:inline-block;padding:10px 18px;border-radius:6px;'
        f'background:#ffa000;color:#2b1f08;font-weight:600;text-decoration:none">{escape(link_label)}</a></p>'
        f'<p style="font-size:13px;color:#6b5a3e">If the button does not work, open this link: {escape(link)}</p>'
        f'<p style="font-size:13px;color:#6b5a3e">{escape(footer)}</p>'
        f"<p>— {APP_NAME}</p></div>"
    )
    return Email(to=to, subject=subject, text=text, html=html)
